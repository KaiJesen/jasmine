#ifndef __JAS_CONV_T_HPP__
#define __JAS_CONV_T_HPP__

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_mat_storage.hpp"
#include "jas_updator_t.hpp"

namespace jasmine {

/**
 * 2-D convolution layer with a complete backward pass (dL/dInput, dL/dWeight, dL/dBias).
 *
 * Why "im2col + GEMM": jasmine's only data carrier is the 2-D `mat_t`; there are no NCHW/NHWC
 * tensors. The layer therefore fixes this convention:
 *
 *     input  x : [C_in,  H * W]            channels first, spatial dims flattened into columns
 *     kernel W : [C_out, C_in * Kh * Kw]   one row per output channel, flattened as (c, kh, kw)
 *     bias   b : [C_out, 1]                broadcast over all output positions
 *     output y : [C_out, H_out * W_out]
 *
 * That is exactly `weight_net_t`'s shape convention (weight [out, in], input [in, T]); only "in" has
 * been expanded by im2col into `C_in*Kh*Kw` and "T" into `H_out*W_out`. The forward pass is then a
 * single `W.dot(col)` (large sizes go straight to BLAS) and the backward pass is two transposed
 * GEMMs plus one col2im scatter -- no new matrix multiply has to be written for this layer, and all
 * of the convolution's complexity lives in the im2col/col2im pair of loops.
 *
 * Note that the bias must be added in **two steps** (`dot`, then `+=`): writing `W.dot(col) + b`
 * pushes the whole multiply back onto a naive triple loop and loses BLAS; see the comment inside
 * `forward` and TESTING.md section 11 for the measurements.
 *
 * 1-D convolution (e.g. over a sequence) is the special case H = 1, Kh = 1, pad_h = 0: feed a
 * sequence of length L as [C_in, 1*L] and neither the shape checks nor the backward logic needs a
 * branch. `one_d(...)` is just a shorthand for that parameter set; it returns an instance of this
 * same class rather than a second implementation.
 *
 * Output sizes (same formula as PyTorch's `Conv2d`):
 *
 *     H_out = (H + 2 * pad_h - dil_h * (Kh - 1) - 1) / stride_h + 1
 *     W_out = (W + 2 * pad_w - dil_w * (Kw - 1) - 1) / stride_w + 1
 *
 * Where the shape comes from: H/W and the kernel size can be neither inferred from `[C_in, H*W]`
 * (H and W are interchangeable) nor derived from C_in alone, so the layer takes them explicitly via
 * `set_param(...)` and the constructor's parameter list matches it exactly. It is deliberately not
 * called `reinit`: `is_reinitable_net` is defined by `requires { net.reinit(std::vector<int>()); }`
 * (see jas_mat_concepts.hpp), and providing one would change the reinit semantics of every complex
 * network containing this layer -- the same reasoning as `layer_norm_net_t`'s `set_param`.
 *
 * Backward cache: forward stores the im2col result in `m_col` ([C_in*Kh*Kw, H_out*W_out]) and
 * backward reuses it. Recomputing im2col during backward would save that memory but pay the im2col
 * traffic twice; this follows the existing layers in the repository (gelu caches its input, mha its
 * projections) and trades space for time.
 */
template <typename input_type, template <typename> class updator_type>
class conv2d_net_t
{
public:
    using val_type = typename input_type::ele_type;

private:
    int m_c_in = 0;
    int m_c_out = 0;
    int m_h = 0;            // input spatial height (1 for 1-D convolution)
    int m_w = 0;            // input spatial width (the sequence length for 1-D convolution)
    int m_kh = 0;
    int m_kw = 0;
    int m_stride_h = 1;
    int m_stride_w = 1;
    int m_pad_h = 0;
    int m_pad_w = 0;
    int m_dil_h = 1;
    int m_dil_w = 1;
    int m_h_out = 0;
    int m_w_out = 0;

    mat_t<val_type> m_weight;       // [C_out, C_in * Kh * Kw]
    mat_t<val_type> m_bias;         // [C_out, 1]
    updator_type<val_type> m_weight_updator;
    updator_type<val_type> m_bias_updator;

    mat_t<val_type> m_col;          // generic path: cached im2col [C_in * Kh * Kw, H_out * W_out]
    mat_t<val_type> m_input;        // zero-copy path: cached input (used to rebuild the window view)

    static int output_size(int const in, int const k, int const stride, int const pad, int const dil)
    {
        return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    }

    /**
     * Parameter checks that do not depend on the output size. They must run before output_size(),
     * which divides by `stride`: with stride == 0 that is not an exception but a division by zero
     * (integer division).
     */
    static void validate_basic(int const c_in, int const c_out, int const h, int const w,
                               int const kh, int const kw,
                               int const stride_h, int const stride_w,
                               int const pad_h, int const pad_w,
                               int const dil_h, int const dil_w)
    {
        if (c_in < 1 || c_out < 1 || h < 1 || w < 1 || kh < 1 || kw < 1)
            throw std::invalid_argument("conv2d_net_t: channels, spatial size and kernel size must be >= 1");
        if (stride_h < 1 || stride_w < 1 || dil_h < 1 || dil_w < 1)
            throw std::invalid_argument("conv2d_net_t: stride and dilation must be >= 1");
        if (pad_h < 0 || pad_w < 0)
            throw std::invalid_argument("conv2d_net_t: padding must be >= 0");
    }

    /**
     * Whether the im2col matrix can be expressed **without copying** as "a reinterpretation of the
     * input itself, transposed".
     *
     * The address identity that has to hold (flat col index == input memory index) is
     *     (oh*W_out + ow)*K + (c*Kh + i)*Kw + j  ==  c*H*W + (oh*sh - ph + i*dh)*W + (ow*sw - pw + j*dw)
     * Comparing term by term, it only holds when the row direction is not convolved at all and the
     * layer has a single channel:
     *   C_in == 1, Kh == 1, stride_h == 1, pad_h == 0, dilation_h == 1 (H is carried through)
     *   and dilation_w == 1, pad_w == 0, stride_w == Kw, W == W_out*Kw
     * Then col = input.reshape_view(H_out*W_out, Kw).t() and no copy is needed at all. That is
     * exactly the geometry of patchify (ViT patch embedding) and of non-overlapping 1-D convolution.
     */
    bool col_as_view() const
    {
        if (m_c_in != 1 || m_kh != 1 || m_stride_h != 1 || m_pad_h != 0 || m_dil_h != 1
            || m_dil_w != 1 || m_pad_w != 0 || m_stride_w != m_kw)
            return false;
        // Single row (H == 1): the window sequence is a **prefix** of the input, of length
        // w_out*Kw <= input length; the remaining tail (the partial window that floor semantics
        // drops) falls outside the view and does not matter.
        // Multiple rows: every row must divide evenly (W == W_out*Kw), otherwise the addresses from
        // the second row on are misaligned.
        return m_h == 1 || m_w == m_w_out * m_kw;
    }

    /**
     * im2col: flatten the (c, kh, kw) receptive field of every output position into a column.
     * Out-of-range padding positions are written as 0; no padded copy of the input is materialised.
     */
    template <typename Src>
    void im2col(Src const& input, mat_t<val_type>& col) const
    {
        const int k = m_c_in * m_kh * m_kw;
        const int n = m_h_out * m_w_out;
        // Same as col2im: avoid reshape(1,1) on an empty matrix, which computes `% 0`
        // (the degenerate convolution where k == n == 1)
        if (col.row_num() != k || col.col_num() != n)
            col = mat_t<val_type>(k, n);
        for (int c = 0; c < m_c_in; ++c)
        {
            for (int kh = 0; kh < m_kh; ++kh)
            {
                for (int kw = 0; kw < m_kw; ++kw)
                {
                    const int r = (c * m_kh + kh) * m_kw + kw;
                    for (int oh = 0; oh < m_h_out; ++oh)
                    {
                        const int ih = oh * m_stride_h - m_pad_h + kh * m_dil_h;
                        for (int ow = 0; ow < m_w_out; ++ow)
                        {
                            const int iw = ow * m_stride_w - m_pad_w + kw * m_dil_w;
                            const bool inside = (ih >= 0 && ih < m_h && iw >= 0 && iw < m_w);
                            col(r, oh * m_w_out + ow) = inside
                                ? static_cast<val_type>(input(c, ih * m_w + iw))
                                : val_type(0);
                        }
                    }
                }
            }
        }
    }

    /** col2im: scatter-add dL/dcol back into dL/dx using the same receptive fields (padding is dropped) */
    void col2im(mat_t<val_type> const& dcol, mat_t<val_type>& dx) const
    {
        if (dx.row_num() != m_c_in || dx.col_num() != m_h * m_w)
            dx = mat_t<val_type>(m_c_in, m_h * m_w);
        dx = val_type(0);               // the branch above does not zero when the size matches, so redo it
        for (int c = 0; c < m_c_in; ++c)
        {
            for (int kh = 0; kh < m_kh; ++kh)
            {
                for (int kw = 0; kw < m_kw; ++kw)
                {
                    const int r = (c * m_kh + kh) * m_kw + kw;
                    for (int oh = 0; oh < m_h_out; ++oh)
                    {
                        const int ih = oh * m_stride_h - m_pad_h + kh * m_dil_h;
                        for (int ow = 0; ow < m_w_out; ++ow)
                        {
                            const int iw = ow * m_stride_w - m_pad_w + kw * m_dil_w;
                            if (ih >= 0 && ih < m_h && iw >= 0 && iw < m_w)
                                dx(c, ih * m_w + iw) += dcol(r, oh * m_w_out + ow);
                        }
                    }
                }
            }
        }
    }

public:
    conv2d_net_t() = default;

    conv2d_net_t(int const c_in, int const c_out, int const h, int const w,
                 int const kh, int const kw,
                 int const stride_h = 1, int const stride_w = 1,
                 int const pad_h = 0, int const pad_w = 0,
                 int const dil_h = 1, int const dil_w = 1)
    {
        set_param(c_in, c_out, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);
    }

    /**
     * Convenience entry point for 1-D convolution: still the same 2-D implementation, with the
     * sequence of length L placed along the width (H = 1, Kh = 1, pad_h = 0) and no second code path.
     *
     *     auto conv = conv_t::one_d(c_in, c_out, L, kernel, stride, pad, dilation);
     *     auto y = conv.forward(x);        // x is [C_in, L], y is [C_out, L_out]
     */
    static conv2d_net_t one_d(int const c_in, int const c_out, int const len, int const k,
                              int const stride = 1, int const pad = 0, int const dilation = 1)
    {
        return conv2d_net_t(c_in, c_out, /*h=*/1, /*w=*/len, /*kh=*/1, /*kw=*/k,
                            /*stride_h=*/1, /*stride_w=*/stride, /*pad_h=*/0, /*pad_w=*/pad,
                            /*dil_h=*/1, /*dil_w=*/dilation);
    }

    /**
     * Configure the shapes and allocate weight/bias (both zeroed; fill them later with
     * init_weight<init_type>()). As with `layer_norm_net_t::set_param`, a weight loader needs to
     * write weight()/bias() before the first forward, so this must be called first -- otherwise
     * those accessors return unallocated empty matrices.
     */
    void set_param(int const c_in, int const c_out, int const h, int const w,
                   int const kh, int const kw,
                   int const stride_h = 1, int const stride_w = 1,
                   int const pad_h = 0, int const pad_w = 0,
                   int const dil_h = 1, int const dil_w = 1)
    {
        validate_basic(c_in, c_out, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);
        const int h_out = output_size(h, kh, stride_h, pad_h, dil_h);
        const int w_out = output_size(w, kw, stride_w, pad_w, dil_w);
        if (h_out < 1 || w_out < 1)
            throw std::invalid_argument("conv2d_net_t: kernel larger than padded input, output would be empty");

        m_c_in = c_in;
        m_c_out = c_out;
        m_h = h;
        m_w = w;
        m_kh = kh;
        m_kw = kw;
        m_stride_h = stride_h;
        m_stride_w = stride_w;
        m_pad_h = pad_h;
        m_pad_w = pad_w;
        m_dil_h = dil_h;
        m_dil_w = dil_w;
        m_h_out = h_out;
        m_w_out = w_out;

        // Assign a fresh matrix wholesale instead of calling reshape: with C_out == 1 the bias is
        // 1x1, and mat_t::reshape(1,1) on a default-constructed empty matrix first reads
        // (*this)(0,0) -- internally `0 % row_num()`, i.e. `% 0` -- before converting to a scalar.
        // A single output channel is a common configuration, so this trap must be avoided.
        m_weight = mat_t<val_type>(c_out, c_in * kh * kw);
        m_bias = mat_t<val_type>(c_out, 1);
        m_col = mat_t<val_type>(c_in * kh * kw, h_out * w_out);
    }

    int c_in() const { return m_c_in; }
    int c_out() const { return m_c_out; }
    int in_h() const { return m_h; }
    int in_w() const { return m_w; }
    int kernel_h() const { return m_kh; }
    int kernel_w() const { return m_kw; }
    int stride_h() const { return m_stride_h; }
    int stride_w() const { return m_stride_w; }
    int pad_h() const { return m_pad_h; }
    int pad_w() const { return m_pad_w; }
    int dilation_h() const { return m_dil_h; }
    int dilation_w() const { return m_dil_w; }
    int out_h() const { return m_h_out; }
    int out_w() const { return m_w_out; }

    /** Whether forward() takes the zero-copy window-view path for this configuration (tests, diagnostics) */
    bool col_view_enabled() const { return col_as_view(); }

    /** weight [C_out, C_in*Kh*Kw]; written directly by weight loaders */
    mat_t<val_type>& weight() { return m_weight; }
    mat_t<val_type> const& weight() const { return m_weight; }
    /** bias [C_out, 1]; written directly by weight loaders or zeroed */
    mat_t<val_type>& bias() { return m_bias; }
    mat_t<val_type> const& bias() const { return m_bias; }

    template <typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        if (m_weight.valid() == false)
            throw std::runtime_error("conv2d_net_t::forward: layer is not configured, call set_param() first");
        if (input.row_num() != m_c_in || input.col_num() != m_h * m_w)
            throw std::invalid_argument("conv2d_net_t::forward: input must be [C_in, H*W]");

        if (col_as_view())
        {
            // The window matrix is a reinterpretation of the input itself: keep the input
            // (to rebuild the view during backward) instead of materialising col
            detail::store_for_backward(m_input, std::forward<Src>(input));
            const int n_pos = m_h_out * m_w_out;
            // Take the first n_pos*Kw elements (a single-row slice, hence dense) and reinterpret
            // them as (N x K): no copy anywhere in this path
            auto used = m_input.view(0, 0, 1, n_pos * m_kw);
            auto windows = used.reshape_view(n_pos, m_kw);                  // (N x K), zero copy
            mat_t<val_type> out = m_weight.dot(windows.t());                // bare dot -> BLAS
            out += m_bias;
            return out;
        }

        im2col(input, m_col);
        // It has to be two steps -- a bare dot, then the bias -- and never
        // `m_weight.dot(m_col) + m_bias`: mat_add_t::clone() evaluates element by element, calling
        // mat_dot_t::operator() for every (i,j), so the whole multiply falls back to a naive loop
        // where each output element scans K on its own and **try_fast_gemm / BLAS is never reached**.
        // Measured at M=64, N=1024, K=288: 98.9 ms as one expression, 7.5 ms as two statements plus
        // a broadcast add (~0.2 ms). The semantics are identical; see TESTING.md section 11.
        mat_t<val_type> out = m_weight.dot(m_col);
        out += m_bias;
        return out;
    }

    /** Stateless layer (no recurrence over space/time): a single column and a whole input share one path */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    /**
     * delta = dL/dy with shape [C_out, H_out*W_out]; returns dL/dx with shape [C_in, H*W].
     * The weight/bias gradients are handed to the updators inside the layer, matching weight_net_t.
     */
    template <typename other_type>
    mat_t<val_type> backward(other_type const& delta)
    {
        if (m_weight.valid() == false)
            throw std::runtime_error("conv2d_net_t::backward: layer is not configured, call set_param() first");
        if (delta.row_num() != m_c_out || delta.col_num() != m_h_out * m_w_out)
            throw std::runtime_error("conv2d_net_t::backward: delta shape mismatch");

        // ∂L/∂W = delta · colᵀ        [C_out, N] · [N, K] → [C_out, K]
        // On the zero-copy path, col^T *is* that (N x K) window view -- nothing to materialise
        mat_t<val_type> delta_weight;
        if (col_as_view())
        {
            const int n_pos = m_h_out * m_w_out;
            auto used = m_input.view(0, 0, 1, n_pos * m_kw);
            delta_weight = delta.dot(used.reshape_view(n_pos, m_kw));
        }
        else
            delta_weight = delta.dot(m_col.t());
        // dL/db = sum over the output positions   [C_out, 1]
        mat_t<val_type> const delta_bias = hsum(delta);
        // ∂L/∂col = Wᵀ · delta       [K, C_out] · [C_out, N] → [K, N]
        mat_t<val_type> const delta_col = m_weight.t().dot(delta);

        mat_t<val_type> delta_input;
        col2im(delta_col, delta_input);

        m_weight_updator.update(delta_weight, m_weight);
        m_bias_updator.update(delta_bias, m_bias);
        return delta_input;
    }

    template <typename init_type>
    void init_weight()
    {
        init_matrix<init_type>(m_weight);
        init_matrix<init_type>(m_bias);
    }

    template <typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_weight_updator.set(std::forward<upr_arg_types>(args)...);
        m_bias_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_weight_updator.set_lr(lr);
        m_bias_updator.set_lr(lr);
    }

    void step()
    {
        m_weight_updator.step();
        m_bias_updator.step();
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "conv2d_net_t:(in:" << m_c_in << "x" << m_h << "x" << m_w
           << ", out:" << m_c_out << "x" << m_h_out << "x" << m_w_out
           << ", kernel:" << m_kh << "x" << m_kw
           << ", stride:" << m_stride_h << "x" << m_stride_w
           << ", pad:" << m_pad_h << "x" << m_pad_w
           << ", dilation:" << m_dil_h << "x" << m_dil_w << ")";
        return ss.str();
    }
};

} // namespace jasmine
#endif
