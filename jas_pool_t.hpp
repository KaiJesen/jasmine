#ifndef __JAS_POOL_T_HPP__
#define __JAS_POOL_T_HPP__

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_mat_express_t.hpp"

namespace jasmine {

/**
 * Pooling modes.
 *
 * - max    : take the maximum inside the window; padding counts as -inf; ties route their gradient
 *            to the first element (see below).
 * - average: sum the window and divide by a divisor; whether that divisor is kh*kw or the number of
 *            valid elements is controlled by count_include_pad.
 */
enum class pool_mode
{
    max = 0,
    average = 1
};

inline const char* pool_mode_name(pool_mode mode)
{
    return mode == pool_mode::average ? "average" : "max";
}

/**
 * 2-D pooling layer (max / average) with a complete backward pass (dL/dInput).
 *
 * The shape convention matches `conv2d_net_t`:
 *
 *     input  x : [C, H * W]        channels first, spatial dims flattened into columns
 *     output y : [C, H_out * W_out]
 *
 * Pooling has **no trainable parameters** and is independent per channel (the output channel count
 * equals the input channel count), so this is a "static layer": like `relu_net_t` / `gelu_net_t` it
 * only takes `input_type`, holds no updator, and implements `init_weight<init_type>()` / `step()` as
 * no-ops (`complex_net_t` calls them unconditionally). Two consequences: `is_updatable_net` is false
 * (the chain's `set_updator` skips this layer), and because it has no `reinit` (its shape comes from
 * `set_param`) it takes no slot in `complex_net_t::reinit`.
 *
 * Output sizes (floor mode, same formula as PyTorch's `MaxPool2d` / `AvgPool2d`):
 *
 *     H_out = (H + 2 * pad_h - Kh) / stride_h + 1
 *     W_out = (W + 2 * pad_w - Kw) / stride_w + 1
 *
 * Three semantics deliberately aligned with PyTorch (each pinned by a unit test):
 *
 * 1. **stride defaults to the kernel size**: passing 0 for `stride_h`/`stride_w` means "same as the
 *    kernel" (PyTorch's `MaxPool2d(2)` is `stride=2`). 0 is never a legal stride elsewhere, so the
 *    sentinel is unambiguous.
 * 2. **average pooling defaults to count_include_pad = true**: the divisor is always `Kh*Kw`, so the
 *    padding zeros enter the denominator as well (PyTorch's `nn.AvgPool2d` default). Setting it to
 *    false divides by the number of **valid** elements instead, which makes the divisor smaller for
 *    border windows and the output larger.
 * 3. **ties go to the first maximum** (in (kh, kw) row-major order); the gradient follows that single
 *    position rather than being split evenly. PyTorch's CPU kernel does the same: with
 *    `x = [[1,1],[1,1]]`, `max_pool2d(2)` gives `x.grad == [1,0,0,0]`, not `[0.25,...]`.
 *
 * Why there is no dilation / ceil_mode: average pooling has no dilation in PyTorch at all, and
 * ceil_mode introduces "is the last window complete" rules that interact with count_include_pad.
 * Both can be added on demand; the common floor + whole-kernel case is done properly first.
 * `pad <= Kh/2` (PyTorch's "pad should be at most half of effective kernel size") is a hard
 * constraint, and it also guarantees that every window covers at least one real element, so max
 * pooling never faces an all-(-inf) window and average pooling never divides by zero.
 *
 * Backward cache: max pooling records the argmax of every output element during forward (the
 * flattened input index, kept in an int vector rather than a float matrix to avoid precision issues
 * on large maps); average pooling's divisor is a function of the output position and is recomputed
 * during backward, so it needs no cache. Both cache `m_channels` to validate delta's row count.
 */
template <typename input_type>
class pool2d_net_t
{
public:
    using val_type = typename input_type::ele_type;

private:
    pool_mode m_mode = pool_mode::max;
    int m_h = 0;                // input spatial height (1 for 1-D pooling)
    int m_w = 0;                // input spatial width (the sequence length for 1-D pooling)
    int m_kh = 0;
    int m_kw = 0;
    int m_stride_h = 0;
    int m_stride_w = 0;
    int m_pad_h = 0;
    int m_pad_w = 0;
    int m_h_out = 0;
    int m_w_out = 0;
    bool m_count_include_pad = true;

    int m_channels = 0;             // cached by forward; backward validates delta's rows against it
    std::vector<int> m_argmax;      // max mode only: flattened input index of every output element
    std::vector<int> m_divisor;     // average mode only: divisor per output position (precomputed)

    static int output_size(int const in, int const k, int const stride, int const pad)
    {
        return (in + 2 * pad - k) / stride + 1;
    }

    /**
     * Parameter checks that do not depend on the output size; they must run before output_size(),
     * which divides by `stride` -- with stride == 0 that is not an exception but a division by zero
     * (integer division).
     */
    static void validate_basic(int const h, int const w, int const kh, int const kw,
                               int const stride_h, int const stride_w,
                               int const pad_h, int const pad_w)
    {
        if (h < 1 || w < 1 || kh < 1 || kw < 1)
            throw std::invalid_argument("pool2d_net_t: spatial size and kernel size must be >= 1");
        if (stride_h < 1 || stride_w < 1)
            throw std::invalid_argument("pool2d_net_t: stride must be >= 1 (or 0 meaning 'same as kernel')");
        if (pad_h < 0 || pad_w < 0)
            throw std::invalid_argument("pool2d_net_t: padding must be >= 0");
        if (2 * pad_h > kh || 2 * pad_w > kw)
            throw std::invalid_argument("pool2d_net_t: pad should be at most half of the kernel size");
    }

    /** Window walk: map element (i, j) of output position (oh, ow) to input coordinates; false if out of range */
    bool window_at(int const oh, int const ow, int const i, int const j,
                   int& ih, int& iw) const
    {
        ih = oh * m_stride_h - m_pad_h + i;
        iw = ow * m_stride_w - m_pad_w + j;
        return (ih >= 0 && ih < m_h && iw >= 0 && iw < m_w);
    }

    /**
     * Divisor used by average pooling: always the window area when include_pad is set, otherwise the
     * number of valid elements in this window. Both cases are precomputed into an (H_out x W_out)
     * table in set_param -- the divisor only depends on the output position, so recomputing it in
     * O(Kh*Kw) for every output element of forward/backward would be pointless.
     */
    int avg_divisor(int const oh, int const ow) const
    {
        return m_divisor[static_cast<std::size_t>(oh) * m_w_out + ow];
    }

public:
    pool2d_net_t() = default;

    pool2d_net_t(pool_mode const mode, int const h, int const w, int const kh, int const kw,
                 int const stride_h = 0, int const stride_w = 0,
                 int const pad_h = 0, int const pad_w = 0,
                 bool const count_include_pad = true)
    {
        set_param(mode, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, count_include_pad);
    }

    /**
     * Configure shapes and windows; this layer has no weights, so it only records and validates and
     * allocates nothing. Passing 0 for `stride_h`/`stride_w` means "same as the kernel in that
     * direction". When it throws, the object keeps its previous state (locals are validated before
     * any member is written).
     */
    void set_param(pool_mode const mode, int const h, int const w, int const kh, int const kw,
                   int const stride_h = 0, int const stride_w = 0,
                   int const pad_h = 0, int const pad_w = 0,
                   bool const count_include_pad = true)
    {
        const int sh = (stride_h == 0) ? kh : stride_h;
        const int sw = (stride_w == 0) ? kw : stride_w;
        validate_basic(h, w, kh, kw, sh, sw, pad_h, pad_w);
        const int h_out = output_size(h, kh, sh, pad_h);
        const int w_out = output_size(w, kw, sw, pad_w);
        if (h_out < 1 || w_out < 1)
            throw std::invalid_argument("pool2d_net_t: kernel larger than padded input, output would be empty");

        m_mode = mode;
        m_h = h;
        m_w = w;
        m_kh = kh;
        m_kw = kw;
        m_stride_h = sh;
        m_stride_w = sw;
        m_pad_h = pad_h;
        m_pad_w = pad_w;
        m_h_out = h_out;
        m_w_out = w_out;
        m_count_include_pad = count_include_pad;
        m_channels = 0;
        m_argmax.clear();

        // Precompute the divisor of every output position (only average mode needs it)
        m_divisor.clear();
        if (m_mode == pool_mode::average)
        {
            m_divisor.resize(static_cast<std::size_t>(m_h_out) * m_w_out);
            for (int oh = 0; oh < m_h_out; ++oh)
            {
                for (int ow = 0; ow < m_w_out; ++ow)
                {
                    if (m_count_include_pad)
                    {
                        m_divisor[static_cast<std::size_t>(oh) * m_w_out + ow] = m_kh * m_kw;
                    }
                    else
                    {
                        int valid = 0;
                        for (int i = 0; i < m_kh; ++i)
                            for (int j = 0; j < m_kw; ++j)
                            {
                                int ih = 0, iw = 0;
                                if (window_at(oh, ow, i, j, ih, iw))
                                    ++valid;
                            }
                        m_divisor[static_cast<std::size_t>(oh) * m_w_out + ow] = valid;
                    }
                }
            }
        }
    }

    /**
     * Convenience entry point for 1-D pooling: still the same 2-D implementation, with the sequence
     * of length L along the width (H = 1, Kh = 1, pad_h = 0) and no second code path.
     *
     *     auto pool = pool_t::one_d(pool_mode::max, L, 2);   // k=2, stride defaults to k
     *     auto y = pool.forward(x);        // x is [C, L], y is [C, L_out]
     */
    static pool2d_net_t one_d(pool_mode const mode, int const len, int const k,
                              int const stride = 0, int const pad = 0,
                              bool const count_include_pad = true)
    {
        return pool2d_net_t(mode, /*h=*/1, /*w=*/len, /*kh=*/1, /*kw=*/k,
                            /*stride_h=*/1, /*stride_w=*/stride, /*pad_h=*/0, /*pad_w=*/pad,
                            count_include_pad);
    }

    pool_mode mode() const { return m_mode; }
    int in_h() const { return m_h; }
    int in_w() const { return m_w; }
    int kernel_h() const { return m_kh; }
    int kernel_w() const { return m_kw; }
    int stride_h() const { return m_stride_h; }
    int stride_w() const { return m_stride_w; }
    int pad_h() const { return m_pad_h; }
    int pad_w() const { return m_pad_w; }
    int out_h() const { return m_h_out; }
    int out_w() const { return m_w_out; }
    bool count_include_pad() const { return m_count_include_pad; }

    template <typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        if (m_kh < 1)
            throw std::runtime_error("pool2d_net_t::forward: layer is not configured, call set_param() first");
        if (input.row_num() < 1 || input.col_num() != m_h * m_w)
            throw std::invalid_argument("pool2d_net_t::forward: input must be [C, H*W]");

        const int c = input.row_num();
        const int n = m_h_out * m_w_out;
        m_channels = c;
        if (m_mode == pool_mode::max)
            m_argmax.assign(static_cast<std::size_t>(c) * n, -1);

        mat_t<val_type> out(c, n);
        for (int ch = 0; ch < c; ++ch)
        {
            for (int oh = 0; oh < m_h_out; ++oh)
            {
                for (int ow = 0; ow < m_w_out; ++ow)
                {
                    const int col = oh * m_w_out + ow;
                    if (m_mode == pool_mode::max)
                    {
                        // Strict `>`: ties keep the one seen first (row-major), matching PyTorch.
                        // The -inf seed only gives an all-padding window a defined value; the
                        // pad <= k/2 constraint rules that case out (and if it ever happened, the
                        // idx < 0 check below would stop it before backward).
                        val_type best = -std::numeric_limits<val_type>::infinity();
                        int best_idx = -1;
                        for (int i = 0; i < m_kh; ++i)
                        {
                            for (int j = 0; j < m_kw; ++j)
                            {
                                int ih = 0, iw = 0;
                                if (!window_at(oh, ow, i, j, ih, iw))
                                    continue;
                                const val_type v = static_cast<val_type>(input(ch, ih * m_w + iw));
                                if (v > best)
                                {
                                    best = v;
                                    best_idx = ih * m_w + iw;
                                }
                            }
                        }
                        out(ch, col) = best;
                        m_argmax[static_cast<std::size_t>(ch) * n + col] = best_idx;
                    }
                    else
                    {
                        val_type sum = val_type(0);
                        for (int i = 0; i < m_kh; ++i)
                        {
                            for (int j = 0; j < m_kw; ++j)
                            {
                                int ih = 0, iw = 0;
                                if (!window_at(oh, ow, i, j, ih, iw))
                                    continue;
                                sum += static_cast<val_type>(input(ch, ih * m_w + iw));
                            }
                        }
                        out(ch, col) = sum / static_cast<val_type>(avg_divisor(oh, ow));
                    }
                }
            }
        }
        return out;
    }

    /** Stateless layer (no recurrence over space/time): a single column and a whole input share one path */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    /**
     * delta = dL/dy with shape [C, H_out*W_out]; returns dL/dx with shape [C, H*W].
     * Max pooling: the gradient goes back through the argmax recorded by forward (only one position
     * per window receives it). Average pooling: `delta / divisor` is scatter-added to every valid
     * position of the window, so overlapping windows accumulate naturally.
     */
    template <typename other_type>
    mat_t<val_type> backward(other_type const& delta)
    {
        if (m_kh < 1)
            throw std::runtime_error("pool2d_net_t::backward: layer is not configured, call set_param() first");
        if (delta.row_num() != m_channels || delta.col_num() != m_h_out * m_w_out)
            throw std::runtime_error("pool2d_net_t::backward: delta shape mismatch");

        const int c = m_channels;
        const int n = m_h_out * m_w_out;
        mat_t<val_type> dx(c, m_h * m_w);       // a freshly constructed matrix is guaranteed zeroed

        for (int ch = 0; ch < c; ++ch)
        {
            for (int oh = 0; oh < m_h_out; ++oh)
            {
                for (int ow = 0; ow < m_w_out; ++ow)
                {
                    const int col = oh * m_w_out + ow;
                    const val_type d = static_cast<val_type>(delta(ch, col));
                    if (m_mode == pool_mode::max)
                    {
                        const int idx = m_argmax[static_cast<std::size_t>(ch) * n + col];
                        if (idx >= 0)
                            dx(ch, idx) += d;
                    }
                    else
                    {
                        const val_type scale = d / static_cast<val_type>(avg_divisor(oh, ow));
                        for (int i = 0; i < m_kh; ++i)
                        {
                            for (int j = 0; j < m_kw; ++j)
                            {
                                int ih = 0, iw = 0;
                                if (window_at(oh, ow, i, j, ih, iw))
                                    dx(ch, ih * m_w + iw) += scale;
                            }
                        }
                    }
                }
            }
        }
        return dx;
    }

    /** Parameterless layer: no-ops like relu_net_t / gelu_net_t, called unconditionally by complex_net_t */
    template <typename init_type>
    void init_weight()
    {
        // no weights
    }

    void step()
    {
        // no weights
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "pool2d_net_t:(" << pool_mode_name(m_mode)
           << ", in:" << m_h << "x" << m_w
           << ", out:" << m_h_out << "x" << m_w_out
           << ", kernel:" << m_kh << "x" << m_kw
           << ", stride:" << m_stride_h << "x" << m_stride_w
           << ", pad:" << m_pad_h << "x" << m_pad_w;
        if (m_mode == pool_mode::average)
            ss << ", count_include_pad:" << (m_count_include_pad ? "true" : "false");
        ss << ")";
        return ss.str();
    }
};

} // namespace jasmine
#endif
