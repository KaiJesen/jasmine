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
 * 二维卷积层，带完整反向传播（∂L/∂输入 / ∂L/∂权重 / ∂L/∂偏置）。
 *
 * 为什么是「im2col + GEMM」：jasmine 的数据载体只有 2 维的 `mat_t`，没有 NCHW/NHWC 张量。
 * 所以本层约定输入为
 *
 *     输入 x   : [C_in,  H * W]            通道先行，空间维展平在列上
 *     卷积核 W : [C_out, C_in * Kh * Kw]   每个输出通道一行，行内按 (c, kh, kw) 展平
 *     偏置 b   : [C_out, 1]                广播到所有输出位置
 *     输出 y   : [C_out, H_out * W_out]
 *
 * 这正是 `weight_net_t` 的形状约定（权重 [out, in]、输入 [in, T]），只不过「in」被 im2col
 * 展开成 `C_in*Kh*Kw`，"T" 被展开成 `H_out*W_out`。于是前向就是一次现成的
 * `W.dot(col)`（大尺寸自动走 BLAS），反向就是两次转置 GEMM + 一次 col2im 散射——
 * 不需要为本层新写任何矩阵乘，卷积的所有复杂度都收敛在 im2col/col2im 这一对循环里。
 *
 * 注意偏置必须**分两步**加（`dot` 之后再 `+=`）：写成 `W.dot(col) + b` 会让整个乘法退回
 * 朴素三重循环、丢掉 BLAS，原因见 `forward` 内的注释与 TESTING.md 第 11 节。
 *
 * 一维卷积（例如序列卷积）是 H = 1、Kh = 1、pad_h = 0 的特例：把长度 L 的序列当作
 * [C_in, 1*L] 喂进来即可，形状检查与反向逻辑都不需要分叉——`one_d(...)` 就是这一串参数的
 * 便捷写法，它返回的仍是本类的实例，不是第二套实现。
 *
 * 输出尺寸（与 PyTorch `Conv2d` 同式）：
 *
 *     H_out = (H + 2 * pad_h - dil_h * (Kh - 1) - 1) / stride_h + 1
 *     W_out = (W + 2 * pad_w - dil_w * (Kw - 1) - 1) / stride_w + 1
 *
 * 形状的来源：H/W/核大小这些既不能从 `[C_in, H*W]` 反推（H 与 W 可互换），也不能只由
 * C_in 决定，所以本层用 `set_param(...)` 显式配置，构造函数的参数表与它完全一致。
 * 刻意不叫 `reinit`：`is_reinitable_net` 靠 `requires { net.reinit(std::vector<int>()); }`
 * 判定（见 jas_mat_concepts.hpp），给了 reinit 会改变所有含本层的复杂网络的 reinit 语义；
 * 这与 `layer_norm_net_t` 用 `set_param` 的理由相同。
 *
 * 反向缓存：forward 缓存 im2col 结果 `m_col`（[C_in*Kh*Kw, H_out*W_out]），backward 直接复用。
 * 换成「反向重算 im2col」可以省下这份内存，但会把 im2col 的访存代价付两遍；这里跟随本仓库
 * 既有层（如 gelu 缓存输入、mha 缓存投影结果）的取舍，用空间换时间。
 */
template <typename input_type, template <typename> class updator_type>
class conv2d_net_t
{
public:
    using val_type = typename input_type::ele_type;

private:
    int m_c_in = 0;
    int m_c_out = 0;
    int m_h = 0;            // 输入空间高（一维卷积取 1）
    int m_w = 0;            // 输入空间宽（一维卷积即序列长度）
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

    mat_t<val_type> m_col;          // 通用路径缓存的 im2col：[C_in * Kh * Kw, H_out * W_out]
    mat_t<val_type> m_input;        // 零拷贝路径缓存的输入本体（重建窗口视图用）

    static int output_size(int const in, int const k, int const stride, int const pad, int const dil)
    {
        return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    }

    /**
     * 与输出尺寸无关的参数自检。必须在 output_size() 之前调用：那里要做
     * `... / stride`，stride 为 0 就不是「抛异常」而是除零崩掉（整数除法）。
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
     * im2col 矩阵能否**零拷贝**表达成「输入本体重新解释后的转置」。
     *
     * 需要的地址等式（col 的扁平下标 == 输入的内存下标）是
     *     (oh*W_out + ow)*K + (c*Kh + i)*Kw + j  ==  c*H*W + (oh*sh - ph + i*dh)*W + (ow*sw - pw + j*dw)
     * 逐项比对可知，只有「行方向完全不卷、单通道」时成立：
     *   C_in == 1、Kh == 1、stride_h == 1、pad_h == 0、dilation_h == 1（H 只是被原样带过）
     *   且 dilation_w == 1、pad_w == 0、stride_w == Kw、W == W_out*Kw
     * 此时 col = input.reshape_view(H_out*W_out, Kw).t()，一次拷贝都不用做。
     * 这也正是 patchify（ViT 的 patch embedding）与非重叠一维卷积的几何。
     */
    bool col_as_view() const
    {
        if (m_c_in != 1 || m_kh != 1 || m_stride_h != 1 || m_pad_h != 0 || m_dil_h != 1
            || m_dil_w != 1 || m_pad_w != 0 || m_stride_w != m_kw)
            return false;
        // 单行（H==1）：窗口序列就是输入的一段**前缀**，长度 w_out*Kw ≤ 输入长度，
        // 余下的尾巴（floor 语义丢掉的那个不满窗）落在视图之外，不影响。
        // 多行：每行必须整除对齐（W == W_out*Kw），否则第 2 行起地址会错位。
        return m_h == 1 || m_w == m_w_out * m_kw;
    }

    /**
     * im2col：把每个输出位置对应的 (c, kh, kw) 感受野摊平成列。
     * 越界的 padding 位置直接写 0，不物化带 padding 的输入副本。
     */
    template <typename Src>
    void im2col(Src const& input, mat_t<val_type>& col) const
    {
        const int k = m_c_in * m_kh * m_kw;
        const int n = m_h_out * m_w_out;
        // 同 col2im：绕开 reshape(1,1) 在空矩阵上的 `% 0`（k == n == 1 的退化卷积）
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

    /** col2im：∂L/∂col 按同一套感受野散射累加回 ∂L/∂x（padding 位置直接丢弃） */
    void col2im(mat_t<val_type> const& dcol, mat_t<val_type>& dx) const
    {
        if (dx.row_num() != m_c_in || dx.col_num() != m_h * m_w)
            dx = mat_t<val_type>(m_c_in, m_h * m_w);
        dx = val_type(0);               // 尺寸恰好相同时上面不会清零，必须显式重来
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
     * 一维卷积便捷入口：`*this` 仍是同一个 2-D 实现，只是把长度 L 的序列放在宽度方向
     * （H = 1、Kh = 1、pad_h = 0），不引入第二条代码路径。
     *
     *     auto conv = conv_t::one_d(c_in, c_out, L, kernel, stride, pad, dilation);
     *     auto y = conv.forward(x);        // x 形状 [C_in, L]，y 形状 [C_out, L_out]
     */
    static conv2d_net_t one_d(int const c_in, int const c_out, int const len, int const k,
                              int const stride = 1, int const pad = 0, int const dilation = 1)
    {
        return conv2d_net_t(c_in, c_out, /*h=*/1, /*w=*/len, /*kh=*/1, /*kw=*/k,
                            /*stride_h=*/1, /*stride_w=*/stride, /*pad_h=*/0, /*pad_w=*/pad,
                            /*dil_h=*/1, /*dil_w=*/dilation);
    }

    /**
     * 配置形状并分配权重/偏置（权重与偏置都置 0，之后用 init_weight<init_type>() 填充）。
     * 与 `layer_norm_net_t::set_param` 一样，权重加载器需要在 forward 之前写入 weight()/bias()，
     * 所以必须先调用本函数完成分配，否则拿到的是未分配的空矩阵。
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

        // 用「整体赋值一份新矩阵」而不是 reshape：C_out == 1 时偏置是 1x1，而
        // mat_t::reshape(1,1) 在默认构造的空矩阵上会先读 (*this)(0,0)（内部是 `0 % row_num()`，
        // 即 `% 0`）才转标量——C_out==1 的卷积是很常见的配置，必须绕开这个坑。
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

    /** 本配置下 forward 是否走零拷贝窗口视图路径（供测试与诊断） */
    bool col_view_enabled() const { return col_as_view(); }

    /** 权重 [C_out, C_in*Kh*Kw]；供权重加载器直接写入 */
    mat_t<val_type>& weight() { return m_weight; }
    mat_t<val_type> const& weight() const { return m_weight; }
    /** 偏置 [C_out, 1]；供权重加载器直接写入或置零 */
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
            // 窗口矩阵就是输入本体的重解释：存输入（反向重建视图用），不物化 col
            detail::store_for_backward(m_input, std::forward<Src>(input));
            const int n_pos = m_h_out * m_w_out;
            // 取前 n_pos*Kw 个元素（单行切片 → 紧凑），再重解释成 (N x K)：全程零拷贝
            auto used = m_input.view(0, 0, 1, n_pos * m_kw);
            auto windows = used.reshape_view(n_pos, m_kw);                  // (N x K) 零拷贝
            mat_t<val_type> out = m_weight.dot(windows.t());                // 裸 dot → BLAS
            out += m_bias;
            return out;
        }

        im2col(input, m_col);
        // 必须写成「先裸 dot、再加偏置」两步，不能写成 `m_weight.dot(m_col) + m_bias`：
        // `mat_add_t::clone()` 是逐元素求值的，它会对每个 (i,j) 调用 mat_dot_t::operator()，
        // 于是整次矩阵乘退回「每个输出元素自己扫一遍 K」的朴素循环，**永远碰不到
        // try_fast_gemm / BLAS**。实测 M=64,N=1024,K=288 时：表达式写法 98.9ms，
        // 拆成两句 7.5ms + 一次广播加（~0.2ms）。语义完全一致，见 TESTING.md 第 11 节。
        mat_t<val_type> out = m_weight.dot(m_col);
        out += m_bias;
        return out;
    }

    /** 无状态（在空间/时间维上无递归）层：单列/整段输入走同一条路径 */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    /**
     * delta = ∂L/∂y，形状 [C_out, H_out*W_out]；返回 ∂L/∂x 形状 [C_in, H*W]。
     * 权重/偏置梯度在层内直接交给 updator 更新，与 weight_net_t 的语义一致。
     */
    template <typename other_type>
    mat_t<val_type> backward(other_type const& delta)
    {
        if (m_weight.valid() == false)
            throw std::runtime_error("conv2d_net_t::backward: layer is not configured, call set_param() first");
        if (delta.row_num() != m_c_out || delta.col_num() != m_h_out * m_w_out)
            throw std::runtime_error("conv2d_net_t::backward: delta shape mismatch");

        // ∂L/∂W = delta · colᵀ        [C_out, N] · [N, K] → [C_out, K]
        // 零拷贝路径下 colᵀ 就是那个 (N x K) 窗口视图本身，同样不需要物化
        mat_t<val_type> delta_weight;
        if (col_as_view())
        {
            const int n_pos = m_h_out * m_w_out;
            auto used = m_input.view(0, 0, 1, n_pos * m_kw);
            delta_weight = delta.dot(used.reshape_view(n_pos, m_kw));
        }
        else
            delta_weight = delta.dot(m_col.t());
        // ∂L/∂b = 对输出位置求和   [C_out, 1]
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
