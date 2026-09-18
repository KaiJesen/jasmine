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
 * 池化模式。
 *
 * - max    ：取窗口内最大值；padding 视作 -inf；并列最大值按「第一个」路由梯度（见下）。
 * - average：窗口内求和后除以除数，除数是 kh*kw 还是「有效元素个数」由 count_include_pad 决定。
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
 * 二维池化层（最大池化 / 平均池化），带完整反向传播（∂L/∂输入）。
 *
 * 形状约定与 `conv2d_net_t` 一致：
 *
 *     输入 x : [C, H * W]        通道先行，空间维展平在列上
 *     输出 y : [C, H_out * W_out]
 *
 * 池化**没有可训练参数**，而且逐通道独立（输出通道数 == 输入通道数），所以本层是「静态层」：
 * 与 `relu_net_t` / `gelu_net_t` 一样只吃 `input_type`、不持有 updator，
 * `init_weight<init_type>()` / `step()` 都是空实现（`complex_net_t` 会无条件调用它们）。
 * 这带来两个直接后果：`is_updatable_net` 为假（链上的 `set_updator` 会跳过本层），
 * 且因为它没有 `reinit`（形状由 `set_param` 给），`complex_net_t::reinit` 也不占槽位。
 *
 * 输出尺寸（floor 模式，与 PyTorch `MaxPool2d` / `AvgPool2d` 同式）：
 *
 *     H_out = (H + 2 * pad_h - Kh) / stride_h + 1
 *     W_out = (W + 2 * pad_w - Kw) / stride_w + 1
 *
 * 三个刻意与 PyTorch 对齐的语义（都有单测钉住）：
 *
 * 1. **stride 缺省等于核大小**：`stride_h/stride_w` 传 0 表示「同核大小」（PyTorch 的
 *    `MaxPool2d(2)` 就是 `stride=2`）。0 在别处不是合法步长，所以拿它当哨兵不会歧义。
 * 2. **平均池化默认 count_include_pad = true**：除数恒为 `Kh*Kw`，padding 的 0 也进分母
 *    （PyTorch `nn.AvgPool2d` 的默认行为）。置 false 则除以窗口内**有效**元素个数，
 *    此时边界窗口的除数更小、输出更大。
 * 3. **并列最大值取第一个**（按 (kh, kw) 行优先顺序），梯度只走那一个位置，不均分。
 *    PyTorch CPU 实现同样如此：`x = [[1,1],[1,1]]`、`max_pool2d(2)` 的 `x.grad` 是
 *    `[1,0,0,0]` 而不是 `[0.25,...]`。
 *
 * 为什么没有 dilation / ceil_mode：平均池化在 PyTorch 里就没有 dilation，ceil_mode 会引入
 * 「最后一个窗口是否算满」的额外规则，和 count_include_pad 交织在一起；这两条按需再加，
 * 现在先把最常用的 floor + 整数核做扎实。`pad <= Kh/2`（PyTorch 的 "pad should be at most
 * half of effective kernel size"）是硬约束，它同时保证每个窗口至少覆盖一个真实元素，
 * 于是最大池化不会遇到「整窗都是 -inf」、平均池化也不会除以 0。
 *
 * 反向缓存：最大池化 forward 时记录每个输出元素的 argmax（展平后的输入下标，存 int 向量，
 * 不用浮点矩阵是为了避免大图上的精度问题）；平均池化的除数是位置函数，反向现算即可，
 * 不需要额外缓存。两者都缓存了 `m_channels` 用于校验 delta 的行数。
 */
template <typename input_type>
class pool2d_net_t
{
public:
    using val_type = typename input_type::ele_type;

private:
    pool_mode m_mode = pool_mode::max;
    int m_h = 0;                // 输入空间高（一维池化取 1）
    int m_w = 0;                // 输入空间宽（一维池化即序列长度）
    int m_kh = 0;
    int m_kw = 0;
    int m_stride_h = 0;
    int m_stride_w = 0;
    int m_pad_h = 0;
    int m_pad_w = 0;
    int m_h_out = 0;
    int m_w_out = 0;
    bool m_count_include_pad = true;

    int m_channels = 0;             // forward 缓存：backward 校验 delta 行数
    std::vector<int> m_argmax;      // max 模式专用：每个输出元素对应的展平输入下标
    std::vector<int> m_divisor;     // average 模式专用：每个输出位置的除数（预存，避免逐个重算）

    static int output_size(int const in, int const k, int const stride, int const pad)
    {
        return (in + 2 * pad - k) / stride + 1;
    }

    /**
     * 与输出尺寸无关的参数自检，必须在 output_size() 之前调用：
     * 那里有 `... / stride`，stride 为 0 就不是「抛异常」而是除零崩掉（整数除法）。
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

    /** 窗口遍历：把 (oh, ow) 的第 (i, j) 个元素映射到输入坐标，越界返回 false */
    bool window_at(int const oh, int const ow, int const i, int const j,
                   int& ih, int& iw) const
    {
        ih = oh * m_stride_h - m_pad_h + i;
        iw = ow * m_stride_w - m_pad_w + j;
        return (ih >= 0 && ih < m_h && iw >= 0 && iw < m_w);
    }

    /**
     * 平均池化的除数：include_pad 时恒为窗口面积，否则是本窗口的有效元素个数。
     * 两种情形都在 set_param 里预存成一张 (H_out x W_out) 的表 —— 除数只是输出位置的函数，
     * 没必要在 forward/backward 的每个输出元素上重算一遍 O(Kh*Kw)。
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
     * 配置形状与窗口；本层无权重，所以这里只记账 + 校验，不分配任何参数。
     * `stride_h/stride_w` 传 0 表示「等于对应方向的核大小」。
     * 抛出时对象保持原状（先算局部量并校验，最后才写入成员）。
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

        // 预存每个输出位置的除数（只有 average 用得上）
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
     * 一维池化便捷入口：`*this` 仍是同一个 2-D 实现，只是把长度 L 的序列放在宽度方向
     * （H = 1、Kh = 1、pad_h = 0），不引入第二条代码路径。
     *
     *     auto pool = pool_t::one_d(pool_mode::max, L, 2);   // k=2, stride 缺省 = k
     *     auto y = pool.forward(x);        // x 形状 [C, L]，y 形状 [C, L_out]
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
                        // 严格 `>`：并列时保留先遇到的那个（行优先），与 PyTorch 一致。
                        // 初值 -inf 只是为了让「整窗都是 padding」也有定义；按 pad <= k/2 的
                        // 约束，这种窗口不会出现（真的出现也会被下面的 idx < 0 挡在反向之前）。
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

    /** 无状态（在空间/时间维上无递归）层：单列/整段输入走同一条路径 */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    /**
     * delta = ∂L/∂y，形状 [C, H_out*W_out]；返回 ∂L/∂x 形状 [C, H*W]。
     * 最大池化：梯度按 forward 记下的 argmax 原路返回（每个窗口只有一个位置拿到梯度）。
     * 平均池化：`delta / 除数` 散射累加到窗口内每个有效位置；重叠窗口自然累加。
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
        mat_t<val_type> dx(c, m_h * m_w);       // 新构造的矩阵一定是清零的

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

    /** 无参数层：与 relu_net_t / gelu_net_t 一样是空实现，供 complex_net_t 无条件调用 */
    template <typename init_type>
    void init_weight()
    {
        // 无权重
    }

    void step()
    {
        // 无权重
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
