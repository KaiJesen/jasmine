#ifndef __JAS_GELU_T_HPP__
#define __JAS_GELU_T_HPP__

#include <cmath>
#include <sstream>
#include <string>

#include "jas_mat_t.hpp"
#include "jas_mat_storage.hpp"

namespace jasmine {

/**
 * GELU 激活层（gelu_new，tanh 近似）。
 *
 * 精确 GELU 用 erf：
 *     gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 * HuggingFace GPT-2 config 里的 `gelu_new`（别名 `gelu_pytorch_tanh`）用 tanh 拟合同一曲线：
 *     gelu(x) = 0.5 * x * (1 + tanh(alpha * (x + beta * x^3)))
 *     alpha = sqrt(2/pi) ~= 0.7978845608,  beta = 0.044715
 * 两者数值差异极小，但做 GPT-2 权重对齐时必须用 tanh 版，否则 logits 会有偏差。
 *
 * 用途：GPT-2 的 FFN 为 Linear -> GELU -> Linear（对照 jasmine 原有的 ReLU 版 base_ffn_t）。
 * 与 relu_net_t 一样是「无状态静态层」：不持有可训练参数，init_weight / step 都是空实现。
 */
template <typename input_type>
class gelu_net_t
{
public:
    // 公开：可位于 complex_net 链首（complex_net_t 从首个成员取 val_type）
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_input;    // 前一次的输入，反向传播时用于计算梯度

public:
    gelu_net_t() = default;

    /** 标量 GELU（tanh 近似）：供矩阵逐元素调用，也便于单测直接比对 */
    static val_type gelu(val_type const& x) noexcept
    {
        constexpr val_type kAlpha = static_cast<val_type>(0.7978845608028654);  // sqrt(2/pi)
        constexpr val_type kBeta = static_cast<val_type>(0.044715);
        constexpr val_type kHalf = static_cast<val_type>(0.5);
        constexpr val_type kOne = static_cast<val_type>(1);
        const val_type inner = kAlpha * (x + kBeta * x * x * x);
        return kHalf * x * (kOne + std::tanh(inner));
    }

    /** 标量 GELU 的导数：d/dx [ 0.5 x (1 + tanh(u)) ]，u = alpha (x + beta x^3) */
    static val_type gelu_grad(val_type const& x) noexcept
    {
        constexpr val_type kAlpha = static_cast<val_type>(0.7978845608028654);
        constexpr val_type kBeta = static_cast<val_type>(0.044715);
        constexpr val_type kHalf = static_cast<val_type>(0.5);
        constexpr val_type kOne = static_cast<val_type>(1);
        constexpr val_type kThree = static_cast<val_type>(3);
        const val_type inner = kAlpha * (x + kBeta * x * x * x);
        const val_type t = std::tanh(inner);
        const val_type du_dx = kAlpha * (kOne + kThree * kBeta * x * x);
        return kHalf * (kOne + t) + kHalf * x * (kOne - t * t) * du_dx;
    }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        mat_t<val_type> out(m_input.row_num(), m_input.col_num());
        for (int i = 0; i < m_input.row_num(); ++i)
            for (int j = 0; j < m_input.col_num(); ++j)
                out(i, j) = gelu(m_input(i, j));
        return out;
    }

    /** 无状态层：单列输入与整段 forward 相同 */
    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        if (delta.row_num() != m_input.row_num() || delta.col_num() != m_input.col_num())
        {
            throw std::runtime_error("delta size does not match input size");
        }
        mat_t<val_type> out(m_input.row_num(), m_input.col_num());
        for (int i = 0; i < m_input.row_num(); ++i)
            for (int j = 0; j < m_input.col_num(); ++j)
                out(i, j) = delta(i, j) * gelu_grad(m_input(i, j));
        return out;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "gelu_net_t(gelu_new)";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 无权重
    }

    void step()
    {
        // 无权重
    }
};

} // namespace jasmine
#endif
