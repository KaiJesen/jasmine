#ifndef __JAS_SILU_T_HPP__
#define __JAS_SILU_T_HPP__

#include <cmath>
#include <sstream>
#include <string>

#include "jas_mat_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_storage.hpp"

namespace jasmine {

/**
 * SiLU 激活层（又名 Swish）。
 *
 *     silu(x) = x * sigmoid(x) = x / (1 + e^-x)
 *
 * 与 GELU 同为「平滑激活」：在负半轴非单调、有下界（x≈-1.278 处取最小值 ≈-0.278），
 * 处处可导、梯度几乎处处非零。和 relu_net_t / sigmoid_net_t 一样是「无状态静态层」：
 * 不持有可训练参数，init_weight / step 都是空实现。
 *
 * 用途：SwiGLU FFN 的门控分支（对齐 LLaMA 系权重）。
 *     SwiGLU(x) = down_proj( silu(gate_proj(x)) ⊙ up_proj(x) )
 * 注意 HF 的命名与论文符号不对应：**过激活的是 `gate_proj`（论文 W1）**，
 * 而 `up_proj`（论文 W3）不过激活，两者逐元素相乘后再过 down_proj。
 *
 * 与 GPT-2 的 GELU 对比（本类不参与 GPT-2 路径，改的是 FFN 里的激活）：
 *     GELU-MLP:  y = W2 · gelu(W1 x)               两个矩阵
 *     SwiGLU-MLP:y = W2 · ( silu(W1 x) ⊙ W3 x )    三个矩阵，中间维取 ≈8/3·d_model
 *
 * 实现说明：逐元素相乘在 jasmine 里就是矩阵的 `operator*`（Hadamard，非矩阵乘），
 * sigmoid 复用 jas_mat_express_t.hpp 的 `sigmoid()`，避免在仓库里出现第二份实现。
 */
template <typename input_type>
class silu_net_t
{
public:
    // 公开：与 gelu_net_t 一致，允许该层位于 complex_net 链首（complex_net_t 从首个成员取 val_type）
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_input;    // 前一次的输入，反向传播时用于计算梯度

public:
    silu_net_t() = default;

    /**
     * 标量 sigmoid。表达式与 mat_sigmoid_t::work 保持一致，保证标量路径与矩阵路径结果相同。
     * 两端饱和时 exp 会溢出为 inf，但 1/(1+inf)=0、1/(1+0)=1，结果仍然正确且不会产生 NaN。
     */
    static val_type sigmoid_v(val_type const& x) noexcept
    {
        constexpr val_type kOne = static_cast<val_type>(1);
        return kOne / (kOne + std::exp(-x));
    }

    /** 标量 SiLU：供矩阵逐元素调用，也便于单测直接比对 */
    static val_type silu(val_type const& x) noexcept
    {
        return x * sigmoid_v(x);
    }

    /**
     * 标量 SiLU 的导数：
     *     d/dx [ x σ(x) ] = σ(x) + x σ(x)(1-σ(x)) = σ(x) [ 1 + x (1 - σ(x)) ]
     * 只需 σ(x)，所以前向缓存 m_input 即可，反向现算 σ 不用额外存一份矩阵。
     */
    static val_type silu_grad(val_type const& x) noexcept
    {
        constexpr val_type kOne = static_cast<val_type>(1);
        const val_type s = sigmoid_v(x);
        return s * (kOne + x * (kOne - s));
    }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        return m_input * sigmoid(m_input);      // x ⊙ σ(x)，逐元素
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
        // σ 与 (1-σ) 先物化，避免表达式嵌套过深、也避免同一子表达式被重复求值
        mat_t<val_type> const s = sigmoid(m_input);
        mat_t<val_type> const one_minus_s = static_cast<val_type>(1) - s;
        mat_t<val_type> const prod = m_input * s * one_minus_s;      // x σ (1-σ)
        mat_t<val_type> const d_silu = s + prod;                      // σ + x σ (1-σ)
        return mat_t<val_type>(delta * d_silu);
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "silu_net_t(silu/swish)";
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
