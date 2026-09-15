#ifndef __JAS_LOSS_T_HPP__
#define __JAS_LOSS_T_HPP__

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "jas_mat_concepts.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_storage.hpp"
#include "jas_mat_utility.hpp"

namespace jasmine {

// 损失函数做成和网络一样的形式，但是只有反向传播有意义，正向传播直接透传
template <typename input_type>
class mat_loss_t
{
public:
    using val_type = typename input_type::ele_type;
    /** complex_net::infer 跳过本层，输入原样传给后续层（训练仍走 forward 透传） */
    static constexpr bool skip_on_infer = true;
    mat_loss_t() = default;

    mat_t<val_type> m_input;

    template <typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        return m_input;
    }

    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }
};

template <typename input_type>
class mse_loss_t : public mat_loss_t<input_type>
{
public:
    using base_type = mat_loss_t<input_type>;
    using val_type = typename base_type::val_type;

    mse_loss_t() = default;

    mat_t<val_type> backward(const mat_t<val_type>& target) const
    {
        return base_type::m_input - target;
    }

    val_type loss(const mat_t<val_type>& target) const
    {
        return mean(pow(base_type::m_input - target));
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "mse_loss_t";
        return ss.str();
    }

    template <typename init_type>
    void init_weight()
    {
    }

    void step()
    {
    }
};

/**
 * Fused softmax + cross-entropy / NLL。
 * forward：透传并缓存 logits（V × T）；infer 经 skip_on_infer 跳过。
 * target：1 × T，元素为类别 id（存于 val_type）。
 * ignore_index：该 id 的位置不计入 loss / 梯度（典型为 pad）。
 * position_mask：可选 1 × T，0 表示忽略（可用于屏蔽未来位等）；与 ignore_index 同时生效。
 * EOS 作为普通类别 id，不再用连续特征维阈值。
 */
template <typename input_type>
class ce_loss_t : public mat_loss_t<input_type>
{
public:
    using base_type = mat_loss_t<input_type>;
    using val_type = typename base_type::val_type;

private:
    int m_ignore_index = -1;
    mat_t<val_type> m_position_mask; // 空 = 全计入

    bool include_position(int t, int label) const
    {
        if (m_ignore_index >= 0 && label == m_ignore_index)
            return false;
        if (m_position_mask.valid() && m_position_mask.col_num() > 0)
        {
            if (t >= m_position_mask.col_num())
                return false;
            if (m_position_mask(0, t) == val_type(0))
                return false;
        }
        return true;
    }

    /** 对单列 logits 做稳定 softmax；若 probs_out 非空则写入该列概率；返回 -log p[y] */
    static val_type softmax_nll_col(mat_t<val_type> const& logits, int t, int y,
                                    mat_t<val_type>* probs_out)
    {
        const int V = logits.row_num();
        val_type vmax = logits(0, t);
        for (int i = 1; i < V; ++i)
            if (logits(i, t) > vmax)
                vmax = logits(i, t);

        val_type sum = 0;
        for (int i = 0; i < V; ++i)
        {
            val_type e = std::exp(logits(i, t) - vmax);
            if (probs_out)
                (*probs_out)(i, t) = e;
            sum += e;
        }
        if (probs_out)
        {
            for (int i = 0; i < V; ++i)
                (*probs_out)(i, t) /= sum;
        }
        return -(logits(y, t) - vmax) + std::log(sum);
    }

public:
    ce_loss_t() = default;

    void set_ignore_index(int idx) { m_ignore_index = idx; }
    int ignore_index() const { return m_ignore_index; }

    /** 1×T，1=计入 loss，0=忽略；传空矩阵清除 */
    void set_position_mask(mat_t<val_type> const& mask) { m_position_mask = mask; }

    void clear_position_mask() { m_position_mask = mat_t<val_type>(); }

    mat_t<val_type> backward(const mat_t<val_type>& target) const
    {
        mat_t<val_type> const& logits = base_type::m_input;
        if (target.row_num() != 1 || target.col_num() != logits.col_num())
            throw std::invalid_argument("ce_loss_t::backward: target must be 1×T matching logits cols");

        const int V = logits.row_num();
        const int T = logits.col_num();
        mat_t<val_type> grad(V, T);
        grad = val_type(0);
        mat_t<val_type> probs(V, T);

        int counted = 0;
        for (int t = 0; t < T; ++t)
        {
            const int y = static_cast<int>(target(0, t));
            if (!include_position(t, y))
                continue;
            if (y < 0 || y >= V)
                throw std::out_of_range("ce_loss_t::backward: label out of range");
            (void)softmax_nll_col(logits, t, y, &probs);
            for (int i = 0; i < V; ++i)
                grad(i, t) = probs(i, t);
            grad(y, t) -= val_type(1);
            ++counted;
        }
        if (counted > 0)
        {
            const val_type inv = val_type(1) / static_cast<val_type>(counted);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < V; ++i)
                    grad(i, t) *= inv;
        }
        return grad;
    }

    val_type loss(const mat_t<val_type>& target) const
    {
        mat_t<val_type> const& logits = base_type::m_input;
        if (target.row_num() != 1 || target.col_num() != logits.col_num())
            throw std::invalid_argument("ce_loss_t::loss: target must be 1×T matching logits cols");

        const int T = logits.col_num();
        val_type total = 0;
        int counted = 0;
        for (int t = 0; t < T; ++t)
        {
            const int y = static_cast<int>(target(0, t));
            if (!include_position(t, y))
                continue;
            if (y < 0 || y >= logits.row_num())
                throw std::out_of_range("ce_loss_t::loss: label out of range");
            total += softmax_nll_col(logits, t, y, nullptr);
            ++counted;
        }
        if (counted == 0)
            return val_type(0);
        return total / static_cast<val_type>(counted);
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "ce_loss_t(ignore_index:" << m_ignore_index << ")";
        return ss.str();
    }

    template <typename init_type>
    void init_weight()
    {
    }

    void step()
    {
    }
};

} // namespace jasmine
#endif
