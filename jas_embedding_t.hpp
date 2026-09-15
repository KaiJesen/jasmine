#ifndef __JAS_EMBEDDING_T_HPP__
#define __JAS_EMBEDDING_T_HPP__

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_mat_utility.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_updator_t.hpp"

namespace jasmine {

/**
 * 离散 token → 稠密向量查表。
 * 权重 E 布局为 d_model × vocab（第 id 列 = 该 token 的向量）。
 * 输入 ids：1 × T，元素为整数 token id（存于 val_type）。
 * 输出：d_model × T。
 *
 * 末端输出投影用 weight_net_t / output_proj_net_t（d_model → vocab）得到 logits V × T。
 */
template <typename input_type, template <typename> class updator_type>
class embedding_net_t
{
public:
    using val_type = typename input_type::ele_type;

private:
    mat_t<val_type> m_weight;          // d_model × vocab
    updator_type<val_type> m_updator;
    mat_t<val_type> m_ids;             // 1 × T，forward 缓存
    int m_vocab = 1;
    int m_d_model = 1;

public:
    embedding_net_t(int vocab = 1, int d_model = 1)
        : m_weight(d_model, vocab), m_updator(), m_vocab(vocab), m_d_model(d_model)
    {
    }

    /** reinit 约定与 weight_net 一致：container[0]=vocab(in)，[1]=d_model(out) */
    void reinit(std::vector<int> const& container)
    {
        if (container.size() < 2)
            throw std::invalid_argument("embedding_net_t::reinit needs {vocab, d_model}");
        m_vocab = container[0];
        m_d_model = container[1];
        m_weight.reshape(m_d_model, m_vocab);
    }

    int vocab_size() const { return m_vocab; }
    int d_model() const { return m_d_model; }
    mat_t<val_type>& weight() { return m_weight; }
    mat_t<val_type> const& weight() const { return m_weight; }

    template <typename Src>
    mat_t<val_type> forward(Src&& ids)
    {
        m_ids = mat_t<val_type>(std::forward<Src>(ids));
        if (m_ids.row_num() != 1)
            throw std::invalid_argument("embedding_net_t: ids must be shape 1×T");

        const int T = m_ids.col_num();
        mat_t<val_type> out(m_d_model, T);
        for (int t = 0; t < T; ++t)
        {
            const int id = static_cast<int>(m_ids(0, t));
            if (id < 0 || id >= m_vocab)
                throw std::out_of_range("embedding_net_t: token id out of range");
            for (int i = 0; i < m_d_model; ++i)
                out(i, t) = m_weight(i, id);
        }
        return out;
    }

    template <typename Src>
    mat_t<val_type> forward_one(Src&& ids)
    {
        return forward(std::forward<Src>(ids));
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        if (delta.row_num() != m_d_model || delta.col_num() != m_ids.col_num())
            throw std::runtime_error("embedding_net_t::backward: delta shape mismatch");

        mat_t<val_type> grad_w(m_d_model, m_vocab);
        grad_w = val_type(0);
        const int T = m_ids.col_num();
        for (int t = 0; t < T; ++t)
        {
            const int id = static_cast<int>(m_ids(0, t));
            for (int i = 0; i < m_d_model; ++i)
                grad_w(i, id) += delta(i, t);
        }
        m_updator.update(grad_w, m_weight);
        // 对离散 id 不回传梯度；返回空矩阵占位（调用方不应再依赖）
        return mat_t<val_type>(1, T);
    }

    template <typename init_type>
    void init_weight()
    {
        init_matrix<init_type>(m_weight);
    }

    template <typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_updator.set_lr(lr);
    }

    void step()
    {
        m_updator.step();
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "embedding_net_t:(vocab:" << m_vocab
           << ", d_model:" << m_d_model << ")";
        return ss.str();
    }
};

} // namespace jasmine

#endif
