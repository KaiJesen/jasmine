#ifndef __MAT_MHA_T_HPP__
#define __MAT_MHA_T_HPP__

#include <cmath>
#include <limits>
#include <memory>
#include <tuple>

#include "mat_t.hpp"
#include "mat_view_t.hpp"
#include "mat_net_t.hpp"
#include "mat_express_t.hpp"
#include "mat_RoPE_t.hpp"

namespace jasmine {

/**
 * 单头 attention 核：输入已是切好的 Q/K/V（d_head × seq）。
 * 负责 RoPE(Q/K)、score、可选 causal mask、softmax、加权求和。
 * 不再包含 W_Q/W_K/W_V（经典 MHA 中投影在 mat_mha_t 全维完成）。
 */
template <typename input_type, template<typename> class updator_type>
class mat_head_gen_t
{
public:
    using val_type = typename input_type::ele_type;
    using ele_type = typename input_type::ele_type;
    using rope_net_type = RoPE_net_t<mat_t<val_type>>;

    struct bwd_pack_t
    {
        mat_t<val_type> delta_q;
        mat_t<val_type> delta_k;
        mat_t<val_type> delta_v;
    };

private:
    using softmax_type = hsoftmax_net_t<input_type>;

    softmax_type m_softmax;
    mat_t<val_type> m_q, m_k, m_v;  // 前向缓存（RoPE 之后）
    bool m_mask;
    std::shared_ptr<rope_net_type> m_rope;  // 来自 rope_registry，按 d_head 共享；空则不做 RoPE
    int m_d_head = 1;

public:
    mat_head_gen_t(int const& d_head = 1, bool const& mask = false, int const& seq_len = 1)
        : m_q(d_head, seq_len), m_k(d_head, seq_len), m_v(d_head, seq_len),
          m_mask(mask), m_rope(nullptr), m_d_head(d_head)
    {
    }

    void set_rope(std::shared_ptr<rope_net_type> rope)
    {
        m_rope = std::move(rope);
    }

    std::shared_ptr<rope_net_type> rope() const
    {
        return m_rope;
    }

    void set_param(int const& d_head, bool const& mask = false, int const& seq_len = 1)
    {
        m_d_head = d_head;
        m_q.reshape(d_head, seq_len);
        m_k.reshape(d_head, seq_len);
        m_v.reshape(d_head, seq_len);
        m_mask = mask;
        // 默认从注册中心取该头维对应的 RoPE（同 d 共享）
        if (d_head > 0 && d_head % 2 == 0)
            m_rope = rope_registry_t<val_type>::instance().get(d_head);
        else
            m_rope.reset();
    }

    template <typename q_t, typename k_t, typename v_t>
    mat_t<val_type> forward(const q_t& q, const k_t& k, const v_t& v)
    {
        m_q = q.clone();
        m_k = k.clone();
        m_v = v.clone();
        if (m_rope)
        {
            m_q = m_rope->forward(m_q);
            m_k = m_rope->forward(m_k);
        }

        auto attn_scores = (m_q.t().dot(m_k) / std::sqrt(static_cast<val_type>(m_q.row_num()))).clone();
        /*!ANCHOR 掩码规则说明
        * 由于scores=Q'K，也就是说scores中的i行j列元素表示的是Q序列中第i个值与K序列中第j个值之间的分数；
        * Q是表示的是当前的查询，K是可关注的历史。那么就需要就针对每个Q让他只能看到之前发生的K。也就是j > i的都设置为无效的
        */
        if (m_mask)
        {
            for (int i = 0; i < attn_scores.row_num(); ++i)
            {
                for (int j = i + 1; j < attn_scores.col_num(); ++j)
                {
                    attn_scores(i, j) = -std::numeric_limits<val_type>::infinity();
                }
            }
        }
        auto attn_weights = m_softmax.forward(attn_scores);
        auto output = m_v.dot(attn_weights.t()).clone();

        return output;
    }

    bwd_pack_t backward(const mat_view_t<mat_t<val_type>>& delta)
    {
        mat_t<val_type> delta_v = delta.dot(m_softmax.m_output);
        mat_t<val_type> delta_attn_weights = delta.t().dot(m_v);
        mat_t<val_type> delta_qt_k = m_softmax.backward(delta_attn_weights);
        /*!LINK - 链接规则说明
        * 由于scores=Q'K，也就是说scores中的i行j列元素表示的是Q序列中第i个值与K序列中第j个值之间的分数；
        * Q是表示的是当前的查询，K是可关注的历史。那么就需要就针对每个Q让他只能看到之前发生的K。也就是j > i的都设置为无效的
        * 反向时对未来位置的梯度置零，与前向 -inf 掩码对应。
        */
        if (m_mask)
        {
            for (int i = 0; i < delta_qt_k.row_num(); ++i)
            {
                for (int j = i + 1; j < delta_qt_k.col_num(); ++j)
                {
                    delta_qt_k(i, j) = 0;
                }
            }
        }
        const val_type scale = static_cast<val_type>(std::sqrt(m_q.row_num()));
        mat_t<val_type> delta_q = m_k.dot(delta_qt_k.t()) / scale;
        mat_t<val_type> delta_k = m_q.dot(delta_qt_k) / scale;
        if (m_rope)
        {
            delta_q = m_rope->backward(delta_q);
            delta_k = m_rope->backward(delta_k);
        }
        return bwd_pack_t{std::move(delta_q), std::move(delta_k), std::move(delta_v)};
    }

    template <typename... upr_arg_types>
    void set_updator(upr_arg_types&&...)
    {
        // attention 核无独立可训练投影
    }

    void set_lr(val_type)
    {
    }

    template <typename init_type>
    void init_weight()
    {
    }

    std::string net_type(int const& indent = 0) const
    {
        return print_indent(indent) + "mat_head_gen_t(attend-only,d_head:" + std::to_string(m_d_head) + ")";
    }

    void step()
    {
    }
};

// 垂直拼接多个 mat_t 对象（按行拼接）
template <typename val_type>
mat_t<val_type> vconcat(const std::vector<mat_t<val_type>>& mats)
{
    if (mats.empty())
        throw std::runtime_error("Cannot concatenate empty vector");

    int cols = mats[0].col_num();
    int total_rows = 0;

    for (const auto& mat : mats)
    {
        if (mat.col_num() != cols)
            throw std::runtime_error("All matrices must have the same number of columns");
        total_rows += mat.row_num();
    }

    mat_t<val_type> result(total_rows, cols);
    int row_offset = 0;

    for (const auto& mat : mats)
    {
        for (int i = 0; i < mat.row_num(); ++i)
        {
            for (int j = 0; j < mat.col_num(); ++j)
            {
                result(row_offset + i, j) = mat(i, j);
            }
        }
        row_offset += mat.row_num();
    }

    return result;
}

// 垂直分割 mat_t 为多个子矩阵（按行分割，返回视图）
template <typename mat_type>
std::vector<mat_view_t<mat_type>> vsplit(mat_type& mat, int num_splits)
{
    if (num_splits <= 0 || mat.row_num() % num_splits != 0)
        throw std::runtime_error("Invalid number of splits or row count mismatch");

    int rows_per_split = mat.row_num() / num_splits;
    std::vector<mat_view_t<mat_type>> views;

    for (int i = 0; i < num_splits; ++i)
    {
        int start_row = i * rows_per_split;
        views.emplace_back(mat, start_row, 0, rows_per_split, mat.col_num());
    }

    return views;
}

/*!SECTION: MHA
*   多头注意力机制和线型变换网络不太一样，线型变换网络初始化时传入的是输入的维度和输出的维度，但是MHA传入的是头的数目以及模型的维度。另外，在transformer中，每一层的MHA的输入和输出都是相等维度的，另外，编码器和解码器的输出维度相等，但是序列长度可能不同。
所以，总体上来说，就维度而言，transformer实际只需要1个整形来表示输入和输出的维度，另外需要1个整形来表示头数量。因此，我们可以将mha认为是一种stable的网络。而stable的网络不能定义reinit函数。
*
* 经典实现路径：全维 QKV 投影 → 按头切分 → 各头 attend → concat → W_O。
* 每头看到的是全输入经投影后的不同切片，而不是把输入特征硬分区后各自投影。
*/
template <typename input_type, template<typename> class updator_type>
class mat_mha_t
{
public:
    using val_type = typename input_type::ele_type;

private:
    using proj_type = weight_net_t<mat_t<val_type>, updator_type>;
    using head_type = mat_head_gen_t<input_type, updator_type>;

    proj_type m_q_net;
    proj_type m_k_net;
    proj_type m_v_net;
    std::vector<head_type> m_heads;
    proj_type m_output_proj;
    int m_num_heads;
    int m_d_model;
    int m_d_head;
    bool m_mask = false;

    // 前向缓存，供 backward 切分梯度
    mat_t<val_type> m_q_full, m_k_full, m_v_full;

public:
    mat_mha_t(int num_heads = 1, int d_model = 1, bool mask = false, int seq_len = 1)
        : m_q_net(d_model, d_model), m_k_net(d_model, d_model), m_v_net(d_model, d_model),
          m_output_proj(d_model, d_model), m_num_heads(num_heads), m_d_model(d_model),
          m_d_head(d_model / num_heads), m_mask(mask)
    {
        /* 设置默认构造参数目的是让其可以没有参数进行构造，以便放入其他复杂网络结构中 */
        if (d_model % num_heads != 0)
            throw std::runtime_error("d_model must be divisible by num_heads");

        for (int i = 0; i < num_heads; ++i)
            m_heads.emplace_back(m_d_head, mask, seq_len);
        bind_rope();
    }

    void set_param(int num_heads, int d_model, bool mask = false, int seq_len = 1)
    {
        m_num_heads = num_heads;
        m_d_model = d_model;
        m_d_head = d_model / num_heads;
        m_mask = mask;

        if (d_model % num_heads != 0)
            throw std::runtime_error("d_model must be divisible by num_heads");

        m_q_net.reinit(std::vector<int>{d_model, d_model});
        m_k_net.reinit(std::vector<int>{d_model, d_model});
        m_v_net.reinit(std::vector<int>{d_model, d_model});
        m_output_proj.reinit(std::vector<int>{d_model, d_model});

        m_heads.resize(num_heads);
        for (int i = 0; i < num_heads; ++i)
            m_heads[i].set_param(m_d_head, mask, seq_len);
        bind_rope();
    }

    /** 显式绑定/刷新各头的 RoPE（同 d_head 共享注册中心条目） */
    void bind_rope(int max_seq_len = 0)
    {
        std::shared_ptr<RoPE_net_t<mat_t<val_type>>> rope;
        if (m_d_head > 0 && m_d_head % 2 == 0)
            rope = rope_registry_t<val_type>::instance().get(m_d_head, max_seq_len);
        for (auto& head : m_heads)
            head.set_rope(rope);
    }

    mat_t<val_type> forward(const input_type& input)
    {
        // Step 1: 检查输入维度是否合法
        if (input.row_num() != m_d_model)
            throw std::runtime_error("Input row dimension must match d_model");

        // Step 2: 全维 QKV 投影（各头共享同一组 W_Q/W_K/W_V）
        m_q_full = m_q_net.forward(input);
        m_k_full = m_k_net.forward(input);
        m_v_full = m_v_net.forward(input);

        // Step 3: 按头切分投影结果（切的是 Q/K/V，不是原始输入特征）
        auto q_splits = vsplit(m_q_full, m_num_heads);
        auto k_splits = vsplit(m_k_full, m_num_heads);
        auto v_splits = vsplit(m_v_full, m_num_heads);

        // Step 4: 每个头独立 attend（RoPE / score / mask / softmax / V）
        std::vector<mat_t<val_type>> head_outputs(m_num_heads);
        for (int i = 0; i < m_num_heads; ++i)
            head_outputs[i] = m_heads[i].forward(q_splits[i], k_splits[i], v_splits[i]);

        // Step 5: 拼接所有头的输出，再经 W_O 映射回 d_model
        mat_t<val_type> concatenated_output = vconcat(head_outputs);
        auto final_output = m_output_proj.forward(concatenated_output);

        return final_output;
    }

    mat_t<val_type> forward(const input_type& input, const input_type& encoder_input)
    {
        // Step 1: 检查输入维度是否合法
        if (input.row_num() != m_d_model)
            throw std::runtime_error("Input row dimension must match d_model");
        if (encoder_input.row_num() != m_d_model)
            throw std::runtime_error("Encoder input row dimension must match d_model");

        // Step 2: 交叉注意力 — Q 来自 decoder，K/V 来自 encoder；交叉注意力不需要 mask 层
        m_q_full = m_q_net.forward(input);
        m_k_full = m_k_net.forward(encoder_input);
        m_v_full = m_v_net.forward(encoder_input);

        // Step 3: 按头切分投影结果
        auto q_splits = vsplit(m_q_full, m_num_heads);
        auto k_splits = vsplit(m_k_full, m_num_heads);
        auto v_splits = vsplit(m_v_full, m_num_heads);

        // Step 4: 每个头独立进行正向传播
        std::vector<mat_t<val_type>> head_outputs(m_num_heads);
        for (int i = 0; i < m_num_heads; ++i)
            head_outputs[i] = m_heads[i].forward(q_splits[i], k_splits[i], v_splits[i]);

        // Step 5: 拼接所有头的输出，再经输出映射网络得到最终结果
        mat_t<val_type> concatenated_output = vconcat(head_outputs);
        auto final_output = m_output_proj.forward(concatenated_output);

        return final_output;
    }

    mat_t<val_type> backward(const input_type& delta)
    {
        // 反向传播到输出线性层 W_O
        auto delta_concat = m_output_proj.backward(delta);

        // 按行分割 delta_concat 为多个子矩阵（视图），对应各头
        auto deltas = vsplit(delta_concat, m_num_heads);

        // 聚合各头的 delta_q/k/v，再经全维投影回传到输入
        mat_t<val_type> delta_q(m_d_model, delta.col_num());
        mat_t<val_type> delta_k(m_d_model, delta.col_num());
        mat_t<val_type> delta_v(m_d_model, delta.col_num());
        delta_q = val_type(0);
        delta_k = val_type(0);
        delta_v = val_type(0);
        auto dq_views = vsplit(delta_q, m_num_heads);
        auto dk_views = vsplit(delta_k, m_num_heads);
        auto dv_views = vsplit(delta_v, m_num_heads);

        for (int i = 0; i < m_num_heads; ++i)
        {
            auto g = m_heads[i].backward(deltas[i]);
            dq_views[i].assign(g.delta_q);
            dk_views[i].assign(g.delta_k);
            dv_views[i].assign(g.delta_v);
        }

        return (m_q_net.backward(delta_q) + m_k_net.backward(delta_k) + m_v_net.backward(delta_v)).clone();
    }

    /*!
     * 交叉注意力反向：encoder_delta 在各层以累加方式收集 K/V 回传梯度，Q 梯度回传到 decoder 输入。
     * 旧版「按特征硬切头」时曾需要最后除以层数做权宜修正；经典全维 QKV 下直接累加即可，不要再除。
     */
    mat_t<val_type> backward(const input_type& delta, mat_t<val_type>& encoder_delta)
    {
        // 反向传播到输出线性层 W_O
        auto delta_concat = m_output_proj.backward(delta);

        // 按行分割 delta_concat 为多个子矩阵（视图）
        auto deltas = vsplit(delta_concat, m_num_heads);

        mat_t<val_type> delta_q(m_d_model, delta.col_num());
        mat_t<val_type> delta_k(m_d_model, m_k_full.col_num());
        mat_t<val_type> delta_v(m_d_model, m_v_full.col_num());
        delta_q = val_type(0);
        delta_k = val_type(0);
        delta_v = val_type(0);
        auto dq_views = vsplit(delta_q, m_num_heads);
        auto dk_views = vsplit(delta_k, m_num_heads);
        auto dv_views = vsplit(delta_v, m_num_heads);

        for (int i = 0; i < m_num_heads; ++i)
        {
            auto g = m_heads[i].backward(deltas[i]);
            dq_views[i].assign(g.delta_q);
            dk_views[i].assign(g.delta_k);
            dv_views[i].assign(g.delta_v);
        }

        encoder_delta += (m_k_net.backward(delta_k) + m_v_net.backward(delta_v));
        return m_q_net.backward(delta_q);
    }

    template <typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_q_net.set_updator(std::forward<upr_arg_types>(args)...);
        m_k_net.set_updator(std::forward<upr_arg_types>(args)...);
        m_v_net.set_updator(std::forward<upr_arg_types>(args)...);
        m_output_proj.set_updator(std::forward<upr_arg_types>(args)...);
        for (auto& head : m_heads)
            head.set_updator(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_q_net.set_lr(lr);
        m_k_net.set_lr(lr);
        m_v_net.set_lr(lr);
        m_output_proj.set_lr(lr);
        for (auto& head : m_heads)
            head.set_lr(lr);
    }

    template <typename init_type>
    void init_weight()
    {
        m_q_net.template init_weight<init_type>();
        m_k_net.template init_weight<init_type>();
        m_v_net.template init_weight<init_type>();
        m_output_proj.template init_weight<init_type>();
        for (auto& head : m_heads)
            head.template init_weight<init_type>();
    }

    std::string net_type(int const& indent = 0) const
    {
        return print_indent(indent) + "MHA(classic,heads:" + std::to_string(m_num_heads)
            + ", d_model:" + std::to_string(m_d_model) + ")";
    }

    void step()
    {
        m_q_net.step();
        m_k_net.step();
        m_v_net.step();
        m_output_proj.step();
        for (auto& head : m_heads)
            head.step();
    }
};

template <typename input_type, template<typename> class updator_type>
class mat_mhca_t : public mat_mha_t<input_type, updator_type>
{
public:
    using base_type = mat_mha_t<input_type, updator_type>;
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type>* m_encoder_output;
    mat_t<val_type>* m_encoder_delta;
public:
    mat_mhca_t(int num_heads = 1, int d_model = 1, bool mask = false, int seq_len = 1)
        : base_type(num_heads, d_model, mask, seq_len)
    {
        m_encoder_output = nullptr;
        m_encoder_delta = nullptr;
    }

    void set_encoder_param(mat_t<val_type>& encoder_output, mat_t<val_type>& encoder_delta)
    {
        m_encoder_output = &encoder_output;
        m_encoder_delta = &encoder_delta;
    }

    mat_t<val_type> forward(const input_type& input)
    {
        if (m_encoder_output == nullptr)
            throw std::runtime_error("Encoder output not set for cross attention");

        return base_type::forward(input, *m_encoder_output);
    }

    mat_t<val_type> backward(const input_type& delta)
    {
        if (m_encoder_delta == nullptr)
            throw std::runtime_error("Encoder delta not set for cross attention");

        return base_type::backward(delta, *m_encoder_delta);
    }

    void step()
    {
        base_type::step();
    }
};

} // namespace jasmine
#endif
