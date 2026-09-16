#ifndef __JAS_MHA_T_HPP__
#define __JAS_MHA_T_HPP__

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>

#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_net_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_RoPE_t.hpp"
#include "jas_kv_cache_t.hpp"
#include "jas_mat_storage.hpp"
#include "jas_mat_gemm.hpp"

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
        return forward_at(q, k, v, 0, 0);
    }

    /**
     * Q 列 j 用绝对位置 q_pos+j，K 列 j 用 k_pos+j 做 RoPE。
     * 训练默认 (0,0)；cross-attn decode 时 Q 用解码绝对位置，K 仍从 0。
     */
    template <typename q_t, typename k_t, typename v_t>
    mat_t<val_type> forward_at(const q_t& q, const k_t& k, const v_t& v,
                               int q_pos, int k_pos)
    {
        detail::store_for_backward(m_q, q);
        detail::store_for_backward(m_k, k);
        detail::store_for_backward(m_v, v);
        if (m_rope)
        {
            m_q = m_rope->forward_at(m_q, q_pos);
            m_k = m_rope->forward_at(m_k, k_pos);
        }

        mat_t<val_type> attn_scores =
            m_q.t().dot(m_k) / std::sqrt(static_cast<val_type>(m_q.row_num()));
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
        return m_v.dot(attn_weights.t());
    }

    /**
     * 推理单步 / 增量：q/k/v 为新 token（可多列 prefill），pos 为第一列的绝对位置。
     * 写入 RoPE 后的 K/V 到 cache，再对 cache 全长做 attention。
     * q_len==1 时无需再填 causal mask（只有一个 query）。
     */
    template <typename q_t, typename k_t, typename v_t>
    mat_t<val_type> forward_one_at(const q_t& q, const k_t& k, const v_t& v,
                                   int pos, kv_cache_t<val_type>& cache)
    {
        mat_t<val_type> q_new(q);
        mat_t<val_type> k_new(k);
        mat_t<val_type> v_new(v);
        if (m_rope)
        {
            q_new = m_rope->forward_at(q_new, pos);
            k_new = m_rope->forward_at(k_new, pos);
        }
        cache.append(k_new, v_new);

        m_q = std::move(q_new);
        auto k_cached = cache.keys();
        auto v_cached = cache.values();

        mat_t<val_type> attn_scores =
            m_q.t().dot(k_cached) / std::sqrt(static_cast<val_type>(m_q.row_num()));
        /*!ANCHOR forward_one 掩码
         * scores 行 = 本步 query（绝对位置 pos..pos+q_len-1），列 = cache 中全部 key（0..len-1）。
         * 多列 prefill 时仍需屏蔽「未来 key」：对 query 行 i，绝对位置 p=pos+i，屏蔽 j > p。
         * 单列时 q_len=1 且 cache 已含当前 key，自然无未来列，可不 mask。
         */
        if (m_mask && attn_scores.row_num() > 1)
        {
            for (int i = 0; i < attn_scores.row_num(); ++i)
            {
                const int abs_pos = pos + i;
                for (int j = abs_pos + 1; j < attn_scores.col_num(); ++j)
                    attn_scores(i, j) = -std::numeric_limits<val_type>::infinity();
            }
        }
        auto attn_weights = m_softmax.forward(attn_scores);
        return v_cached.dot(attn_weights.t());
    }

    /**
     * cross-attn 推理：Q 为新 token，K/V 已在 cache 中（encode 时写入，不再 append）。
     * q 列 j 使用绝对位置 pos+j 做 RoPE；K 侧 RoPE 已在写入 cache 时完成。
     */
    template <typename q_t>
    mat_t<val_type> attend_cached(const q_t& q, int pos, kv_cache_t<val_type>& cache)
    {
        if (cache.length() == 0)
            throw std::runtime_error("attend_cached: empty KV cache");

        mat_t<val_type> q_new(q);
        if (m_rope)
            q_new = m_rope->forward_at(q_new, pos);

        m_q = std::move(q_new);
        auto k_cached = cache.keys();
        auto v_cached = cache.values();

        mat_t<val_type> attn_scores =
            m_q.t().dot(k_cached) / std::sqrt(static_cast<val_type>(m_q.row_num()));
        // cross-attn 通常 m_mask=false；若误开 mask，多列时按绝对位置屏蔽未来 key
        if (m_mask && attn_scores.row_num() > 1)
        {
            for (int i = 0; i < attn_scores.row_num(); ++i)
            {
                const int abs_pos = pos + i;
                for (int j = abs_pos + 1; j < attn_scores.col_num(); ++j)
                    attn_scores(i, j) = -std::numeric_limits<val_type>::infinity();
            }
        }
        auto attn_weights = m_softmax.forward(attn_scores);
        return v_cached.dot(attn_weights.t());
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
*
* GQA（Grouped-Query Attention）不是另一套算法，而是把「K/V 头数」从 Q 头数上解耦：
*   - n_kv_heads == n_heads  → 经典 MHA（默认，行为与旧版逐位一致）
*   - 1 < n_kv_heads < n_heads → GQA：连续 group_size = n_heads/n_kv_heads 个 Q 头共享一个 KV 头
*   - n_kv_heads == 1        → MQA
* 与 MHA 的差异只有三处：① K/V 投影输出宽度变成 n_kv_heads*d_head；② KV cache 只存 n_kv_heads
* 份（省显存之处）；③ 反向时共享同一 KV 头的多个 Q 头梯度要**累加**。
* 注意力核 mat_head_gen_t 完全不需要改动 —— 它本来就只认切好的 Q/K/V。
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
    int m_num_heads;        // Q 头数
    int m_num_kv_heads;     // K/V 头数：== m_num_heads 即经典 MHA，== 1 即 MQA，中间即 GQA
    int m_group_size;       // 每个 KV 头被多少个 Q 头共享 = m_num_heads / m_num_kv_heads
    int m_d_model;
    int m_d_head;           // 每个头的宽度 = d_model / num_heads
    int m_d_kv;             // K/V 投影输出宽度 = m_num_kv_heads * m_d_head（GQA 下 < d_model）
    bool m_mask = false;
    bool m_use_rope = true;   // false = 绝对位置模型（GPT-2），Q/K 不做 RoPE
    rope_pair_layout m_rope_layout = rope_pair_layout::interleaved;

    // 前向缓存，供 backward 切分梯度
    mat_t<val_type> m_q_full, m_k_full, m_v_full;

    // decoder self-attn 推理用：每个 **KV 头**一份（投影+RoPE 后）。
    // GQA 下 m_num_kv_heads < m_num_heads，这正是 KV cache 的省显存之处。
    std::vector<kv_cache_t<val_type>> m_kv_caches;

    void ensure_kv_caches()
    {
        if (static_cast<int>(m_kv_caches.size()) != m_num_kv_heads)
            m_kv_caches.resize(m_num_kv_heads);
    }

    /** Q 头 i 对应的 K/V 头下标（连续 group_size 个 Q 头共享一个 KV 头） */
    int kv_head_of(int const& q_head) const { return q_head / m_group_size; }

    /**
     * 把 src 累加到 dst 的 [row0, row0+src.row_num()) 行。
     * mat_view_t 只有 assign 没有 +=，而 GQA 下多个 Q 头要共享同一个 KV 头的梯度，必须累加。
     */
    static void add_rows(mat_t<val_type>& dst, int const& row0, const mat_t<val_type>& src)
    {
        for (int i = 0; i < src.row_num(); ++i)
            for (int j = 0; j < src.col_num(); ++j)
                dst(row0 + i, j) += src(i, j);
    }

    /**
     * 把一个 KV 头的新列写入 cache：K 做 RoPE，V 不旋转（与 fill_kv_cache 口径一致）。
     * GQA 下**每个 KV 头只能调用一次** —— 共享该 KV 头的多个 Q 头若各自调用，
     * 同一份 K/V 会被重复 append，cache 长度与内容都会错。
     */
    void append_kv_head(int const& kv_head, const mat_t<val_type>& k_h,
                        const mat_t<val_type>& v_h, int const& pos)
    {
        mat_t<val_type> k_rot(k_h);
        if (auto rope = m_heads[kv_head].rope())
            k_rot = rope->forward_at(k_rot, pos);
        m_kv_caches[kv_head].append(k_rot, v_h);
    }

public:
    mat_mha_t(int num_heads = 1, int d_model = 1, bool mask = false, int seq_len = 1,
              int n_kv_heads = 0)
    {
        /* 设置默认构造参数目的是让其可以没有参数进行构造，以便放入其他复杂网络结构中 */
        set_param(num_heads, d_model, mask, seq_len, n_kv_heads);
    }

    /**
     * 配置维度。
     * n_kv_heads == 0（默认）表示与 num_heads 相同 → 经典 MHA，行为与旧版逐位一致；
     * 0 < n_kv_heads < num_heads → GQA；n_kv_heads == 1 → MQA。
     * 约束：d_model % num_heads == 0 且 num_heads % n_kv_heads == 0。
     */
    void set_param(int num_heads, int d_model, bool mask = false, int seq_len = 1,
                   int n_kv_heads = 0)
    {
        if (d_model % num_heads != 0)
            throw std::runtime_error("d_model must be divisible by num_heads");
        if (n_kv_heads <= 0)
            n_kv_heads = num_heads;      // 默认退化为 MHA
        if (num_heads % n_kv_heads != 0)
            throw std::runtime_error("num_heads must be divisible by n_kv_heads");

        m_num_heads = num_heads;
        m_num_kv_heads = n_kv_heads;
        m_group_size = num_heads / n_kv_heads;
        m_d_model = d_model;
        m_d_head = d_model / num_heads;
        m_d_kv = m_num_kv_heads * m_d_head;
        m_mask = mask;

        m_q_net.reinit(std::vector<int>{d_model, d_model});
        // GQA 关键差异：K/V 只投影出 n_kv_heads 个头，而不是 d_model
        m_k_net.reinit(std::vector<int>{d_model, m_d_kv});
        m_v_net.reinit(std::vector<int>{d_model, m_d_kv});
        m_output_proj.reinit(std::vector<int>{d_model, d_model});

        m_heads.resize(num_heads);
        for (int i = 0; i < num_heads; ++i)
            m_heads[i].set_param(m_d_head, mask, seq_len);
        ensure_kv_caches();
        if (seq_len > 0)
            reserve_kv_cache(seq_len);
        clear_kv_cache();
        bind_rope();
    }

    void set_kv_cache_mode(kv_cache_mode mode)
    {
        ensure_kv_caches();
        for (auto& c : m_kv_caches)
            c.set_mode(mode);
    }

    void reserve_kv_cache(int max_seq)
    {
        ensure_kv_caches();
        for (auto& c : m_kv_caches)
            c.reserve(m_d_head, max_seq);
    }

    void clear_kv_cache()
    {
        for (auto& c : m_kv_caches)
            c.clear();
    }

    int kv_cache_length() const
    {
        return m_kv_caches.empty() ? 0 : m_kv_caches.front().length();
    }

    /**
     * 用整段 K/V（已按 KV 头拼接的 m_d_kv×seq）填满各 KV 头 cache。
     * K 在写入前按 k_start_pos 做 RoPE；V 不旋转。
     * 供 cross-attn：encode 后一次性写入，之后 decode 只读。
     */
    void fill_kv_cache(mat_t<val_type> k_full, mat_t<val_type> v_full, int k_start_pos = 0)
    {
        if (k_full.row_num() != m_d_kv || v_full.row_num() != m_d_kv)
            throw std::runtime_error("fill_kv_cache: row dim must be n_kv_heads * d_head");
        if (k_full.col_num() != v_full.col_num())
            throw std::runtime_error("fill_kv_cache: K/V seq length mismatch");

        ensure_kv_caches();
        const int seq = k_full.col_num();
        for (auto& c : m_kv_caches)
            c.reserve(m_d_head, std::max(1, seq));
        clear_kv_cache();

        auto k_splits = vsplit(k_full, m_num_kv_heads);
        auto v_splits = vsplit(v_full, m_num_kv_heads);
        for (int i = 0; i < m_num_kv_heads; ++i)
        {
            if (auto rope = m_heads[i].rope())
            {
                mat_t<val_type> k_h = rope->forward_at(k_splits[i], k_start_pos);
                m_kv_caches[i].append(k_h, v_splits[i]);
            }
            else
            {
                m_kv_caches[i].append(k_splits[i], v_splits[i]);
            }
        }
    }

    /** 对固定 memory 做 W_K/W_V 投影后写入 KV cache（cross-attn prepare） */
    void cache_kv_from_memory(const input_type& memory, int k_start_pos = 0)
    {
        if (memory.row_num() != m_d_model)
            throw std::runtime_error("cache_kv_from_memory: row dim must match d_model");
        fill_kv_cache(m_k_net.forward(memory), m_v_net.forward(memory), k_start_pos);
    }

    /**
     * 推理：只用 Q 投影，对已填充的 KV cache 做 attention（不 append）。
     * cross-attn 在 prepare / cache_kv_from_memory 之后调用。
     */
    mat_t<val_type> forward_one_cached_kv(const input_type& input, int q_pos)
    {
        if (input.row_num() != m_d_model)
            throw std::runtime_error("Input row dimension must match d_model");
        if (kv_cache_length() == 0)
            throw std::runtime_error("forward_one_cached_kv: KV cache empty; call cache_kv_from_memory first");

        m_q_full = m_q_net.forward(input);
        auto q_splits = vsplit(m_q_full, m_num_heads);

        std::vector<mat_t<val_type>> head_outputs(m_num_heads);
        const bool par_heads =
            detail::mha_heads_should_parallel(m_num_heads, kv_cache_length(), m_d_head);
#ifdef JASMINE_USE_OPENMP
#pragma omp parallel for schedule(static) if(par_heads)
#endif
        for (int i = 0; i < m_num_heads; ++i)
            head_outputs[i] = m_heads[i].attend_cached(
                q_splits[i], q_pos, m_kv_caches[kv_head_of(i)]);

        return m_output_proj.forward(vconcat(head_outputs));
    }

    /** 显式绑定/刷新各头的 RoPE（同 d_head + 同配对约定共享注册中心条目） */
    void bind_rope(int max_seq_len = 0)
    {
        std::shared_ptr<RoPE_net_t<mat_t<val_type>>> rope;
        if (m_use_rope && m_d_head > 0 && m_d_head % 2 == 0)
            rope = rope_registry_t<val_type>::instance().get(m_d_head, max_seq_len, m_rope_layout);
        for (auto& head : m_heads)
            head.set_rope(rope);
    }

    /**
     * 把各头共享的 RoPE 缓存切成 static 并预留到 max_seq 个位置。
     *
     * 注册中心按 (d_head, layout) 共享，`set_default_mode` 只作用于**新建**的条目，
     * 所以这里直接对拿到的那一个对象设置，不依赖调用顺序，也不改全局默认模式
     * （否则会波及同进程里其它维度的用法）。
     *
     * 预留长度取单调最大值：同一个 d_head 被多个模型共享时，先来的大长度不会被后来的小长度缩掉。
     *
     * static 之后越界是**抛异常**而不是默默扩容 —— 这是把「上下文上限」变成可检查的不变量，
     * 而不是一句注释。因此要在生成之前调用（此时还没有视图被别人持有）。
     */
    void reserve_rope_cache(int max_seq, rope_cache_mode mode = rope_cache_mode::static_fixed)
    {
        if (max_seq <= 0)
            throw std::invalid_argument("RoPE reserve length must be positive");
        if (!m_use_rope || m_d_head <= 0 || m_d_head % 2 != 0)
            return;                     // 绝对位置模型（GPT-2 等）没有 RoPE 可预留

        auto& reg = rope_registry_t<val_type>::instance();
        std::shared_ptr<RoPE_net_t<mat_t<val_type>>> rope =
            reg.get(m_d_head, 0, m_rope_layout);
        if (!rope)
            return;

        // 同一个 d_head 被多层/多模型共享，上层会逐层调到这里。已经就位就直接返回，
        // 否则 reserve() 在 static 模式下会重置记账并整表重填，白白重算 n_layers 遍。
        const int already = rope->cache_max_seq_len();
        if (rope->cache_mode() == mode && already >= max_seq)
            return;

        const int want = std::max(max_seq, already);      // 先读后切：set_cache_mode 会清掉这个值
        if (rope->cache_mode() != mode)
            rope->set_cache_mode(mode);
        rope->reserve(want);

        for (auto& head : m_heads)      // 头们共享同一个指针，这里只是保证绑定关系是最新的
            head.set_rope(rope);
    }

    /**
     * 选择 RoPE 的特征配对约定（见 rope_pair_layout）。
     * 导出 HF LLaMA 系权重时必须用 half_split；jasmine 原生训练保持 interleaved。
     */
    void set_rope_pair_layout(rope_pair_layout layout)
    {
        if (m_rope_layout == layout)
            return;
        m_rope_layout = layout;
        bind_rope();
    }

    rope_pair_layout pair_layout() const
    {
        return m_rope_layout;
    }

    /**
     * 开关 RoPE。默认 true（jasmine 原有行为）。
     * 绝对位置模型（如 GPT-2，用 wpe 位置嵌入）必须置 false，否则 Q/K 会被额外旋转，
     * 与参考实现的 logits 对不上。
     */
    void set_use_rope(bool use)
    {
        m_use_rope = use;
        bind_rope();
    }

    /** W_Q 投影 [d_model, d_model]；供权重加载器写入 */
    proj_type& q_proj() { return m_q_net; }
    proj_type const& q_proj() const { return m_q_net; }
    /** W_K 投影 [m_d_kv, d_model]：GQA 下输出宽度是 n_kv_heads*d_head（< d_model） */
    proj_type& k_proj() { return m_k_net; }
    proj_type const& k_proj() const { return m_k_net; }
    /** W_V 投影 [m_d_kv, d_model]：同 W_K */
    proj_type& v_proj() { return m_v_net; }
    proj_type const& v_proj() const { return m_v_net; }
    /** W_O 输出投影 [d_model, d_model] */
    proj_type& out_proj() { return m_output_proj; }
    proj_type const& out_proj() const { return m_output_proj; }

    // ---- GQA 维度自省（权重加载器据此切分 fused QKV、测试据此断言分组） ----
    /** Q 头数 */
    int num_heads() const { return m_num_heads; }
    /** K/V 头数；== num_heads() 即 MHA，== 1 即 MQA */
    int num_kv_heads() const { return m_num_kv_heads; }
    /** 每个 KV 头被多少个 Q 头共享（MHA 下恒为 1） */
    int group_size() const { return m_group_size; }
    /** 每个头的宽度 */
    int d_head() const { return m_d_head; }
    /** K/V 投影输出宽度 = num_kv_heads() * d_head() */
    int d_kv() const { return m_d_kv; }
    /** 单个 Q 头（测试/自省用，可查其绑定的 RoPE 等） */
    head_type& head(int i) { return m_heads.at(static_cast<std::size_t>(i)); }
    const head_type& head(int i) const { return m_heads.at(static_cast<std::size_t>(i)); }

    mat_t<val_type> forward(const input_type& input)
    {
        // Step 1: 检查输入维度是否合法
        if (input.row_num() != m_d_model)
            throw std::runtime_error("Input row dimension must match d_model");

        // Step 2: 全维 QKV 投影（各头共享同一组 W_Q/W_K/W_V）
        {
            const bool par_proj =
                detail::gemm_should_parallel(m_d_model, input.col_num(), m_d_model);
#ifdef JASMINE_USE_OPENMP
            if (par_proj)
            {
#pragma omp parallel sections
                {
#pragma omp section
                    m_q_full = m_q_net.forward(input);
#pragma omp section
                    m_k_full = m_k_net.forward(input);
#pragma omp section
                    m_v_full = m_v_net.forward(input);
                }
            }
            else
#endif
            {
                m_q_full = m_q_net.forward(input);
                m_k_full = m_k_net.forward(input);
                m_v_full = m_v_net.forward(input);
            }
        }

        // Step 3: 按头切分投影结果（切的是 Q/K/V，不是原始输入特征）
        // Q 切成 n_q_heads 份（d_model 宽），K/V 切成 n_kv_heads 份（d_kv 宽）
        auto q_splits = vsplit(m_q_full, m_num_heads);
        auto k_splits = vsplit(m_k_full, m_num_kv_heads);
        auto v_splits = vsplit(m_v_full, m_num_kv_heads);

        // Step 4: 每个头独立 attend（RoPE / score / mask / softmax / V）
        // GQA：连续的 group_size 个 Q 头共用同一个 KV 头
        std::vector<mat_t<val_type>> head_outputs(m_num_heads);
        const int seq_len = input.col_num();
        const bool par_heads = detail::mha_heads_should_parallel(m_num_heads, seq_len, m_d_head);
#ifdef JASMINE_USE_OPENMP
#pragma omp parallel for schedule(static) if(par_heads)
#endif
        for (int i = 0; i < m_num_heads; ++i)
            head_outputs[i] = m_heads[i].forward(
                q_splits[i], k_splits[kv_head_of(i)], v_splits[kv_head_of(i)]);

        // Step 5: 拼接所有头的输出，再经 W_O 映射回 d_model
        mat_t<val_type> concatenated_output = vconcat(head_outputs);
        auto final_output = m_output_proj.forward(concatenated_output);

        return final_output;
    }

    /**
     * 推理 forward_one（decoder self-attn）：只投影本步 token，追加 KV cache，attend 历史。
     * 输入通常为 d_model×1；也支持多列 prefill（pos = 当前 cache.length()）。
     * 训练 / teacher forcing 仍用 forward()，不触碰 cache。
     */
    mat_t<val_type> forward_one(const input_type& input)
    {
        if (input.row_num() != m_d_model)
            throw std::runtime_error("Input row dimension must match d_model");
        ensure_kv_caches();

        const int pos = kv_cache_length();
        const int seq_len = input.col_num();
        m_q_full = m_q_net.forward(input);
        m_k_full = m_k_net.forward(input);
        m_v_full = m_v_net.forward(input);

        auto q_splits = vsplit(m_q_full, m_num_heads);
        auto k_splits = vsplit(m_k_full, m_num_kv_heads);
        auto v_splits = vsplit(m_v_full, m_num_kv_heads);

        /*!ANCHOR GQA 的 KV cache 必须"每个 KV 头 append 一次"
         * 共享同一个 KV 头的多个 Q 头如果各自 append，同一份 K/V 会被写入两次：
         * 形状不报错、单步看似可用，但 cache 长度与内容全错，到多轮对话才炸。
         * 因此拆成两步：先按 KV 头写入（K 做 RoPE），再让各 Q 头只读不写（attend_cached）。
         * group_size == 1（MHA）时，这与原来的 forward_one_at 完全等价。
         */
        for (int g = 0; g < m_num_kv_heads; ++g)
            append_kv_head(g, k_splits[g], v_splits[g], pos);

        std::vector<mat_t<val_type>> head_outputs(m_num_heads);
        // decode 单列时 cache 长度决定 attend 工作量
        const int attend_seq = std::max(seq_len, kv_cache_length());
        const bool par_heads = detail::mha_heads_should_parallel(m_num_heads, attend_seq, m_d_head);
#ifdef JASMINE_USE_OPENMP
#pragma omp parallel for schedule(static) if(par_heads)
#endif
        for (int i = 0; i < m_num_heads; ++i)
            head_outputs[i] = m_heads[i].attend_cached(
                q_splits[i], pos, m_kv_caches[kv_head_of(i)]);

        mat_t<val_type> concatenated_output = vconcat(head_outputs);
        return m_output_proj.forward(concatenated_output);
    }

    mat_t<val_type> forward(const input_type& input, const input_type& encoder_input)
    {
        return forward_at(input, encoder_input, 0);
    }

    /**
     * 交叉注意力；q_pos 为 decoder 侧 Q 的 RoPE 起点（训练为 0，decode 为绝对时间步）。
     */
    mat_t<val_type> forward_at(const input_type& input, const input_type& encoder_input, int q_pos)
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
        auto k_splits = vsplit(m_k_full, m_num_kv_heads);
        auto v_splits = vsplit(m_v_full, m_num_kv_heads);

        // Step 4: 每个头独立进行正向传播（Q 用 q_pos，K 从 0）；GQA 下多个 Q 头共享 KV 头
        std::vector<mat_t<val_type>> head_outputs(m_num_heads);
        const int seq_len = input.col_num();
        const bool par_heads = detail::mha_heads_should_parallel(m_num_heads, seq_len, m_d_head);
#ifdef JASMINE_USE_OPENMP
#pragma omp parallel for schedule(static) if(par_heads)
#endif
        for (int i = 0; i < m_num_heads; ++i)
            head_outputs[i] = m_heads[i].forward_at(
                q_splits[i], k_splits[kv_head_of(i)], v_splits[kv_head_of(i)], q_pos, 0);

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
        // delta_q 是 Q 头数宽（d_model），delta_k/v 是 KV 头数宽（d_kv）
        const int T = delta.col_num();
        mat_t<val_type> delta_q(m_d_model, T);
        mat_t<val_type> delta_k(m_d_kv, T);
        mat_t<val_type> delta_v(m_d_kv, T);
        delta_q = val_type(0);
        delta_k = val_type(0);
        delta_v = val_type(0);
        auto dq_views = vsplit(delta_q, m_num_heads);

        for (int i = 0; i < m_num_heads; ++i)
        {
            auto g = m_heads[i].backward(deltas[i]);
            dq_views[i].assign(g.delta_q);
            // GQA：多个 Q 头共享一个 KV 头 → 梯度必须**累加**（assign 会只剩最后一个头）
            const int kv = kv_head_of(i);
            add_rows(delta_k, kv * m_d_head, g.delta_k);
            add_rows(delta_v, kv * m_d_head, g.delta_v);
        }

        return mat_t<val_type>(
            m_q_net.backward(delta_q) + m_k_net.backward(delta_k) + m_v_net.backward(delta_v));
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
        mat_t<val_type> delta_k(m_d_kv, m_k_full.col_num());
        mat_t<val_type> delta_v(m_d_kv, m_v_full.col_num());
        delta_q = val_type(0);
        delta_k = val_type(0);
        delta_v = val_type(0);
        auto dq_views = vsplit(delta_q, m_num_heads);

        for (int i = 0; i < m_num_heads; ++i)
        {
            auto g = m_heads[i].backward(deltas[i]);
            dq_views[i].assign(g.delta_q);
            // GQA：共享 KV 头的多个 Q 头梯度累加
            const int kv = kv_head_of(i);
            add_rows(delta_k, kv * m_d_head, g.delta_k);
            add_rows(delta_v, kv * m_d_head, g.delta_v);
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
        std::string tag = "MHA(classic";
        if (m_num_kv_heads == 1)
            tag = "MQA";
        else if (m_num_kv_heads != m_num_heads)
            tag = "GQA";
        return print_indent(indent) + tag
            + ",q_heads:" + std::to_string(m_num_heads)
            + ",kv_heads:" + std::to_string(m_num_kv_heads)
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
    // decode 时 Q 的 RoPE 绝对位置（与同层 self-attn 时间步对齐）；clear 时归零
    int m_q_rope_pos = 0;
public:
    mat_mhca_t(int num_heads = 1, int d_model = 1, bool mask = false, int seq_len = 1,
               int n_kv_heads = 0)
        : base_type(num_heads, d_model, mask, seq_len, n_kv_heads)
    {
        m_encoder_output = nullptr;
        m_encoder_delta = nullptr;
    }

    void set_encoder_param(mat_t<val_type>& encoder_output, mat_t<val_type>& encoder_delta)
    {
        m_encoder_output = &encoder_output;
        m_encoder_delta = &encoder_delta;
    }

    void reset_cross_q_pos()
    {
        m_q_rope_pos = 0;
    }

    /**
     * encode 之后调用：把 encoder memory 投影为 K/V（RoPE 后）写入 cache。
     * 之后 forward_one 只算 Q，复用这份 K/V。
     */
    void prepare_cross_kv()
    {
        if (m_encoder_output == nullptr)
            throw std::runtime_error("Encoder output not set for cross attention");
        this->cache_kv_from_memory(*m_encoder_output, 0);
        m_q_rope_pos = 0;
    }

    mat_t<val_type> forward(const input_type& input)
    {
        if (m_encoder_output == nullptr)
            throw std::runtime_error("Encoder output not set for cross attention");

        return base_type::forward(input, *m_encoder_output);
    }

    /**
     * cross-attn 推理：K/V 来自 prepare_cross_kv 的 cache；本步只投影 Q。
     */
    mat_t<val_type> forward_one(const input_type& input)
    {
        if (m_encoder_output == nullptr)
            throw std::runtime_error("Encoder output not set for cross attention");
        if (this->kv_cache_length() == 0)
            prepare_cross_kv();

        auto out = this->forward_one_cached_kv(input, m_q_rope_pos);
        m_q_rope_pos += input.col_num();
        return out;
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
