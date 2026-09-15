#ifndef __JAS_GPT2_T_HPP__
#define __JAS_GPT2_T_HPP__

#include <sstream>
#include <string>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_mat_utility.hpp"
#include "jas_net_t.hpp"
#include "jas_mha_t.hpp"
#include "jas_embedding_t.hpp"
#include "jas_gelu_t.hpp"

namespace jasmine {

/*
 * GPT-2 前向对齐所需的模块（inference only）。
 *
 * 与 jasmine 原有 enc-dec / decoder_only 栈的关键差异：
 *   1. Pre-norm：LayerNorm 在残差分支内侧，即 x + SubLayer(LN(x))；
 *      jasmine 原有的 res_mha_norm_t 是 post-norm：LN(SubLayer(x) + x)。
 *   2. 绝对位置编码：wte(ids) + wpe(position)，不是 RoPE；因此注意力必须 set_use_rope(false)。
 *   3. FFN 激活是 gelu_new（tanh 近似），不是 ReLU。
 *   4. 栈尾多一个 ln_f，lm_head 与 wte 权重绑定且无 bias。
 *
 * 层级（与 encoder_layer_t 同形：两个子块，各含 LN + 主体）：
 *   block[0] = residual( LayerNorm -> MHA(causal, no RoPE) )
 *   block[1] = residual( LayerNorm -> Linear -> GELU -> Linear )
 */

template<typename val_type, template<typename> class updator_type>
using gpt2_attn_branch_t = complex_net_builder_t<val_type>
    ::template push_back_updatable<layer_norm_net_t, updator_type>
    ::template push_back_updatable<mat_mha_t, updator_type>
    ::type;

template<typename val_type, template<typename> class updator_type>
using gpt2_res_attn_t = residual_net_t<gpt2_attn_branch_t<val_type, updator_type>>;

template<typename val_type, template<typename> class updator_type>
using gpt2_ffn_branch_t = complex_net_builder_t<val_type>
    ::template push_back_updatable<layer_norm_net_t, updator_type>
    ::template push_back_updatable<weight_net_t, updator_type>
    ::template push_back_staticnet<gelu_net_t>
    ::template push_back_updatable<weight_net_t, updator_type>
    ::type;

template<typename val_type, template<typename> class updator_type>
using gpt2_res_ffn_t = residual_net_t<gpt2_ffn_branch_t<val_type, updator_type>>;

template<typename val_type, template<typename> class updator_type>
using gpt2_block_t = complex_net_builder_t<val_type>
    ::template push_back_impl<gpt2_res_attn_t<val_type, updator_type>>
    ::template push_back_impl<gpt2_res_ffn_t<val_type, updator_type>>
    ::type;

template <typename input_type, template<typename> class updator_type>
class gpt2_model_t
{
public:
    using val_type = typename input_type::ele_type;
    using norm_type = layer_norm_net_t<mat_t<val_type>, updator_type>;
    using attn_type = mat_mha_t<mat_t<val_type>, updator_type>;
    using linear_type = weight_net_t<mat_t<val_type>, updator_type>;
    using embed_type = embedding_net_t<mat_t<val_type>, updator_type>;
    using block_type = gpt2_block_t<val_type, updator_type>;

private:
    embed_type m_wte;                       // [d_model, vocab]
    embed_type m_wpe;                       // [d_model, n_pos]
    std::vector<block_type> m_blocks;
    norm_type m_ln_f;
    linear_type m_lm_head;                  // [vocab, d_model]，与 wte 绑定，bias 恒为 0

    int m_n_layers = 1;
    int m_n_heads = 1;
    int m_d_model = 1;
    int m_d_ff = 0;
    int m_vocab = 1;
    int m_n_pos = 1;

public:
    gpt2_model_t(int n_layers = 1, int n_heads = 1, int d_model = 1, int d_ff = 0,
                 int vocab = 1, int n_pos = 1)
    {
        set_param(n_layers, n_heads, d_model, d_ff, vocab, n_pos);
    }

    /**
     * 配置全部子模块维度。mat_mha_t 是 stable 网络（不可 reinit），因此这里逐个显式设置，
     * 不走 complex_net_t::reinit（后者会跳过 residual 包裹的层）。
     */
    void set_param(int n_layers, int n_heads, int d_model, int d_ff, int vocab, int n_pos)
    {
        if (d_ff == 0) d_ff = d_model * 4;
        m_n_layers = n_layers;
        m_n_heads = n_heads;
        m_d_model = d_model;
        m_d_ff = d_ff;
        m_vocab = vocab;
        m_n_pos = n_pos;

        m_wte.reinit(std::vector<int>{vocab, d_model});
        m_wpe.reinit(std::vector<int>{n_pos, d_model});
        m_ln_f.set_param(d_model);
        m_lm_head.reinit(std::vector<int>{d_model, vocab});

        m_blocks.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            ln_1(i).set_param(d_model);
            ln_2(i).set_param(d_model);
            attn(i).set_param(n_heads, d_model, true, 1);
            // GPT-2 用绝对位置嵌入，注意力侧不做 RoPE
            attn(i).set_use_rope(false);
            mlp_fc(i).reinit(std::vector<int>{d_model, d_ff});
            mlp_proj(i).reinit(std::vector<int>{d_ff, d_model});
        }
        // lm_head 无 bias
        m_lm_head.bias() = val_type(0);
    }

    /** 嵌入层：ids 1×T -> wte(ids) + wpe(0..T-1)，输出 d_model×T */
    mat_t<val_type> embed(const mat_t<val_type>& ids)
    {
        const int T = ids.col_num();
        mat_t<val_type> h = m_wte.forward(ids);

        mat_t<val_type> pos_ids(1, T);
        for (int t = 0; t < T; ++t)
            pos_ids(0, t) = static_cast<val_type>(t);
        return (h + m_wpe.forward(pos_ids)).clone();
    }

    /** 输出头：ln_f -> lm_head(tied)，d_model×T -> vocab×T */
    mat_t<val_type> head(const mat_t<val_type>& hidden)
    {
        return m_lm_head.forward(m_ln_f.forward(hidden));
    }

    /** 整段前向：ids 1×T -> logits vocab×T */
    mat_t<val_type> forward(const mat_t<val_type>& ids)
    {
        mat_t<val_type> h = embed(ids);
        for (auto& block : m_blocks)
            h = block.forward(h);
        return head(h);
    }

    /**
     * 逐层输出（ln_f 之前）：[嵌入输出, block0 输出, ..., blockN-1 输出]。
     * 共 n_layers+1 项，用于按层与参考实现比对，快速定位不一致发生在哪一层。
     *
     * 注意：HuggingFace 的 `output_hidden_states=True` 里 `hidden_states[-1]` 是
     * **经过 ln_f 之后**的值（transformers>=5），不能直接拿来当最后一个 block 的输出；
     * `tools/export_gpt2.py` 已手动补算真正的 pre-ln_f 末层值。
     * 这也说明只比 logits 不够：ln_f 会掩盖仿射级别的偏差。
     */
    std::vector<mat_t<val_type>> forward_stages(const mat_t<val_type>& ids)
    {
        std::vector<mat_t<val_type>> stages;
        stages.reserve(m_blocks.size() + 1);
        stages.push_back(embed(ids));
        for (auto& block : m_blocks)
            stages.push_back(block.forward(stages.back()));
        return stages;
    }

    /** 单个 block 的前向（逐层对齐 / 调试用） */
    mat_t<val_type> block_forward(int const& i, const mat_t<val_type>& hidden)
    {
        return m_blocks[i].forward(hidden);
    }

    /**
     * 增量推理：ids 1×1（单个新 token），pos 为其绝对位置（用于索引 wpe）。
     * 各层 self-attn 走 KV cache；调用前需 clear_kv_cache()，预填阶段先用 forward_one 喂 prompt。
     */
    mat_t<val_type> forward_one(const mat_t<val_type>& ids, int pos)
    {
        mat_t<val_type> h = m_wte.forward(ids);
        mat_t<val_type> pos_ids(1, 1);
        pos_ids(0, 0) = static_cast<val_type>(pos);
        h = (h + m_wpe.forward(pos_ids)).clone();

        for (auto& block : m_blocks)
            h = block.forward_one(h);

        h = m_ln_f.forward_one(h);
        return m_lm_head.forward_one(h);
    }

    /** prompt 预填：整段前向并填充 KV cache（等价于对 0..T-1 逐步 forward_one） */
    mat_t<val_type> prefill(const mat_t<val_type>& ids)
    {
        clear_kv_cache();
        const int T = ids.col_num();
        mat_t<val_type> logits;
        for (int t = 0; t < T; ++t)
            logits = forward_one(ids.view(0, t, 1, 1).clone(), t);
        return logits;
    }

    void clear_kv_cache()
    {
        for (auto& block : m_blocks)
            attn_of(block).clear_kv_cache();
    }

    void reserve_kv_cache(int max_seq)
    {
        for (auto& block : m_blocks)
            attn_of(block).reserve_kv_cache(max_seq);
    }

    void set_kv_cache_mode(kv_cache_mode mode)
    {
        for (auto& block : m_blocks)
            attn_of(block).set_kv_cache_mode(mode);
    }

    int kv_cache_length() const
    {
        return m_blocks.empty() ? 0 : attn_of(m_blocks.front()).kv_cache_length();
    }

    /** lm_head.weight = wte.weight 的转置（GPT-2 权重绑定） */
    void tie_word_embeddings()
    {
        m_lm_head.weight() = m_wte.weight().t().clone();
        m_lm_head.bias() = val_type(0);
    }

    template <typename init_type>
    void init_weight()
    {
        m_wte.template init_weight<init_type>();
        m_wpe.template init_weight<init_type>();
        m_ln_f.template init_weight<init_type>();
        m_lm_head.template init_weight<init_type>();
        for (auto& block : m_blocks)
            block.template init_weight<init_type>();
    }

    template<typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        for (auto& block : m_blocks)
            block.set_updator(std::forward<upr_arg_types>(args)...);
        m_ln_f.set_updator(std::forward<upr_arg_types>(args)...);
        m_wte.set_updator(std::forward<upr_arg_types>(args)...);
        m_wpe.set_updator(std::forward<upr_arg_types>(args)...);
        m_lm_head.set_updator(std::forward<upr_arg_types>(args)...);
    }

    // ---- 权重加载器所需的访问器 ----

    embed_type& wte() { return m_wte; }
    embed_type const& wte() const { return m_wte; }
    embed_type& wpe() { return m_wpe; }
    embed_type const& wpe() const { return m_wpe; }
    norm_type& ln_f() { return m_ln_f; }
    norm_type const& ln_f() const { return m_ln_f; }
    linear_type& lm_head() { return m_lm_head; }
    linear_type const& lm_head() const { return m_lm_head; }

    /** ln_1（attention 前的 LayerNorm），对应 GPT-2 `ln_1.*` */
    norm_type& ln_1(int const& i) { return m_blocks[i].template get<0, 0>(); }
    norm_type const& ln_1(int const& i) const { return m_blocks[i].template get<0, 0>(); }
    /** attention 本体 `attn.*` */
    attn_type& attn(int const& i) { return m_blocks[i].template get<0, 1>(); }
    attn_type const& attn(int const& i) const { return m_blocks[i].template get<0, 1>(); }
    /** ln_2（FFN 前的 LayerNorm），对应 GPT-2 `ln_2.*` */
    norm_type& ln_2(int const& i) { return m_blocks[i].template get<1, 0>(); }
    norm_type const& ln_2(int const& i) const { return m_blocks[i].template get<1, 0>(); }
    /** FFN 第一层，对应 GPT-2 `mlp.c_fc` */
    linear_type& mlp_fc(int const& i) { return m_blocks[i].template get<1, 1>(); }
    linear_type const& mlp_fc(int const& i) const { return m_blocks[i].template get<1, 1>(); }
    /** FFN 第二层，对应 GPT-2 `mlp.c_proj` */
    linear_type& mlp_proj(int const& i) { return m_blocks[i].template get<1, 3>(); }
    linear_type const& mlp_proj(int const& i) const { return m_blocks[i].template get<1, 3>(); }

    int n_layers() const { return m_n_layers; }
    int n_heads() const { return m_n_heads; }
    int d_model() const { return m_d_model; }
    int d_ff() const { return m_d_ff; }
    int vocab_size() const { return m_vocab; }
    int n_pos() const { return m_n_pos; }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "gpt2_model_t:(layers:" << m_n_layers
           << ", heads:" << m_n_heads << ", d_model:" << m_d_model
           << ", d_ff:" << m_d_ff << ", vocab:" << m_vocab
           << ", n_pos:" << m_n_pos << ", pre-norm, abs-pos, gelu_new)\n";
        ss << print_indent(indent + 2) << "wte(embed): " << m_wte.net_type() << "\n";
        ss << print_indent(indent + 2) << "wpe(embed): " << m_wpe.net_type() << "\n";
        for (int i = 0; i < m_n_layers; ++i)
        {
            ss << print_indent(indent + 2) << "block " << i
               << ": ln_1 -> attn(causal,no-rope) -> ln_2 -> mlp(linear,gelu,linear)\n";
        }
        ss << print_indent(indent + 2) << "ln_f -> lm_head(tied)";
        return ss.str();
    }

private:
    /** 非模板辅助：const / 非 const 的 block 取 attention */
    static attn_type& attn_of(block_type& block)
    {
        return block.template get<0, 1>();
    }
    static attn_type const& attn_of(block_type const& block)
    {
        return block.template get<0, 1>();
    }
};

} // namespace jasmine
#endif
