#ifndef __JAS_LLAMA_T_HPP__
#define __JAS_LLAMA_T_HPP__

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_mat_utility.hpp"
#include "jas_net_t.hpp"
#include "jas_mha_t.hpp"
#include "jas_embedding_t.hpp"
#include "jas_silu_t.hpp"

namespace jasmine {

/*
 * LLaMA 系（LLaMA / TinyLlama / Mistral / Qwen2 …）前向所需的模块。
 *
 * 与 GPT-2（jas_gpt2_t.hpp）的差异 —— 每一处都对应一个已验证的构件：
 *   1. **RMSNorm 取代 LayerNorm**：不减均值、无 beta（jas_net_t.hpp 的 rms_norm_net_t）。
 *      残差仍是 pre-norm：x + SubLayer(Norm(x))。
 *   2. **RoPE 取代绝对位置嵌入**：没有 wpe，位置信息由注意力内部的旋转提供
 *      （因此 mat_mha_t 保持默认的 use_rope = true，而 GPT-2 要显式关掉）。
 *   3. **SwiGLU FFN**：silu(gate_proj(x)) ⊙ up_proj(x) 再经 down_proj，
 *      三个矩阵而不是 GPT-2 的两个（gated_net_t + silu_net_t）。
 *   4. **GQA**：K/V 头数少于 Q 头数（TinyLlama: 32 Q 头 / 4 KV 头）。
 *      mat_mha_t 的 n_kv_heads 参数化，注意力核不变。
 *   5. **没有 bias**：注意力与 FFN 的全部线性层都无偏置，因此这些 bias 必须显式置零。
 *   6. **lm_head 通常不绑定**（TinyLlama 的 tie_word_embeddings=false），
 *      与 GPT-2 相反；需要绑定的变体可调 tie_word_embeddings()。
 *
 * 层级结构：
 *   block[i]   = residual( RMSNorm -> MHA(causal, RoPE, GQA) )
 *              + residual( RMSNorm -> gated(Linear->SiLU, Linear) -> down_proj )
 *   motif      = wte(ids) -> block[0..N-1] -> RMSNorm -> lm_head
 *
 * 与 gpt2_model_t 一样是 **inference only**：这里不追求训练配方，只求前向数值与
 * 开源权重对齐（黄金值比对见 tests/test_llama_weights.cpp）。
 */

/** attention 分支：RMSNorm -> MHA（RoPE 由 mat_mha_t 默认开启） */
template<typename val_type, template<typename> class updator_type>
using llama_attn_branch_t = complex_net_builder_t<val_type>
    ::template push_back_updatable<rms_norm_net_t, updator_type>
    ::template push_back_updatable<mat_mha_t, updator_type>
    ::type;

template<typename val_type, template<typename> class updator_type>
using llama_res_attn_t = residual_net_t<llama_attn_branch_t<val_type, updator_type>>;

/**
 * FFN 分支：RMSNorm -> SwiGLU(gated) -> down_proj。
 *
 * 中间那个 gated 容器只负责「分叉 + 逐元素乘」；down_proj 是普通层，接在容器之后
 * （与 residual_net_t 只管加 skip、不管分支内部层同理）。
 * 把 silu_net_t 换成 gelu_net_t / relu_net_t 即得 GEGLU / ReGLU。
 */
template<typename val_type, template<typename> class updator_type>
using llama_ffn_branch_t = complex_net_builder_t<val_type>
    ::template push_back_updatable<rms_norm_net_t, updator_type>
    ::template push_back_impl<gated_ffn_branches_t<val_type, updator_type, silu_net_t>>
    ::template push_back_updatable<weight_net_t, updator_type>
    ::type;

template<typename val_type, template<typename> class updator_type>
using llama_res_ffn_t = residual_net_t<llama_ffn_branch_t<val_type, updator_type>>;

template<typename val_type, template<typename> class updator_type>
using llama_block_t = complex_net_builder_t<val_type>
    ::template push_back_impl<llama_res_attn_t<val_type, updator_type>>
    ::template push_back_impl<llama_res_ffn_t<val_type, updator_type>>
    ::type;

template <typename input_type, template<typename> class updator_type>
class llama_model_t
{
public:
    using val_type = typename input_type::ele_type;
    using norm_type = rms_norm_net_t<mat_t<val_type>, updator_type>;
    using attn_type = mat_mha_t<mat_t<val_type>, updator_type>;
    using linear_type = weight_net_t<mat_t<val_type>, updator_type>;
    using embed_type = embedding_net_t<mat_t<val_type>, updator_type>;
    using block_type = llama_block_t<val_type, updator_type>;

    /** 默认 eps；对齐权重时必须用模型 config 里的 rms_norm_eps */
    static constexpr val_type kDefaultRmsEps = norm_type::kDefaultEps;

private:
    embed_type m_wte;                       // [d_model, vocab]；**没有** wpe
    std::vector<block_type> m_blocks;
    norm_type m_ln_f;
    linear_type m_lm_head;                  // [vocab, d_model]
    bool m_tied = false;                    // LLaMA 系多数不绑定（TinyLlama: false）

    int m_n_layers = 1;
    int m_n_heads = 1;
    int m_n_kv_heads = 1;
    int m_d_model = 1;
    int m_d_ff = 0;
    int m_vocab = 1;
    int m_n_pos = 1;
    val_type m_rms_eps = kDefaultRmsEps;

public:
    llama_model_t() = default;

    /**
     * 配置全部子模块维度。这些子模块（mat_mha_t / rms_norm_net_t）都是 **stable 网络**
     * （不接受 reinit），所以这里逐个显式 set_param/reinit，不走 complex_net_t::reinit。
     *
     * @param n_kv_heads 0 表示等于 n_heads（经典 MHA）；TinyLlama = 4 而 n_heads = 32
     * @param rms_eps    RMSNorm 的 eps，默认 1e-5（TinyLlama 正是 1e-5）
     */
    void set_param(int n_layers, int n_heads, int d_model, int d_ff,
                   int vocab, int n_pos, int n_kv_heads = 0,
                   val_type rms_eps = kDefaultRmsEps)
    {
        if (n_layers <= 0) throw std::runtime_error("n_layers must be positive");
        if (d_model <= 0) throw std::runtime_error("d_model must be positive");
        m_n_layers = n_layers;
        m_n_heads = n_heads;
        m_d_model = d_model;
        m_d_ff = d_ff;
        m_vocab = vocab;
        m_n_pos = n_pos;
        m_rms_eps = rms_eps;

        m_wte.reinit(std::vector<int>{vocab, d_model});
        m_ln_f.set_param(d_model, rms_eps);
        m_lm_head.reinit(std::vector<int>{d_model, vocab});

        m_blocks.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            ln_1(i).set_param(d_model, rms_eps);
            // seq_len=1：mat_mha_t 是 stable 网络，前向按输入列数走；mask=true 为因果
            attn(i).set_param(n_heads, d_model, true, 1, n_kv_heads);
            // LLaMA 用 RoPE，不要关（与 GPT-2 相反）
            attn(i).set_use_rope(true);
            // HF 的 LlamaRotaryEmbedding 用 (i, i + d/2) 配对，与 jasmine 原生的交错配对不同
            attn(i).set_rope_pair_layout(rope_pair_layout::half_split);

            ln_2(i).set_param(d_model, rms_eps);
            mlp_gate(i).reinit(std::vector<int>{d_model, d_ff});
            mlp_up(i).reinit(std::vector<int>{d_model, d_ff});
            mlp_down(i).reinit(std::vector<int>{d_ff, d_model});
        }
        m_n_kv_heads = attn(0).num_kv_heads();

        zero_all_biases();
        m_tied = false;
    }

    /**
     * LLaMA 系没有线性层偏置。weight_net_t 一律带 bias，所以必须显式置零，
     * 否则 logits 会有一个恒定的整体偏移（且加载权重时不会有任何报错）。
     */
    void zero_all_biases()
    {
        for (int i = 0; i < m_n_layers; ++i)
        {
            mlp_gate(i).bias() = val_type(0);
            mlp_up(i).bias() = val_type(0);
            mlp_down(i).bias() = val_type(0);
            attn(i).q_proj().bias() = val_type(0);
            attn(i).k_proj().bias() = val_type(0);
            attn(i).v_proj().bias() = val_type(0);
            attn(i).out_proj().bias() = val_type(0);
        }
        m_lm_head.bias() = val_type(0);
    }

    /** 嵌入层：ids 1×T -> wte(ids)，输出 d_model×T。位置由 RoPE 提供，故无 wpe */
    mat_t<val_type> embed(const mat_t<val_type>& ids)
    {
        return m_wte.forward(ids);
    }

    /** 输出头：ln_f -> lm_head，d_model×T -> vocab×T */
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
     * 逐层输出（ln_f 之前）：[嵌入输出, block0 输出, ..., blockN-1 输出]，共 n_layers+1 项。
     * 与 GPT-2 同样的注意点：transformers>=5 的 hidden_states[-1] 是 ln_f **之后**的值，
     * 导出脚本已补算真正的 pre-ln_f 末层值；只比 logits 不够（RMSNorm 同样会掩盖仿射级偏差）。
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
     * 增量推理：ids 1×T（T 可为整段 prompt 或单个新 token）。
     *
     * 与 GPT-2 不同，**不需要 pos 参数**：LLaMA 的位置来自 RoPE，而 mat_mha_t 内部
     * 用 kv_cache_length() 作为本步的起始绝对位置，语义天然正确。
     * 返回最后一个位置的 logits（vocab×1）。
     */
    mat_t<val_type> forward_one(const mat_t<val_type>& ids)
    {
        mat_t<val_type> h = m_wte.forward(ids);
        for (auto& block : m_blocks)
            h = block.forward_one(h);
        h = m_ln_f.forward_one(h);
        const int T = ids.col_num();
        return m_lm_head.forward_one(h.view(0, T - 1, h.row_num(), 1).clone());
    }

    /**
     * prompt 预填。可以整段一次喂入（多列 forward_one）：mat_head_gen_t 的
     * forward_one 支持多列，并会按绝对位置自行补 causal mask，等价于逐 token 调用
     * （见 LlamaStructure.PrefillMatchesPerTokenStepping）。
     */
    mat_t<val_type> prefill(const mat_t<val_type>& ids)
    {
        clear_kv_cache();
        return forward_one(ids);
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

    /** lm_head.weight = wte.weight 的转置（GPT-2 的绑定方式；LLaMA 系多数不需要） */
    void tie_word_embeddings()
    {
        m_lm_head.weight() = m_wte.weight().t().clone();
        m_lm_head.bias() = val_type(0);
        m_tied = true;
    }

    bool tied_word_embeddings() const { return m_tied; }

    template <typename init_type>
    void init_weight()
    {
        m_wte.template init_weight<init_type>();
        m_ln_f.template init_weight<init_type>();
        m_lm_head.template init_weight<init_type>();
        for (auto& block : m_blocks)
            block.template init_weight<init_type>();
        // weight_net_t::init_weight 会把 bias 也随机初始化，而 LLaMA 根本没有 bias。
        // 在这里再清一次，使「无 bias」成为无论调用顺序如何都成立的不变量
        // （set_param 里那次清零只覆盖 set_param → 立刻加载权重 的路径）。
        zero_all_biases();
    }

    /** 载入权重后调用：把配置有但模型没有的 bias 再次清零（幂等，便宜） */
    void finalize_after_load()
    {
        zero_all_biases();
    }

    template<typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        for (auto& block : m_blocks)
            block.set_updator(std::forward<upr_arg_types>(args)...);
        m_ln_f.set_updator(std::forward<upr_arg_types>(args)...);
        m_wte.set_updator(std::forward<upr_arg_types>(args)...);
        m_lm_head.set_updator(std::forward<upr_arg_types>(args)...);
    }

    // ---- 权重加载器所需的访问器 ----

    embed_type& wte() { return m_wte; }
    embed_type const& wte() const { return m_wte; }
    norm_type& ln_f() { return m_ln_f; }
    norm_type const& ln_f() const { return m_ln_f; }
    linear_type& lm_head() { return m_lm_head; }
    linear_type const& lm_head() const { return m_lm_head; }

    /** attention 前的 RMSNorm，对应 HF `model.layers.{i}.input_layernorm` */
    norm_type& ln_1(int const& i) { return m_blocks[i].template get<0, 0>(); }
    norm_type const& ln_1(int const& i) const { return m_blocks[i].template get<0, 0>(); }
    /** 注意力本体 `self_attn`（含 GQA 的 q/k/v/o_proj） */
    attn_type& attn(int const& i) { return m_blocks[i].template get<0, 1>(); }
    attn_type const& attn(int const& i) const { return m_blocks[i].template get<0, 1>(); }
    /** FFN 前的 RMSNorm，对应 HF `post_attention_layernorm` */
    norm_type& ln_2(int const& i) { return m_blocks[i].template get<1, 0>(); }
    norm_type const& ln_2(int const& i) const { return m_blocks[i].template get<1, 0>(); }

    /** SwiGLU 的 gate 分支线性层 `mlp.gate_proj` [d_ff, d_model] */
    linear_type& mlp_gate(int const& i) { return m_blocks[i].template get<1, 1, 0, 0>(); }
    linear_type const& mlp_gate(int const& i) const { return m_blocks[i].template get<1, 1, 0, 0>(); }
    /** SwiGLU 的 up 分支线性层 `mlp.up_proj` [d_ff, d_model] */
    linear_type& mlp_up(int const& i) { return m_blocks[i].template get<1, 1, 1>(); }
    linear_type const& mlp_up(int const& i) const { return m_blocks[i].template get<1, 1, 1>(); }
    /** SwiGLU 的输出投影 `mlp.down_proj` [d_model, d_ff] */
    linear_type& mlp_down(int const& i) { return m_blocks[i].template get<1, 2>(); }
    linear_type const& mlp_down(int const& i) const { return m_blocks[i].template get<1, 2>(); }

    /**
     * 整个 FFN 残差块（RMSNorm -> SwiGLU -> down_proj，外裹 residual）。
     * 主要给测试用：`.base_net().forward(x)` 可以只跑分支本身，不叠加 skip。
     */
    auto& ffn_res(int const& i) { return m_blocks[i].template get<1>(); }
    auto const& ffn_res(int const& i) const { return m_blocks[i].template get<1>(); }
    /** 整个注意力残差块（RMSNorm -> MHA，外裹 residual） */
    auto& attn_res(int const& i) { return m_blocks[i].template get<0>(); }
    auto const& attn_res(int const& i) const { return m_blocks[i].template get<0>(); }

    int n_layers() const { return m_n_layers; }
    int n_heads() const { return m_n_heads; }
    int n_kv_heads() const { return m_n_kv_heads; }
    int group_size() const { return m_n_heads / std::max(1, m_n_kv_heads); }
    int d_model() const { return m_d_model; }
    int d_ff() const { return m_d_ff; }
    int d_head() const { return m_n_heads > 0 ? m_d_model / m_n_heads : 0; }
    int vocab_size() const { return m_vocab; }
    int n_pos() const { return m_n_pos; }
    val_type rms_eps() const { return m_rms_eps; }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "llama_model_t:(layers:" << m_n_layers
           << ", heads:" << m_n_heads << ", kv_heads:" << m_n_kv_heads
           << ", d_model:" << m_d_model << ", d_ff:" << m_d_ff
           << ", vocab:" << m_vocab << ", n_pos:" << m_n_pos
           << ", rms_eps:" << m_rms_eps
           << ", pre-norm, rope, swiglu, no-bias"
           << (m_tied ? ", tied-head" : ", untied-head") << ")\n";
        ss << print_indent(indent + 2) << "wte(embed): " << m_wte.net_type() << "\n";
        for (int i = 0; i < m_n_layers; ++i)
            ss << print_indent(indent + 2) << "block " << i
               << ": rms -> attn(causal,rope,gqa) -> rms -> swiglu(gate,up->down)\n";
        ss << print_indent(indent + 2) << "ln_f(rms) -> lm_head";
        return ss.str();
    }

private:
    static attn_type& attn_of(block_type& block)
    {
        return block.template get<0, 1>();
    }
    static attn_type const& attn_of(block_type const& block)
    {
        return block.template get<0, 1>();
    }
};

/**
 * RoPE 的基频（θ 的分母底数）目前在 jas_RoPE_t.hpp 里是按 LLaMA 原始的 10000 硬编码的。
 * TinyLlama-1.1B-Chat-v1.0 的 rope_theta 正是 10000.0，因此可以直接用；
 * 但 LLaMA-3.x 等用 500000 的变体会对不上，导入前必须显式检查，避免「logits 看着像但其实错了」。
 */
constexpr double kJasmineRoPESupportedBase = 10000.0;

inline void require_supported_rope_theta(double theta)
{
    if (std::abs(theta - kJasmineRoPESupportedBase) > 1e-6)
        throw std::runtime_error(
            "RoPE base (rope_theta) " + std::to_string(theta) + " is not supported: "
            "jas_RoPE_t.hpp hard-codes " + std::to_string(kJasmineRoPESupportedBase) +
            ". Extend mat_RoPE_t to take a configurable base before loading this model.");
}

} // namespace jasmine
#endif
