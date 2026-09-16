#ifndef __JAS_CUDA_LLAMA_HPP__
#define __JAS_CUDA_LLAMA_HPP__

/**
 * 设备端 LLaMA 系模型：`llama_model_t` 的 GPU 对应物。
 *
 * 前身是「各个零件都上设备了，但模型本身还是主机实现」（CUDA.md 第 9.5 节）。
 * 这个头把那层适配补上：**参数搬运 + 设备路径的 forward / forward_one / backward**。
 *
 * ## 权重加载仍然留在主机
 *
 * 这是刻意的：主机端已经有一整套经过黄金值验证的加载器（`jas_weight_io.hpp`、
 * `llama_model_t::finalize_after_load`），还有 `bind_rope()` 依赖的进程级单例
 * `rope_registry_t`。把它照搬到设备端只会多出一份要维护的解析逻辑。
 * 所以流程是：
 *
 *   llama_model_t 加载权重（主机，已有测试）  →  dev_llama_t::upload_from(host)  →  设备端前向 / 训练
 *
 * 搬过去的只有数值，而数值搬运是可以逐元素对拍的 —— `tests/test_cuda_llama.cu`
 * 里第一步就是「搬完之后两边的前向逐元素相同」。
 *
 * ## 与主机结构的对应
 *
 *   | 主机                                   | 设备                                   |
 *   |----------------------------------------|----------------------------------------|
 *   | `block = residual(rms → MHA)` + `residual(rms → SwiGLU → down)` | `dev_llama_block_t`（同名四件套） |
 *   | `wte`（`embedding_net_t`）              | `dev_embedding_t`                      |
 *   | `ln_f` / `lm_head`                      | `dev_rms_norm_t` / `dev_linear_t`      |
 *
 * 单层里唯一「不复用容器」的地方是**残差加法**：写成显式的 `eval_fused(a + skip, ...)`
 * 而不是套 `dev_residual_t`。理由是这一层同时要缓存**两份** skip（注意力支与 FFN 支），
 * 套容器反而要在外面再包一层显式的加法，得不偿失。`dev_residual_t` 本身仍然可用、
 * 也仍然有独立用例。
 *
 * ## 位置信息
 *
 * LLaMA 系没有位置嵌入，位置由注意力内部的 RoPE 提供。所以整模型**共享一个
 * `dev_rope_t`**（主机端也是全模型共享注册中心里同一份），`set_param` 时建好、
 * 绑到每层的注意力上。配对约定默认 `half_split` —— 与 HF 的
 * `LlamaRotaryEmbedding` 一致，也是主机 `llama_model_t` 的选择。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_embedding.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_mha.hpp"
#include "jas_cuda_net.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_rope.hpp"
#include "jas_cuda_updator.hpp"
#include "jas_llama_t.hpp"
#include "jas_mat_express_t.hpp"

namespace jasmine {
namespace cuda {

// ---------------------------------------------------------------------------
// 单层：llama_block_t 的设备对应物
// ---------------------------------------------------------------------------

/**
 * 一个 LLaMA block：
 *
 *   h1 = x  + attn(rms_1(x))                                 ← 注意力支
 *   h2 = h1 + down(silu(gate(rms_2(h1))) ⊙ up(rms_2(h1)))    ← SwiGLU FFN 支
 *
 * 两支都是 **pre-norm + 残差**：归一化在支路入口，残差把输入原样加回去。
 * 所以反向时上游梯度有两条路回到输入 —— 一条穿过支路、一条原样直通，
 * `eval_fused(d_branch + delta, ...)` 那一行就是这个恒等项。
 */
template <typename T, template <typename> class updator_type>
class dev_llama_block_t
{
public:
    using ele_type = T;
    using norm_type = dev_rms_norm_t<T, updator_type>;
    using attn_type = dev_mha_t<T, updator_type>;
    using linear_type = dev_linear_t<T, updator_type>;
    /** SwiGLU 的 gate 分支就是「线性层 → SiLU」，用通用串联容器表达（GELU 变体同理） */
    using gate_branch_type = dev_chain_t<linear_type, dev_silu_t<T>>;

    dev_llama_block_t() = default;

    void set_param(int n_heads, int d_model, int d_ff, int n_kv_heads, T rms_eps)
    {
        m_ln1.set_param(d_model, rms_eps);
        m_attn.set_param(n_heads, d_model, /*mask=*/true, n_kv_heads);
        m_ln2.set_param(d_model, rms_eps);
        m_swiglu.gate().first().set_param(d_model, d_ff);
        m_swiglu.up().set_param(d_model, d_ff);
        m_down.set_param(d_ff, d_model);
    }

    void set_rope(dev_rope_t<T>* rope) { m_attn.set_rope(rope); }

    /** 训练 / 预填：整段序列一次前向（因果掩码在注意力层内部施加）。 */
    template <typename X>
    dev_matrix_t<T> forward(const X& x)
    {
        // ---- 注意力支 ----
        detail::materialize_input(x, m_skip1);
        dev_matrix_t<T> n1 = m_ln1.forward(m_skip1.const_leaf());
        dev_matrix_t<T> a = m_attn.forward(n1.const_leaf());
        dev_matrix_t<T> h1(m_skip1.row_num(), m_skip1.col_num());
        if (h1.row_num() > 0 && h1.col_num() > 0)
            eval_fused(a.leaf() + m_skip1.leaf(), h1.buffer());

        // ---- SwiGLU FFN 支 ----
        detail::materialize_input(h1, m_skip2);
        dev_matrix_t<T> n2 = m_ln2.forward(m_skip2.const_leaf());
        dev_matrix_t<T> sw = m_swiglu.forward(n2.const_leaf());
        dev_matrix_t<T> d = m_down.forward(sw.const_leaf());
        dev_matrix_t<T> h2(m_skip2.row_num(), m_skip2.col_num());
        if (h2.row_num() > 0 && h2.col_num() > 0)
            eval_fused(d.leaf() + m_skip2.leaf(), h2.buffer());
        return h2;
    }

    /** 反向：顺序与 forward 严格相反，两支的梯度在各自残差点相加。 */
    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        // FFN 支（后做的先反）
        dev_matrix_t<T> d_sw = m_down.backward(delta);
        dev_matrix_t<T> d_h1 = m_swiglu.backward(d_sw);
        dev_matrix_t<T> d_n2 = m_ln2.backward(d_h1);
        dev_matrix_t<T> d_x2(d_n2.row_num(), d_n2.col_num());
        if (d_x2.row_num() > 0 && d_x2.col_num() > 0)
            eval_fused(d_n2.leaf() + delta.const_leaf(), d_x2.buffer());

        // 注意力支
        dev_matrix_t<T> d_a = m_attn.backward(d_x2);
        dev_matrix_t<T> d_n1 = m_ln1.backward(d_a);
        dev_matrix_t<T> d_x1(d_n1.row_num(), d_n1.col_num());
        if (d_x1.row_num() > 0 && d_x1.col_num() > 0)
            eval_fused(d_n1.leaf() + d_x2.const_leaf(), d_x1.buffer());
        return d_x1;
    }

    /**
     * 增量推理：本步 token 走一遍，K/V 追加进 cache。
     *
     * 与前向的唯一实质差别在注意力：这里走 `dev_mha_t::forward_one`（cache 路径），
     * 位置信息来自 cache 长度，所以**不需要任何位置参数**。
     */
    template <typename X>
    dev_matrix_t<T> forward_one(const X& x)
    {
        detail::materialize_input(x, m_skip1);
        dev_matrix_t<T> n1 = m_ln1.forward(m_skip1.const_leaf());
        dev_matrix_t<T> a = m_attn.forward_one(n1.const_leaf());
        dev_matrix_t<T> h1(m_skip1.row_num(), m_skip1.col_num());
        if (h1.row_num() > 0 && h1.col_num() > 0)
            eval_fused(a.leaf() + m_skip1.leaf(), h1.buffer());

        detail::materialize_input(h1, m_skip2);
        dev_matrix_t<T> n2 = m_ln2.forward(m_skip2.const_leaf());
        dev_matrix_t<T> sw = m_swiglu.forward(n2.const_leaf());
        dev_matrix_t<T> d = m_down.forward(sw.const_leaf());
        dev_matrix_t<T> h2(m_skip2.row_num(), m_skip2.col_num());
        if (h2.row_num() > 0 && h2.col_num() > 0)
            eval_fused(d.leaf() + m_skip2.leaf(), h2.buffer());
        return h2;
    }

    void clear_kv_cache() { m_attn.clear_kv_cache(); }
    void reserve_kv_cache(int max_seq) { m_attn.reserve_kv_cache(max_seq); }
    int kv_cache_length() const { return m_attn.kv_cache_length(); }

    norm_type& ln_1() { return m_ln1; }
    const norm_type& ln_1() const { return m_ln1; }
    attn_type& attn() { return m_attn; }
    const attn_type& attn() const { return m_attn; }
    norm_type& ln_2() { return m_ln2; }
    const norm_type& ln_2() const { return m_ln2; }
    /** SwiGLU 的两个分支（gate 是「线性 → SiLU」，up 是纯线性） */
    gate_branch_type& mlp_gate() { return m_swiglu.gate(); }
    const gate_branch_type& mlp_gate() const { return m_swiglu.gate(); }
    linear_type& mlp_up() { return m_swiglu.up(); }
    const linear_type& mlp_up() const { return m_swiglu.up(); }
    linear_type& mlp_down() { return m_down; }
    const linear_type& mlp_down() const { return m_down; }

    void step()
    {
        m_ln1.step();
        m_attn.step();
        m_ln2.step();
        m_swiglu.step();
        m_down.step();
    }

    void set_lr(T lr)
    {
        m_ln1.set_lr(lr);
        m_attn.set_lr(lr);
        m_ln2.set_lr(lr);
        m_swiglu.set_lr(lr);
        m_down.set_lr(lr);
    }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_ln1.set_updator(std::forward<arg_types>(args)...);
        m_attn.set_updator(std::forward<arg_types>(args)...);
        m_ln2.set_updator(std::forward<arg_types>(args)...);
        m_swiglu.set_updator(std::forward<arg_types>(args)...);
        m_down.set_updator(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        m_ln1.template init_weight<init_type>();
        m_attn.template init_weight<init_type>();
        m_ln2.template init_weight<init_type>();
        m_swiglu.template init_weight<init_type>();
        m_down.template init_weight<init_type>();
        zero_biases();
    }

    /** LLaMA 系没有线性层偏置；加载权重后必须调用（主机端是 `zero_all_biases`）。 */
    void zero_biases()
    {
        m_attn.zero_biases();
        m_swiglu.gate().first().zero_bias();
        m_swiglu.up().zero_bias();
        m_down.zero_bias();
    }

    std::string net_type(int indent = 0) const
    {
        const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');
        return pad + "dev_llama_block_t(\n" + m_ln1.net_type(indent + 1) + "\n"
               + m_attn.net_type(indent + 1) + "\n" + m_ln2.net_type(indent + 1) + "\n"
               + m_swiglu.net_type(indent + 1) + "\n" + m_down.net_type(indent + 1) + "\n" + pad
               + ")";
    }

private:
    norm_type m_ln1;
    attn_type m_attn;
    norm_type m_ln2;
    dev_gated_t<gate_branch_type, linear_type> m_swiglu;  // gate/up + 逐元素相乘
    linear_type m_down;
    dev_matrix_t<T> m_skip1, m_skip2;  // 两个残差的 skip 缓存（反向的直通项要用）
};

// ---------------------------------------------------------------------------
// 模型
// ---------------------------------------------------------------------------

/**
 * 设备端 LLaMA：`wte → block[0..N-1] → ln_f → lm_head`。
 *
 * 支持两条路径：
 *   - `forward` / `backward` —— 训练与预填（整段序列、因果掩码、不动 cache）；
 *   - `forward_one` —— 增量解码（KV cache，位置来自 cache 长度）。
 *
 * 与主机一样，`forward_one` 只返回**最后一个位置**的 logits（`vocab × 1`）。
 */
template <typename T, template <typename> class updator_type>
class dev_llama_t
{
public:
    using ele_type = T;
    using host_type = llama_model_t<mat_t<T>, updator_type>;
    using block_type = dev_llama_block_t<T, updator_type>;
    using norm_type = dev_rms_norm_t<T, updator_type>;
    using linear_type = dev_linear_t<T, updator_type>;
    using embed_type = dev_embedding_t<T, updator_type>;

    static constexpr T kDefaultRmsEps = static_cast<T>(1e-5);

    dev_llama_t() = default;

    /**
     * 配置全部子模块。参数顺序与主机 `llama_model_t::set_param` 一致，方便对照。
     *
     * `n_kv_heads == 0` 表示等于 `n_heads`（经典 MHA）。
     */
    void set_param(int n_layers, int n_heads, int d_model, int d_ff, int vocab, int n_pos,
                   int n_kv_heads = 0, T rms_eps = kDefaultRmsEps,
                   rope_pair_layout layout = rope_pair_layout::half_split)
    {
        if (n_layers <= 0 || n_heads <= 0 || d_model <= 0)
            throw std::invalid_argument("dev_llama_t::set_param: 维度必须为正");
        if (d_model % n_heads != 0)
            throw std::invalid_argument("dev_llama_t::set_param: d_model 必须能被 n_heads 整除");

        m_n_layers = n_layers;
        m_n_heads = n_heads;
        m_n_kv_heads = (n_kv_heads > 0) ? n_kv_heads : n_heads;
        m_d_model = d_model;
        m_d_ff = d_ff;
        m_vocab = vocab;
        m_n_pos = n_pos;
        m_rms_eps = rms_eps;
        m_d_head = d_model / n_heads;
        m_rope_layout = layout;

        // 全模型共享一份 RoPE（主机端也是注册中心里的同一份）
        m_rope.set_param(m_d_head, layout);
        m_rope.reserve(n_pos > 0 ? n_pos : 1);

        m_wte.set_param(vocab, d_model);
        m_ln_f.set_param(d_model, rms_eps);
        m_lm_head.set_param(d_model, vocab);

        m_blocks.clear();
        m_blocks.resize(static_cast<std::size_t>(n_layers));
        for (auto& b : m_blocks)
        {
            b.set_param(n_heads, d_model, d_ff, m_n_kv_heads, rms_eps);
            b.set_rope(&m_rope);
            b.zero_biases();  // LLaMA 没有 bias，从构造起就成立，不依赖调用顺序
        }
    }

    dev_rope_t<T>& rope() { return m_rope; }
    const dev_rope_t<T>& rope() const { return m_rope; }

    /** 换 RoPE 配对约定（导出 HF 权重用 half_split，jasmine 原生训练是 interleaved）。 */
    void set_rope_pair_layout(rope_pair_layout layout)
    {
        if (m_rope_layout == layout)
            return;
        m_rope_layout = layout;
        m_rope.set_pair_layout(layout);
    }

    rope_pair_layout pair_layout() const { return m_rope_layout; }

    /**
     * 把主机模型（已加载权重）的参数搬到设备上。
     *
     * 逐层逐矩阵搬运，**只搬数值**；形状不对会立刻报错，而不是等到前向数值不对。
     */
    /**
     * 主机模型的类型。它是**模板参数的另一半**（更新器也是模板参数），
     * 所以 `upload_from` 写成泛型：主机端用什么更新器不影响「参数长什么样」，
     * 没必要逼调用方把两边的更新器凑成同一个类型。
     */
    template <typename HostModel>
    void upload_from(const HostModel& host)
    {
        if (host.n_layers() != m_n_layers || host.d_model() != m_d_model)
            throw std::invalid_argument("dev_llama_t::upload_from: 主机模型的形状与设备端不一致"
                                        "（先按同一套参数调 set_param）");

        m_wte.upload_weight(host.wte().weight());

        for (int i = 0; i < m_n_layers; ++i)
        {
            block_type& b = m_blocks[static_cast<std::size_t>(i)];
            b.ln_1().upload_gama(host.ln_1(i).gama());
            b.attn().q_proj().upload_weight(host.attn(i).q_proj().weight());
            b.attn().k_proj().upload_weight(host.attn(i).k_proj().weight());
            b.attn().v_proj().upload_weight(host.attn(i).v_proj().weight());
            b.attn().out_proj().upload_weight(host.attn(i).out_proj().weight());
            b.ln_2().upload_gama(host.ln_2(i).gama());
            b.mlp_gate().first().upload_weight(host.mlp_gate(i).weight());
            b.mlp_up().upload_weight(host.mlp_up(i).weight());
            b.mlp_down().upload_weight(host.mlp_down(i).weight());
            // LLaMA 无 bias：主机那边加载后是零，设备端也清一次（幂等）
            b.zero_biases();
        }

        m_ln_f.upload_gama(host.ln_f().gama());
        m_lm_head.upload_weight(host.lm_head().weight());
        m_lm_head.zero_bias();
    }

    // ---- 前向 ----

    /** 整段前向：`ids`（1 × T）→ logits（vocab × T）。 */
    template <typename Ids>
    dev_matrix_t<T> forward(const Ids& ids)
    {
        dev_matrix_t<T> h = m_wte.forward(ids);
        for (auto& b : m_blocks)
            h = b.forward(h.const_leaf());
        return head(h);
    }

    /**
     * 逐层输出（`ln_f` **之前**）：`[嵌入输出, block0 输出, ...]`，共 `n_layers + 1` 项。
     *
     * 与主机 `forward_stages` 同序同义。对拍时比 logits 更有用：RMSNorm 与 lm_head
     * 都是仿射变换，偏差会被它们掩盖或放大，逐层输出能把出错的位置直接指出来。
     */
    template <typename Ids>
    std::vector<dev_matrix_t<T>> forward_stages(const Ids& ids)
    {
        std::vector<dev_matrix_t<T>> stages;
        stages.reserve(static_cast<std::size_t>(m_n_layers) + 1);
        stages.push_back(m_wte.forward(ids));
        for (auto& b : m_blocks)
            stages.push_back(b.forward(stages.back().const_leaf()));
        return stages;
    }

    /**
     * 增量推理：`ids`（1 × T，T 可为整段 prompt 或单个新 token）→ 最后一个位置的 logits。
     *
     * 与主机一样**不需要位置参数**：位置来自 RoPE，而起始绝对位置由 cache 长度决定。
     */
    template <typename Ids>
    dev_matrix_t<T> forward_one(const Ids& ids)
    {
        dev_matrix_t<T> h = m_wte.forward(ids);
        for (auto& b : m_blocks)
            h = b.forward_one(h.const_leaf());
        dev_matrix_t<T> n = m_ln_f.forward(h.const_leaf());
        const int seq = n.col_num();
        if (seq <= 0)
            return dev_matrix_t<T>(m_vocab, 0);
        // 只要最后一列：零拷贝取尾列（d_model × 1），不必先拷一份紧凑副本
        return m_lm_head.forward(col_slice(n.const_leaf(), seq - 1, 1));
    }

    /** prompt 预填：清空 cache 后整段喂入（多列 `forward_one`，与主机同义）。 */
    template <typename Ids>
    dev_matrix_t<T> prefill(const Ids& ids)
    {
        clear_kv_cache();
        return forward_one(ids);
    }

    /** 输出头：`ln_f` → `lm_head`。 */
    dev_matrix_t<T> head(const dev_matrix_t<T>& hidden)
    {
        dev_matrix_t<T> n = m_ln_f.forward(hidden.const_leaf());
        return m_lm_head.forward(n.const_leaf());
    }

    // ---- 反向 ----

    /**
     * 反向：`lm_head → ln_f → block[N-1..0] → wte`。
     *
     * 返回值是 `(1 × T)` 的占位 —— 与主机 `embedding_net_t::backward` 一致，
     * 离散 id 没有梯度，**参数更新已经在各层的 `backward` 里就地完成了**。
     */
    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        dev_matrix_t<T> d = m_lm_head.backward(delta);
        dev_matrix_t<T> dn = m_ln_f.backward(d);
        for (int i = m_n_layers - 1; i >= 0; --i)
            dn = m_blocks[static_cast<std::size_t>(i)].backward(dn);
        return m_wte.backward(dn);
    }

    // ---- KV cache ----

    void clear_kv_cache()
    {
        for (auto& b : m_blocks)
            b.clear_kv_cache();
    }

    void reserve_kv_cache(int max_seq)
    {
        for (auto& b : m_blocks)
            b.reserve_kv_cache(max_seq);
    }

    int kv_cache_length() const
    {
        return m_blocks.empty() ? 0 : m_blocks.front().kv_cache_length();
    }

    // ---- 训练接口 ----

    void step()
    {
        m_wte.step();
        for (auto& b : m_blocks)
            b.step();
        m_ln_f.step();
        m_lm_head.step();
    }

    void set_lr(T lr)
    {
        m_wte.set_lr(lr);
        for (auto& b : m_blocks)
            b.set_lr(lr);
        m_ln_f.set_lr(lr);
        m_lm_head.set_lr(lr);
    }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_wte.set_updator(std::forward<arg_types>(args)...);
        for (auto& b : m_blocks)
            b.set_updator(std::forward<arg_types>(args)...);
        m_ln_f.set_updator(std::forward<arg_types>(args)...);
        m_lm_head.set_updator(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        m_wte.template init_weight<init_type>();
        for (auto& b : m_blocks)
            b.template init_weight<init_type>();
        m_ln_f.template init_weight<init_type>();
        m_lm_head.template init_weight<init_type>();
        zero_all_biases();
    }

    /** 把配置有、但 LLaMA 模型没有的 bias 全部清零（对应主机 `zero_all_biases`）。 */
    void zero_all_biases()
    {
        for (auto& b : m_blocks)
            b.zero_biases();
        m_lm_head.zero_bias();
    }

    // ---- 访问器 ----

    embed_type& wte() { return m_wte; }
    const embed_type& wte() const { return m_wte; }
    norm_type& ln_f() { return m_ln_f; }
    const norm_type& ln_f() const { return m_ln_f; }
    linear_type& lm_head() { return m_lm_head; }
    const linear_type& lm_head() const { return m_lm_head; }
    block_type& block(int i) { return m_blocks.at(static_cast<std::size_t>(i)); }
    const block_type& block(int i) const { return m_blocks.at(static_cast<std::size_t>(i)); }

    int n_layers() const { return m_n_layers; }
    int n_heads() const { return m_n_heads; }
    int n_kv_heads() const { return m_n_kv_heads; }
    int group_size() const { return m_n_heads / (m_n_kv_heads > 0 ? m_n_kv_heads : 1); }
    int d_model() const { return m_d_model; }
    int d_ff() const { return m_d_ff; }
    int d_head() const { return m_d_head; }
    int vocab_size() const { return m_vocab; }
    int n_pos() const { return m_n_pos; }
    T rms_eps() const { return m_rms_eps; }

    std::string net_type(int indent = 0) const
    {
        const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');
        std::string s = pad + "dev_llama_t(layers:" + std::to_string(m_n_layers)
                        + ", heads:" + std::to_string(m_n_heads)
                        + ", kv_heads:" + std::to_string(m_n_kv_heads)
                        + ", d_model:" + std::to_string(m_d_model)
                        + ", d_ff:" + std::to_string(m_d_ff)
                        + ", vocab:" + std::to_string(m_vocab)
                        + ", rms_eps:" + std::to_string(m_rms_eps) + ")\n";
        s += m_wte.net_type(indent + 1) + "\n";
        for (const auto& b : m_blocks)
            s += b.net_type(indent + 1) + "\n";
        s += m_ln_f.net_type(indent + 1) + "\n" + m_lm_head.net_type(indent + 1);
        return s;
    }

private:
    embed_type m_wte;
    std::vector<block_type> m_blocks;
    norm_type m_ln_f;
    linear_type m_lm_head;
    dev_rope_t<T> m_rope;
    int m_n_layers = 0;
    int m_n_heads = 0;
    int m_n_kv_heads = 0;
    int m_d_model = 0;
    int m_d_ff = 0;
    int m_d_head = 0;
    int m_vocab = 0;
    int m_n_pos = 0;
    T m_rms_eps = kDefaultRmsEps;
    rope_pair_layout m_rope_layout = rope_pair_layout::half_split;
};

} // namespace cuda
} // namespace jasmine

#endif
