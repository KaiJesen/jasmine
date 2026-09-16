/**
 * LLaMA 系（TinyLlama-1.1B-Chat-v1.0）接入测试。
 *
 * 分三层：
 *   1. 结构层（随机权重，不需要大文件）：拓扑、形状、因果性、KV cache 一致性、
 *      以及**没有 bias / 有 SwiGLU / K/V 投影更窄**这些 LLaMA 特有的性质。
 *   2. 加载层（合成小权重文件）：把 load_llama 的命名映射逐个张量钉住 —— 这类错误
 *      在真实权重上只会表现为「logits 差一点」，很难定位；用合成文件可以精确断言。
 *   3. 对齐层（需要 export_llama.py 导出的真实权重 + HF 黄金值）：逐层 hidden 与 logits。
 *
 * 第 3 层在没有权重文件时 GTEST_SKIP，因此 CI 不依赖 4.4GB 的大文件。
 */

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_llama_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_weight_io.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

template <typename val_type>
using llama_upr_tpl = cache_updator_t<val_type, nadam_t>;

using llama_type = llama_model_t<mat_t<double>, llama_upr_tpl>;

namespace
{

double MaxAbsDiff(const mat_t<double>& a, const mat_t<double>& b)
{
    double m = 0.0;
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < a.col_num(); ++j)
            m = std::max(m, std::abs(a(i, j) - b(i, j)));
    return m;
}

/** 矩阵元素绝对值的最大值（用来断言某个张量「全为 0」） */
double MaxAbsValue(const mat_t<double>& a)
{
    double m = 0.0;
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < a.col_num(); ++j)
            m = std::max(m, std::abs(a(i, j)));
    return m;
}

/**
 * 小配置 LLaMA：d_model=16, heads=4, kv_heads=2 -> d_head=4, group_size=2（真正的 GQA）。
 * 默认参数刻意选成与 TinyLlama 同构（kv<heads、SwiGLU、无 bias），只是尺寸缩小。
 */
llama_type make_small_llama(int layers = 2, int heads = 4, int kv_heads = 2,
                            int d_model = 16, int d_ff = 32, int vocab = 13, int n_pos = 32)
{
    llama_type model;
    model.set_param(layers, heads, d_model, d_ff, vocab, n_pos, kv_heads, 1e-5);
    model.init_weight<xavier_gaussian_t>();
    return model;
}

/**
 * 建一个与 make_small_llama 同形状的合成权重文件：每个张量填成**互不相同的常数**，
 * 这样任何命名映射错位都会立刻暴露成数值不符，而不是「差一点点」。
 * 返回值：文件路径 + 每个张量的期望常数，供逐张量断言。
 */
std::string write_synthetic_llama_weights(int layers, int heads, int kv_heads,
                                          int d_model, int d_ff, int vocab, int n_pos,
                                          bool tied = false)
{
    const int d_head = d_model / heads;
    const int d_kv = kv_heads * d_head;
    const std::string path = "test_llama_synthetic_weights.bin";

    weight_writer_t w;
    double next_val = 1.0;
    auto add_const = [&](const std::string& name, int rows, int cols) {
        // 注意：不能写 `mat_t<double> m(rows, cols, next_val)` ——
        // mat_t 有 mat_t(int, int, bool row_first) 构造，double 会被隐式转成 bool，
        // 于是矩阵根本没被填值（静默拿到未初始化/全 0 的数据）。
        // 必须显式构造再用 operator=(scalar) 填充。
        mat_t<double> m(rows, cols);
        m = next_val;                               // 全填同一个常数
        w.add(name, m);
        next_val += 1.0;
    };

    w.add("cfg.n_layers", mat_t<double>(1, 1, {static_cast<double>(layers)}));
    w.add("cfg.n_heads", mat_t<double>(1, 1, {static_cast<double>(heads)}));
    w.add("cfg.n_kv_heads", mat_t<double>(1, 1, {static_cast<double>(kv_heads)}));
    w.add("cfg.d_model", mat_t<double>(1, 1, {static_cast<double>(d_model)}));
    w.add("cfg.d_ff", mat_t<double>(1, 1, {static_cast<double>(d_ff)}));
    w.add("cfg.vocab", mat_t<double>(1, 1, {static_cast<double>(vocab)}));
    w.add("cfg.n_pos", mat_t<double>(1, 1, {static_cast<double>(n_pos)}));
    w.add("cfg.rms_eps", mat_t<double>(1, 1, {1e-5}));
    w.add("cfg.rope_theta", mat_t<double>(1, 1, {10000.0}));
    w.add("cfg.tie_word_embeddings", mat_t<double>(1, 1, {tied ? 1.0 : 0.0}));

    add_const("wte.weight", d_model, vocab);
    for (int i = 0; i < layers; ++i)
    {
        const std::string p = "h." + std::to_string(i) + ".";
        add_const(p + "ln_1.weight", d_model, 1);
        add_const(p + "attn.q.weight", d_model, d_model);
        add_const(p + "attn.k.weight", d_kv, d_model);
        add_const(p + "attn.v.weight", d_kv, d_model);
        add_const(p + "attn.out.weight", d_model, d_model);
        add_const(p + "ln_2.weight", d_model, 1);
        add_const(p + "mlp.gate.weight", d_ff, d_model);
        add_const(p + "mlp.up.weight", d_ff, d_model);
        add_const(p + "mlp.down.weight", d_model, d_ff);
    }
    add_const("ln_f.weight", d_model, 1);
    if (!tied)
        add_const("lm_head.weight", vocab, d_model);

    w.write(path);
    return path;
}

/** 找一个可用的 LLaMA 权重文件（含 HF 黄金值）。找不到就 GTEST_SKIP */
std::string find_llama_weights()
{
    std::vector<std::string> candidates;
    if (const char* env = std::getenv("JASMINE_LLAMA_WEIGHTS"))
        candidates.emplace_back(env);
    for (const char* name : {"tinyllama_weights.bin"})
    {
        candidates.emplace_back(name);
        candidates.emplace_back(std::string("../") + name);
        candidates.emplace_back(std::string("build/") + name);
    }
    for (const auto& c : candidates)
        if (!c.empty() && std::filesystem::exists(c))
            return c;
    return {};
}

} // namespace

// ---------------------------------------------------------------------------
// 结构层
// ---------------------------------------------------------------------------

TEST(LlamaStructure, ForwardShape)
{
    auto model = make_small_llama();
    mat_t<double> ids(1, 4, {1.0, 2.0, 3.0, 4.0});
    auto logits = model.forward(ids);
    ExpectShape(logits, 13, 4);
    for (int i = 0; i < logits.row_num(); ++i)
        for (int j = 0; j < logits.col_num(); ++j)
            EXPECT_TRUE(std::isfinite(logits(i, j)));
}

TEST(LlamaStructure, GqaKvProjectionIsNarrow)
{
    auto model = make_small_llama(/*layers=*/1, /*heads=*/4, /*kv_heads=*/2,
                                  /*d_model=*/16, /*d_ff=*/32, /*vocab=*/13, /*n_pos=*/32);
    EXPECT_EQ(model.n_heads(), 4);
    EXPECT_EQ(model.n_kv_heads(), 2);
    EXPECT_EQ(model.group_size(), 2);
    EXPECT_EQ(model.d_head(), 4);

    // Q/O 仍是 d_model；K/V 只有 n_kv_heads*d_head = 8
    ExpectShape(model.attn(0).q_proj().weight(), 16, 16);
    ExpectShape(model.attn(0).k_proj().weight(), 8, 16);
    ExpectShape(model.attn(0).v_proj().weight(), 8, 16);
    ExpectShape(model.attn(0).out_proj().weight(), 16, 16);
}

TEST(LlamaStructure, SwiGluHasThreeProjections)
{
    auto model = make_small_llama();
    // gate/up: d_model -> d_ff；down: d_ff -> d_model（GPT-2 只有两个矩阵）
    ExpectShape(model.mlp_gate(0).weight(), 32, 16);
    ExpectShape(model.mlp_up(0).weight(), 32, 16);
    ExpectShape(model.mlp_down(0).weight(), 16, 32);
}

TEST(LlamaStructure, SwiGluZeroGateMakesFfnIdentity)
{
    // SwiGLU 的 gate 分支带 SiLU：SiLU(0)=0。所以 gate_proj 恒等 0 时
    //   down( silu(0) ⊙ up(x) ) = down(0) = 0   （bias 也已置零）
    // 于是：分支输出为 0，而整个残差块退化为恒等映射。
    // 这一条同时钉住四件事：确实经过 gate 分支、逐元素乘（而非矩阵乘）在起作用、
    // bias 真的是 0、以及 residual 的 skip 确实加上了。
    auto model = make_small_llama(/*layers=*/1);
    model.mlp_gate(0).weight() = 0.0;
    model.mlp_up(0).weight() = 1.0;
    model.mlp_down(0).weight() = 1.0;

    mat_t<double> x(16, 3);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 3; ++j)
            x(i, j) = 0.1 * (i + 1) - 0.2 * j;

    // 只跑 FFN 分支本身（RMSNorm -> SwiGLU -> down），不含 residual skip
    auto branch_out = model.ffn_res(0).base_net().forward(x);
    for (int i = 0; i < branch_out.row_num(); ++i)
        for (int j = 0; j < branch_out.col_num(); ++j)
            EXPECT_NEAR(branch_out(i, j), 0.0, 1e-12)
                << "gate 全 0 时分支输出应为 0，(" << i << "," << j << ")";

    // 套上 residual：out = branch(x) + x = x
    auto block_out = model.ffn_res(0).forward(x);
    ExpectNearMat(block_out, x, 1e-12);
}

TEST(LlamaStructure, AllBiasesAreZero)
{
    // LLaMA 没有线性层 bias；weight_net_t 一律带 bias，所以必须显式置零。
    // 漏掉不会报错，只会让 logits 有一个恒定偏移。
    auto model = make_small_llama();
    for (int i = 0; i < model.n_layers(); ++i)
    {
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.mlp_gate(i).bias()), 0.0) << "gate " << i;
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.mlp_up(i).bias()), 0.0) << "up " << i;
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.mlp_down(i).bias()), 0.0) << "down " << i;
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.attn(i).q_proj().bias()), 0.0) << "q " << i;
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.attn(i).k_proj().bias()), 0.0) << "k " << i;
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.attn(i).v_proj().bias()), 0.0) << "v " << i;
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.attn(i).out_proj().bias()), 0.0) << "o " << i;
    }
    EXPECT_DOUBLE_EQ(MaxAbsValue(model.lm_head().bias()), 0.0);
}

TEST(LlamaStructure, RmsNormAffineLoadableBeforeForward)
{
    // set_param 必须显式分配 gamma（RMSNorm 没有 beta）
    llama_type model;
    model.set_param(1, 4, 16, 32, 13, 32, 2, 1e-6);
    for (int i = 0; i < model.n_layers(); ++i)
    {
        ASSERT_TRUE(model.ln_1(i).gama().valid());
        ASSERT_TRUE(model.ln_2(i).gama().valid());
        ExpectShape(model.ln_1(i).gama(), 16, 1);
        EXPECT_DOUBLE_EQ(model.ln_1(i).eps(), 1e-6);
    }
    ASSERT_TRUE(model.ln_f().gama().valid());
    ExpectShape(model.ln_f().gama(), 16, 1);
    EXPECT_DOUBLE_EQ(model.rms_eps(), 1e-6);
}

TEST(LlamaStructure, RopeIsWiredAndChangesOutput)
{
    // 最直接的「RoPE 不是空操作」检查：同一个多 token 输入，开/关 RoPE 输出必须不同。
    // GPT-2 那边是反过来测的（默认关，打开后必须变），因为绝对位置模型不能有 RoPE。
    auto model = make_small_llama();
    mat_t<double> ids(1, 4, {3.0, 8.0, 1.0, 5.0});
    auto with_rope = model.forward(ids);

    for (int i = 0; i < model.n_layers(); ++i)
        model.attn(i).set_use_rope(false);
    auto without = model.forward(ids);
    EXPECT_GT(MaxAbsDiff(with_rope, without), 1e-6)
        << "关掉 RoPE 后输出没变，说明 RoPE 根本没接上";

    for (int i = 0; i < model.n_layers(); ++i)
        model.attn(i).set_use_rope(true);
    auto back_on = model.forward(ids);
    EXPECT_LT(MaxAbsDiff(with_rope, back_on), 1e-12) << "重新开启后应回到原结果";
}

TEST(LlamaStructure, SingleTokenOutputIsPositionIndependent)
{
    // 这条记录一个**容易误解**的性质：RoPE 只编码相对位置。
    //   (R_m q)·(R_n k) = q · R_{n-m} k
    // 单 token 推理时 q 与 k 来自同一个 token（m == n），相对距离恒为 0，
    // 旋转对分数毫无影响 → 单 token 的输出本来就与绝对位置无关。
    //
    // 所以「同一 token 在不同位置应当不同」是**绝对位置编码(wpe)**的行为，
    // 不能拿来测 RoPE（我第一版就写错了）。RoPE 的位置效应只在多 key 的相对距离上体现，
    // 见 RopeEncodesRelativeDistance。
    auto model = make_small_llama();
    model.clear_kv_cache();
    auto at0 = model.forward_one(mat_t<double>(1, 1, {7.0}));
    model.clear_kv_cache();
    model.forward_one(mat_t<double>(1, 1, {7.0}));      // 占掉位置 0
    model.forward_one(mat_t<double>(1, 1, {7.0}));      // 占掉位置 1
    auto at2 = model.forward_one(mat_t<double>(1, 1, {7.0}));
    EXPECT_LT(MaxAbsDiff(at0, at2), 1e-12)
        << "单 token 输出不应依赖绝对位置（RoPE 是相对位置编码）";
}


TEST(LlamaStructure, RopeEncodesRelativeDistance)
{
    // 在注意力层上直接验证 RoPE 的语义：把同一组 K/V 固定在相同位置，
    // 只把 query 往后移 —— query 到各 key 的**相对距离**变了，输出必须改变。
    // 反过来，关掉 RoPE 后同样的位移不改变任何分数，输出应完全一致。
    using mha_t = mat_mha_t<mat_t<double>, llama_upr_tpl>;
    mha_t mha(2, 8, /*mask=*/true, 1, /*n_kv_heads=*/2);
    mha.set_use_rope(true);

    // 固定权重（非零、确定），避免随机初始化让结论不可复现
    auto fill = [](mat_t<double>& m, double base) {
        for (int i = 0; i < m.row_num(); ++i)
            for (int j = 0; j < m.col_num(); ++j)
                m(i, j) = base * std::sin(0.7 * (i + 1) * (j + 2)) + 0.1 * (i - j);
    };
    fill(mha.q_proj().weight(), 0.31);
    fill(mha.k_proj().weight(), 0.27);
    fill(mha.v_proj().weight(), 0.23);
    fill(mha.out_proj().weight(), 0.19);

    const int d_kv = mha.d_kv();          // 2 heads * 4 = 8
    ASSERT_EQ(d_kv, 8);
    mat_t<double> k_full(d_kv, 2), v_full(d_kv, 2);
    fill(k_full, 0.5);
    fill(v_full, 0.6);
    mat_t<double> q(8, 1);
    fill(q, 0.7);

    mha.fill_kv_cache(k_full, v_full, /*k_start_pos=*/0);
    auto near_out = mha.forward_one_cached_kv(q, /*q_pos=*/2);    // 相对距离 2 和 1
    auto far_out = mha.forward_one_cached_kv(q, /*q_pos=*/10);    // 相对距离 10 和 9
    EXPECT_GT(MaxAbsDiff(near_out, far_out), 1e-9)
        << "相对距离变化后输出未变，RoPE 未生效";

    // 关掉 RoPE：绝对位置不再影响任何东西，两次调用必须一致
    mha.set_use_rope(false);
    mha.fill_kv_cache(k_full, v_full, 0);
    auto plain_near = mha.forward_one_cached_kv(q, 2);
    auto plain_far = mha.forward_one_cached_kv(q, 10);
    EXPECT_LT(MaxAbsDiff(plain_near, plain_far), 1e-12)
        << "关掉 RoPE 后不应再依赖位置";

    // 且此时应与「开了 RoPE 但距离为 0」的旋转无关性一致 —— 用来确认上面的差异
    // 确实来自旋转，而不是别的东西碰巧变了
    EXPECT_GT(MaxAbsDiff(near_out, plain_near), 1e-9);
}

TEST(LlamaStructure, CausalFirstPositionStable)
{
    auto model = make_small_llama();
    auto out1 = model.forward(mat_t<double>(1, 1, {5.0}));
    auto out4 = model.forward(mat_t<double>(1, 4, {5.0, 7.0, 2.0, 9.0}));
    for (int i = 0; i < out1.row_num(); ++i)
        EXPECT_NEAR(out1(i, 0), out4(i, 0), 1e-9) << "row " << i;
}

TEST(LlamaStructure, FutureTokensDoNotAffectPast)
{
    auto model = make_small_llama();
    auto oa = model.forward(mat_t<double>(1, 3, {1.0, 2.0, 3.0}));
    auto ob = model.forward(mat_t<double>(1, 3, {1.0, 2.0, 10.0}));
    for (int i = 0; i < oa.row_num(); ++i)
    {
        EXPECT_NEAR(oa(i, 0), ob(i, 0), 1e-9);
        EXPECT_NEAR(oa(i, 1), ob(i, 1), 1e-9);
    }
}

TEST(LlamaStructure, ForwardOneMatchesFullForward)
{
    auto model = make_small_llama(2, 4, 2, 16, 32, 13, 32);
    mat_t<double> ids(1, 5, {1.0, 4.0, 2.0, 7.0, 3.0});
    auto ref = model.forward(ids);

    // 逐 token
    model.clear_kv_cache();
    for (int t = 0; t < 5; ++t)
    {
        auto step = model.forward_one(ids.view(0, t, 1, 1).clone());
        ExpectShape(step, 13, 1);
        for (int i = 0; i < step.row_num(); ++i)
            EXPECT_NEAR(step(i, 0), ref(i, t), 1e-9) << "t=" << t << " row=" << i;
    }
    EXPECT_EQ(model.kv_cache_length(), 5);
}

TEST(LlamaStructure, PrefillMatchesPerTokenStepping)
{
    // prefill 是整段多列喂入（走 mat_mha_t 的多列 forward_one，内部自行补 causal mask）。
    // 它必须与逐 token 喂入完全一致 —— 否则 demo 里「预填 prompt」和「逐个 decode」
    // 的结果会不同，而这种差异只在长 prompt 上显现。
    auto model = make_small_llama(2, 4, 2, 16, 32, 13, 32);
    mat_t<double> ids(1, 6, {1.0, 4.0, 2.0, 7.0, 3.0, 5.0});
    auto ref = model.forward(ids);

    auto last = model.prefill(ids);
    ExpectShape(last, 13, 1);
    for (int i = 0; i < last.row_num(); ++i)
        EXPECT_NEAR(last(i, 0), ref(i, 5), 1e-9) << "row=" << i;
    EXPECT_EQ(model.kv_cache_length(), 6);

    // 与逐 token 路径对照（同一段 prompt，两条推理路径）
    model.clear_kv_cache();
    mat_t<double> step;
    for (int t = 0; t < 6; ++t)
        step = model.forward_one(ids.view(0, t, 1, 1).clone());
    EXPECT_LT(MaxAbsDiff(step, last), 1e-9);
}

TEST(LlamaStructure, MultiTurnCacheMatchesFullForward)
{
    // 交互式对话的关键不变量：跨轮复用 KV cache（中途不 clear）必须等价于把整段拼起来
    // 做一次 forward。这里的位置来自 RoPE 且由 kv_cache_length 推出，若位置推进有误，
    // 多轮之后就会漂移。
    auto model = make_small_llama(2, 4, 2, 16, 32, 13, 32);
    mat_t<double> all(1, 9, {1.0, 4.0, 2.0, 7.0, 3.0, 5.0, 0.0, 6.0, 9.0});
    auto ref = model.forward(all);

    model.clear_kv_cache();
    const int chunk_sizes[] = {2, 3, 4};
    int pos = 0;
    for (int ci = 0; ci < 3; ++ci)
    {
        // 每轮用「多列」喂入（模拟把该轮的新 token 一次 prefill）
        mat_t<double> chunk(1, chunk_sizes[ci]);
        for (int k = 0; k < chunk_sizes[ci]; ++k)
            chunk(0, k) = all(0, pos + k);
        auto out = model.forward_one(chunk);
        for (int r = 0; r < out.row_num(); ++r)
            EXPECT_NEAR(out(r, 0), ref(r, pos + chunk_sizes[ci] - 1), 1e-9)
                << "chunk " << ci << " row " << r;
        pos += chunk_sizes[ci];
    }
    EXPECT_EQ(model.kv_cache_length(), 9);

    // reset 后重新走一遍，结果必须一致（不残留状态）
    model.clear_kv_cache();
    EXPECT_EQ(model.kv_cache_length(), 0);
    pos = 0;
    for (int ci = 0; ci < 3; ++ci)
    {
        mat_t<double> chunk(1, chunk_sizes[ci]);
        for (int k = 0; k < chunk_sizes[ci]; ++k)
            chunk(0, k) = all(0, pos + k);
        auto out = model.forward_one(chunk);
        for (int r = 0; r < out.row_num(); ++r)
            EXPECT_NEAR(out(r, 0), ref(r, pos + chunk_sizes[ci] - 1), 1e-9);
        pos += chunk_sizes[ci];
    }
}

TEST(LlamaStructure, ForwardStagesMatchBlockByBlock)
{
    auto model = make_small_llama(3);
    mat_t<double> ids(1, 4, {1.0, 2.0, 3.0, 4.0});
    auto stages = model.forward_stages(ids);
    EXPECT_EQ(stages.size(), 4u);   // 嵌入 + 3 层

    mat_t<double> h = model.embed(ids);
    EXPECT_LT(MaxAbsDiff(h, stages[0]), 1e-12);
    for (int i = 0; i < model.n_layers(); ++i)
    {
        h = model.block_forward(i, h);
        EXPECT_LT(MaxAbsDiff(h, stages[i + 1]), 1e-12) << "layer " << i;
    }
    EXPECT_LT(MaxAbsDiff(model.head(stages.back()), model.forward(ids)), 1e-12);
}

TEST(LlamaStructure, HeadIsUntiedByDefault)
{
    // TinyLlama 的 tie_word_embeddings=false，与 GPT-2 相反
    auto model = make_small_llama();
    EXPECT_FALSE(model.tied_word_embeddings());
    ExpectShape(model.lm_head().weight(), 13, 16);

    // 显式绑定后应与 wte 的转置一致，且可解绑语义上仍是「显式绑定过」
    model.tie_word_embeddings();
    EXPECT_TRUE(model.tied_word_embeddings());
    for (int v = 0; v < 13; ++v)
        for (int d = 0; d < 16; ++d)
            EXPECT_NEAR(model.lm_head().weight()(v, d), model.wte().weight()(d, v), 1e-12);
}

TEST(LlamaStructure, RopeThetaMustBeSupported)
{
    // jas_RoPE_t.hpp 把基频 10000 硬编码；导入 500000 的模型（LLaMA-3.x）必须显式报错，
    // 而不是产出一个「看着像但其实错了」的 logits。
    EXPECT_NO_THROW(require_supported_rope_theta(10000.0));
    EXPECT_THROW(require_supported_rope_theta(500000.0), std::runtime_error);
    EXPECT_THROW(require_supported_rope_theta(1e6), std::runtime_error);
}

TEST(LlamaStructure, NetTypeMentionsLlamaTraits)
{
    auto model = make_small_llama();
    const std::string s = model.net_type();
    EXPECT_NE(s.find("llama_model_t"), std::string::npos);
    EXPECT_NE(s.find("rope"), std::string::npos);
    EXPECT_NE(s.find("swiglu"), std::string::npos);
    EXPECT_NE(s.find("kv_heads"), std::string::npos);
    EXPECT_NE(s.find("rms"), std::string::npos);
}

// ---------------------------------------------------------------------------
// 加载层：合成权重文件，逐个张量钉住 load_llama 的命名映射
// ---------------------------------------------------------------------------

TEST(LlamaLoad, SyntheticFileMapsEveryTensor)
{
    // 每个张量填成不同的常数：命名映射错位会直接表现为数值不符
    const int layers = 2, heads = 4, kv_heads = 2, d_model = 8, d_ff = 16, vocab = 10, n_pos = 16;
    const int d_head = d_model / heads;
    const int d_kv = kv_heads * d_head;
    const std::string path = write_synthetic_llama_weights(
        layers, heads, kv_heads, d_model, d_ff, vocab, n_pos, /*tied=*/false);

    weight_file_t wf;
    wf.load(path);
    const auto cfg = read_llama_config(wf);
    EXPECT_EQ(cfg.n_layers, layers);
    EXPECT_EQ(cfg.n_heads, heads);
    EXPECT_EQ(cfg.n_kv_heads, kv_heads);
    EXPECT_EQ(cfg.d_model, d_model);
    EXPECT_EQ(cfg.d_ff, d_ff);
    EXPECT_EQ(cfg.vocab, vocab);
    EXPECT_EQ(cfg.n_pos, n_pos);
    // 权重文件是 f32，所以 1e-5 会被舍入成 9.9999997473787516e-06（相对误差 2.5e-8）。
    // 不能用 EXPECT_DOUBLE_EQ —— 但也无需担心：这点 eps 差异对前向的影响远小于对齐容差。
    EXPECT_NEAR(cfg.rms_eps, 1e-5, 1e-12);
    EXPECT_DOUBLE_EQ(cfg.rope_theta, 10000.0);
    EXPECT_FALSE(cfg.tied);

    llama_type model;
    model.set_param(cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff,
                    cfg.vocab, cfg.n_pos, cfg.n_kv_heads, cfg.rms_eps);
    load_llama(model, wf);

    // 承重断言：所有 bias 必须仍是 0（加载流程的最后一步）
    for (int i = 0; i < model.n_layers(); ++i)
    {
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.mlp_gate(i).bias()), 0.0);
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.mlp_down(i).bias()), 0.0);
        EXPECT_DOUBLE_EQ(MaxAbsValue(model.attn(i).q_proj().bias()), 0.0);
    }
    EXPECT_DOUBLE_EQ(MaxAbsValue(model.lm_head().bias()), 0.0);

    // 各张量形状必须与权重文件一致（read_into 会在不一致时抛异常，这里再显式确认一遍）
    ExpectShape(model.wte().weight(), d_model, vocab);
    ExpectShape(model.lm_head().weight(), vocab, d_model);
    for (int i = 0; i < layers; ++i)
    {
        ExpectShape(model.ln_1(i).gama(), d_model, 1);
        ExpectShape(model.ln_2(i).gama(), d_model, 1);
        ExpectShape(model.mlp_gate(i).weight(), d_ff, d_model);
        ExpectShape(model.mlp_up(i).weight(), d_ff, d_model);
        ExpectShape(model.mlp_down(i).weight(), d_model, d_ff);
        ExpectShape(model.attn(i).k_proj().weight(), d_kv, d_model);
    }

    // 数值：把加载进来的常数与文件里的常数对齐（用 wf 自己再读一遍做交叉核对）
    mat_t<double> ref(d_model, vocab);
    wf.read_into("wte.weight", ref);
    ExpectNearMat(model.wte().weight(), ref, 0.0);
    ASSERT_GT(ref.row_num(), 0);
    EXPECT_DOUBLE_EQ(ref(0, 0), 1.0);      // 第一个张量的常数

    mat_t<double> ref_gate(d_ff, d_model);
    wf.read_into("h.1.mlp.gate.weight", ref_gate);
    ExpectNearMat(model.mlp_gate(1).weight(), ref_gate, 0.0);
    // 不同层必须是不同的数据（防止把 h.0 写进了 h.1）
    mat_t<double> ref_gate0(d_ff, d_model);
    wf.read_into("h.0.mlp.gate.weight", ref_gate0);
    EXPECT_NE(ref_gate(0, 0), ref_gate0(0, 0));

    std::filesystem::remove(path);
}

TEST(LlamaLoad, TiedHeadVariantHasNoLmHeadTensor)
{
    const int layers = 1, heads = 2, kv_heads = 1, d_model = 4, d_ff = 8, vocab = 6, n_pos = 8;
    const std::string path = write_synthetic_llama_weights(
        layers, heads, kv_heads, d_model, d_ff, vocab, n_pos, /*tied=*/true);

    weight_file_t wf;
    wf.load(path);
    EXPECT_FALSE(wf.has("lm_head.weight"));
    EXPECT_TRUE(read_llama_config(wf).tied);

    llama_type model;
    model.set_param(layers, heads, d_model, d_ff, vocab, n_pos, kv_heads, 1e-5);
    ASSERT_NO_THROW(load_llama(model, wf));

    // 无独立 lm_head 张量时必须退化为绑定 wte
    EXPECT_TRUE(model.tied_word_embeddings());
    for (int v = 0; v < vocab; ++v)
        for (int d = 0; d < d_model; ++d)
            EXPECT_NEAR(model.lm_head().weight()(v, d), model.wte().weight()(d, v), 1e-12);

    std::filesystem::remove(path);
}

TEST(LlamaLoad, MissingTensorIsRejected)
{
    // 少一个张量必须在加载时就炸，而不是留下一个未初始化的权重静默跑出错误结果
    const int layers = 1, heads = 2, kv_heads = 1, d_model = 4, d_ff = 8, vocab = 6, n_pos = 8;
    const std::string path = write_synthetic_llama_weights(
        layers, heads, kv_heads, d_model, d_ff, vocab, n_pos, false);

    // 用一个更大的 d_ff 去读，形状对不上
    weight_file_t wf;
    wf.load(path);
    llama_type model;
    model.set_param(layers, heads, d_model, /*d_ff=*/d_ff * 2, vocab, n_pos, kv_heads, 1e-5);
    EXPECT_THROW(load_llama(model, wf), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(LlamaLoad, ReleaseFreesBlobAndBlocksFurtherReads)
{
    // 大模型加载完成后应能显式释放数据区（TinyLlama 的 f32 权重约 4.4GB）
    const int layers = 1, heads = 2, kv_heads = 1, d_model = 4, d_ff = 8, vocab = 6, n_pos = 8;
    const std::string path = write_synthetic_llama_weights(
        layers, heads, kv_heads, d_model, d_ff, vocab, n_pos, false);

    weight_file_t wf;
    wf.load(path);
    EXPECT_FALSE(wf.released());
    mat_t<double> w(d_model, vocab);
    ASSERT_NO_THROW(wf.read_into("wte.weight", w));

    wf.release();
    EXPECT_TRUE(wf.released());
    EXPECT_TRUE(wf.has("wte.weight"));       // 索引仍在
    EXPECT_THROW(wf.read_into("wte.weight", w), std::runtime_error);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// 对齐层：与 HuggingFace 的黄金值比对（需要 export_llama.py 导出的真实权重）
// ---------------------------------------------------------------------------

namespace
{

/** 需要导出权重（含黄金值）的测试基类 */
class LlamaAlignmentTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_path = find_llama_weights();
        if (m_path.empty())
        {
            GTEST_SKIP() << "no LLaMA weight file found; run "
                            "`python tools/export_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
                            "--out build/tinyllama_weights.bin --golden-prompt \"The capital of France is\" "
                            "--golden-chat \"What is the capital of France?\"` "
                            "or set JASMINE_LLAMA_WEIGHTS";
        }
        ASSERT_NO_THROW(m_wf.load(m_path)) << "failed to load " << m_path;
        ASSERT_TRUE(m_wf.has("wte.weight")) << "weight file looks malformed: " << m_path;
    }

    weight_file_t m_wf;
    std::string m_path;

    llama_type make_model() const
    {
        auto cfg = read_llama_config(m_wf);
        require_supported_rope_theta(cfg.rope_theta);
        llama_type model;
        model.set_param(cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff,
                        cfg.vocab, cfg.n_pos, cfg.n_kv_heads,
                        static_cast<double>(cfg.rms_eps));
        return model;
    }

    /** 取出某个 golden 组的输入 token 与参考 logits */
    void load_golden(const std::string& tag, mat_t<double>& ids, mat_t<double>& ref_logits) const
    {
        const std::string in = "golden." + tag + ".input_ids";
        const std::string lg = "golden." + tag + ".logits";
        ASSERT_TRUE(m_wf.has(in)) << "no " << in << "; re-export with --golden-*";
        ASSERT_TRUE(m_wf.has(lg)) << "no " << lg;
        const auto& ie = m_wf.entry(in);
        const auto& le = m_wf.entry(lg);
        ids = mat_t<double>(ie.rows, ie.cols);
        ref_logits = mat_t<double>(le.rows, le.cols);
        m_wf.read_into(in, ids);
        m_wf.read_into(lg, ref_logits);
    }

    static void check_argmax(const mat_t<double>& got, const mat_t<double>& ref,
                             const std::string& what)
    {
        ASSERT_EQ(got.row_num(), ref.row_num());
        ASSERT_EQ(got.col_num(), ref.col_num());
        for (int t = 0; t < ref.col_num(); ++t)
        {
            int j_ours = 0, j_ref = 0;
            for (int v = 1; v < ref.row_num(); ++v)
            {
                if (got(v, t) > got(j_ours, t)) j_ours = v;
                if (ref(v, t) > ref(j_ref, t)) j_ref = v;
            }
            EXPECT_EQ(j_ours, j_ref) << what << ": argmax mismatch at position " << t;
        }
    }
};

} // namespace

TEST_F(LlamaAlignmentTest, RawHiddenStatesMatchLayerByLayer)
{
    auto model = make_model();
    load_llama(model, m_wf);

    mat_t<double> ids, ref_logits;
    load_golden("raw", ids, ref_logits);

    auto stages = model.forward_stages(ids);
    ASSERT_EQ(static_cast<int>(stages.size()), model.n_layers() + 1);

    const double tol = 2e-3;
    for (int i = 0; i < static_cast<int>(stages.size()); ++i)
    {
        const std::string name = "golden.raw.hidden." + std::to_string(i);
        if (!m_wf.has(name)) continue;
        const auto& e = m_wf.entry(name);
        mat_t<double> ref(e.rows, e.cols);
        m_wf.read_into(name, ref);
        const double diff = MaxAbsDiff(stages[i], ref);
        const std::string label = (i == 0) ? "embeddings" : ("block " + std::to_string(i - 1));
        EXPECT_LT(diff, tol) << "hidden stage " << i << " (" << label
                             << ") mismatch, max_abs_diff=" << diff;
    }
}

TEST_F(LlamaAlignmentTest, RawGoldenLogitsMatch)
{
    auto model = make_model();
    load_llama(model, m_wf);

    // 锁定 RoPE 配对约定：HF LLaMA 用 (i, i + d/2)，若写成 jasmine 原生的交错配对，
    // 除位置 0 以外所有位置都会错位（且位置 0 仍然「看起来是对的」，极易漏掉）。
    EXPECT_EQ(model.attn(0).pair_layout(), rope_pair_layout::half_split);

    mat_t<double> ids, ref;
    load_golden("raw", ids, ref);

    auto logits = model.forward(ids);
    ExpectShape(logits, ref.row_num(), ref.col_num());

    // 实测 ~3e-5，量级正好是 float32 参考值本身的舍入噪声；
    // 取 1e-3 留足跨平台/BLAS 归约顺序的余量，同时足以拦住配对约定之类的结构性错误（会造成 O(1) 偏差）
    const double diff = MaxAbsDiff(logits, ref);
    EXPECT_LT(diff, 1e-3) << "logits max_abs_diff=" << diff;
    check_argmax(logits, ref, "raw");
}

TEST_F(LlamaAlignmentTest, KVCacheDecodeMatchesGolden)
{
    auto model = make_model();
    load_llama(model, m_wf);

    mat_t<double> ids, ref;
    load_golden("raw", ids, ref);
    const int T = ids.col_num();

    // 逐 token 走 KV cache，每步都应与黄金 logits 对齐
    model.clear_kv_cache();
    for (int t = 0; t < T; ++t)
    {
        auto step = model.forward_one(ids.view(0, t, 1, 1).clone());
        ExpectShape(step, ref.row_num(), 1);
        double diff = 0.0;
        for (int v = 0; v < step.row_num(); ++v)
            diff = std::max(diff, std::abs(step(v, 0) - ref(v, t)));
        EXPECT_LT(diff, 1e-3) << "step " << t << " max_abs_diff=" << diff;
    }
}

TEST_F(LlamaAlignmentTest, PrefillMatchesGoldenLastPosition)
{
    // demo 实际走的路径：整段 prefill 一次，取最后一个位置的 logits
    auto model = make_model();
    load_llama(model, m_wf);

    mat_t<double> ids, ref;
    load_golden("raw", ids, ref);
    const int T = ids.col_num();

    auto last = model.prefill(ids);
    ExpectShape(last, ref.row_num(), 1);
    double diff = 0.0;
    for (int v = 0; v < last.row_num(); ++v)
        diff = std::max(diff, std::abs(last(v, 0) - ref(v, T - 1)));
    EXPECT_LT(diff, 1e-3) << "prefill vs golden last position max_abs_diff=" << diff;
}

TEST_F(LlamaAlignmentTest, ChatTemplateLogitsMatch)
{
    // 黄金值来自 tokenizer.apply_chat_template(...)，正是 llama_chat 喂给模型的序列。
    // 这条把「对话 demo 的输入构造」也纳入验证范围，而不只是裸文本前向。
    if (!m_wf.has("golden.chat.input_ids"))
        GTEST_SKIP() << "no golden.chat.* in the weight file; re-export with --golden-chat";

    auto model = make_model();
    load_llama(model, m_wf);

    mat_t<double> ids, ref;
    load_golden("chat", ids, ref);

    auto logits = model.forward(ids);
    const double diff = MaxAbsDiff(logits, ref);
    EXPECT_LT(diff, 1e-3) << "chat-template logits max_abs_diff=" << diff;
    check_argmax(logits, ref, "chat");
}

TEST_F(LlamaAlignmentTest, ConfigMatchesTinyLlamaShapes)
{
    auto cfg = read_llama_config(m_wf);
    if (cfg.d_model != 2048)
        GTEST_SKIP() << "not TinyLlama-1.1B (d_model=" << cfg.d_model << ")";

    EXPECT_EQ(cfg.n_layers, 22);
    EXPECT_EQ(cfg.n_heads, 32);
    EXPECT_EQ(cfg.n_kv_heads, 4);
    EXPECT_EQ(cfg.d_ff, 5632);
    EXPECT_EQ(cfg.vocab, 32000);
    EXPECT_EQ(cfg.n_pos, 2048);
    // rms_eps 在 config.json 里是 float32 字面量 1e-5，读回来会有 float32 舍入
    // (9.9999997473787516e-06)，因此用近邻比较
    EXPECT_NEAR(cfg.rms_eps, 1e-5, 1e-12);
    EXPECT_DOUBLE_EQ(cfg.rope_theta, 10000.0);
    EXPECT_FALSE(cfg.tied);
}
