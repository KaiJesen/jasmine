/**
 * GQA（Grouped-Query Attention）单测。
 *
 * 设计前提：GQA 不是新算法，而是把「K/V 头数」从 Q 头数上解耦。
 *   n_kv_heads == n_heads    → 经典 MHA（默认，与旧实现逐位一致）
 *   1 < n_kv_heads < n_heads → GQA
 *   n_kv_heads == 1          → MQA
 *
 * 因此测试重点不是"再验一遍注意力数学"（那由 test_mha / test_causal_attention 覆盖），
 * 而是钉住 GQA 特有的三处差异：
 *   1. K/V 投影变窄（n_kv_heads*d_head）且分组映射正确；
 *   2. KV cache 每个 **KV 头** 只 append 一次（naive 复用 forward_one_at 会重复写入）；
 *   3. 反向时共享同一 KV 头的多个 Q 头梯度必须**累加**。
 *
 * 最强判据是「GQA == 把 K/V 复制 group_size 份的 MHA」：一次校验前向分组与反向累加语义。
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_mha_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

using dmat = mat_t<double>;
using mha_t = mat_mha_t<dmat, nadam_t>;
using mha_sgd_t = mat_mha_t<dmat, sgd_t>;

/** 确定性伪随机填充（不依赖全局随机引擎，保证可复现） */
void fill_lin(dmat& w, dmat& b, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (int i = 0; i < w.row_num(); ++i)
    {
        b(i, 0) = dist(rng);
        for (int j = 0; j < w.col_num(); ++j)
            w(i, j) = dist(rng);
    }
}

dmat make_mat(int rows, int cols, double scale, unsigned seed)
{
    dmat m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = dist(rng);
    return m;
}

/**
 * 完全独立的朴素 GQA 参考实现（不经过 mat_mha_t 的任何代码路径）。
 * 显式按 Q 头循环、显式行 softmax、显式按 kv_head = h / group 取 K/V。
 */
dmat naive_gqa_forward(const dmat& Wq, const dmat& bq,
                       const dmat& Wk, const dmat& bk,
                       const dmat& Wv, const dmat& bv,
                       const dmat& Wo, const dmat& bo,
                       const dmat& x, int n_q, int n_kv, bool mask)
{
    const int d_model = Wq.col_num();
    const int dh = d_model / n_q;
    const int group = n_q / n_kv;
    const int T = x.col_num();

    const dmat Q = Wq.dot(x) + bq;   // d_model × T
    const dmat K = Wk.dot(x) + bk;   // n_kv*dh × T
    const dmat V = Wv.dot(x) + bv;

    dmat concat(d_model, T);
    for (int h = 0; h < n_q; ++h)
    {
        const int kv = h / group;      // 分组映射：连续 group 个 Q 头共用一个 KV 头
        dmat q_h(dh, T), k_g(dh, T), v_g(dh, T);
        for (int r = 0; r < dh; ++r)
            for (int t = 0; t < T; ++t)
            {
                q_h(r, t) = Q(h * dh + r, t);
                k_g(r, t) = K(kv * dh + r, t);
                v_g(r, t) = V(kv * dh + r, t);
            }

        dmat scores = q_h.t().dot(k_g) / std::sqrt(static_cast<double>(dh));   // T × T
        if (mask)
            for (int i = 0; i < T; ++i)
                for (int j = i + 1; j < T; ++j)
                    scores(i, j) = -std::numeric_limits<double>::infinity();

        dmat attn(T, T);
        for (int i = 0; i < T; ++i)
        {
            double mx = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < T; ++j)
                mx = std::max(mx, scores(i, j));
            double sum = 0.0;
            for (int j = 0; j < T; ++j)
            {
                attn(i, j) = std::exp(scores(i, j) - mx);
                sum += attn(i, j);
            }
            for (int j = 0; j < T; ++j)
                attn(i, j) /= sum;
        }

        const dmat out_h = v_g.dot(attn.t());     // dh × T
        for (int r = 0; r < dh; ++r)
            for (int t = 0; t < T; ++t)
                concat(h * dh + r, t) = out_h(r, t);
    }
    return Wo.dot(concat) + bo;
}

/**
 * 把 GQA 的 K/V 权重按 group 复制成等价 MHA 的权重：
 * MHA 的 KV 头 g（共 n_q 个）应等于 GQA 的 KV 头 g / group。
 */
void replicate_kv(const dmat& gqa_w, const dmat& gqa_b, int dh, int group,
                  dmat& mha_w, dmat& mha_b)
{
    const int n_q = mha_w.row_num() / dh;
    for (int g = 0; g < n_q; ++g)
    {
        const int src = (g / group) * dh;
        for (int r = 0; r < dh; ++r)
        {
            mha_b(g * dh + r, 0) = gqa_b(src + r, 0);
            for (int j = 0; j < gqa_w.col_num(); ++j)
                mha_w(g * dh + r, j) = gqa_w(src + r, j);
        }
    }
}

/** 用同一套权重搭建 (GQA, 等价的 K/V 复制版 MHA)，Q/O 同构 */
void wire_equivalent_pair(mha_t& gqa, mha_t& mha, int dh, int group)
{
    fill_lin(gqa.q_proj().weight(), gqa.q_proj().bias(), 1);
    fill_lin(gqa.k_proj().weight(), gqa.k_proj().bias(), 2);
    fill_lin(gqa.v_proj().weight(), gqa.v_proj().bias(), 3);
    fill_lin(gqa.out_proj().weight(), gqa.out_proj().bias(), 4);

    mha.q_proj().weight() = gqa.q_proj().weight();
    mha.q_proj().bias() = gqa.q_proj().bias();
    mha.out_proj().weight() = gqa.out_proj().weight();
    mha.out_proj().bias() = gqa.out_proj().bias();
    replicate_kv(gqa.k_proj().weight(), gqa.k_proj().bias(), dh, group,
                 mha.k_proj().weight(), mha.k_proj().bias());
    replicate_kv(gqa.v_proj().weight(), gqa.v_proj().bias(), dh, group,
                 mha.v_proj().weight(), mha.v_proj().bias());
}

} // namespace

TEST(Gqa, DefaultsToClassicMha)
{
    // 不传 n_kv_heads 时必须退化为经典 MHA，维度与旧实现一致
    mha_t m(4, 8, true, 2);
    EXPECT_EQ(m.num_heads(), 4);
    EXPECT_EQ(m.num_kv_heads(), 4);
    EXPECT_EQ(m.group_size(), 1);
    EXPECT_EQ(m.d_head(), 2);
    EXPECT_EQ(m.d_kv(), 8);
    EXPECT_EQ(m.k_proj().weight().row_num(), 8);
    EXPECT_EQ(m.v_proj().weight().row_num(), 8);
    EXPECT_EQ(m.d_kv(), m.k_proj().weight().row_num());
}

TEST(Gqa, KvProjectionIsNarrower)
{
    const int d_model = 8, n_q = 4, n_kv = 2;
    mha_t gqa(n_q, d_model, true, 2, n_kv);
    EXPECT_EQ(gqa.num_kv_heads(), 2);
    EXPECT_EQ(gqa.group_size(), 2);
    EXPECT_EQ(gqa.d_kv(), 4);                    // n_kv * d_head = 2 * 2

    // K/V 投影变窄，Q/O 仍为全宽
    EXPECT_EQ(gqa.k_proj().weight().row_num(), 4);
    EXPECT_EQ(gqa.v_proj().weight().row_num(), 4);
    EXPECT_EQ(gqa.k_proj().weight().col_num(), d_model);
    EXPECT_EQ(gqa.q_proj().weight().row_num(), d_model);
    EXPECT_EQ(gqa.out_proj().weight().row_num(), d_model);

    // 输出仍是 d_model 宽（concat 的是 n_q 个头，不是 n_kv 个）
    dmat x = make_mat(d_model, 3, 0.5, 1);
    ExpectShape(gqa.forward(x), d_model, 3);
}

TEST(Gqa, MatchesNaiveReference)
{
    // 与完全独立的朴素实现逐元素比对；关掉 RoPE 以便参考实现保持简单
    const int d_model = 8, n_q = 4, n_kv = 2, T = 3;
    for (bool mask : {false, true})
    {
        mha_t gqa(n_q, d_model, mask, T, n_kv);
        gqa.set_use_rope(false);
        fill_lin(gqa.q_proj().weight(), gqa.q_proj().bias(), 11);
        fill_lin(gqa.k_proj().weight(), gqa.k_proj().bias(), 12);
        fill_lin(gqa.v_proj().weight(), gqa.v_proj().bias(), 13);
        fill_lin(gqa.out_proj().weight(), gqa.out_proj().bias(), 14);

        dmat x = make_mat(d_model, T, 0.5, 15);
        dmat ref = naive_gqa_forward(
            gqa.q_proj().weight(), gqa.q_proj().bias(),
            gqa.k_proj().weight(), gqa.k_proj().bias(),
            gqa.v_proj().weight(), gqa.v_proj().bias(),
            gqa.out_proj().weight(), gqa.out_proj().bias(),
            x, n_q, n_kv, mask);

        ExpectNearMat(gqa.forward(x), ref, 1e-12);
        SCOPED_TRACE(std::string("mask=") + std::to_string(mask));
    }
}

TEST(Gqa, EquivalentToMhaWithReplicatedKv)
{
    // GQA 的定义就是「K/V 头被多个 Q 头共享」，等价于把 K/V 复制 group_size 份的 MHA。
    // RoPE 开/关都测：shared K 在每个 Q 头里各自旋转，必须与 MHA 各副本一致。
    const int d_model = 8, n_q = 4, n_kv = 2, dh = 2, group = 2, T = 3;
    for (bool use_rope : {false, true})
    {
        for (bool mask : {false, true})
        {
            mha_t gqa(n_q, d_model, mask, T, n_kv);
            mha_t mha(n_q, d_model, mask, T, 0);     // n_kv = n_q
            gqa.set_use_rope(use_rope);
            mha.set_use_rope(use_rope);
            wire_equivalent_pair(gqa, mha, dh, group);

            dmat x = make_mat(d_model, T, 0.5, 42);
            ExpectNearMat(gqa.forward(x), mha.forward(x), 1e-12);
            SCOPED_TRACE(std::string("rope=") + std::to_string(use_rope)
                         + " mask=" + std::to_string(mask));
        }
    }
}

TEST(Gqa, BackwardSumsKvGradientsAcrossGroup)
{
    // 核心语义测试：GQA 一个 KV 头的梯度 == 共享它的各个 Q 头在 MHA 中等价位置的梯度之和。
    // 若把 add_rows 误写成 assign（覆盖），这里只会剩下组内最后一个 Q 头的贡献，立刻暴露。
    const int d_model = 8, n_q = 4, n_kv = 2, dh = 2, group = 2, T = 3;

    mha_sgd_t gqa(n_q, d_model, true, T, n_kv);
    mha_sgd_t mha(n_q, d_model, true, T, 0);
    // sgd lr=1 → weight_after == weight_before - grad，可直接读出梯度
    gqa.set_updator(1.0);
    mha.set_updator(1.0);
    gqa.set_use_rope(false);
    mha.set_use_rope(false);

    // 这里两边类型不同（sgd），单独接线
    fill_lin(gqa.q_proj().weight(), gqa.q_proj().bias(), 1);
    fill_lin(gqa.k_proj().weight(), gqa.k_proj().bias(), 2);
    fill_lin(gqa.v_proj().weight(), gqa.v_proj().bias(), 3);
    fill_lin(gqa.out_proj().weight(), gqa.out_proj().bias(), 4);
    mha.q_proj().weight() = gqa.q_proj().weight();
    mha.q_proj().bias() = gqa.q_proj().bias();
    mha.out_proj().weight() = gqa.out_proj().weight();
    mha.out_proj().bias() = gqa.out_proj().bias();
    replicate_kv(gqa.k_proj().weight(), gqa.k_proj().bias(), dh, group,
                 mha.k_proj().weight(), mha.k_proj().bias());
    replicate_kv(gqa.v_proj().weight(), gqa.v_proj().bias(), dh, group,
                 mha.v_proj().weight(), mha.v_proj().bias());

    dmat x = make_mat(d_model, T, 0.5, 7);
    dmat delta = make_mat(d_model, T, 0.3, 8);

    // 前向先对齐，否则后面的梯度比较没有意义
    ExpectNearMat(gqa.forward(x), mha.forward(x), 1e-12);

    const dmat gqa_kw = gqa.k_proj().weight();
    const dmat gqa_vw = gqa.v_proj().weight();
    gqa.backward(delta);
    const dmat dgqa_k = gqa_kw - gqa.k_proj().weight();
    const dmat dgqa_v = gqa_vw - gqa.v_proj().weight();

    const dmat mha_kw = mha.k_proj().weight();
    const dmat mha_vw = mha.v_proj().weight();
    mha.backward(delta);
    const dmat dmha_k = mha_kw - mha.k_proj().weight();
    const dmat dmha_v = mha_vw - mha.v_proj().weight();

    for (int j = 0; j < n_kv; ++j)
    {
        for (int r = 0; r < dh; ++r)
        {
            for (int c = 0; c < d_model; ++c)
            {
                double sum_k = 0.0, sum_v = 0.0;
                for (int g = j * group; g < (j + 1) * group; ++g)   // 组内各 Q 头
                {
                    sum_k += dmha_k(g * dh + r, c);
                    sum_v += dmha_v(g * dh + r, c);
                }
                EXPECT_NEAR(dgqa_k(j * dh + r, c), sum_k, 1e-12)
                    << "kv_head=" << j << " r=" << r << " c=" << c;
                EXPECT_NEAR(dgqa_v(j * dh + r, c), sum_v, 1e-12)
                    << "kv_head=" << j << " r=" << r << " c=" << c;
            }
        }
    }
}

TEST(Gqa, KvCacheGetsOneAppendPerKvHead)
{
    // 回归测试：共享 KV 头的多个 Q 头若各自调用 forward_one_at，同一份 K/V 会被写 group 次。
    // 正确实现必须每个 KV 头只 append 一次 → cache 长度恰为步数，而不是 步数*group_size。
    const int d_model = 8, n_q = 4, n_kv = 2, group = 2, T = 5;
    ASSERT_GT(group, 1) << "本测试的前提是存在共享";

    mha_t gqa(n_q, d_model, true, T, n_kv);
    gqa.init_weight<xavier_gaussian_t>();
    gqa.clear_kv_cache();

    for (int t = 0; t < T; ++t)
        gqa.forward_one(make_mat(d_model, 1, 0.5, 100 + t));

    EXPECT_EQ(gqa.kv_cache_length(), T)
        << "共享 KV 头被重复 append；长度应为 " << T << "，而不是 " << T * group;
}

TEST(Gqa, CacheMatchesFullForward)
{
    // 增量推理（KV cache，每个 KV 头只写一次）必须与整段训练前向逐列一致
    const int d_model = 8, n_q = 4, n_kv = 2, T = 4;
    mha_t gqa(n_q, d_model, true, T, n_kv);
    gqa.init_weight<xavier_gaussian_t>();

    dmat x = make_mat(d_model, T, 0.5, 55);
    dmat full = gqa.forward(x);            // 训练路径：一次整段

    gqa.clear_kv_cache();
    dmat last;
    for (int t = 0; t < T; ++t)
        last = gqa.forward_one(x.view(0, t, d_model, 1).clone());

    ExpectNearMat(last, full.view(0, T - 1, d_model, 1), 1e-12);
    EXPECT_EQ(gqa.kv_cache_length(), T);
}

TEST(Gqa, BackwardMatchesNumericalGradient)
{
    // 端到端数值梯度：覆盖 MQA（n_kv == 1），此时 group_size == n_q，累加最"重"
    const int d_model = 6, n_q = 3, n_kv = 1, T = 2;
    mha_sgd_t gqa(n_q, d_model, true, T, n_kv);
    ASSERT_EQ(gqa.group_size(), n_q);
    gqa.set_updator(1.0);      // sgd lr=1
    gqa.set_use_rope(false);

    fill_lin(gqa.q_proj().weight(), gqa.q_proj().bias(), 21);
    fill_lin(gqa.k_proj().weight(), gqa.k_proj().bias(), 22);
    fill_lin(gqa.v_proj().weight(), gqa.v_proj().bias(), 23);
    fill_lin(gqa.out_proj().weight(), gqa.out_proj().bias(), 24);

    dmat x = make_mat(d_model, T, 0.5, 25);
    // 损失 L = Σ 0.5·out²  ⇒  dL/dout = out。
    // 必须在 gqa 自身上先 forward：backward 依赖 forward 留下的内部缓存
    // （softmax 输出、各头的 m_q/m_k/m_v、投影层的 m_input）。
    const dmat delta = gqa.forward(x);
    // forward 不改权重；此后快照即"解析梯度所基于的参数状态"
    mha_sgd_t pristine = gqa;

    // 损失 L = Σ 0.5·out²  ⇒  dL/dout = out（delta 已在上方由 gqa.forward 得到）
    const dmat wq0 = pristine.q_proj().weight();
    const dmat wk0 = pristine.k_proj().weight();
    const dmat wv0 = pristine.v_proj().weight();
    const dmat wo0 = pristine.out_proj().weight();

    gqa.backward(delta);
    const dmat dwq = wq0 - gqa.q_proj().weight();
    const dmat dwk = wk0 - gqa.k_proj().weight();
    const dmat dwv = wv0 - gqa.v_proj().weight();
    const dmat dwo = wo0 - gqa.out_proj().weight();

    auto loss_of = [&](const dmat& wq, const dmat& wk, const dmat& wv, const dmat& wo) {
        auto g = pristine;
        g.q_proj().weight() = wq;
        g.k_proj().weight() = wk;
        g.v_proj().weight() = wv;
        g.out_proj().weight() = wo;
        const dmat o = g.forward(x);
        double s = 0.0;
        for (int i = 0; i < o.row_num(); ++i)
            for (int j = 0; j < o.col_num(); ++j)
                s += 0.5 * o(i, j) * o(i, j);
        return s;
    };

    const double h = 1e-6;
    // 对四个投影逐个权重量做中心差分；which: 0=Wq 1=Wk 2=Wv 3=Wo
    auto check = [&](int which, const dmat& w0, const dmat& analytic) {
        for (int i = 0; i < w0.row_num(); ++i)
            for (int j = 0; j < w0.col_num(); ++j)
            {
                dmat wp = w0, wm = w0;
                wp(i, j) += h;
                wm(i, j) -= h;
                dmat a = wq0, b = wk0, c = wv0, d = wo0;
                dmat a2 = wq0, b2 = wk0, c2 = wv0, d2 = wo0;
                switch (which)
                {
                case 0: a = wp; a2 = wm; break;
                case 1: b = wp; b2 = wm; break;
                case 2: c = wp; c2 = wm; break;
                default: d = wp; d2 = wm; break;
                }
                const double num = (loss_of(a, b, c, d) - loss_of(a2, b2, c2, d2)) / (2 * h);
                EXPECT_NEAR(analytic(i, j), num, 1e-6)
                    << "which=" << which << " (" << i << "," << j << ")";
            }
    };

    check(0, wq0, dwq);
    check(1, wk0, dwk);
    check(2, wv0, dwv);
    check(3, wo0, dwo);
}

TEST(Gqa, MqaSingleKvHead)
{
    const int d_model = 8, n_q = 4, n_kv = 1, T = 3;
    mha_t mqa(n_q, d_model, true, T, n_kv);
    EXPECT_EQ(mqa.num_kv_heads(), 1);
    EXPECT_EQ(mqa.group_size(), n_q);        // 所有 Q 头共享唯一 KV 头
    EXPECT_EQ(mqa.d_kv(), 2);
    mqa.init_weight<xavier_gaussian_t>();

    dmat x = make_mat(d_model, T, 0.5, 9);
    ExpectShape(mqa.forward(x), d_model, T);

    // 反向可跑通（group_size == n_q 的极端累加）。
    // 注意顺序：backward 依赖训练路径 forward 留下的头内缓存（m_v 等），
    // 必须在 forward_one 之前做 —— forward_one 走 attend_cached，不填这些缓存。
    EXPECT_NO_THROW(mqa.backward(make_mat(d_model, T, 0.1, 30)));

    // KV cache 只有 1 份，长度等于步数
    mqa.clear_kv_cache();
    for (int t = 0; t < T; ++t)
        mqa.forward_one(make_mat(d_model, 1, 0.5, 20 + t));
    EXPECT_EQ(mqa.kv_cache_length(), T);
}

TEST(Gqa, RejectsIndivisibleKvHeads)
{
    // num_heads 必须被 n_kv_heads 整除，d_model 必须被 num_heads 整除
    EXPECT_THROW((mha_t(4, 8, true, 2, 3)), std::runtime_error);
    EXPECT_THROW((mha_t(3, 8, true, 2, 0)), std::runtime_error);
    // 合法取值
    EXPECT_NO_THROW((mha_t(4, 8, true, 2, 4)));   // == n_q → MHA
    EXPECT_NO_THROW((mha_t(4, 8, true, 2, 2)));   // GQA
    EXPECT_NO_THROW((mha_t(4, 8, true, 2, 1)));   // MQA
}

TEST(Gqa, NetTypeNamesTheVariant)
{
    mha_t mha(4, 8, true, 2, 4);
    mha_t gqa(4, 8, true, 2, 2);
    mha_t mqa(4, 8, true, 2, 1);
    EXPECT_NE(mha.net_type().find("MHA"), std::string::npos);
    EXPECT_NE(gqa.net_type().find("GQA"), std::string::npos);
    EXPECT_NE(mqa.net_type().find("MQA"), std::string::npos);
    EXPECT_NE(gqa.net_type().find("kv_heads:2"), std::string::npos);
}
