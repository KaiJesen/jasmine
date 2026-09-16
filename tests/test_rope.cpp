#include <algorithm>
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include "jas_RoPE_t.hpp"
#include "jas_mha_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"


using namespace jasmine;
TEST(RoPE, UniteAndNetForwardFinite)
{
    mat_RoPE_t<float> rope(64);
    auto u0 = rope.forward_unite(0, 0);
    ExpectShape(u0, 2, 2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_TRUE(std::isfinite(u0(i, j)));

    RoPE_net_t<mat_t<double>> rope_net(4);
    mat_t<double> input(4, 4, {
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6
    });
    auto out = rope_net.forward(input);
    ExpectShape(out, 4, 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_TRUE(std::isfinite(out(i, j)));
}

TEST(RoPE, UniteSharesSingleTheta)
{
    // d=4, pair=0 → θ = m / 10000^0 = m；pair=1 → θ = m / 10000^(2/4) = m / 100
    mat_RoPE_t<double> rope(4);

    auto u00 = rope.forward_unite(0, 0); // φ = 0
    EXPECT_NEAR(u00(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(u00(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(u00(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(u00(1, 1), 1.0, 1e-12);

    auto u01 = rope.forward_unite(0, 1); // φ = 1
    const double c1 = std::cos(1.0);
    const double s1 = std::sin(1.0);
    EXPECT_NEAR(u01(0, 0), c1, 1e-12);
    EXPECT_NEAR(u01(0, 1), -s1, 1e-12);
    EXPECT_NEAR(u01(1, 0), s1, 1e-12);
    EXPECT_NEAR(u01(1, 1), c1, 1e-12);

    auto u11 = rope.forward_unite(1, 1); // φ = 1 / 100
    const double c2 = std::cos(0.01);
    const double s2 = std::sin(0.01);
    EXPECT_NEAR(u11(0, 0), c2, 1e-12);
    EXPECT_NEAR(u11(0, 1), -s2, 1e-12);
    EXPECT_NEAR(u11(1, 0), s2, 1e-12);
    EXPECT_NEAR(u11(1, 1), c2, 1e-12);
}

TEST(RoPE, NetForwardMatchesManualRotation)
{
    RoPE_net_t<mat_t<double>> rope_net(2);
    mat_t<double> input(2, 2, {
        1.0, 0.0,
        0.0, 1.0
    });
    // col0 m=0: identity; col1 m=1, pair0: rotate (0,1) by φ=1
    auto out = rope_net.forward(input);
    EXPECT_NEAR(out(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(out(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(out(0, 1), -std::sin(1.0), 1e-12);
    EXPECT_NEAR(out(1, 1),  std::cos(1.0), 1e-12);
}

TEST(RoPE, StaticCacheMatchesDynamic)
{
    mat_t<double> input(4, 3, {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    });

    RoPE_net_t<mat_t<double>> dyn(4);
    auto out_dyn = dyn.forward(input);

    RoPE_net_t<mat_t<double>> stat(4);
    stat.set_cache_mode(rope_cache_mode::static_fixed);
    stat.reserve(8);
    auto out_stat = stat.forward(input);

    ExpectNearMat(out_dyn, out_stat, 1e-12);
}

TEST(RoPE, StaticCacheRejectsOverflow)
{
    mat_RoPE_t<double> rope(4);
    rope.set_cache_mode(rope_cache_mode::static_fixed);
    rope.reserve(2); // positions m=0,1 only

    EXPECT_NO_THROW(rope.forward_unite(0, 1));
    EXPECT_THROW(rope.forward_unite(0, 2), std::out_of_range);
}

TEST(RoPE, RegistrySharesSameInstanceByD)
{
    auto& reg = rope_registry_t<double>::instance();
    reg.clear();
    auto a = reg.get(4);
    auto b = reg.get(4);
    auto c = reg.get(8);
    EXPECT_EQ(a.get(), b.get());
    EXPECT_NE(a.get(), c.get());
    EXPECT_EQ(reg.size(), 2u);
    reg.clear();
}

TEST(RoPE, MhaHeadsShareRegistryRope)
{
    auto& reg = rope_registry_t<double>::instance();
    reg.clear();

    mat_mha_t<mat_t<double>, nadam_t> mha(2, 4, false, 2); // d_head = 2
    mat_mha_t<mat_t<double>, nadam_t> mha2(2, 4, false, 3);
    EXPECT_TRUE(reg.contains(2));
    EXPECT_EQ(reg.size(), 1u);

    auto rope = reg.get(2);
    ASSERT_NE(rope, nullptr);
    EXPECT_EQ(rope->get_d(), 2);
    reg.clear();
}

TEST(RoPE, BackwardIsTransposeRotation)
{
    // d=2, seq=2: position m=1 uses φ=1, R = [[c,-s],[s,c]], backward applies R^T
    RoPE_net_t<mat_t<double>> rope_net(2);
    mat_t<double> delta(2, 2, 0.0);
    delta(0, 1) = 1.0; // col1 = [1, 0]^T
    auto grad = rope_net.backward(delta);
    const double c = std::cos(1.0);
    const double s = std::sin(1.0);
    // col0 m=0: R^T = I
    EXPECT_NEAR(grad(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(grad(1, 0), 0.0, 1e-12);
    // col1: R^T * [1,0]^T = [c, -s]^T
    EXPECT_NEAR(grad(0, 1), c, 1e-12);
    EXPECT_NEAR(grad(1, 1), -s, 1e-12);
}

/**
 * 下面一组测例锁定 RoPE 的「特征配对约定」。
 *
 * 背景：θ_i = m / 10000^(2i/d) 在两种约定下完全一致，区别只在于第 i 个角作用在哪两个特征上。
 * jasmine 原生实现（interleaved）作用在 (2i, 2i+1)；HuggingFace 的 LLaMA
 * （LlamaRotaryEmbedding + rotate_half）作用在 (i, i + d/2)。
 * 当初接 TinyLlama 时正是这里写错，导致除位置 0 以外所有位置的 logits 都对不上。
 */
TEST(RoPE, HalfSplitPairsOffsetFeatures)
{
    // half_split：第 i 个角精确作用在 (i, i + d/2) 上，θ_i = m / 10000^(2i/d)
    const int d = 4, T = 3;
    mat_t<double> input(d, T, {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    });

    RoPE_net_t<mat_t<double>> rope_net(d);
    rope_net.set_pair_layout(rope_pair_layout::half_split);
    EXPECT_EQ(rope_net.pair_layout(), rope_pair_layout::half_split);
    auto out = rope_net.forward(input);

    const int half = d / 2;
    for (int j = 0; j < T; ++j)
    {
        for (int i = 0; i < half; ++i)
        {
            const double angle = static_cast<double>(j) / std::pow(10000.0, (2.0 * i) / d);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            // 独立公式，不引用实现：row i 与 row i+half 构成一对
            EXPECT_NEAR(out(i, j), c * input(i, j) - s * input(i + half, j), 1e-12);
            EXPECT_NEAR(out(i + half, j), s * input(i, j) + c * input(i + half, j), 1e-12);
        }
    }
}

TEST(RoPE, LayoutsAgreeAtPositionZeroOnly)
{
    // 位置 m=0 时旋转矩阵是单位阵，两种约定输出逐元素相同。
    // 这正是配对约定写错却「看起来对」的原因：只比对单 token / 位置 0 永远发现不了，
    // 必须用多 token 序列才能暴露。
    const int d = 4, T = 3;
    mat_t<double> input(d, T, {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    });

    RoPE_net_t<mat_t<double>> inter(d);
    RoPE_net_t<mat_t<double>> half(d);
    half.set_pair_layout(rope_pair_layout::half_split);
    EXPECT_EQ(inter.pair_layout(), rope_pair_layout::interleaved);

    auto out_inter = inter.forward(input);
    auto out_half = half.forward(input);

    for (int r = 0; r < d; ++r)
    {
        EXPECT_NEAR(out_inter(r, 0), input(r, 0), 1e-12);
        EXPECT_NEAR(out_half(r, 0), input(r, 0), 1e-12);
    }

    // 位置 >= 1 必须出现差异，否则说明配对约定没有生效
    double diff = 0.0;
    for (int r = 0; r < d; ++r)
        for (int j = 1; j < T; ++j)
            diff = std::max(diff, std::abs(out_inter(r, j) - out_half(r, j)));
    EXPECT_GT(diff, 1e-3);
}

TEST(RoPE, BackwardIsAdjointForBothLayouts)
{
    // backward 必须正好是 forward 的转置（伴随）：<forward(x), δ> == <x, backward(δ)>
    // 与实现细节无关，且对两种配对约定都要成立（尤其 half_split 的搬运方向不能反）。
    const int d = 4, T = 3;
    std::mt19937 gen(20240916);
    std::normal_distribution<double> nd(0.0, 1.0);
    mat_t<double> x(d, T, 0.0), delta(d, T, 0.0);
    for (int r = 0; r < d; ++r)
        for (int j = 0; j < T; ++j)
        {
            x(r, j) = nd(gen);
            delta(r, j) = nd(gen);
        }

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        RoPE_net_t<mat_t<double>> rope_net(d);
        rope_net.set_pair_layout(layout);
        auto y = rope_net.forward(x);
        auto g = rope_net.backward(delta);

        double lhs = 0.0, rhs = 0.0;
        for (int r = 0; r < d; ++r)
            for (int j = 0; j < T; ++j)
            {
                lhs += y(r, j) * delta(r, j);
                rhs += x(r, j) * g(r, j);
            }
        EXPECT_NEAR(lhs, rhs, 1e-12)
            << "layout=" << rope_pair_layout_name(layout);
    }
}

TEST(RoPE, RegistrySeparatesPairLayouts)
{
    // 同 d、不同配对约定必须是两份独立缓存，不能互相复用
    auto& reg = rope_registry_t<double>::instance();
    reg.clear();

    auto a = reg.get(4, 0, rope_pair_layout::interleaved);
    auto b = reg.get(4, 0, rope_pair_layout::half_split);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a.get(), b.get());
    EXPECT_EQ(a->pair_layout(), rope_pair_layout::interleaved);
    EXPECT_EQ(b->pair_layout(), rope_pair_layout::half_split);
    EXPECT_EQ(reg.size(), 2u);
    EXPECT_TRUE(reg.contains(4, rope_pair_layout::half_split));
    EXPECT_EQ(reg.get(4, 0, rope_pair_layout::half_split).get(), b.get());
    reg.clear();
}

TEST(RoPE, MhaPropagatesPairLayoutToHeads)
{
    auto& reg = rope_registry_t<double>::instance();
    reg.clear();

    mat_mha_t<mat_t<double>, nadam_t> mha(2, 4, false, 2); // d_head = 2
    EXPECT_EQ(mha.pair_layout(), rope_pair_layout::interleaved);
    EXPECT_EQ(reg.size(), 1u); // 构造时已注册 interleaved

    mha.set_rope_pair_layout(rope_pair_layout::half_split);
    EXPECT_EQ(mha.pair_layout(), rope_pair_layout::half_split);
    EXPECT_TRUE(reg.contains(2, rope_pair_layout::half_split));
    EXPECT_EQ(reg.size(), 2u);

    // 头拿到的必须是 half_split 那一份
    auto rope = reg.get(2, 0, rope_pair_layout::half_split);
    for (int i = 0; i < mha.num_heads(); ++i)
        ASSERT_EQ(mha.head(i).rope().get(), rope.get());

    // 重复设置同一约定不应新增缓存条目
    mha.set_rope_pair_layout(rope_pair_layout::half_split);
    EXPECT_EQ(reg.size(), 2u);
    reg.clear();
}
