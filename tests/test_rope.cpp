#include <cmath>
#include <gtest/gtest.h>
#include "mat_RoPE_t.hpp"
#include "mat_mha_t.hpp"
#include "mat_updator_t.hpp"
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
