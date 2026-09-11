#include <cmath>
#include <gtest/gtest.h>
#include "mat_RoPE_t.hpp"
#include "test_helpers.hpp"

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
