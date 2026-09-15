#include <cmath>
#include <gtest/gtest.h>
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"


using namespace jasmine;
TEST(LayerNorm, PerColumnZeroMeanUnitVariance)
{
    layer_norm_net_t<mat_t<double>, nadam_t> ln;
    mat_t<double> input(4, 2, {
        1.0, 10.0,
        2.0, 20.0,
        3.0, 30.0,
        4.0, 40.0
    });

    auto out = ln.forward(input);
    ExpectShape(out, 4, 2);

    for (int j = 0; j < 2; ++j)
    {
        double mean = 0.0;
        for (int i = 0; i < 4; ++i)
            mean += out(i, j);
        mean /= 4.0;
        EXPECT_NEAR(mean, 0.0, 1e-6);

        double var = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            double d = out(i, j) - mean;
            var += d * d;
        }
        var /= 4.0;
        EXPECT_NEAR(var, 1.0, 1e-5);
    }
}

TEST(LayerNorm, BackwardFiniteAndSameShape)
{
    layer_norm_net_t<mat_t<double>, nadam_t> ln;
    ln.set_updator(0.01);
    mat_t<double> input(3, 2, {
        0.2, 1.5,
        -0.4, 0.3,
        0.8, -1.2
    });
    auto out = ln.forward(input);
    mat_t<double> grad = out;
    grad = 1.0;
    auto din = ln.backward(grad);
    ExpectShape(din, 3, 2);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_TRUE(std::isfinite(din(i, j)));
}
