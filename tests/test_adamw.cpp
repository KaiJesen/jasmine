/**
 * Unit tests for adamw_t: decoupled weight decay.
 *
 * Three key properties:
 *   1. with wd = 0 it must match adam_t bit for bit (proving that the only difference is the decay);
 *   2. with a zero gradient the decay still applies: theta <- theta * (1 - lr*wd) -- exactly what
 *      "decoupled" means;
 *   3. the decay does not pass through the moment estimate: theta1 = theta0 - lr*(g_hat + wd*theta0),
 *      where g_hat is Adam's normalised gradient step.
 */

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "jas_mat_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{
using dmat = mat_t<double>;

dmat grad_at(int step, int rows, int cols)
{
    dmat g(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            g(i, j) = std::sin(step * 0.7 + i * 1.3 + j * 0.5) * 0.5;
    return g;
}
} // namespace

TEST(AdamW, MatchesAdamWhenWeightDecayIsZero)
{
    dmat w_adam(3, 2, {1.0, -2.0, 3.0, -4.0, 5.0, -6.0});
    dmat w_adamw = w_adam;
    adam_t<double> adam(0.01);
    adamw_t<double> adamw(0.01, 0.9, 0.999, 1e-8, /*weight_decay=*/0.0);

    for (int step = 0; step < 5; ++step)
    {
        const dmat g = grad_at(step, 3, 2);
        adam.update(g, w_adam);
        adamw.update(g, w_adamw);
    }
    ExpectNearMat(w_adamw, w_adam, 0.0);          // bit for bit
}

TEST(AdamW, DecayAppliesWithZeroGradient)
{
    dmat w(1, 3, {1.0, 2.0, 3.0});
    adamw_t<double> opt(0.1, 0.9, 0.999, 1e-8, /*weight_decay=*/0.5);
    dmat zero(1, 3);
    zero = 0.0;
    opt.update(zero, w);
    // the moments are zero, so g_hat = 0 and only the decoupled decay remains: theta*(1-lr*wd)
    EXPECT_NEAR(w(0, 0), 0.95, 1e-12);
    EXPECT_NEAR(w(0, 1), 1.90, 1e-12);
    EXPECT_NEAR(w(0, 2), 2.85, 1e-12);
}

TEST(AdamW, DecayIsDecoupledFromTheMomentEstimate)
{
    dmat w(1, 1, {2.0});
    const double lr = 0.1, wd = 0.25;
    adamw_t<double> opt(lr, 0.9, 0.999, 1e-8, wd);
    dmat g(1, 1, {0.5});                          // after one step: m_hat = g, v_hat = g^2, g_hat = g/(|g|+eps) ~ 1
    opt.update(g, w);

    const double g_hat = 0.5 / (0.5 + 1e-8);
    const double expected = 2.0 - lr * (g_hat + wd * 2.0);   // the decay acts directly on theta
    EXPECT_NEAR(w(0, 0), expected, 1e-9) << "expected=" << expected;

    // for contrast: the L2 form of Adam adds wd*theta to the gradient, so |g| normalises the decay
    // and its strength is completely different
    dmat w_l2(1, 1, {2.0});
    adam_t<double> adam(lr);
    dmat g_l2(1, 1, {0.5 + wd * 2.0});
    adam.update(g_l2, w_l2);
    EXPECT_NE(w_l2(0, 0), w(0, 0));
}

TEST(AdamW, SetAndAccessors)
{
    adamw_t<double> opt(0.01, 0.9, 0.999, 1e-8, 0.02);
    EXPECT_NEAR(opt.weight_decay(), 0.02, 1e-15);
    opt.set_weight_decay(0.5);
    EXPECT_NEAR(opt.weight_decay(), 0.5, 1e-15);
    opt.set(0.001, 0.9, 0.999, 1e-8, 0.03);
    EXPECT_NEAR(opt.weight_decay(), 0.03, 1e-15);
    opt.set_lr(0.002);                            // not throwing is all we check
    opt.step();
    SUCCEED();
}
