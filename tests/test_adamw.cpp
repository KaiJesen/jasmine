/**
 * adamw_t 单测：解耦权重衰减（decoupled weight decay）。
 *
 * 三个关键性质：
 *   1. wd = 0 时必须与 adam_t 逐位一致（说明差别只在衰减项）；
 *   2. 梯度为 0 时，衰减仍然生效：θ ← θ·(1 - lr·wd)（这正是「解耦」的语义）；
 *   3. 衰减强度与矩估计无关：θ₁ = θ₀ - lr·(ĝ + wd·θ₀)，ĝ 是 Adam 的归一化梯度步。
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
    ExpectNearMat(w_adamw, w_adam, 0.0);          // 逐位一致
}

TEST(AdamW, DecayAppliesWithZeroGradient)
{
    dmat w(1, 3, {1.0, 2.0, 3.0});
    adamw_t<double> opt(0.1, 0.9, 0.999, 1e-8, /*weight_decay=*/0.5);
    dmat zero(1, 3);
    zero = 0.0;
    opt.update(zero, w);
    // 矩估计为 0 → ĝ = 0；只剩解耦衰减 θ·(1 - lr·wd) = θ·0.95
    EXPECT_NEAR(w(0, 0), 0.95, 1e-12);
    EXPECT_NEAR(w(0, 1), 1.90, 1e-12);
    EXPECT_NEAR(w(0, 2), 2.85, 1e-12);
}

TEST(AdamW, DecayIsDecoupledFromTheMomentEstimate)
{
    dmat w(1, 1, {2.0});
    const double lr = 0.1, wd = 0.25;
    adamw_t<double> opt(lr, 0.9, 0.999, 1e-8, wd);
    dmat g(1, 1, {0.5});                          // 一步之后：m̂ = g, v̂ = g², ĝ = g/(|g|+ε) ≈ 1
    opt.update(g, w);

    const double g_hat = 0.5 / (0.5 + 1e-8);
    const double expected = 2.0 - lr * (g_hat + wd * 2.0);   // 衰减直接作用在 θ 上
    EXPECT_NEAR(w(0, 0), expected, 1e-9) << "expected=" << expected;

    // 对照：Adam 的 L2 版本会把 wd·θ 加进梯度，衰减被 |g| 归一化后强度完全不同
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
    opt.set_lr(0.002);                            // 不抛异常即可
    opt.step();
    SUCCEED();
}
