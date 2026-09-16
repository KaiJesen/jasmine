/**
 * silu_net_t（SiLU / Swish）单测。
 *
 * 基准值取自 PyTorch：
 *     import torch; y = torch.nn.functional.silu(x); y.sum().backward()
 * 与 tools/ 下的 GPT-2 对齐不同，SiLU 不依赖任何权重文件，所以这些测试在 CI 里始终会跑。
 */

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_silu_t.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

using silu_t = silu_net_t<mat_t<double>>;

// cache_updator_t 有两个模板参数，要先绑定 nadam_t 才能作为 updator_tpl 传入
// complex_net_builder（与 jas_gpt2_t.hpp 里的 gpt2_upr_tpl 同理）。
template<typename val_type>
using silu_upr_tpl = cache_updator_t<val_type, nadam_t>;

} // namespace

TEST(SiLU, MatchesDefinition)
{
    // silu(x) == x * sigmoid(x) == x / (1 + e^-x)
    for (double x : {-8.0, -3.0, -1.278, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0})
    {
        const double expect = x / (1.0 + std::exp(-x));
        EXPECT_NEAR(silu_t::silu(x), expect, 1e-12) << "x=" << x;
    }
}

TEST(SiLU, KnownValues)
{
    // 与 torch.nn.functional.silu(float64) 一致
    EXPECT_NEAR(silu_t::silu(0.0), 0.0, 1e-12);
    EXPECT_NEAR(silu_t::silu(1.0), 0.7310585786300049, 1e-12);
    EXPECT_NEAR(silu_t::silu(-1.0), -0.2689414213699951, 1e-12);
    EXPECT_NEAR(silu_t::silu(0.5), 0.3112296656009273, 1e-12);
    EXPECT_NEAR(silu_t::silu(-0.5), -0.1887703343990727, 1e-12);
    EXPECT_NEAR(silu_t::silu(-3.0), -0.14227761953270035, 1e-12);
    EXPECT_NEAR(silu_t::silu(2.0), 1.7615941559557646, 1e-12);
    // 大正数应趋近恒等。注意相对偏差本来就存在（silu(x)/x = 1 - e^-x），x=30 时真值偏差
    // 已是 e^-30 ≈ 9.36e-14，不是数值误差，所以容差取 1e-12 而不是机器精度级别。
    EXPECT_NEAR(silu_t::silu(8.0), 7.997317198956269, 1e-12);
    EXPECT_LT(std::abs(silu_t::silu(30.0) / 30.0 - 1.0), 1e-12);
}

TEST(SiLU, KnownGradients)
{
    // d/dx silu(x) = sigmoid(x)[1 + x(1 - sigmoid(x))]，基准同样来自 torch
    EXPECT_NEAR(silu_t::silu_grad(0.0), 0.5, 1e-12);
    EXPECT_NEAR(silu_t::silu_grad(1.0), 0.9276705118714869, 1e-12);
    EXPECT_NEAR(silu_t::silu_grad(-1.0), 0.07232948812851325, 1e-12);
    EXPECT_NEAR(silu_t::silu_grad(0.5), 0.7399611873026519, 1e-12);
    EXPECT_NEAR(silu_t::silu_grad(-0.5), 0.2600388126973482, 1e-12);
    EXPECT_NEAR(silu_t::silu_grad(-3.0), -0.08810410601516962, 1e-12);
    EXPECT_NEAR(silu_t::silu_grad(2.0), 1.0907842487848955, 1e-12);
    // 大正数梯度趋近 1
    EXPECT_NEAR(silu_t::silu_grad(8.0), 1.0023465512355845, 1e-12);
}

TEST(SiLU, IsNonMonotonicWithKnownMinimum)
{
    // SiLU 的招牌性质：负半轴非单调，在 x≈-1.27846 处取最小值 ≈-0.27846
    // （torch.linspace(-1.5,-1.0,200001).argmin() -> -1.278465）
    double best_x = 0.0, best_v = 1e9;
    for (int i = 0; i <= 200000; ++i)
    {
        const double x = -1.5 + 1.5 * i / 200000.0;
        const double v = silu_t::silu(x);
        if (v < best_v) { best_v = v; best_x = x; }
    }
    EXPECT_NEAR(best_x, -1.278465, 1e-4);
    EXPECT_NEAR(best_v, -0.27846451925429261, 1e-6);

    // 最小值处导数应≈0（符号改变）
    EXPECT_NEAR(silu_t::silu_grad(best_x), 0.0, 1e-4);
    // 两侧各取一点，确认非单调：最小值的邻居都**大于**最小值本身。
    // 注意这些数是负数，"更大"意味着更接近 0，所以断言方向是 GT 而非 LT。
    EXPECT_GT(silu_t::silu(-1.4), silu_t::silu(-1.278465));
    EXPECT_GT(silu_t::silu(-1.2), silu_t::silu(-1.278465));
    // 完整数值排序：min < silu(-1.2) < silu(-1.4) < silu(-1.1)
    // -1.4 与 -1.2 分列最小值两侧，但**左侧更高** —— 最小值附近是不对称的。
    EXPECT_LT(silu_t::silu(-1.278465), silu_t::silu(-1.2));
    EXPECT_LT(silu_t::silu(-1.2), silu_t::silu(-1.4));
    // 越过最小值后随 x 增大单调回升，趋向 0
    EXPECT_LT(silu_t::silu(-1.4), silu_t::silu(-1.1));
}

TEST(SiLU, ForwardShapeAndValues)
{
    silu_t act;
    mat_t<double> in(3, 2, {-1.0, 0.5, 2.0, -0.25, 0.0, 3.0});
    auto out = act.forward(in);
    ExpectShape(out, 3, 2);
    // 矩阵路径与标量路径必须一致
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(out(i, j), silu_t::silu(in(i, j)), 1e-15)
                << "at (" << i << "," << j << ")";
}

TEST(SiLU, ForwardOneMatchesForward)
{
    // 无状态层：单列输入与整段 forward 结果相同（KV cache 解码路径会走 forward_one）
    silu_t act;
    mat_t<double> in(2, 4, {-1.0, 0.5, 2.0, -0.25, 0.0, 3.0, -2.0, 1.0});
    auto full = act.forward(in);
    for (int j = 0; j < in.col_num(); ++j)
    {
        auto one = act.forward_one(in.view(0, j, in.row_num(), 1));
        ExpectShape(one, in.row_num(), 1);
        for (int i = 0; i < in.row_num(); ++i)
            EXPECT_NEAR(one(i, 0), full(i, j), 1e-15);
    }
}

TEST(SiLU, BackwardMatchesNumericalGradient)
{
    const double h = 1e-6;
    for (double x : {-3.0, -1.278, -1.0, -0.5, 0.0, 0.7, 2.0, 5.0})
    {
        silu_t act;
        mat_t<double> in(1, 1, {x});
        act.forward(in);
        const double analytic = act.backward(mat_t<double>(1, 1, {1.0}))(0, 0);

        const double num = (silu_t::silu(x + h) - silu_t::silu(x - h)) / (2 * h);
        EXPECT_NEAR(analytic, num, 1e-6) << "x=" << x;
    }
}

TEST(SiLU, BackwardScalesWithDelta)
{
    // 反向是逐元素乘，对 delta 必须线性
    silu_t act;
    mat_t<double> in(1, 3, {-1.0, 0.0, 2.0});
    act.forward(in);
    mat_t<double> d1(1, 3, {1.0, 1.0, 1.0});
    mat_t<double> d2(1, 3, {2.5, -3.0, 0.5});
    auto g1 = act.backward(d1);
    auto g2 = act.backward(d2);
    for (int j = 0; j < 3; ++j)
        EXPECT_NEAR(g2(0, j), g1(0, j) * d2(0, j), 1e-12) << "j=" << j;
}

TEST(SiLU, BackwardRejectsWrongShape)
{
    silu_t act;
    act.forward(mat_t<double>(2, 3, {1, 2, 3, 4, 5, 6}));
    EXPECT_THROW(act.backward(mat_t<double>(1, 3, {1, 1, 1})), std::runtime_error);
    EXPECT_THROW(act.backward(mat_t<double>(2, 2, {1, 1, 1, 1})), std::runtime_error);
}

TEST(SiLU, SaturatesWithoutNaN)
{
    // 两端饱和时 exp 会溢出成 inf，但结果必须干净（0 / 1），不能是 NaN
    silu_t act;
    mat_t<double> in(1, 4, {-1e4, -1e3, 1e3, 1e4});
    auto out = act.forward(in);
    for (int j = 0; j < out.col_num(); ++j)
        EXPECT_FALSE(std::isnan(out(0, j))) << "j=" << j;

    EXPECT_DOUBLE_EQ(out(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(out(0, 1), 0.0);
    EXPECT_NEAR(out(0, 2), 1e3, 1e-6);
    EXPECT_NEAR(out(0, 3), 1e4, 1e-6);

    // 反向同样不能出 NaN
    auto grad = act.backward(mat_t<double>(1, 4, {1.0, 1.0, 1.0, 1.0}));
    for (int j = 0; j < grad.col_num(); ++j)
        EXPECT_FALSE(std::isnan(grad(0, j))) << "j=" << j;
    // 极负端梯度≈0，极正端梯度≈1
    EXPECT_NEAR(grad(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(grad(0, 3), 1.0, 1e-12);
}

TEST(SiLU, FloatInstantiationSaturatesCleanly)
{
    // float 的 exp 更早溢出（|x|>88），确认单精度路径也不会产生 NaN
    silu_net_t<mat_t<float>> act;
    mat_t<float> in(1, 4, {-100.0f, -50.0f, 50.0f, 100.0f});
    auto out = act.forward(in);
    for (int j = 0; j < out.col_num(); ++j)
        EXPECT_FALSE(std::isnan(out(0, j))) << "j=" << j;
    EXPECT_FLOAT_EQ(out(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(out(0, 3), 100.0f);
}

TEST(SiLU, NetTypeMentionsSilu)
{
    silu_t act;
    EXPECT_NE(act.net_type().find("silu"), std::string::npos);
    // 带缩进时缩进应生效
    EXPECT_NE(act.net_type(4).find("silu"), std::string::npos);
    EXPECT_EQ(act.net_type(4).substr(0, 4), "    ");
}

TEST(SiLU, UsableAsStaticLayerInComplexNet)
{
    // SwiGLU 会把 SiLU 接到 Linear 后面（Linear -> SiLU -> ...），这里验证它满足
    // complex_net 静态层的接口要求：push_back_staticnet 能被实例化、forward 能跑通。
    using chain_t = complex_net_builder_t<double>
        ::template push_back_updatable<weight_net_t, silu_upr_tpl>
        ::template push_back_staticnet<silu_net_t>
        ::type;

    chain_t net;
    net.reinit(std::vector<int>{3, 3});
    net.init_weight<xavier_gaussian_t>();

    mat_t<double> in(3, 2, {-1.0, 0.5, 2.0, -0.25, 0.0, 3.0});
    mat_t<double> out = net.forward(in);
    ExpectShape(out, 3, 2);
    for (int i = 0; i < out.row_num(); ++i)
        for (int j = 0; j < out.col_num(); ++j)
            EXPECT_FALSE(std::isnan(out(i, j)));

    // 反向通路也要能跑（SiLU 的 backward 会被链式调用）
    mat_t<double> delta(3, 2, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    EXPECT_NO_THROW(net.backward(delta));
    net.step();
}
