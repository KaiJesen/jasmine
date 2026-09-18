/**
 * mean_pool_net_t 单测：把 [d_model, T] 的 token 序列压成 [d_model, 1]。
 *
 * 它是「Transformer encoder → 分类头」之间的那一层：forward 对 token 取平均，
 * backward 把梯度平均分摊回每个 token（与 mean 的定义一致），并且必须是无参数静态层。
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "jas_mat_concepts.hpp"
#include "jas_mat_t.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{
using dmat = mat_t<double>;
using mean_t = mean_pool_net_t<dmat>;
} // namespace

TEST(MeanPool, ForwardAveragesEachRow)
{
    mean_t mp;
    const dmat x(3, 4, {1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12});
    const dmat y = mp.forward(x);
    ExpectShape(y, 3, 1);
    EXPECT_DOUBLE_EQ(y(0, 0), 2.5);
    EXPECT_DOUBLE_EQ(y(1, 0), 6.5);
    EXPECT_DOUBLE_EQ(y(2, 0), 10.5);
}

TEST(MeanPool, BackwardSpreadsGradientEvenly)
{
    mean_t mp;
    const dmat x(2, 4, {1, 2, 3, 4, 5, 6, 7, 8});
    mp.forward(x);
    const dmat delta(2, 1, {8, 12});
    const dmat dx = mp.backward(delta);
    ExpectShape(dx, 2, 4);
    for (int t = 0; t < 4; ++t)
    {
        EXPECT_DOUBLE_EQ(dx(0, t), 8.0 / 4.0);    // 每个 token 分摊 1/T
        EXPECT_DOUBLE_EQ(dx(1, t), 12.0 / 4.0);
    }

    // 形状不符抛错
    EXPECT_THROW(mp.backward(dmat(2, 2)), std::runtime_error);
    EXPECT_THROW(mp.backward(dmat(3, 1)), std::runtime_error);
}

TEST(MeanPool, IsStaticLayerWithoutReinit)
{
    static_assert(!is_updatable_net<mean_t>);
    static_assert(!is_reinitable_net<mean_t>);
    SUCCEED();
}

TEST(MeanPool, ChainWithLinearHead)
{
    // mean pool → linear 的小链：反向要能穿过池化回到序列
    using chain_t = complex_net_builder_t<double>
        ::push_back_staticnet<mean_pool_net_t>
        ::push_back_updatable<weight_net_t, sgd_t>
        ::type;
    chain_t net;
    net.reinit(std::vector<int>{4, 3});          // fc: 4 → 3（容器只作用于 weight_net）
    net.get<1>().set_updator(1.0);
    net.get<1>().weight() = 0.5;
    net.get<1>().bias() = 0.0;

    dmat x(4, 5);
    for (int i = 0; i < 4; ++i)
        for (int t = 0; t < 5; ++t) x(i, t) = static_cast<double>(i * 5 + t + 1);

    const dmat pooled = net.get<0>().forward(x);
    ExpectShape(pooled, 4, 1);
    const dmat y = net.forward(x);
    ExpectShape(y, 3, 1);

    const dmat delta(3, 1, {1, 2, 3});
    const dmat dx = net.backward(delta);
    ExpectShape(dx, 4, 5);                       // 梯度回到了原始序列形状
}

TEST(MeanPool, NetTypeMentionsTokenCount)
{
    mean_t mp;
    mp.forward(dmat(4, 7));
    const std::string s = mp.net_type();
    EXPECT_NE(s.find("mean_pool_net_t"), std::string::npos);
    EXPECT_NE(s.find("tokens:7"), std::string::npos);
}
