/**
 * flatten_net_t 单测：CNN 尾部「特征图 → 向量」那一层，也是 conv+relu+pool+flatten+encoder
 * 这条静态层堆叠链里唯一新增的层。
 *
 * 三件事：展平顺序（行优先，与 conv/pool 的布局一致）、反向还原形状、以及它必须被识别为
 * 无参数静态层（否则会占掉 complex_net_t::reinit 的容器槽位）。
 */

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_conv_t.hpp"
#include "jas_mat_concepts.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{
using dmat = mat_t<double>;
using flat_t = flatten_net_t<dmat>;
template <typename T> using upr_tpl = cache_updator_t<T, sgd_t>;
} // namespace

TEST(Flatten, ForwardIsRowMajorFlatten)
{
    flat_t flatten;
    flatten.set_param(2, 3);
    const dmat x(2, 3, {1, 2, 3,
                        4, 5, 6});
    const dmat y = flatten.forward(x);
    ExpectShape(y, 6, 1);
    for (int i = 0; i < 6; ++i)
        EXPECT_DOUBLE_EQ(y(i, 0), static_cast<double>(i + 1)) << "i=" << i;
}

TEST(Flatten, BackwardRestoresShape)
{
    flat_t flatten;
    flatten.set_param(2, 3);
    const dmat x(2, 3, {1, 2, 3, 4, 5, 6});
    flatten.forward(x);
    const dmat delta(6, 1, {10, 20, 30, 40, 50, 60});
    const dmat dx = flatten.backward(delta);
    ExpectShape(dx, 2, 3);
    EXPECT_DOUBLE_EQ(dx(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(dx(0, 2), 30.0);
    EXPECT_DOUBLE_EQ(dx(1, 0), 40.0);
    EXPECT_DOUBLE_EQ(dx(1, 2), 60.0);

    // 形状不符必须抛错
    EXPECT_THROW(flatten.backward(dmat(5, 1)), std::runtime_error);
    EXPECT_THROW(flatten.backward(dmat(6, 2)), std::runtime_error);
}

TEST(Flatten, LazyShapeFromFirstForwardAndRejectsMismatch)
{
    flat_t flatten;                       // 不调用 set_param
    const dmat x(3, 2, {1, 2, 3, 4, 5, 6});
    ExpectShape(flatten.forward(x), 6, 1);
    EXPECT_THROW(flatten.forward(dmat(2, 3)), std::invalid_argument);   // 形状变了要显式 set_param
}

TEST(Flatten, IsStaticLayerWithoutReinit)
{
    static_assert(!is_updatable_net<flat_t>);
    static_assert(!is_reinitable_net<flat_t>);
    SUCCEED();
}

TEST(Flatten, ChainWithConvAndLinear)
{
    // conv → flatten → fc 的静态层堆叠链：反向要能穿过 flatten 回到卷积
    using chain_t = complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, upr_tpl>
        ::push_back_staticnet<flatten_net_t>
        ::push_back_updatable<weight_net_t, upr_tpl>
        ::type;

    chain_t net;
    net.get<0>().set_param(1, 2, 3, 3, 3, 3, 1, 1, 1, 1);   // → [2, 9]
    net.get<1>().set_param(2, 9);
    net.reinit(std::vector<int>{18, 4});                     // fc: 18 → 4
    net.get<0>().set_updator(1.0);
    net.get<2>().set_updator(1.0);
    net.get<0>().weight() = 0.5;
    net.get<0>().bias() = 0.0;
    net.get<2>().weight() = 0.25;
    net.get<2>().bias() = 0.0;

    dmat x(1, 9);
    for (int j = 0; j < 9; ++j) x(0, j) = static_cast<double>(j + 1);
    const dmat y = net.forward(x);
    ExpectShape(y, 4, 1);

    // 手工前向核对：conv → 展平 → fc
    const dmat conv_out = net.get<0>().forward(x);   // 注意：这会再次缓存，仅用于数值核对
    dmat flat(18, 1);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 9; ++j)
            flat(i * 9 + j, 0) = conv_out(i, j);
    const dmat fc_out = net.get<2>().forward(flat);
    ExpectNearMat(y, fc_out, 1e-12);

    const dmat delta(4, 1, {1, 2, 3, 4});
    const dmat dx = net.backward(delta);
    ExpectShape(dx, 1, 9);                            // 梯度穿过了 flatten 回到卷积输入
}

TEST(Flatten, NetTypeReportsShape)
{
    flat_t flatten;
    flatten.set_param(16, 49);
    const std::string s = flatten.net_type();
    EXPECT_NE(s.find("flatten_net_t"), std::string::npos);
    EXPECT_NE(s.find("16x49"), std::string::npos);
    EXPECT_NE(s.find("784x1"), std::string::npos);
}
