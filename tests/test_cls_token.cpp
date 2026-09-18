/**
 * Unit tests for cls_token_net_t (prepends a learnable CLS vector) and take_token_net_t (takes one
 * column).
 *
 * Together they form a ViT-style classification head: prepend CLS before the encoder and take the
 * CLS column after it.
 */

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_mat_concepts.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{
using dmat = mat_t<double>;
using cls_t = cls_token_net_t<dmat, sgd_t>;
using take_t = take_token_net_t<dmat>;
} // namespace

TEST(ClsToken, ForwardPrependsTheLearnedVector)
{
    cls_t cls;
    cls.set_param(3);
    cls.token() = dmat(3, 1, {9, 8, 7});          // write the CLS vector directly so the check is exact
    const dmat x(3, 4, {1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12});
    const dmat y = cls.forward(x);
    ExpectShape(y, 3, 5);                          // 4 tokens + 1 CLS
    EXPECT_DOUBLE_EQ(y(0, 0), 9.0);
    EXPECT_DOUBLE_EQ(y(2, 0), 7.0);
    for (int i = 0; i < 3; ++i)
        for (int t = 0; t < 4; ++t)
            EXPECT_DOUBLE_EQ(y(i, t + 1), x(i, t)) << "tokens must be shifted by one, unchanged";
}

TEST(ClsToken, BackwardFeedsTheTokenUpdatorAndPassesTokensThrough)
{
    cls_t cls;
    cls.set_param(2);
    cls.set_updator(1.0);                          // sgd lr=1 → token_after == token_before - grad
    cls.token() = dmat(2, 1, {0.5, -0.5});
    const dmat x(2, 3);
    cls.forward(x);

    const dmat delta(2, 4, {1, 2, 3, 4,
                            5, 6, 7, 8});
    const dmat dx = cls.backward(delta);

    // the CLS gradient is column 0 of delta
    EXPECT_NEAR(cls.token()(0, 0), 0.5 - 1.0, 1e-12);
    EXPECT_NEAR(cls.token()(1, 0), -0.5 - 5.0, 1e-12);
    // the returned gradient is delta without column 0
    ExpectShape(dx, 2, 3);
    for (int i = 0; i < 2; ++i)
        for (int t = 0; t < 3; ++t)
            EXPECT_DOUBLE_EQ(dx(i, t), delta(i, t + 1));

    EXPECT_THROW(cls.backward(dmat(2, 3)), std::runtime_error);
}

TEST(ClsToken, IsUpdatableButNotReinitDetected)
{
    static_assert(is_updatable_net<cls_t>);
    static_assert(!is_reinitable_net<cls_t>);      // set_param instead of reinit -> no container slot
    SUCCEED();
}

TEST(TakeToken, PicksTheRequestedColumn)
{
    const dmat x(2, 4, {1, 2, 3, 4,
                        5, 6, 7, 8});
    take_t first;
    first.set_param(0);
    const dmat y0 = first.forward(x);
    ExpectShape(y0, 2, 1);
    EXPECT_DOUBLE_EQ(y0(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(y0(1, 0), 5.0);

    take_t third;
    third.set_param(2);
    const dmat y2 = third.forward(x);
    EXPECT_DOUBLE_EQ(y2(0, 0), 3.0);

    EXPECT_THROW(third.forward(dmat(2, 2)), std::out_of_range);   // index 2 >= 2 columns
}

TEST(TakeToken, BackwardScattersIntoOneColumn)
{
    take_t tk;
    tk.set_param(1);
    tk.forward(dmat(3, 4));
    const dmat delta(3, 1, {1, 2, 3});
    const dmat dx = tk.backward(delta);
    ExpectShape(dx, 3, 4);
    for (int i = 0; i < 3; ++i)
        for (int t = 0; t < 4; ++t)
        {
            if (t == 1) EXPECT_DOUBLE_EQ(dx(i, t), delta(i, 0));
            else EXPECT_DOUBLE_EQ(dx(i, t), 0.0);
        }
    EXPECT_THROW(tk.backward(dmat(3, 2)), std::runtime_error);
}

TEST(TakeToken, IsStaticLayerWithoutReinit)
{
    static_assert(!is_updatable_net<take_t>);
    static_assert(!is_reinitable_net<take_t>);
    SUCCEED();
}

TEST(ClsToken, NetTypes)
{
    cls_t cls;
    cls.set_param(8);
    EXPECT_NE(cls.net_type().find("cls_token_net_t"), std::string::npos);
    EXPECT_NE(cls.net_type().find("d_model:8"), std::string::npos);
    take_t tk;
    tk.set_param(0);
    EXPECT_NE(tk.net_type().find("take_token_net_t"), std::string::npos);
}
