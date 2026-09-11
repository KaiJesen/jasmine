#include <cmath>
#include <gtest/gtest.h>
#include "mat_t.hpp"
#include "mat_view_t.hpp"
#include "mat_express_t.hpp"
#include "test_helpers.hpp"


using namespace jasmine;
TEST(MatBasic, ShapeAndTransposeView)
{
    mat_t<double> m(3, 3, {
        1.1, 1.2, 1.3,
        2.1, 2.2, 2.3,
        3.1, 3.2, 3.3
    });
    mat_view_t<mat_t<double>> mv(m, 1, 1, 2, 1);
    ExpectShape(mv, 2, 1);
    EXPECT_NEAR(mv(0, 0), 2.2, 1e-12);
    EXPECT_NEAR(mv(1, 0), 3.2, 1e-12);
}

TEST(MatBasic, ExpressionMaterializesToMat)
{
    mat_t<double> m1{3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }};
    mat_t<double> m2{3, 3, {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    }};

    mat_t<double> m3 = ((m1 + m2 - m1) * m2 / m1).dot(m2).dot(m2);
    ExpectShape(m3, 3, 3);
    // Smoke: finite values after chained expression materialization.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_TRUE(std::isfinite(m3(i, j)));
}
