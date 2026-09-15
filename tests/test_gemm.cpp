#include <cmath>
#include <gtest/gtest.h>
#include "mat_t.hpp"
#include "mat_express_t.hpp"
#include "mat_view_t.hpp"

using namespace jasmine;

namespace {

mat_t<double> naive_dot(mat_t<double> const& a, mat_t<double> const& b)
{
    mat_t<double> c(a.row_num(), b.col_num());
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < b.col_num(); ++j)
        {
            double s = 0.0;
            for (int k = 0; k < a.col_num(); ++k)
                s += a(i, k) * b(k, j);
            c(i, j) = s;
        }
    return c;
}

} // namespace

TEST(MatGemm, SquareMatchesNaive)
{
    constexpr int n = 48; // above gemm_fast_threshold (16^3)
    mat_t<double> a(n, n);
    mat_t<double> b(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            a(i, j) = 0.01 * i + 0.02 * j;
            b(i, j) = 0.03 * i - 0.01 * j;
        }

    mat_t<double> got = a.dot(b).clone();
    mat_t<double> exp = naive_dot(a, b);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(got(i, j), exp(i, j), 1e-9) << "i=" << i << " j=" << j;
}

TEST(MatGemm, RectangularAndFloat)
{
    mat_t<float> a(32, 64);
    mat_t<float> b(64, 16);
    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 64; ++j)
            a(i, j) = static_cast<float>(i - j) * 0.1f;
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 16; ++j)
            b(i, j) = static_cast<float>(i + 2 * j) * 0.05f;

    mat_t<float> got = a.dot(b).clone();
    ASSERT_EQ(got.row_num(), 32);
    ASSERT_EQ(got.col_num(), 16);

    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 16; ++j)
        {
            float s = 0.f;
            for (int k = 0; k < 64; ++k)
                s += a(i, k) * b(k, j);
            EXPECT_NEAR(got(i, j), s, 1e-4f);
        }
}

TEST(MatGemm, TransposeOperandMatchesNaive)
{
    mat_t<double> a(40, 24);
    mat_t<double> b(40, 18);
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 24; ++j)
            a(i, j) = 0.1 * i + j;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 18; ++j)
            b(i, j) = 0.2 * i - j;

    // a.t() is 24x40, b is 40x18 → 24x18
    mat_t<double> got = a.t().dot(b).clone();
    mat_t<double> at = a.t().clone();
    mat_t<double> exp = naive_dot(at, b);
    for (int i = 0; i < got.row_num(); ++i)
        for (int j = 0; j < got.col_num(); ++j)
            EXPECT_NEAR(got(i, j), exp(i, j), 1e-9);
}

TEST(MatGemm, SmallFallsBackStillCorrect)
{
    mat_t<double> a(3, 2, {1, 2, 3, 4, 5, 6});
    mat_t<double> b(2, 3, {1, 0, 2, 0, 1, 3});
    mat_t<double> got = a.dot(b).clone();
    EXPECT_NEAR(got(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(got(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(got(0, 2), 8.0, 1e-12);
    EXPECT_NEAR(got(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(got(2, 0), 5.0, 1e-12);
}
