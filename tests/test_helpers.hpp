#ifndef JASMINE_TEST_HELPERS_HPP
#define JASMINE_TEST_HELPERS_HPP

#include <cmath>
#include <gtest/gtest.h>
#include "jas_mat_t.hpp"


using namespace jasmine;
template <typename MatA, typename MatB>
void ExpectNearMat(const MatA& a, const MatB& b, double tol = 1e-5)
{
    ASSERT_EQ(a.row_num(), b.row_num());
    ASSERT_EQ(a.col_num(), b.col_num());
    for (int i = 0; i < a.row_num(); ++i)
    {
        for (int j = 0; j < a.col_num(); ++j)
        {
            EXPECT_NEAR(static_cast<double>(a(i, j)),
                        static_cast<double>(b(i, j)),
                        tol)
                << "mismatch at (" << i << ", " << j << ")";
        }
    }
}

template <typename Mat>
void ExpectShape(const Mat& m, int rows, int cols)
{
    EXPECT_EQ(m.row_num(), rows);
    EXPECT_EQ(m.col_num(), cols);
}

#endif
