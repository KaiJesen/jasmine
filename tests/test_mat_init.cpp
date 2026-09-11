#include <cmath>
#include <gtest/gtest.h>
#include "mat_t.hpp"
#include "mat_init_t.hpp"

TEST(MatInit, StrategiesProduceFiniteNonZero)
{
    mat_t<double> m(3, 4);

    init_matrix<xavier_gaussian_t>(m);
    bool any_nonzero = false;
    for (int i = 0; i < m.row_num(); ++i)
        for (int j = 0; j < m.col_num(); ++j)
        {
            EXPECT_TRUE(std::isfinite(m(i, j)));
            any_nonzero = any_nonzero || (m(i, j) != 0.0);
        }
    EXPECT_TRUE(any_nonzero);

    init_matrix<xavier_uniform_t>(m);
    init_matrix<he_gaussian_t>(m);
    init_matrix<he_uniform_t>(m);
    for (int i = 0; i < m.row_num(); ++i)
        for (int j = 0; j < m.col_num(); ++j)
            EXPECT_TRUE(std::isfinite(m(i, j)));
}
