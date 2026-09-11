#include <cmath>
#include <gtest/gtest.h>
#include "mat_mha_t.hpp"
#include "mat_init_t.hpp"
#include "mat_updator_t.hpp"
#include "test_helpers.hpp"

static double mse(const mat_t<double>& a, const mat_t<double>& b)
{
    double s = 0.0;
    int n = a.row_num() * a.col_num();
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < a.col_num(); ++j)
        {
            double d = a(i, j) - b(i, j);
            s += d * d;
        }
    return s / n;
}

TEST(Mha, HeadGenCanFitSimpleTarget)
{
    mat_head_gen_t<mat_view_t<mat_t<double>>, adam_t> head(4);
    head.init_weight<xavier_gaussian_t>();
    head.set_updator(0.01);

    mat_t<double> input(4, 2, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
    mat_t<double> expected(4, 2, {0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});

    double loss0 = mse(head.forward(input.view()), expected);
    for (int i = 0; i < 2000; ++i)
    {
        auto output = head.forward(input.view());
        head.backward((output - expected).clone().view());
        head.step();
    }
    double loss1 = mse(head.forward(input.view()), expected);
    EXPECT_LT(loss1, loss0);
}

TEST(Mha, MultiHeadForwardShapeAndTrain)
{
    constexpr int num_heads = 4;
    constexpr int d_model = 8;
    constexpr int seq_len = 2;

    mat_mha_t<mat_t<double>, nadam_t> mha(num_heads, d_model, false, seq_len);
    mha.init_weight<xavier_gaussian_t>();
    mha.set_updator(0.1);

    mat_t<double> input(d_model, seq_len, {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    });
    mat_t<double> label(d_model, seq_len, {
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    });

    auto out0 = mha.forward(input);
    ExpectShape(out0, d_model, seq_len);
    double loss0 = mse(out0, label);

    for (int i = 0; i < 300; ++i)
    {
        auto output = mha.forward(input);
        mha.backward((output - label).clone().view());
        mha.step();
    }
    EXPECT_LT(mse(mha.forward(input), label), loss0);
}
