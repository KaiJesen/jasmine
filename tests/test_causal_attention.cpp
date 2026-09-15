#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include "jas_mha_t.hpp"
#include "jas_transformer_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_mat_express_t.hpp"
#include "test_helpers.hpp"


using namespace jasmine;
using val_type = float;

TEST(CausalAttention, MaskedMhaFirstPositionStable)
{
    // With causal mask, length-1 and length-2 inputs should agree on position 0.
    mat_mha_t<mat_t<val_type>, nadam_t> mha(2, 6, true, 10);
    mha.init_weight<xavier_gaussian_t>();

    mat_t<val_type> input1(6, 1, {
        0.5f, 0.3f, 0.2f, 0.1f, 0.3f, 0.5f
    });
    mat_t<val_type> input2(6, 2, {
        0.5f, 0.8f,
        0.3f, 0.7f,
        0.2f, 0.4f,
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    });

    auto out1 = mha.forward(input1);
    auto out2 = mha.forward(input2);

    ExpectShape(out1, 6, 1);
    ExpectShape(out2, 6, 2);
    ExpectNearMat(out1, out2.front_col(), 1e-4);
}

TEST(CausalAttention, DecoderFirstPositionStable)
{
    transformer_base_t<mat_t<val_type>, nadam_t> tf_base(2, 3, 2, 6, 24);
    tf_base.init_weight<xavier_gaussian_t>();

    mat_t<val_type> en_input(6, 3, {
        0.5f, 0.8f, 0.3f,
        0.7f, 0.2f, 0.4f,
        0.6f, 0.8f, 0.1f,
        0.9f, 0.3f, 0.7f,
        0.3f, 0.7f, 0.2f,
        0.4f, 0.5f, 0.6f
    });
    mat_t<val_type> input1(6, 1, {
        0.5f, 0.3f, 0.2f, 0.1f, 0.3f, 0.5f
    });
    mat_t<val_type> input2(6, 2, {
        0.5f, 0.8f,
        0.3f, 0.7f,
        0.2f, 0.4f,
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    });

    tf_base.encoder_forward(en_input);
    auto out1 = tf_base.forward(input1);
    auto out2 = tf_base.forward(input2);
    ExpectNearMat(out1, out2.front_col(), 1e-4);
}

TEST(CausalAttention, MaskedSoftmaxMatchesHandCalc)
{
    // scores = Q'K layout: row i = query pos, col j = key pos
    // After causal mask j>i → -inf, then row-wise softmax.
    mat_t<double> scores(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });
    for (int i = 0; i < scores.row_num(); ++i)
        for (int j = i + 1; j < scores.col_num(); ++j)
            scores(i, j) = -std::numeric_limits<double>::infinity();

    hsoftmax_net_t<mat_t<double>> sm;
    auto w = sm.forward(scores);

    // row0: only key0 visible → [1, 0]
    EXPECT_NEAR(w(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(w(0, 1), 0.0, 1e-12);

    // row1: softmax([3, 4])
    const double e3 = std::exp(3.0);
    const double e4 = std::exp(4.0);
    EXPECT_NEAR(w(1, 0), e3 / (e3 + e4), 1e-12);
    EXPECT_NEAR(w(1, 1), e4 / (e3 + e4), 1e-12);
}

TEST(CausalAttention, FutureTokensDoNotAffectPastOutputs)
{
    // Causal contract: mutating tokens after position t must not change outputs at ≤ t.
    // Use even d_head so RoPE is active (d_model=4, heads=2 → d_head=2).
    mat_mha_t<mat_t<double>, nadam_t> mha(2, 4, true, 8);
    mha.init_weight<xavier_gaussian_t>();

    mat_t<double> base(4, 3, {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    });
    auto out_base = mha.forward(base);

    mat_t<double> mutate_future = base.clone();
    mutate_future(0, 2) = 9.9;
    mutate_future(1, 2) = -9.9;
    mutate_future(2, 2) = 8.8;
    mutate_future(3, 2) = -8.8;
    auto out_mut = mha.forward(mutate_future);

    // positions 0 and 1 unchanged; position 2 may change
    ExpectNearMat(out_base.view(0, 0, 4, 2), out_mut.view(0, 0, 4, 2), 1e-10);
}
