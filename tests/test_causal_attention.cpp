#include <gtest/gtest.h>
#include "mat_mha_t.hpp"
#include "mat_transformer_t.hpp"
#include "mat_init_t.hpp"
#include "test_helpers.hpp"

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
