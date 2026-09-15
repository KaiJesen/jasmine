#include <cmath>
#include <gtest/gtest.h>
#include "jas_transformer_kernel_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_loss_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"


using namespace jasmine;
template <typename val_type>
using tf_upr_tpl = cache_updator_t<val_type, nadam_t>;

TEST(TransformerKernel, FfnResidualShape)
{
    using val_type = float;
    mat_t<val_type> input(4, 2, {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f});

    res_ffn_t<val_type, nadam_t> res_ffn;
    res_ffn.base_net().reinit(std::vector<int>{4, 16, 4});
    res_ffn.base_net().set_updator(0.01);
    res_ffn.init_weight<xavier_gaussian_t>();

    auto out = res_ffn.forward(input);
    ExpectShape(out, 4, 2);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_TRUE(std::isfinite(out(i, j)));
}

TEST(TransformerKernel, EncoderTrainLossDecreases)
{
    using encoder_type = encoder_t<double, adam_t>;
    using net_type = complex_net_builder_t<double>
        ::push_back_impl<encoder_type>
        ::push_back_staticnet<mse_loss_t>
        ::type;

    net_type cnet;
    cnet.template get<0>().set_param(2, 2, 4, 16, 1);
    cnet.set_updator(0.01);
    cnet.init_weight<xavier_gaussian_t>();

    mat_t<double> input(4, 2, {
        0.1, 0.4,
        0.2, 0.3,
        0.3, 0.2,
        0.4, 0.1
    });
    mat_t<double> target = input.clone();

    cnet.forward(input);
    double loss0 = cnet.back().loss(target);
    for (int i = 0; i < 200; ++i)
    {
        cnet.forward(input);
        cnet.backward(target);
        cnet.step();
    }
    cnet.forward(input);
    EXPECT_LT(cnet.back().loss(target), loss0);
}

TEST(TransformerKernel, TfKernelTrainLossDecreases)
{
    using val_type = double;
    using tf_type = transformer_kernel_t<val_type, tf_upr_tpl>;
    using net_type = complex_net_builder_t<val_type>
        ::push_back_impl<tf_type>
        ::push_back_staticnet<mse_loss_t>
        ::type;

    net_type cnet;
    auto& tf = cnet.template get<0>();
    tf.set_param(2, 2, 2, 4, 16);
    cnet.init_weight<xavier_gaussian_t>();
    cnet.set_updator(0.01);

    mat_t<val_type> encoder_input(4, 1, {0.1, 0.2, 0.3, 0.4});
    mat_t<val_type> decoder_input(4, 1, {0.1, 0.2, 0.3, 0.4});
    mat_t<val_type> decoder_target(4, 1, {0.1, 0.2, 0.3, 0.4});

    tf.encoder_forward(encoder_input);
    cnet.forward(decoder_input);
    double loss0 = cnet.back().loss(decoder_target);
    for (int i = 0; i < 80; ++i)
    {
        cnet.forward(decoder_input);
        cnet.backward(decoder_target);
        cnet.step();
    }
    cnet.forward(decoder_input);
    EXPECT_LT(cnet.back().loss(decoder_target), loss0);
}

TEST(TransformerKernel, DecoderOnlyTrainLossDecreases)
{
    using val_type = double;
    using net_type = complex_net_builder_t<val_type>
        ::push_back_impl<decoder_only_t<val_type, adam_t>>
        ::push_back_staticnet<mse_loss_t>
        ::type;

    net_type cnet;
    cnet.template get<0>().set_param(2, 2, 4, 16, 1);
    cnet.set_updator(0.01);
    cnet.init_weight<xavier_gaussian_t>();

    mat_t<val_type> input(4, 2, {
        0.1, 0.4,
        0.2, 0.3,
        0.3, 0.2,
        0.4, 0.1
    });
    mat_t<val_type> target = input.clone();

    cnet.forward(input);
    double loss0 = cnet.back().loss(target);
    for (int i = 0; i < 200; ++i)
    {
        cnet.forward(input);
        cnet.backward(target);
        cnet.step();
    }
    cnet.forward(input);
    EXPECT_LT(cnet.back().loss(target), loss0);
}

TEST(TransformerKernel, DecoderOnlyCausalFirstPositionStable)
{
    using val_type = float;
    decoder_only_t<val_type, nadam_t> dec(2, 2, 6, 24, 10);
    dec.init_weight<xavier_gaussian_t>();

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

    auto out1 = dec.forward(input1);
    auto out2 = dec.forward(input2);
    ExpectNearMat(out1, out2.front_col(), 1e-4);
}

TEST(TransformerKernel, DecoderOnlyForwardOneMatchesFull)
{
    using val_type = double;
    constexpr int d_model = 4;
    constexpr int seq = 4;
    decoder_only_t<val_type, nadam_t> dec(2, 2, d_model, 16, seq);
    dec.init_weight<xavier_gaussian_t>();

    mat_t<val_type> input(d_model, seq, {
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6
    });

    auto ref = dec.forward(input);
    dec.clear_kv_cache();
    mat_t<val_type> step_out(d_model, seq);
    for (int t = 0; t < seq; ++t)
    {
        auto y = dec.forward_one(input.view(0, t, d_model, 1).clone());
        ExpectShape(y, d_model, 1);
        for (int i = 0; i < d_model; ++i)
            step_out(i, t) = y(i, 0);
    }
    ExpectNearMat(ref, step_out, 1e-9);
}
