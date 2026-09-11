#include <cmath>
#include <gtest/gtest.h>
#include "mat_transformer_kernel_t.hpp"
#include "mat_init_t.hpp"
#include "mat_loss_t.hpp"
#include "mat_updator_t.hpp"
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
