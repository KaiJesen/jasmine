#include <gtest/gtest.h>
#include "mat_net_t.hpp"
#include "mat_init_t.hpp"
#include "mat_loss_t.hpp"
#include "test_helpers.hpp"


using namespace jasmine;
template <size_t N>
struct test_net_t
{
    using val_type = int;
    int forward(int x)
    {
        return x + 1;
    }

    int backward(int delta)
    {
        return delta - 1;
    }
};

TEST(Net, WeightStackLossDecreases)
{
    using net_type = complex_net_builder_t<double>
        ::push_back_updatable<weight_net_t, nadam_t>
        ::push_back_updatable<weight_net_t, adam_t>
        ::push_back_staticnet<sigmoid_net_t>
        ::push_back_staticnet<mse_loss_t>
        ::type;

    net_type net;
    net.reinit(std::vector<int>{2, 3, 3});
    net.init_weight<xavier_gaussian_t>();
    net.set_updator(0.1);

    mat_t<double> input(2, 2, {0.5, 0.8, 0.3, 0.7});
    mat_t<double> label(3, 2, {0.2, 0.4, 0.6, 0.8, 0.1, 0.9});

    net.forward(input);
    double loss0 = net.back().loss(label);
    for (int i = 0; i < 200; ++i)
    {
        net.forward(input);
        net.backward(label);
        net.step();
    }
    net.forward(input);
    EXPECT_LT(net.back().loss(label), loss0);
}

TEST(Net, ComplexPipelineForwardShape)
{
    auto pipe = complex_net_t<test_net_t<1>, test_net_t<2>, test_net_t<3>>();
    EXPECT_EQ(pipe.forward(0), 3);
    EXPECT_EQ(pipe.backward(3), 0);
}

TEST(Net, InferSkipsLossAndMatchesTrainPrefix)
{
    using net_type = complex_net_builder_t<double>
        ::push_back_updatable<weight_net_t, nadam_t>
        ::push_back_updatable<weight_net_t, adam_t>
        ::push_back_staticnet<sigmoid_net_t>
        ::push_back_staticnet<mse_loss_t>
        ::type;

    net_type net;
    net.reinit(std::vector<int>{2, 3, 3});
    net.init_weight<xavier_gaussian_t>();

    mat_t<double> input(2, 1, {0.5, 0.8});
    auto train_out = net.forward(input);
    auto infer_out = net.infer(input);
    ExpectNearMat(train_out, infer_out, 1e-12);

    // 单列 infer 与整段 forward 的最后一列一致（无 KV 层时）
    mat_t<double> seq(2, 2, {0.5, 0.2, 0.8, 0.4});
    auto full = net.forward(seq);
    auto col1 = net.infer(seq.view(0, 1, 2, 1).clone());
    ExpectNearMat(full.view(0, 1, full.row_num(), 1), col1, 1e-12);
}
