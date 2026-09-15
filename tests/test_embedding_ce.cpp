#include <cmath>
#include <gtest/gtest.h>

#include "jas_embedding_t.hpp"
#include "jas_net_t.hpp"
#include "jas_loss_t.hpp"
#include "jas_mat_init_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

TEST(Embedding, GatherShapeAndValues)
{
    embedding_net_t<mat_t<double>, nadam_t> emb(4, 3); // vocab=4, d_model=3
    emb.weight()(0, 0) = 1;
    emb.weight()(1, 0) = 2;
    emb.weight()(2, 0) = 3;
    emb.weight()(0, 2) = 10;
    emb.weight()(1, 2) = 20;
    emb.weight()(2, 2) = 30;

    mat_t<double> ids(1, 2, {0.0, 2.0});
    auto out = emb.forward(ids);
    ExpectShape(out, 3, 2);
    EXPECT_NEAR(out(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(out(2, 0), 3.0, 1e-12);
    EXPECT_NEAR(out(0, 1), 10.0, 1e-12);
    EXPECT_NEAR(out(1, 1), 20.0, 1e-12);
}

TEST(Embedding, BackwardScattersToCorrectColumns)
{
    embedding_net_t<mat_t<double>, sgd_t> emb(3, 2);
    emb.set_updator(1.0); // lr=1 → weight -= grad
    emb.weight() = 0.0;

    mat_t<double> ids(1, 2, {1.0, 1.0}); // both look up id 1
    emb.forward(ids);

    mat_t<double> delta(2, 2);
    delta(0, 0) = 0.5;
    delta(1, 0) = 1.0;
    delta(0, 1) = 0.5;
    delta(1, 1) = 1.0;
    emb.backward(delta);
    // grad_w[:,1] = (1, 2); update with sgd lr=1: w -= grad
    EXPECT_NEAR(emb.weight()(0, 1), -1.0, 1e-12);
    EXPECT_NEAR(emb.weight()(1, 1), -2.0, 1e-12);
    EXPECT_NEAR(emb.weight()(0, 0), 0.0, 1e-12);
}

TEST(CeLoss, SoftmaxGradMatchesOneHot)
{
    ce_loss_t<mat_t<double>> ce;
    mat_t<double> logits(3, 1, {2.0, 1.0, 0.1});
    ce.forward(logits);
    mat_t<double> target(1, 1, {0.0}); // class 0

    auto grad = ce.backward(target);
    // p = softmax(logits); grad = (p - y) / 1
    double m = std::max({2.0, 1.0, 0.1});
    double e0 = std::exp(2.0 - m), e1 = std::exp(1.0 - m), e2 = std::exp(0.1 - m);
    double s = e0 + e1 + e2;
    EXPECT_NEAR(grad(0, 0), e0 / s - 1.0, 1e-9);
    EXPECT_NEAR(grad(1, 0), e1 / s, 1e-9);
    EXPECT_NEAR(grad(2, 0), e2 / s, 1e-9);

    double nll = -std::log(e0 / s);
    EXPECT_NEAR(ce.loss(target), nll, 1e-9);
}

TEST(CeLoss, IgnoreIndexSkipsPad)
{
    ce_loss_t<mat_t<double>> ce;
    constexpr int PAD = 0;
    ce.set_ignore_index(PAD);

    // logits 2 classes × 3 positions; only middle is real (label 1)
    mat_t<double> logits(2, 3);
    logits = 0.0;
    logits(1, 1) = 5.0; // strong preference for class 1 at t=1
    ce.forward(logits);

    mat_t<double> target(1, 3, {0.0, 1.0, 0.0}); // pad, class1, pad
    double loss_masked = ce.loss(target);
    auto grad = ce.backward(target);

    // pad columns should have zero grad
    EXPECT_NEAR(grad(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(grad(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(grad(0, 2), 0.0, 1e-12);
    EXPECT_NEAR(grad(1, 2), 0.0, 1e-12);
    // only one counted position
    EXPECT_LT(loss_masked, 0.1);

    ce.set_ignore_index(-1);
    double loss_all = ce.loss(target);
    EXPECT_GT(loss_all, loss_masked);
}

TEST(CeLoss, PositionMaskSkipsFuture)
{
    ce_loss_t<mat_t<double>> ce;
    mat_t<double> logits(2, 3);
    logits = 0.0;
    logits(0, 0) = 3.0;
    logits(0, 1) = 3.0;
    logits(0, 2) = -10.0; // would be high NLL if counted with label 0
    ce.forward(logits);

    mat_t<double> target(1, 3, {0.0, 0.0, 0.0});
    mat_t<double> mask(1, 3, {1.0, 1.0, 0.0}); // hide "future" last step
    ce.set_position_mask(mask);

    double loss = ce.loss(target);
    auto grad = ce.backward(target);
    EXPECT_NEAR(grad(0, 2), 0.0, 1e-12);
    EXPECT_NEAR(grad(1, 2), 0.0, 1e-12);
    EXPECT_LT(loss, 1.0);
}

TEST(TokenPipeline, EmbeddingProjCeLossDecreases)
{
    // ids → emb → output_proj → ce_loss；学把固定 id 映射到固定标签
    constexpr int PAD = 0;
    constexpr int V = 5;
    constexpr int D = 8;

    using net_type = complex_net_builder_t<double>
        ::push_back_updatable<embedding_net_t, nadam_t>
        ::push_back_updatable<output_proj_net_t, nadam_t>
        ::push_back_staticnet<ce_loss_t>
        ::type;

    net_type net;
    net.reinit(std::vector<int>{V, D, V});
    net.init_weight<xavier_gaussian_t>();
    net.set_updator(0.05);
    net.template get<2>().set_ignore_index(PAD);

    // sequence: pad, token2, token3, pad — labels same (identity-ish classification)
    mat_t<double> ids(1, 4, {0.0, 2.0, 3.0, 0.0});
    mat_t<double> labels(1, 4, {0.0, 2.0, 3.0, 0.0});

    net.forward(ids);
    double loss0 = net.template get<2>().loss(labels);
    for (int i = 0; i < 80; ++i)
    {
        net.forward(ids);
        net.backward(labels);
        net.step();
    }
    net.forward(ids);
    double loss1 = net.template get<2>().loss(labels);
    EXPECT_LT(loss1, loss0);

    // infer skips CE：输出为 logits V×T
    auto logits = net.infer(ids);
    ExpectShape(logits, V, 4);
}

TEST(TokenPipeline, InferSkipsCeLoss)
{
    using net_type = complex_net_builder_t<double>
        ::push_back_updatable<embedding_net_t, adam_t>
        ::push_back_updatable<weight_net_t, adam_t>
        ::push_back_staticnet<ce_loss_t>
        ::type;

    net_type net;
    net.reinit(std::vector<int>{4, 3, 4});
    net.init_weight<xavier_uniform_t>();

    mat_t<double> ids(1, 2, {1.0, 2.0});
    auto train_out = net.forward(ids);
    auto infer_out = net.infer(ids);
    ExpectNearMat(train_out, infer_out, 1e-12);
}
