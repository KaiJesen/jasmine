/**
 * Unit tests for rbm_net_t (RBM) and dbn_net_t (a statically stacked DBN).
 *
 * An RBM plays two roles in this library and both have to be pinned down:
 *   1. as a **layer**: the P(h=1|v)=sigma(Wv+c) forward plus the "linear + sigmoid" backward (the
 *      path DBN supervised fine-tuning takes);
 *   2. as an **RBM**: the CD-k update rule (the path layer-wise greedy pretraining takes).
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include <cstdio>
#include <gtest/gtest.h>

#include "jas_rbm_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{
using dmat = mat_t<double>;
using rbm_t = rbm_net_t<dmat, sgd_t>;
using val_type_t = double;
template <typename T> using adamw_upr_tpl = cache_updator_t<T, adamw_t>;
} // namespace

TEST(Rbm, HiddenProbabilityIsSigmoidOfLinearTerm)
{
    rbm_t rbm(2, 2);
    rbm.weight() = dmat(2, 2, {1, 0,
                               0, 2});
    rbm.visible_bias() = 0.0;
    rbm.hidden_bias() = 0.0;

    const dmat v(2, 1, {1, 2});
    const dmat h = rbm.forward(v);
    ExpectShape(h, 2, 1);
    EXPECT_NEAR(h(0, 0), 1.0 / (1.0 + std::exp(-1.0)), 1e-12);
    EXPECT_NEAR(h(1, 0), 1.0 / (1.0 + std::exp(-4.0)), 1e-12);

    // the symmetric form P(v|h): W^T projection plus the visible bias
    const dmat back = rbm.visible_prob(h);
    ExpectShape(back, 2, 1);
    EXPECT_NEAR(back(0, 0), 1.0 / (1.0 + std::exp(-(1.0 * h(0, 0)))), 1e-12);
}

TEST(Rbm, LayerBackwardMatchesNumericalGradient)
{
    // treat the RBM as a "linear + sigmoid" layer and check numerical gradients
    // (with SGD lr=1 the parameter change IS the gradient)
    rbm_t rbm(3, 2);
    rbm.set_updator(1.0);
    rbm.weight() = dmat(2, 3, {0.5, -0.3, 0.2,
                               0.1, 0.4, -0.6});
    rbm.visible_bias() = dmat(3, 1, {0.1, -0.2, 0.3});
    rbm.hidden_bias() = dmat(2, 1, {-0.1, 0.2});

    const dmat v(3, 1, {1.0, -2.0, 0.5});
    const dmat h = rbm.forward(v);

    const dmat w_before = rbm.weight(), b_before = rbm.visible_bias(), c_before = rbm.hidden_bias();
    const dmat dv = rbm.backward(h);            // L = 0.5·Σh² → delta = h
    const dmat grad_w = w_before - rbm.weight();
    const dmat grad_b = b_before - rbm.visible_bias();
    const dmat grad_c = c_before - rbm.hidden_bias();

    // numerical gradient: L(v;W,b,c) = 0.5 * sum sigma(Wv+c)^2
    auto loss = [&](const dmat& W, const dmat& b, const dmat& c) {
        const dmat hh = sigmoid(W.dot(v) + c);
        double s = 0.0;
        for (int i = 0; i < hh.row_num(); ++i) s += 0.5 * hh(i, 0) * hh(i, 0);
        (void)b;
        return s;
    };
    const double eps = 1e-6;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            dmat wp = w_before, wm = w_before;
            wp(i, j) += eps; wm(i, j) -= eps;
            EXPECT_NEAR(grad_w(i, j), (loss(wp, b_before, c_before) - loss(wm, b_before, c_before)) / (2 * eps), 1e-6);
        }
        dmat cp = c_before, cm = c_before;
        cp(i, 0) += eps; cm(i, 0) -= eps;
        EXPECT_NEAR(grad_c(i, 0), (loss(w_before, b_before, cp) - loss(w_before, b_before, cm)) / (2 * eps), 1e-6);
    }
    // the visible bias does not affect h, so its gradient is 0 (b only appears in P(v|h))
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(grad_b(i, 0), 0.0, 1e-12);

    // input gradient: dL/dv = W^T (h * (1-h) * delta), with delta = h here
    dmat expect_dv(3, 1);
    for (int i = 0; i < 3; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < 2; ++j)
        {
            const double hj = h(j, 0);
            s += w_before(j, i) * hj * (1 - hj) * hj;
        }
        expect_dv(i, 0) = s;
    }
    ExpectNearMat(dv, expect_dv, 1e-12);
}

TEST(Rbm, ContrastiveDivergenceFollowsTheCdRule)
{
    // deterministic CD-1 (sample=false): dW = eta(h1 v1^T - h0 v0^T), db = eta(v1-v0), dc = eta(h1-h0)
    // with SGD lr=1 the parameter change is the negated CD increment; check it term by term.
    rbm_t rbm(3, 2);
    rbm.set_updator(1.0);
    rbm.weight() = dmat(2, 3, {0.3, -0.2, 0.5,
                               0.1, 0.4, -0.3});
    rbm.visible_bias() = dmat(3, 1, {0.05, -0.1, 0.2});
    rbm.hidden_bias() = dmat(2, 1, {0.15, -0.25});
    const dmat v0(3, 1, {0.8, 0.3, 0.6});

    const dmat w0 = rbm.weight(), b0 = rbm.visible_bias(), c0 = rbm.hidden_bias();
    const dmat h0 = rbm.hidden_prob(v0);
    const dmat v1 = rbm.visible_prob(h0);
    const dmat h1 = rbm.hidden_prob(v1);

    rbm.contrastive_divergence(v0, 1, /*sample=*/false);

    // the parameter change IS the CD increment: dW = h0 v0^T - h1 v1^T, db = v0-v1, dc = h0-h1
    // (the updator receives the negated increment and computes p <- p - lr*g, so the two cancel)
    const dmat dW = rbm.weight() - w0;
    const dmat db = rbm.visible_bias() - b0;
    const dmat dc = rbm.hidden_bias() - c0;
    for (int j = 0; j < 2; ++j)
    {
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(dW(j, i), h0(j, 0) * v0(i, 0) - h1(j, 0) * v1(i, 0), 1e-12)
                << "W(" << j << "," << i << ")";
        EXPECT_NEAR(dc(j, 0), h0(j, 0) - h1(j, 0), 1e-12);
    }
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(db(i, 0), v0(i, 0) - v1(i, 0), 1e-12);

    // shape guard: a gradient must match its parameter (checked internally); verify nothing resized
    ExpectShape(rbm.weight(), 2, 3);
    ExpectShape(rbm.visible_bias(), 3, 1);
    ExpectShape(rbm.hidden_bias(), 2, 1);
}

TEST(Rbm, ReconstructionImprovesWithTraining)
{
    rbm_t rbm(6, 4);
    rbm.set_updator(0.5);
    g_random_engine.seed(3);
    rbm.init_weight<xavier_uniform_t>();

    // three separable binary patterns, trained with repeated deterministic CD-1
    dmat data(6, 3, {1, 0, 0,
                     1, 0, 0,
                     0, 1, 0,
                     0, 1, 0,
                     0, 0, 1,
                     0, 0, 1});
    auto mean_recon = [&]() {
        double s = 0.0;
        for (int t = 0; t < data.col_num(); ++t)
        {
            const dmat col = column_as_vector(data, t);
            const dmat rec = rbm.reconstruct(col);
            for (int i = 0; i < rec.row_num(); ++i) s += std::abs(rec(i, 0) - col(i, 0));
        }
        return s / (data.row_num() * data.col_num());
    };

    const double before = mean_recon();
    for (int epoch = 0; epoch < 200; ++epoch)
    {
        for (int t = 0; t < data.col_num(); ++t)
            rbm.contrastive_divergence(column_as_vector(data, t), 1, /*sample=*/false);
        rbm.step();
    }
    const double after = mean_recon();
    EXPECT_LT(after, before) << "before=" << before << " after=" << after;
    EXPECT_LT(after, 0.15) << "the reconstruction error should drop markedly after training";
}

TEST(Rbm, ConceptsAndReinit)
{
    static_assert(is_updatable_net<rbm_t>);
    static_assert(is_reinitable_net<rbm_t>);      // has weights -> participates in the {in,out} protocol
    rbm_t rbm;
    rbm.reinit(std::vector<int>{5, 3});
    ExpectShape(rbm.weight(), 3, 5);
    ExpectShape(rbm.visible_bias(), 5, 1);
    ExpectShape(rbm.hidden_bias(), 3, 1);
    EXPECT_THROW(rbm.forward(dmat(4, 1)), std::invalid_argument);
    EXPECT_THROW(rbm.visible_prob(dmat(4, 1)), std::invalid_argument);
}

TEST(Dbn, GreedyPretrainReducesReconstruction)
{
    // DBN = 2 statically stacked RBMs + a classifier head; container {n_visible, n_hidden1, n_hidden2, n_class}
    dbn_net_t<2, adamw_upr_tpl> dbn;
    dbn.reinit(std::vector<int>{8, 5, 4, 3});
    ExpectShape(dbn.template get<0>().weight(), 5, 8);
    ExpectShape(dbn.template get<1>().weight(), 4, 5);
    ExpectShape(dbn.template get<2>().weight(), 3, 4);

    g_random_engine.seed(11);
    dbn.template get<0>().init_weight<xavier_uniform_t>();
    dbn.template get<1>().init_weight<xavier_uniform_t>();
    dbn.set_updator(0.05);

    dmat data(8, 3, {1, 0, 0,
                     1, 0, 0,
                     1, 0, 0,
                     0, 1, 0,
                     0, 1, 0,
                     0, 1, 0,
                     0, 0, 1,
                     0, 0, 1});

    // Use the explicit reconstruction error, not the last mini-batch error CD returns -- the latter
    // jitters with the learning rate and produced an unstable assertion (the first version of this
    // test fell for exactly that).
    auto recon_error = [&]() {
        double sum = 0.0;
        for (int t = 0; t < data.col_num(); ++t)
        {
            const dmat col = column_as_vector(data, t);
            const dmat rec = dbn.template get<0>().reconstruct(col);
            for (int i = 0; i < rec.row_num(); ++i)
                sum += std::abs(rec(i, 0) - col(i, 0));
        }
        return sum / (data.row_num() * data.col_num());
    };

    const double before = recon_error();
    dbn_pretrain<2>(dbn, data, /*cd_k=*/1, /*epochs=*/5);
    const double after = recon_error();
    EXPECT_LT(after, before) << "greedy pretraining should reduce layer 0's reconstruction error";

    // forward through the stack: RBM0(8->5) -> RBM1(5->4) -> head(4->3)
    const dmat logits = dbn.forward(data);
    ExpectShape(logits, 3, 3);
    ExpectShape(dbn.template get<0>().weight(), 5, 8);   // the updators did not change the shapes
    ExpectShape(dbn.template get<1>().weight(), 4, 5);
    ExpectShape(dbn.template get<2>().weight(), 3, 4);
}

TEST(Dbn, FullChainBackward)
{
    // Whole-chain backward: CE -> classifier -> RBM1 -> RBM0, the gradient has to reach the visible
    // layer. This test used to fail in the -O3 full binary: several TUs declared same-named
    // anonymous-namespace alias templates with different targets (adamw_t vs sgd_t), GCC mangles
    // anonymous namespaces identically (_GLOBAL__N_1), so the linker merged two different layouts of
    // weight_net_t<..., upr_tpl> and the classifier's cache was read at the wrong offset. Giving the
    // aliases unique names fixed it; the full investigation is in TESTING.md 15.7.

    dbn_net_t<2, adamw_upr_tpl> dbn;
    dbn.reinit(std::vector<int>{8, 5, 4, 3});
    dbn.set_updator(0.01);

    dmat data(8, 3);
    data = 0.0;
    for (int t = 0; t < 3; ++t) data(t, t) = 1.0;

    const dmat logits = dbn.forward(data);
    ExpectShape(logits, 3, 3);

    dmat labels(1, 3, {0, 1, 2});
    const dmat dx = dbn.backward(labels);
    ExpectShape(dx, 8, 3);                       // the gradient is back at the visible layer
    dbn.step();

    // the shapes are still intact after the update (the updator must not resize parameters)
    ExpectShape(dbn.template get<0>().weight(), 5, 8);
    ExpectShape(dbn.template get<1>().weight(), 4, 5);
    ExpectShape(dbn.template get<2>().weight(), 3, 4);

    // forward once more: the cache is rebuilt correctly and the chain still works
    ExpectShape(dbn.forward(data), 3, 3);
}
