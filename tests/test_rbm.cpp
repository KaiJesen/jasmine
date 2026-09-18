/**
 * rbm_net_t（RBM）与 dbn_net_t（静态堆叠的 DBN）单测。
 *
 * RBM 在本库里有两个身份，两条都要钉住：
 *   1. 作为**层**：P(h=1|v)=σ(Wv+c) 的前向 + 「线性+sigmoid」的反向（DBN 监督微调走这条）；
 *   2. 作为**RBM 自己**：CD-k 的更新规则（逐层贪心预训练走这条）。
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

    // P(v|h) 是它的对称形式：Wᵀ 投影 + 可见偏置
    const dmat back = rbm.visible_prob(h);
    ExpectShape(back, 2, 1);
    EXPECT_NEAR(back(0, 0), 1.0 / (1.0 + std::exp(-(1.0 * h(0, 0)))), 1e-12);
}

TEST(Rbm, LayerBackwardMatchesNumericalGradient)
{
    // 把 RBM 当「线性+sigmoid」层做数值梯度对拍（SGD lr=1 时参数变化量就是梯度）
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

    // 数值梯度：L(v;W,b,c) = 0.5·Σ σ(Wv+c)²
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
    // 可见偏置不影响 h → 梯度为 0（RBM 的 b 只在 P(v|h) 里出现）
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(grad_b(i, 0), 0.0, 1e-12);

    // 输入梯度：∂L/∂v = Wᵀ (h ⊙ (1-h) ⊙ delta)，这里 delta = h
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
    // 确定性 CD-1（sample=false）：ΔW = η(h1v1ᵀ - h0v0ᵀ)，Δb = η(v1-v0)，Δc = η(h1-h0)
    // 取 SGD lr=1，参数变化量就是「负的 CD 增量」，逐项对拍。
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

    // 参数变化量就是 CD 增量本身：ΔW = h0v0ᵀ - h1v1ᵀ、Δb = v0-v1、Δc = h0-h1
    // （内部给 updator 传的是「负增量」，updator 做 p ← p - lr·g，两者相抵。）
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

    // 形状护栏：梯度与参数形状必须一致（内部会检查），这里确认调用没有破坏形状
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

    // 三个可分的二值模式，反复做确定性 CD-1
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
    EXPECT_LT(after, 0.15) << "训练后重建误差应当明显变小";
}

TEST(Rbm, ConceptsAndReinit)
{
    static_assert(is_updatable_net<rbm_t>);
    static_assert(is_reinitable_net<rbm_t>);      // 带权重 → 参与容器的 {in,out} 协议
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
    // DBN = 2 个 RBM 静态堆叠 + 分类头；容器 {n_visible, n_hidden1, n_hidden2, n_class}
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

    // 用「显式重构误差」而不是 CD 返回的最后一个 mini-batch 误差做判据 —— 后者随 lr 抖，
    // 会写出不稳定的断言（第一版就吃过这个亏）。
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
    EXPECT_LT(after, before) << "贪心预训练应当降低第 0 层的重建误差";

    // 堆叠后的前向：RBM0(8→5) → RBM1(5→4) → 分类头(4→3)
    const dmat logits = dbn.forward(data);
    ExpectShape(logits, 3, 3);
    ExpectShape(dbn.template get<0>().weight(), 5, 8);   // 形状没被 updator 改掉
    ExpectShape(dbn.template get<1>().weight(), 4, 5);
    ExpectShape(dbn.template get<2>().weight(), 3, 4);
}

TEST(Dbn, FullChainBackward)
{
    // 整链反向：CE → 分类头 → RBM1 → RBM0，梯度要一路回到可见层。
    //
    // 已知问题：**只在 `-O3 -DNDEBUG` + 完整 unit_tests 二进制里失败**（Release）；
    // 单独编译本文件、或整包在无优化下编译都通过。失败时分类头的前向缓存 m_input
    // 在 backward 时已被写坏（dims/指针被覆盖），排查详见 TESTING.md 15.6。
    // 这里 skip，避免让 Release 套件带着一个未定位的失败；断言保留，修好后删掉这一行即可。

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
    ExpectShape(dx, 8, 3);                       // 梯度回到可见层
    dbn.step();

    // 更新之后形状仍然完好（updator 不会把参数 resize 掉）
    ExpectShape(dbn.template get<0>().weight(), 5, 8);
    ExpectShape(dbn.template get<1>().weight(), 4, 5);
    ExpectShape(dbn.template get<2>().weight(), 3, 4);

    // 再前向一次：缓存被正确重建，链路仍然可用
    ExpectShape(dbn.forward(data), 3, 3);
}
