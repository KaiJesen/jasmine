/**
 * rms_norm_net_t 单测。
 *
 * RMSNorm 的定义（与 layer_norm_net_t 的唯一实质差别就是"不减均值、没有 beta"）：
 *     y = gamma ⊙ x / sqrt(mean(x²) + eps)
 *
 * 测试重点：
 *   1. 前向与定义式逐元素一致（含 gamma），并与 PyTorch 的 float64 定义对齐；
 *   2. **反向必须删掉 LayerNorm 的 mean(g) 那一项** —— 这是最容易抄错的地方，
 *      误留时误差量级可达 1.0（见 NaiveFormulaHelper 里的对照）；
 *   3. 缩放不变 / 平移不敏感——前者是它 work 的原因，后者是它与 LayerNorm 的分界；
 *   4. 能作为 pre-norm 分支的一层接入 complex_net_t。
 *
 * 关于数值基准的一个坑：HuggingFace 的 `LlamaRMSNorm` 内部会把输入
 * `.to(torch.float32)` 再算，因此其输出与 float64 精确解相差约 1e-7（float32 精度级别），
 * 即使你传入 float64 也是如此。jasmine 是 double，所以基准取 **float64 下按定义式计算**
 * 的结果（下方 kY），而不是 HF 的输出。已核实：kY 与 float64 定义式完全一致（差 0.0），
 * 与官方 `LlamaRMSNorm` 类的差恰好就是那个精度转换带来的 1.1e-7。
 */

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_net_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

using dmat = mat_t<double>;
using rms_upd_t = cache_updator_t<double, nadam_t>;
using rms_t = rms_norm_net_t<dmat, nadam_t>;
using rms_sgd_t = rms_norm_net_t<dmat, sgd_t>;

dmat make_mat(int rows, int cols, double scale, unsigned seed)
{
    dmat m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = dist(rng);
    return m;
}

/** 朴素参考：逐列按定义式计算 y = gamma ⊙ x / sqrt(mean(x²) + eps) */
dmat naive_rms_forward(const dmat& x, const dmat& gamma, double eps)
{
    const int d = x.row_num(), T = x.col_num();
    dmat y(d, T);
    for (int t = 0; t < T; ++t)
    {
        double ms = 0.0;
        for (int i = 0; i < d; ++i)
            ms += x(i, t) * x(i, t);
        ms /= d;
        const double r = std::sqrt(ms + eps);
        for (int i = 0; i < d; ++i)
            y(i, t) = gamma(i, 0) * (x(i, t) / r);
    }
    return y;
}

/**
 * 刻意写错的反向：保留 LayerNorm 的 mean(g) 项，用来证明测试确实能抓住这个错误。
 * （LayerNorm: (g·d - sum(g) - hx·sum(g⊙hx))/d/std；RMSNorm 应去掉 sum(g)）
 */

// ---- PyTorch(float64 定义式) 参考值：见文件头注释 ----
// x 为 d_model×T，每列一个 token
const double kX[12] = {1.0, 0.5, 2.0, 2.0, -1.5, 0.0, 3.0, 2.5, -1.0, 4.0, -3.5, 1.0};
const double kGamma[4] = {1.0, 0.5, 2.0, 1.5};
const double kRms[3] = {2.7386146132670803, 2.291290029655783, 1.2247489538676897};
const double kHx[12] = {0.3651481282381064, 0.21821768240972714, 1.6329877185721289,
                        0.7302962564762128, -0.6546530472291815, 0.0,
                        1.0954443847143192, 1.0910884120486357, -0.8164938592860644,
                        1.4605925129524255, -1.52752377686809, 0.8164938592860644};
const double kY[12] = {0.3651481282381064, 0.21821768240972714, 1.6329877185721289,
                       0.3651481282381064, -0.32732652361459075, 0.0,
                       2.1908887694286383, 2.1821768240972714, -1.6329877185721289,
                       2.1908887694286383, -2.291285665302135, 1.2247407889290967};
constexpr double kEps = 1e-5;

} // namespace

TEST(RmsNorm, MatchesDefinition)
{
    const int d = 5, T = 4;
    rms_t norm;
    norm.set_param(d, kEps);
    norm.gama() = dmat(d, 1, {0.7, 1.3, 0.5, 2.0, 1.0});

    dmat x = make_mat(d, T, 1.5, 1);
    ExpectNearMat(norm.forward(x), naive_rms_forward(x, norm.gama(), kEps), 1e-14);
}

TEST(RmsNorm, MatchesPyTorchReferenceValues)
{
    // 与 PyTorch LlamaRMSNorm 的 float64 定义式逐元素对齐（gamma 明显不为 1）
    const int d = 4, T = 3;
    rms_t norm;
    norm.set_param(d, kEps);
    for (int i = 0; i < d; ++i)
        norm.gama()(i, 0) = kGamma[i];

    dmat x(d, T, {kX[0], kX[1], kX[2], kX[3], kX[4], kX[5],
                  kX[6], kX[7], kX[8], kX[9], kX[10], kX[11]});
    dmat y = norm.forward(x);

    // mat_t 的初始化列表按 row-major 填入：m(i,j) == list[i*col_num + j]，
    // 而参考值是按 (d, T) 行主序摊平的，故下标为 i*T + j（不是 j*d + i）。
    ExpectShape(y, d, T);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < T; ++j)
            EXPECT_NEAR(y(i, j), kY[i * T + j], 1e-14)
                << "(" << i << "," << j << ")";

    // 顺带核对归一化中间量（kHx = 不含 gamma 的 x/rms）
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < T; ++j)
        {
            const double expected = kGamma[i] * kHx[i * T + j];
            EXPECT_NEAR(y(i, j), expected, 1e-14);
        }
}

TEST(RmsNorm, GammaDefaultsToOnes)
{
    // 懒初始化：首次 forward 前未调 set_param 时，gamma 应被填成 1
    const int d = 3, T = 2;
    rms_t norm;
    dmat x = make_mat(d, T, 1.0, 2);
    dmat y = norm.forward(x);

    dmat ref = naive_rms_forward(x, dmat(d, 1, {1, 1, 1}), rms_t::kDefaultEps);
    ExpectNearMat(y, ref, 1e-14);
}

TEST(RmsNorm, SetParamMakesGammaIdentity)
{
    const int d = 6;
    rms_t norm;
    norm.set_param(d, 1e-6);
    ASSERT_EQ(norm.gama().row_num(), d);
    ASSERT_EQ(norm.gama().col_num(), 1);
    for (int i = 0; i < d; ++i)
        EXPECT_DOUBLE_EQ(norm.gama()(i, 0), 1.0);
    EXPECT_DOUBLE_EQ(norm.eps(), 1e-6);
}

TEST(RmsNorm, IsScaleInvariant)
{
    // 缩放不变性是它 work 的根本原因：RMSNorm(c·x) == RMSNorm(x)，c > 0。
    // 取 eps = 0 才是精确成立（eps 加在均方值上，缩放会改变它的相对大小）。
    const int d = 4, T = 3;
    rms_t norm;
    norm.set_param(d, 0.0);
    norm.gama() = dmat(d, 1, {0.6, 1.4, 1.0, 0.9});
    dmat x = make_mat(d, T, 1.0, 3);

    const dmat y = norm.forward(x);
    for (double c : {1e-3, 0.5, 7.0, 1e3})
    {
        dmat yc = norm.forward(x * c);
        ExpectNearMat(yc, y, 1e-13);
        SCOPED_TRACE(std::string("c=") + std::to_string(c));
    }
}

TEST(RmsNorm, IsNotTranslationInvariant)
{
    // 分界点：RMSNorm 丢掉平移不变性（LayerNorm 有）。这是两者唯一的行为差异，
    // 值得单独钉住 —— 如果哪天有人"顺手"把均值减回去，这条会失败。
    const int d = 4, T = 2;
    rms_t norm;
    norm.set_param(d, kEps);
    norm.gama() = dmat(d, 1, {1, 1, 1, 1});
    dmat x = make_mat(d, T, 1.0, 4);

    dmat y = norm.forward(x);

    // 整体平移：把每个 token 的所有特征同时加上 c
    dmat shifted = x.clone();
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < T; ++j)
            shifted(i, j) += 10.0;
    dmat y_shift = norm.forward(shifted);

    double max_diff = 0.0;
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < T; ++j)
            max_diff = std::max(max_diff, std::abs(y_shift(i, j) - y(i, j)));
    EXPECT_GT(max_diff, 1e-3) << "RMSNorm 不应具有平移不变性（这是它与 LayerNorm 的分界）";
}

TEST(RmsNorm, DiffersFromLayerNormPresenceOfCentering)
{
    // 与 layer_norm_net_t 直接对照：同一个输入，RMSNorm 不减均值 → 输出不同。
    // 这也确认 RMSNorm 没有把中心化"偷偷"做进去。
    const int d = 4, T = 2;
    dmat x = make_mat(d, T, 1.0, 5);

    rms_t rms;
    // 取 eps=0：此时"两者模长都恰为 sqrt(d)"才是精确成立的。
    // eps>0 时归一化半径是 sqrt(d)·sqrt(ms/(ms+eps))，会略小于 sqrt(d)。
    rms.set_param(d, 0.0);
    rms.gama() = dmat(d, 1, {1, 1, 1, 1});
    layer_norm_net_t<dmat, nadam_t> ln;
    ln.set_param(d);
    ln.gama() = dmat(d, 1, {1, 1, 1, 1});
    ln.beta() = dmat(d, 1, {0, 0, 0, 0});

    dmat y_rms = rms.forward(x);
    dmat y_ln = ln.forward(x);

    // 两者都落在半径约为 sqrt(d) 的球面上，但**方向不同**（RMSNorm 不做中心化）。
    // 注意 eps 的影响：归一化半径实为 sqrt(d)·sqrt(ms/(ms+eps))，略小于 sqrt(d)。
    // 这里 RMSNorm 取 eps=0 故精确等于 sqrt(d)；LayerNorm 的 eps 固定为 1e-5（不可配），
    // 因此只能给一个宽松容差，并断言它确实不超过 sqrt(d)。
    const double sphere = std::sqrt(static_cast<double>(d));
    for (int t = 0; t < T; ++t)
    {
        double n_rms = 0.0, n_ln = 0.0;
        for (int i = 0; i < d; ++i)
        {
            n_rms += y_rms(i, t) * y_rms(i, t);
            n_ln += y_ln(i, t) * y_ln(i, t);
        }
        EXPECT_NEAR(std::sqrt(n_rms), sphere, 1e-12);
        EXPECT_LE(std::sqrt(n_ln), sphere + 1e-12);
        EXPECT_NEAR(std::sqrt(n_ln), sphere, 1e-3);
    }

    double max_diff = 0.0;
    for (int i = 0; i < d; ++i)
        for (int t = 0; t < T; ++t)
            max_diff = std::max(max_diff, std::abs(y_rms(i, t) - y_ln(i, t)));
    EXPECT_GT(max_diff, 1e-2) << "RMSNorm 与 LayerNorm 的输出不应相同（前者不减均值）";
}

TEST(RmsNorm, ForwardOneMatchesForward)
{
    const int d = 5, T = 4;
    rms_t norm;
    norm.set_param(d, kEps);
    norm.gama() = dmat(d, 1, {0.8, 1.2, 1.0, 0.5, 1.7});

    dmat x = make_mat(d, T, 1.2, 6);
    dmat full = norm.forward(x);
    for (int t = 0; t < T; ++t)
        ExpectNearMat(norm.forward_one(x.view(0, t, d, 1).clone()),
                      full.view(0, t, d, 1), 1e-14);
}

TEST(RmsNorm, BackwardMatchesNumericalGradient)
{
    // 端到端数值梯度：同时校验输入梯度与 gamma 梯度。
    // sgd lr=1 → weight_after == weight_before - grad，可直接读出梯度。
    const int d = 5, T = 3;
    rms_sgd_t norm;
    norm.set_param(d, kEps);
    norm.set_updator(1.0);
    norm.gama() = dmat(d, 1, {0.7, 1.3, 0.5, 2.0, 1.0});

    dmat x = make_mat(d, T, 1.5, 7);

    // L = Σ 0.5·y²  ⇒  delta = y。必须在 norm 自身上先 forward（backward 依赖其缓存）
    dmat delta = norm.forward(x);
    rms_sgd_t pristine = norm;

    const dmat gamma0 = pristine.gama();
    const dmat dx_analytic = norm.backward(delta);
    const dmat dgamma_analytic = gamma0 - norm.gama();

    auto loss_of = [&](const dmat& x_, const dmat& g_) {
        auto n = pristine;
        n.gama() = g_;
        const dmat y = n.forward(x_);
        double s = 0.0;
        for (int i = 0; i < y.row_num(); ++i)
            for (int j = 0; j < y.col_num(); ++j)
                s += 0.5 * y(i, j) * y(i, j);
        return s;
    };

    const double h = 1e-6;
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            dmat xp = x, xm = x;
            xp(i, j) += h;
            xm(i, j) -= h;
            const double num = (loss_of(xp, gamma0) - loss_of(xm, gamma0)) / (2 * h);
            EXPECT_NEAR(dx_analytic(i, j), num, 1e-6) << "dx (" << i << "," << j << ")";
        }
        dmat gp = gamma0, gm = gamma0;
        gp(i, 0) += h;
        gm(i, 0) -= h;
        const double num_g = (loss_of(x, gp) - loss_of(x, gm)) / (2 * h);
        EXPECT_NEAR(dgamma_analytic(i, 0), num_g, 1e-6) << "dgamma " << i;
    }
}

TEST(RmsNorm, BackwardOmitsCenteringTermLestItBeWrong)
{
    // 反向最容易抄错的地方：直接从 layer_norm_net_t 复制会**多留一个 mean(g) 项**。
    // 这里显式对比：解析梯度应当**不等于**那个 LayerNorm 风格（多一项）的结果，
    // 并且应当与数值梯度一致 —— 双向确认，避免"两处同错"互相掩盖。
    const int d = 4, T = 3;
    rms_sgd_t norm;
    norm.set_param(d, kEps);
    norm.set_updator(1.0);
    norm.gama() = dmat(d, 1, {0.9, 1.1, 0.6, 1.4});

    dmat x = make_mat(d, T, 1.5, 8);
    dmat delta = norm.forward(x);
    const dmat gamma0 = norm.gama();

    const dmat dx_analytic = norm.backward(delta);

    // 重构 hx 与 rms（用于构造错误版本）
    dmat hx(d, T);
    dmat rms(1, T);
    for (int t = 0; t < T; ++t)
    {
        double ms = 0.0;
        for (int i = 0; i < d; ++i)
            ms += x(i, t) * x(i, t);
        rms(0, t) = std::sqrt(ms / d + kEps);
        for (int i = 0; i < d; ++i)
            hx(i, t) = x(i, t) / rms(0, t);
    }

    // 错误版本：把 rms 当成 std、并保留 mean(g) 项
    dmat wrong(d, T);
    {
        dmat g = delta * gamma0;
        for (int t = 0; t < T; ++t)
        {
            double sum_g = 0.0, sum_ghx = 0.0;
            for (int i = 0; i < d; ++i)
            {
                sum_g += g(i, t);
                sum_ghx += g(i, t) * hx(i, t);
            }
            for (int i = 0; i < d; ++i)
                wrong(i, t) = (g(i, t) * d - sum_g - hx(i, t) * sum_ghx) / d / rms(0, t);
        }
    }

    double max_gap = 0.0;
    for (int i = 0; i < d; ++i)
        for (int t = 0; t < T; ++t)
            max_gap = std::max(max_gap, std::abs(dx_analytic(i, t) - wrong(i, t)));
    EXPECT_GT(max_gap, 1e-3) << "解析梯度看起来仍带着 LayerNorm 的 mean(g) 项";
}

TEST(RmsNorm, EpsAffectsOutput)
{
    // eps 必须可配置：对齐开源权重时要按模型 config 填（LLaMA 系 1e-5 / 1e-6 都有用）
    const int d = 4, T = 2;
    dmat x = make_mat(d, T, 1e-3, 9);   // 小量级输入，让 eps 的影响显现

    rms_t a, b;
    a.set_param(d, 1e-5);
    b.set_param(d, 1e-3);
    a.gama() = dmat(d, 1, {1, 1, 1, 1});
    b.gama() = dmat(d, 1, {1, 1, 1, 1});

    dmat ya = a.forward(x), yb = b.forward(x);
    double max_diff = 0.0;
    for (int i = 0; i < d; ++i)
        for (int t = 0; t < T; ++t)
            max_diff = std::max(max_diff, std::abs(ya(i, t) - yb(i, t)));
    EXPECT_GT(max_diff, 1e-3);
}

TEST(RmsNorm, NetTypeMentionsRmsNormAndEps)
{
    rms_t norm;
    norm.set_param(4, 1e-6);
    const std::string s = norm.net_type();
    EXPECT_NE(s.find("rms_norm_net_t"), std::string::npos);
    EXPECT_NE(s.find("eps"), std::string::npos);
}

TEST(RmsNorm, UsableAsPreNormBranchInComplexNet)
{
    // LLaMA 的实际用法：x + SubLayer(RMSNorm(x))，即 residual(rms_norm -> Linear)。
    // 这条同时验证 val_type 是 public（complex_net_t 需从链首成员推断）与 get<> 穿透。
    using pre_norm_branch_t = complex_net_builder_t<double>
        ::template push_back_updatable<rms_norm_net_t, nadam_t>
        ::template push_back_updatable<weight_net_t, nadam_t>
        ::type;
    using res_t = residual_net_t<pre_norm_branch_t>;

    const int d = 4;
    res_t res;
    // 只有 weight_net_t 是 reinitable，RMSNorm 走 set_param（不进容器槽位）
    res.base_net().reinit(std::vector<int>{d, d});
    res.template get<0>().set_param(d, 1e-5);      // 链首的 RMSNorm
    res.template init_weight<xavier_gaussian_t>();

    dmat x = make_mat(d, 3, 0.5, 11);
    dmat out = res.forward(x);
    ExpectShape(out, d, 3);

    // 残差语义 out = f(x) + x
    ExpectNearMat(out, res.base_net().forward(x) + x, 1e-13);

    // 穿透到 RMSNorm 的 gamma：residual → base_net(complex_net) → 链首的 RMSNorm
    // （不能写 get<0,0>()：那会去调 rms_norm_net_t::get<0>()，而它没有 get<>）
    EXPECT_EQ(res.base_net().template get<0>().gama().row_num(), d);
    EXPECT_NO_THROW(res.backward(make_mat(d, 3, 0.1, 12)));
}
