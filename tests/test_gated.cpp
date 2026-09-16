/**
 * gated_net_t 门控容器单测。
 *
 * 容器语义：out = gate_branch(x) ⊙ up_branch(x)（逐元素乘，非矩阵乘）。
 * 它是 SwiGLU 的中间部分，down_proj 接在容器后面。
 *
 * 数值基准用 PyTorch 语义：y = silu(gate(x)) ⊙ up(x)。
 * 不依赖任何权重文件，所以这些测试在 CI 里始终会跑。
 */

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_mat_concepts.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_net_t.hpp"
#include "jas_silu_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

template<typename val_type>
using upr_tpl = cache_updator_t<val_type, nadam_t>;

using dmat = mat_t<double>;

/** SwiGLU 的两条分支：gate = Linear→SiLU，up = Linear */
template<typename val_type>
using swiglu_branches_t = gated_ffn_branches_t<val_type, upr_tpl, silu_net_t>;

/** 同上但用纯 SGD，便于让 `weight_after == weight_before - grad` 直接读出梯度 */
template<typename val_type>
using swiglu_sgd_t = gated_ffn_branches_t<val_type, sgd_t, silu_net_t>;

/** 确定性伪随机填充：不依赖全局随机引擎，保证可复现 */
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

/** 朴素参考实现：out = silu(gate(x)) ⊙ up(x)，逐元素三重大小循环 */
dmat reference_forward(const dmat& gate_w, const dmat& gate_b,
                       const dmat& up_w, const dmat& up_b,
                       const dmat& x)
{
    const int d_ff = gate_w.row_num();
    const int T = x.col_num();
    dmat out(d_ff, T);
    for (int i = 0; i < d_ff; ++i)
        for (int t = 0; t < T; ++t)
        {
            double g = gate_b(i, 0), u = up_b(i, 0);
            for (int j = 0; j < x.row_num(); ++j)
            {
                g += gate_w(i, j) * x(j, t);
                u += up_w(i, j) * x(j, t);
            }
            out(i, t) = silu_net_t<dmat>::silu(g) * u;     // 逐元素乘
        }
    return out;
}

} // namespace

TEST(Gated, ForwardMatchesReference)
{
    // 手工填权重，避免依赖随机初始化，逐元素与朴素参考实现比对
    const int d_model = 4, d_ff = 6, T = 3;
    swiglu_branches_t<double> gated;
    gated.reinit(std::vector<int>{d_model, d_ff});

    auto& gate_lin = gated.gate_branch().template get<0>();   // Linear
    auto& up_lin = gated.up_branch();                         // Linear
    auto& gw = gate_lin.weight();
    auto& gb = gate_lin.bias();
    auto& uw = up_lin.weight();
    auto& ub = up_lin.bias();

    for (int i = 0; i < d_ff; ++i)
    {
        gb(i, 0) = 0.1 * i - 0.3;
        ub(i, 0) = -0.05 * i + 0.2;
        for (int j = 0; j < d_model; ++j)
        {
            gw(i, j) = 0.13 * (i + 1) - 0.07 * j;
            uw(i, j) = -0.09 * i + 0.11 * (j + 1);
        }
    }

    dmat x(d_model, T);
    for (int j = 0; j < d_model; ++j)
        for (int t = 0; t < T; ++t)
            x(j, t) = 0.3 * j - 0.2 * t + 0.05 * (j * t);

    dmat out = gated.forward(x);
    ExpectShape(out, d_ff, T);
    ExpectNearMat(out, reference_forward(gw, gb, uw, ub, x), 1e-12);
}

TEST(Gated, ForwardIsElementwiseProductNotMatmul)
{
    // 两分支都退化成恒等映射（权重为单位阵、bias=0），此时
    //     out = silu(x) ⊙ x      （逐元素）
    // 若误用矩阵乘，结果会是 silu(x) · x（矩阵积），两者在方阵上都能算出来但数值不同。
    const int n = 3;
    swiglu_branches_t<double> gated;
    gated.reinit(std::vector<int>{n, n});

    auto& gate_lin = gated.gate_branch().template get<0>();
    auto& up_lin = gated.up_branch();
    gate_lin.weight() = 0.0;
    gate_lin.bias() = 0.0;
    up_lin.weight() = 0.0;
    up_lin.bias() = 0.0;
    for (int i = 0; i < n; ++i)
    {
        gate_lin.weight()(i, i) = 1.0;
        up_lin.weight()(i, i) = 1.0;
    }

    dmat x(n, n, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    dmat out = gated.forward(x);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const double xv = x(i, j);
            EXPECT_NEAR(out(i, j), silu_net_t<dmat>::silu(xv) * xv, 1e-12)
                << "at (" << i << "," << j << ")";
        }

    // 明确排除"矩阵乘"读法：silu(x)·x 的 (0,0) 元素是 Σ_k silu(x)_{0k} x_{k0}
    double matmul_00 = 0.0;
    for (int k = 0; k < n; ++k)
        matmul_00 += silu_net_t<dmat>::silu(x(0, k)) * x(k, 0);
    EXPECT_GT(std::abs(out(0, 0) - matmul_00), 1e-6)
        << "out 看起来像矩阵乘，说明合并算子用错了";
}

TEST(Gated, BackwardMatchesNumericalGradient)
{
    // 端到端数值梯度：对 gate/up 权重逐项做有限差分，与 backward 回传的梯度比对。
    // 这同时验证了容器反向里的「乘对方」以及两分支梯度相加。
    // 用 SGD(lr=1) 是为了让 weight_after == weight_before - grad，能直接读出梯度。
    const int d_model = 3, d_ff = 4, T = 2;

    swiglu_sgd_t<double> gated;
    gated.reinit(std::vector<int>{d_model, d_ff});
    gated.set_updator(1.0);            // sgd, lr = 1

    auto& gate_lin = gated.gate_branch().template get<0>();
    auto& up_lin = gated.up_branch();
    gate_lin.weight() = make_mat(d_ff, d_model, 0.5, 100);
    gate_lin.bias() = make_mat(d_ff, 1, 0.3, 101);
    up_lin.weight() = make_mat(d_ff, d_model, 0.5, 102);
    up_lin.bias() = make_mat(d_ff, 1, 0.3, 103);

    dmat x = make_mat(d_model, T, 0.5, 104);

    // 用平方和损失 L = Σ 0.5·out²，则 delta = ∂L/∂out = out
    const dmat delta = gated.forward(x);

    // 快照：backward 会**原地**改动权重与偏置（weight_net_t 对 bias 也做 update），
    // 数值梯度必须基于同一份参数状态，否则偏置漂移会污染权重的梯度比对。
    const swiglu_sgd_t<double> pristine = gated;

    const dmat gate_w_before = pristine.gate_branch().template get<0>().weight();
    const dmat up_w_before = pristine.up_branch().weight();

    gated.backward(delta);

    const dmat gate_grad = gate_w_before - gated.gate_branch().template get<0>().weight();
    const dmat up_grad = up_w_before - gated.up_branch().weight();

    // 数值梯度：从快照复制（偏置等非受扰参数与解析梯度时的状态一致）
    auto loss_of = [&](const dmat& gw, const dmat& uw) {
        auto g = pristine;
        g.gate_branch().template get<0>().weight() = gw;
        g.up_branch().weight() = uw;
        const dmat o = g.forward(x);
        double s = 0.0;
        for (int i = 0; i < o.row_num(); ++i)
            for (int j = 0; j < o.col_num(); ++j)
                s += 0.5 * o(i, j) * o(i, j);
        return s;
    };

    const double h = 1e-6;
    for (int i = 0; i < d_ff; ++i)
    {
        for (int j = 0; j < d_model; ++j)
        {
            dmat gp = gate_w_before, gm = gate_w_before;
            gp(i, j) += h; gm(i, j) -= h;
            const double num_gate = (loss_of(gp, up_w_before) - loss_of(gm, up_w_before)) / (2 * h);

            dmat up_p = up_w_before, up_m = up_w_before;
            up_p(i, j) += h; up_m(i, j) -= h;
            const double num_up = (loss_of(gate_w_before, up_p) - loss_of(gate_w_before, up_m)) / (2 * h);

            EXPECT_NEAR(gate_grad(i, j), num_gate, 1e-6)
                << "gate weight (" << i << "," << j << ")";
            EXPECT_NEAR(up_grad(i, j), num_up, 1e-6)
                << "up weight (" << i << "," << j << ")";
        }
    }
}

TEST(Gated, BackwardRejectsWrongShape)
{
    swiglu_branches_t<double> gated;
    gated.reinit(std::vector<int>{3, 4});
    gated.forward(dmat(3, 2, {1, 2, 3, 4, 5, 6}));
    // 正确形状：d_ff×T = 4×2
    EXPECT_NO_THROW(gated.backward(dmat(4, 2, {1, 1, 1, 1, 1, 1, 1, 1})));
    // 行数按 d_model 而非 d_ff
    EXPECT_THROW(gated.backward(dmat(3, 2, {1, 1, 1, 1, 1, 1})), std::runtime_error);
    // 列数不符
    EXPECT_THROW(gated.backward(dmat(4, 3, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})),
                 std::runtime_error);
}

TEST(Gated, ForwardOneMatchesForward)
{
    // 推理路径（KV cache 逐列解码）走 forward_one，必须与整段 forward 一致
    const int d_model = 3, d_ff = 5;
    swiglu_branches_t<double> gated;
    gated.reinit(std::vector<int>{d_model, d_ff});
    gated.template init_weight<xavier_gaussian_t>();

    dmat x = make_mat(d_model, 4, 0.5, 999);
    dmat full = gated.forward(x);

    for (int t = 0; t < x.col_num(); ++t)
    {
        dmat one = gated.forward_one(x.view(0, t, x.row_num(), 1));
        ExpectShape(one, d_ff, 1);
        for (int i = 0; i < d_ff; ++i)
            EXPECT_NEAR(one(i, 0), full(i, t), 1e-12) << "i=" << i << " t=" << t;
    }
}

TEST(Gated, ReinitShapesBothBranches)
{
    // gate 与 up 形状相同，一份 {in, out} 应同时配置好两个分支
    const int d_model = 7, d_ff = 11;
    swiglu_branches_t<double> gated;
    gated.reinit(std::vector<int>{d_model, d_ff});

    auto& gate_lin = gated.gate_branch().template get<0>();
    EXPECT_EQ(gate_lin.weight().row_num(), d_ff);
    EXPECT_EQ(gate_lin.weight().col_num(), d_model);
    EXPECT_EQ(gate_lin.bias().row_num(), d_ff);
    EXPECT_EQ(gated.up_branch().weight().row_num(), d_ff);
    EXPECT_EQ(gated.up_branch().weight().col_num(), d_model);
    EXPECT_EQ(gated.up_branch().bias().row_num(), d_ff);
}

TEST(Gated, ReinitIsDetectedByConcept)
{
    // gated_net_t 含可训练分支时必须被 is_reinitable_net 判定为真，
    // 否则 complex_net_t::reinit 会跳过它、权重永远不被配置形状。
    EXPECT_TRUE(is_reinitable_net<swiglu_branches_t<double>>);
    EXPECT_TRUE(is_updatable_net<swiglu_branches_t<double>>);
}

TEST(Gated, NetTypeListsBothBranches)
{
    swiglu_branches_t<double> gated;
    const std::string s = gated.net_type();
    EXPECT_NE(s.find("gated_net_t"), std::string::npos);
    EXPECT_NE(s.find("gate"), std::string::npos);
    EXPECT_NE(s.find("up"), std::string::npos);
    EXPECT_NE(s.find("silu_net_t"), std::string::npos);
}

TEST(Gated, ChainWithDownProjectionInsideComplexNet)
{
    // 完整 SwiGLU FFN：gated(gate=Linear→SiLU, up=Linear) → Linear(down)
    // 验证容器能作为 complex_net 的一层参与 reinit / forward / backward / step
    using swiglu_ffn_t = complex_net_builder_t<double>
        ::template push_back_impl<swiglu_branches_t<double>>
        ::template push_back_updatable<weight_net_t, upr_tpl>
        ::type;

    const int d_model = 4, d_ff = 6, T = 3;
    swiglu_ffn_t ffn;
    // 容器吃掉 {d_model, d_ff}，down_proj 再吃掉 {d_ff, d_model}
    ffn.reinit(std::vector<int>{d_model, d_ff, d_model});
    ffn.template init_weight<xavier_gaussian_t>();

    EXPECT_EQ(ffn.template get<1>().weight().row_num(), d_model);
    EXPECT_EQ(ffn.template get<1>().weight().col_num(), d_ff);

    dmat x = make_mat(d_model, T, 0.5, 7);
    dmat out = ffn.forward(x);
    ExpectShape(out, d_model, T);
    for (int i = 0; i < out.row_num(); ++i)
        for (int j = 0; j < out.col_num(); ++j)
            EXPECT_FALSE(std::isnan(out(i, j)));

    EXPECT_NO_THROW(ffn.backward(make_mat(d_model, T, 0.1, 8)));
    ffn.step();

    // get<> 穿透：get<0,0> = 容器的 gate 分支（complex_net），get<0,0,0> = 其中的 Linear
    EXPECT_EQ((ffn.template get<0, 0, 0>().weight().row_num()), d_ff);
    EXPECT_EQ((ffn.template get<0, 1>().weight().row_num()), d_ff);
}

TEST(Gated, WorksAsResidualWrappedBlock)
{
    // LLaMA 的实际形态：residual(gated → down)，模型级访问器要能穿透 residual 与容器
    using swiglu_ffn_t = complex_net_builder_t<double>
        ::template push_back_impl<swiglu_branches_t<double>>
        ::template push_back_updatable<weight_net_t, upr_tpl>
        ::type;
    using res_t = residual_net_t<swiglu_ffn_t>;

    const int d_model = 3, d_ff = 5;
    res_t res;
    res.base_net().reinit(std::vector<int>{d_model, d_ff, d_model});
    res.template init_weight<xavier_gaussian_t>();

    dmat x = make_mat(d_model, 2, 0.5, 11);
    dmat out = res.forward(x);
    ExpectShape(out, d_model, 2);

    // 残差语义：out = f(x) + x
    dmat fx = res.base_net().forward(x);
    ExpectNearMat(out, fx + x, 1e-12);

    // 穿透访问器：residual → complex_net<0> = gated → <1> = up 分支
    EXPECT_EQ((res.template get<0, 1>().weight().row_num()), d_ff);
    EXPECT_NO_THROW(res.backward(make_mat(d_model, 2, 0.1, 12)));
}

TEST(Gated, EndToEndMatchesPyTorchSwiglu)
{
    // 完整 SwiGLU FFN 的三层结构逐元素比对（基准由 PyTorch 计算，见下方常量）：
    //     y = down( silu(gate(x)) ⊙ up(x) )
    // 这里固定小规模权重并硬编码 PyTorch 的结果，作为跨实现的一致性锚点。
    const int d_model = 2, d_ff = 3, T = 2;

    swiglu_branches_t<double> gated;
    gated.reinit(std::vector<int>{d_model, d_ff});
    weight_net_t<dmat, upr_tpl> down;
    down.reinit(std::vector<int>{d_ff, d_model});

    auto& gate_lin = gated.gate_branch().template get<0>();
    // 权重取简单整数/半整数，便于人工核对
    gate_lin.weight() = dmat(d_ff, d_model, {1.0, 0.0, 0.0, 1.0, 0.5, -0.5});
    gate_lin.bias() = dmat(d_ff, 1, {0.1, -0.2, 0.3});
    gated.up_branch().weight() = dmat(d_ff, d_model, {0.0, 1.0, 1.0, 0.0, -0.5, 0.5});
    gated.up_branch().bias() = dmat(d_ff, 1, {0.05, 0.15, -0.25});
    down.weight() = dmat(d_model, d_ff, {1.0, -1.0, 0.5, 0.25, 1.5, -0.75});
    down.bias() = dmat(d_model, 1, {0.01, -0.02});

    dmat x(d_model, T, {1.0, -1.0, 2.0, 0.5});

    dmat inter = gated.forward(x);       // d_ff×T
    dmat y = down.forward(inter);        // d_model×T
    ExpectShape(inter, d_ff, T);
    ExpectShape(y, d_model, T);

    // gate 分支：gate(x) = W_gate·x + b_gate
    //   t=0: [1·1+0·2+0.1, 0·1+1·2-0.2, 0.5·1-0.5·2+0.3] = [1.1, 1.8, -0.2]
    //   t=1: [1·(-1)+0.1,   1·0.5-0.2,   0.5·(-1)-0.5·0.5+0.3] = [-0.9, 0.3, -0.45]
    // up 分支：up(x) = W_up·x + b_up
    //   t=0: [0·1+1·2+0.05, 1·1+0·2+0.15, -0.5·1+0.5·2-0.25] = [2.05, 1.15, 0.25]
    //   t=1: [1·0.5+0.05, 1·(-1)+0.15, -0.5·(-1)+0.5·0.5-0.25] = [0.55, -0.85, 0.5]
    const double eg0[3] = {1.1, 1.8, -0.2}, eg1[3] = {-0.9, 0.3, -0.45};
    const double eu0[3] = {2.05, 1.15, 0.25}, eu1[3] = {0.55, -0.85, 0.5};
    for (int i = 0; i < d_ff; ++i)
    {
        EXPECT_NEAR(inter(i, 0), silu_net_t<dmat>::silu(eg0[i]) * eu0[i], 1e-12) << "i=" << i;
        EXPECT_NEAR(inter(i, 1), silu_net_t<dmat>::silu(eg1[i]) * eu1[i], 1e-12) << "i=" << i;
    }

    // down 投影：y = W_down·inter + b_down，逐项手算
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < d_model; ++r)
        {
            double expect = down.bias()(r, 0);
            for (int i = 0; i < d_ff; ++i)
                expect += down.weight()(r, i) * inter(i, t);
            EXPECT_NEAR(y(r, t), expect, 1e-12) << "r=" << r << " t=" << t;
        }
}
