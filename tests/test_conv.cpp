/**
 * conv2d_net_t（im2col + GEMM 的二维卷积）单测。
 *
 * 覆盖三件事：
 *   1. 前向与朴素卷积参考实现逐点一致（含 stride / padding / dilation / 1x1 通道混合 / 一维特例）；
 *   2. 反向与有限差分一致——输入、权重、偏置三份梯度都要对拍，不能只钉头尾；
 *   3. 作为 `complex_net_t` 的一层参与 forward / backward / step（不占用 reinit 的容器槽位）。
 *
 * 参考值不依赖任何权重文件或外部工具，CI 里始终会跑。
 */

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_conv_t.hpp"
#include "jas_mat_concepts.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

using dmat = mat_t<double>;
using conv_t = conv2d_net_t<dmat, nadam_t>;
/** 纯 SGD 便于让 `weight_after == weight_before - grad` 直接读出梯度 */
using conv_sgd_t = conv2d_net_t<dmat, sgd_t>;

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

/** 朴素参考实现：六重循环直接按定义累加，作为前向的唯一裁判 */
dmat reference_conv(const dmat& w, const dmat& b, const dmat& x,
                    int h, int w_in, int kh, int kw,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w)
{
    const int c_in = x.row_num();
    const int c_out = w.row_num();
    const int h_out = (h + 2 * pad_h - dil_h * (kh - 1) - 1) / stride_h + 1;
    const int w_out = (w_in + 2 * pad_w - dil_w * (kw - 1) - 1) / stride_w + 1;

    dmat y(c_out, h_out * w_out);
    for (int o = 0; o < c_out; ++o)
    {
        for (int oh = 0; oh < h_out; ++oh)
        {
            for (int ow = 0; ow < w_out; ++ow)
            {
                double s = b(o, 0);
                for (int c = 0; c < c_in; ++c)
                {
                    for (int i = 0; i < kh; ++i)
                    {
                        for (int j = 0; j < kw; ++j)
                        {
                            const int ih = oh * stride_h - pad_h + i * dil_h;
                            const int iw = ow * stride_w - pad_w + j * dil_w;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w_in)
                                s += w(o, (c * kh + i) * kw + j) * x(c, ih * w_in + iw);
                        }
                    }
                }
                y(o, oh * w_out + ow) = s;
            }
        }
    }
    return y;
}

/** 一组配置：把「构造 -> 前向对拍」的样板收在一处 */
struct conv_case
{
    int c_in, c_out, h, w, kh, kw, sh, sw, ph, pw, dh, dw;
    unsigned seed;
};

} // namespace

TEST(Conv2d, ForwardMatchesReference)
{
    // 覆盖：无 padding 单位步长、stride=2 + padding、dilation=2、非方输入、以及
    // 大到会走 BLAS 快速路径的尺寸（M*N*K >= 16^3）。
    const std::vector<conv_case> cases = {
        {2, 3, 3, 4, 2, 2, 1, 1, 0, 0, 1, 1, 11},
        {2, 3, 5, 6, 3, 3, 2, 2, 1, 1, 1, 1, 12},
        {1, 2, 6, 6, 3, 3, 1, 1, 0, 0, 2, 2, 13},
        {3, 4, 4, 7, 2, 3, 1, 2, 0, 0, 1, 1, 14},
        {4, 5, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 15},   // M*N*K = 5*64*36 > 4096，走 BLAS
    };

    for (const auto& c : cases)
    {
        conv_t conv(c.c_in, c.c_out, c.h, c.w, c.kh, c.kw,
                    c.sh, c.sw, c.ph, c.pw, c.dh, c.dw);
        conv.weight() = make_mat(c.c_out, c.c_in * c.kh * c.kw, 0.5, c.seed);
        conv.bias() = make_mat(c.c_out, 1, 0.25, c.seed + 1);
        const dmat x = make_mat(c.c_in, c.h * c.w, 0.5, c.seed + 2);

        const dmat y = conv.forward(x);
        const dmat ref = reference_conv(conv.weight(), conv.bias(), x,
                                        c.h, c.w, c.kh, c.kw,
                                        c.sh, c.sw, c.ph, c.pw, c.dh, c.dw);
        ExpectNearMat(y, ref, 1e-9);
    }
}

TEST(Conv2d, OneByOneIsChannelMix)
{
    // 1x1 卷积没有空间混合：y = W · x，与 weight_net_t 完全同形
    const int c_in = 3, c_out = 2, h = 2, w = 3;
    conv_t conv(c_in, c_out, h, w, 1, 1);
    conv.weight() = make_mat(c_out, c_in, 0.5, 21);
    conv.bias() = make_mat(c_out, 1, 0.25, 22);
    const dmat x = make_mat(c_in, h * w, 0.5, 23);

    const dmat y = conv.forward(x);
    ExpectShape(y, c_out, h * w);
    const dmat ref = conv.weight().dot(x) + conv.bias();     // 1x1 核就是一次通道混合
    ExpectNearMat(y, ref, 1e-12);
}

TEST(Conv2d, Conv1dIsOneRowSpecialCase)
{
    // 序列卷积：H=1、Kh=1、pad_h=0，长度 L 的序列当作 [C_in, 1*L]
    const int c_in = 2, c_out = 3, L = 7, k = 3, stride = 2, pad = 1;
    conv_t conv = conv_t::one_d(c_in, c_out, L, k, stride, pad);
    conv.weight() = make_mat(c_out, c_in * k, 0.5, 31);
    conv.bias() = make_mat(c_out, 1, 0.25, 32);

    const dmat x = make_mat(c_in, L, 0.5, 33);
    const dmat y = conv.forward(x);

    // 便捷入口不能改掉底层配置：H/Kh 必须是 1，H_out 必须是 1
    EXPECT_EQ(conv.in_h(), 1);
    EXPECT_EQ(conv.kernel_h(), 1);
    EXPECT_EQ(conv.out_h(), 1);

    const int l_out = (L + 2 * pad - k) / stride + 1;
    ExpectShape(y, c_out, l_out);
    ExpectNearMat(y, reference_conv(conv.weight(), conv.bias(), x, 1, L, 1, k,
                                    1, stride, 0, pad, 1, 1),
                  1e-12);
}

TEST(Conv2d, BackwardMatchesNumericalGradient)
{
    // 端到端数值梯度：对输入、权重、偏置逐项做中心差分，与 backward 的解析梯度比对。
    // 损失取 L = Σ 0.5·y²，于是 ∂L/∂y = y，直接拿前向输出当 delta。
    // 用 SGD(lr=1)，令 weight_after == weight_before - grad，直接从参数差读出梯度。
    // 故意混入 stride=2 + padding + 非方输入：被 padding 丢弃的位置与跳过的位置
    // 都必须贡献**恰好 0** 的梯度，这是 col2im 最容易写错的地方。
    const int c_in = 2, c_out = 3, h = 4, w = 5, kh = 3, kw = 2;
    const int sh = 2, sw = 1, ph = 1, pw = 0;

    conv_sgd_t conv;
    conv.set_param(c_in, c_out, h, w, kh, kw, sh, sw, ph, pw, 1, 1);
    conv.set_updator(1.0);                       // sgd, lr = 1
    conv.weight() = make_mat(c_out, c_in * kh * kw, 0.5, 41);
    conv.bias() = make_mat(c_out, 1, 0.3, 42);
    const dmat x = make_mat(c_in, h * w, 0.5, 43);

    const dmat y = conv.forward(x);

    // 快照：backward 会原地更新权重与偏置，数值梯度必须基于同一份参数状态
    const conv_sgd_t pristine = conv;
    const dmat w_before = conv.weight();
    const dmat b_before = conv.bias();

    const dmat dx = conv.backward(y);
    ExpectShape(dx, c_in, h * w);

    const dmat grad_w = w_before - conv.weight();
    const dmat grad_b = b_before - conv.bias();

    auto loss_of = [&](const dmat& wp, const dmat& bp, const dmat& xp) {
        conv_sgd_t c = pristine;
        c.weight() = wp;
        c.bias() = bp;
        const dmat o = c.forward(xp);
        double s = 0.0;
        for (int i = 0; i < o.row_num(); ++i)
            for (int j = 0; j < o.col_num(); ++j)
                s += 0.5 * o(i, j) * o(i, j);
        return s;
    };

    const double eps = 1e-6;
    for (int o = 0; o < c_out; ++o)
    {
        for (int idx = 0; idx < c_in * kh * kw; ++idx)
        {
            dmat wp = w_before, wm = w_before;
            wp(o, idx) += eps;
            wm(o, idx) -= eps;
            const double num = (loss_of(wp, b_before, x) - loss_of(wm, b_before, x)) / (2 * eps);
            EXPECT_NEAR(grad_w(o, idx), num, 1e-6) << "weight (" << o << "," << idx << ")";
        }

        dmat bp = b_before, bm = b_before;
        bp(o, 0) += eps;
        bm(o, 0) -= eps;
        const double num_b = (loss_of(w_before, bp, x) - loss_of(w_before, bm, x)) / (2 * eps);
        EXPECT_NEAR(grad_b(o, 0), num_b, 1e-6) << "bias (" << o << ")";
    }

    for (int c = 0; c < c_in; ++c)
    {
        for (int idx = 0; idx < h * w; ++idx)
        {
            dmat xp = x, xm = x;
            xp(c, idx) += eps;
            xm(c, idx) -= eps;
            const double num = (loss_of(w_before, b_before, xp) - loss_of(w_before, b_before, xm)) / (2 * eps);
            EXPECT_NEAR(dx(c, idx), num, 1e-6) << "input (" << c << "," << idx << ")";
        }
    }
}

TEST(Conv2d, BackwardAccumulatesOverlappingPatches)
{
    // 单位步长、大核：同一个输入像素被多个输出窗口共享，∂L/∂x 必须是**累加**而非覆盖。
    // 取全 1 权重 / 0 偏置 / 全 1 delta，解析梯度就等于「该像素参与的窗口数」，
    // 这正是重叠计数，任何"写而不是加"的实现都会在这里露馅。
    const int c_in = 1, c_out = 1, h = 3, w = 3, k = 2;
    conv_sgd_t conv(c_in, c_out, h, w, k, k);
    conv.weight() = 1.0;
    conv.bias() = 0.0;

    dmat x(1, h * w);
    x = 1.0;
    conv.forward(x);
    const dmat dx = conv.backward(dmat(c_out, (h - k + 1) * (w - k + 1)) = 1.0);

    // 2x2 核在 3x3 输入上：四角 1 次、边中 2 次、中心 4 次
    ExpectNearMat(dx, dmat(1, 9, {1, 2, 1, 2, 4, 2, 1, 2, 1}), 1e-12);
}

TEST(Conv2d, DegenerateShapesStillWork)
{
    // 退化到「单通道 + 1x1 输入 + 1x1 核」时，mat_t 会把权重/偏置/im2col 全当成标量矩阵。
    // 这条路径专门用来钉住标量分支：既不能崩（reshape(1,1) 会读空矩阵的 (0,0)），
    // 也不能把梯度算错——C_out==1 的偏置在真实网络里非常常见。
    conv_sgd_t conv(1, 1, 1, 1, 1, 1);
    conv.set_updator(1.0);
    conv.weight() = 2.0;
    conv.bias() = 3.0;

    const dmat x(1, 1, {5.0});
    const dmat y = conv.forward(x);
    ExpectShape(y, 1, 1);
    EXPECT_NEAR(y(0, 0), 2.0 * 5.0 + 3.0, 1e-12);

    // L = 0.5·y²，delta = y = 13；∂L/∂x = delta·w = 26，∂L/∂w = delta·x = 65，∂L/∂b = delta = 13
    const dmat dx = conv.backward(y);
    EXPECT_NEAR(dx(0, 0), 26.0, 1e-12);
    EXPECT_NEAR(2.0 - conv.weight()(0, 0), 65.0, 1e-12);
    EXPECT_NEAR(3.0 - conv.bias()(0, 0), 13.0, 1e-12);
}

TEST(Conv2d, ZeroCopyColViewForPatchifyGeometry)
{
    // C_in=1、行方向不卷、宽度按核大小不重叠 —— 这时 col 就是输入本体的重解释（patchify / ViT
    // patch embedding / 非重叠一维卷积），层必须走零拷贝视图路径。
    const int c_out = 4, h = 2, w = 12, kh = 1, kw = 3;
    conv_t conv(/*c_in=*/1, c_out, h, w, kh, kw, /*sh=*/1, /*sw=*/kw, /*ph=*/0, /*pw=*/0);
    EXPECT_TRUE(conv.col_view_enabled());

    // 只有单通道才成立：多通道时内存是通道优先，折不成一个矩阵
    conv_t multi(2, c_out, h, w, kh, kw, 1, kw, 0, 0);
    EXPECT_FALSE(multi.col_view_enabled());

    // 重叠窗口（stride < kw）不成立
    conv_t overlap(1, c_out, h, w, 1, kw, 1, 1, 0, 0);
    EXPECT_FALSE(overlap.col_view_enabled());

    // 非 1x1 的核高（kh > 1）也不成立
    conv_t tall(1, c_out, h, w, 2, kw, 1, kw, 0, 0);
    EXPECT_FALSE(tall.col_view_enabled());

    conv.weight() = make_mat(c_out, kw, 0.5, 91);
    conv.bias() = make_mat(c_out, 1, 0.25, 92);
    const dmat x = make_mat(1, h * w, 0.5, 93);

    const dmat y = conv.forward(x);
    const dmat ref = reference_conv(conv.weight(), conv.bias(), x, h, w, kh, kw,
                                    1, kw, 0, 0, 1, 1);
    ExpectNearMat(y, ref, 1e-12);

    // 显式按视图语义算一遍，确认走的就是这条等式
    const int n = (h - kh + 1) * ((w - kw) / kw + 1);
    dmat col(kw, n);
    for (int oh = 0; oh < h - kh + 1; ++oh)
        for (int ow = 0; ow < w / kw; ++ow)
            for (int j = 0; j < kw; ++j)
                col(j, oh * (w / kw) + ow) = x(0, oh * w + ow * kw + j);
    dmat ref_col = conv.weight().dot(col) + conv.bias();
    ExpectNearMat(y, ref_col, 1e-12);

    // 反向：dW 同样来自视图（delta · (colᵀ) = delta · windows）
    conv_sgd_t sgd_conv(1, c_out, h, w, kh, kw, 1, kw, 0, 0);
    sgd_conv.weight() = conv.weight();
    sgd_conv.bias() = conv.bias();
    sgd_conv.set_updator(1.0);
    const dmat w_before = sgd_conv.weight();
    const dmat delta = make_mat(c_out, n, 0.5, 94);
    sgd_conv.forward(x);                       // 视图路径的 backward 要靠 forward 缓存的输入
    const dmat dx = sgd_conv.backward(delta);
    ExpectShape(dx, 1, h * w);

    // 朴素参考梯度
    dmat ref_dw(c_out, kw);
    ref_dw = 0.0;
    dmat ref_dx(1, h * w);
    ref_dx = 0.0;
    for (int o = 0; o < c_out; ++o)
        for (int oh = 0; oh < h - kh + 1; ++oh)
            for (int ow = 0; ow < w / kw; ++ow)
            {
                const double d = delta(o, oh * (w / kw) + ow);
                for (int j = 0; j < kw; ++j)
                {
                    ref_dw(o, j) += d * x(0, oh * w + ow * kw + j);
                    ref_dx(0, oh * w + ow * kw + j) += d * w_before(o, j);
                }
            }
    ExpectNearMat(w_before - sgd_conv.weight(), ref_dw, 1e-12);
    ExpectNearMat(dx, ref_dx, 1e-12);

    // 视图路径与通用路径（等价几何、用 multi 配置跑同一份权重）结果一致
    conv_sgd_t generic(1, c_out, h, w, kh, kw, 1, kw, 0, 0);
    generic.set_param(1, c_out, h, w, kh, kw, 1, kw, 0, 0);
    generic.weight() = conv.weight();
    generic.bias() = conv.bias();
    ExpectNearMat(generic.forward(x), y, 1e-12);
}

TEST(Conv2d, OneDZeroCopyMatchesReference)
{
    // 非重叠一维卷积（H=1）—— 便捷入口 one_d 也应吃到零拷贝路径
    const int c_out = 8, L = 60, k = 3, stride = k;
    conv_t conv = conv_t::one_d(1, c_out, L, k, stride, 0);
    EXPECT_TRUE(conv.col_view_enabled());
    conv.weight() = make_mat(c_out, k, 0.5, 95);
    conv.bias() = make_mat(c_out, 1, 0.25, 96);
    const dmat x = make_mat(1, L, 0.5, 97);
    ExpectNearMat(conv.forward(x),
                  reference_conv(conv.weight(), conv.bias(), x, 1, L, 1, k, 1, stride, 0, 0, 1, 1),
                  1e-12);
}

TEST(Conv2d, RejectsBadShapes)
{
    conv_sgd_t conv(2, 3, 3, 3, 2, 2);
    conv.forward(make_mat(2, 9, 0.1, 51));

    // 前向：必须是 [C_in, H*W]
    EXPECT_THROW(conv.forward(make_mat(3, 9, 0.1, 52)), std::invalid_argument);
    EXPECT_THROW(conv.forward(make_mat(2, 8, 0.1, 53)), std::invalid_argument);

    // 反向：必须是 [C_out, H_out*W_out] = [3, 2*2]
    EXPECT_NO_THROW(conv.backward(make_mat(3, 4, 0.1, 54)));
    EXPECT_THROW(conv.backward(make_mat(2, 4, 0.1, 55)), std::runtime_error);
    EXPECT_THROW(conv.backward(make_mat(3, 5, 0.1, 56)), std::runtime_error);
}

TEST(Conv2d, UnconfiguredLayerFailsFast)
{
    conv_sgd_t conv;
    EXPECT_THROW(conv.forward(make_mat(1, 4, 0.1, 61)), std::runtime_error);
    EXPECT_THROW(conv.backward(make_mat(1, 4, 0.1, 62)), std::runtime_error);
    // set_param 之后立即可用（权重/偏置为 0）
    conv.set_param(1, 1, 2, 2, 1, 1);
    ExpectNearMat(conv.forward(make_mat(1, 4, 0.5, 63)), dmat(1, 4) = 0.0, 1e-12);
}

TEST(Conv2d, SetParamRejectsInvalidConfig)
{
    conv_sgd_t conv;
    EXPECT_THROW(conv.set_param(0, 1, 3, 3, 1, 1), std::invalid_argument);   // 通道为 0
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 0, 1), std::invalid_argument);   // 核为 0
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 2, 2, 0, 1), std::invalid_argument);  // stride 为 0
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 2, 2, 1, 1, -1, 0), std::invalid_argument); // 负 pad
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 4, 4), std::invalid_argument);   // 核比输入大
}

TEST(Conv2d, InitWeightFillsConfiguredShapes)
{
    conv_t conv;
    conv.set_param(3, 4, 5, 6, 2, 3);
    conv.init_weight<he_gaussian_t>();

    ExpectShape(conv.weight(), 4, 3 * 2 * 3);
    ExpectShape(conv.bias(), 4, 1);
    double norm = 0.0;
    for (int i = 0; i < conv.weight().row_num(); ++i)
        for (int j = 0; j < conv.weight().col_num(); ++j)
            norm += std::abs(conv.weight()(i, j));
    EXPECT_GT(norm, 0.0) << "init_weight 之后权重不应全为 0";
}

TEST(Conv2d, IsUpdatableButNotReinitDetectedByConcept)
{
    // set_param 而非 reinit：本层形状来自自身配置，不该占用 complex_net_t::reinit 的容器槽位
    static_assert(is_updatable_net<conv_t>);
    static_assert(!is_reinitable_net<conv_t>);
    SUCCEED();
}

TEST(Conv2d, ChainInsideComplexNet)
{
    // conv -> relu 作为 complex_net_t 的一整条链：前向出形状、反向回形状、step 不炸
    using conv_relu_t = complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, sgd_t>
        ::push_back_staticnet<relu_net_t>
        ::type;

    const int c_in = 2, c_out = 3, h = 3, w = 3;
    conv_relu_t net;
    auto& conv = net.get<0>();
    conv.set_param(c_in, c_out, h, w, 2, 2);
    conv.set_updator(0.01);
    conv.weight() = make_mat(c_out, c_in * 2 * 2, 0.5, 71);
    conv.bias() = make_mat(c_out, 1, 0.3, 72);

    const dmat x = make_mat(c_in, h * w, 0.5, 73);
    const dmat y = net.forward(x);
    ExpectShape(y, c_out, 2 * 2);

    const dmat delta = make_mat(c_out, 4, 0.5, 74);
    const dmat dx = net.backward(delta);
    ExpectShape(dx, c_in, h * w);
    EXPECT_NO_THROW(net.step());
    EXPECT_THROW(net.backward(make_mat(c_out, 3, 0.5, 75)), std::runtime_error);
}

TEST(Conv2d, BackwardOnTwelveByTwelveHitsGemmFastPath)
{
    // 尺寸足以走 BLAS/分块 GEMM（M*N*K 远超阈值），确保两条实现路径给出同一结果：
    // 解析梯度与朴素参考实现算出的梯度一致。
    const int c_in = 3, c_out = 4, h = 12, w = 12, k = 3;
    conv_sgd_t conv(c_in, c_out, h, w, k, k);
    conv.set_updator(1.0);
    conv.weight() = make_mat(c_out, c_in * k * k, 0.4, 81);
    conv.bias() = make_mat(c_out, 1, 0.2, 82);
    const dmat x = make_mat(c_in, h * w, 0.4, 83);

    const dmat y = conv.forward(x);
    const dmat w_before = conv.weight();
    const dmat b_before = conv.bias();
    const dmat dx = conv.backward(y);

    // 朴素参考的反向：直接对定义式求导，逐点累加 ∂L/∂x
    const int h_out = h - k + 1, w_out = w - k + 1;
    dmat ref_dx(c_in, h * w);
    ref_dx = 0.0;
    dmat ref_dw(c_out, c_in * k * k);
    ref_dw = 0.0;
    for (int o = 0; o < c_out; ++o)
    {
        for (int oh = 0; oh < h_out; ++oh)
        {
            for (int ow = 0; ow < w_out; ++ow)
            {
                const double d = y(o, oh * w_out + ow);       // ∂L/∂y，L = 0.5·Σy²
                for (int c = 0; c < c_in; ++c)
                {
                    for (int i = 0; i < k; ++i)
                    {
                        for (int j = 0; j < k; ++j)
                        {
                            ref_dw(o, (c * k + i) * k + j) += d * x(c, (oh + i) * w + (ow + j));
                            ref_dx(c, (oh + i) * w + (ow + j)) += d * w_before(o, (c * k + i) * k + j);
                        }
                    }
                }
            }
        }
    }

    ExpectNearMat(dx, ref_dx, 1e-9);
    ExpectNearMat(w_before - conv.weight(), ref_dw, 1e-9);
    dmat ref_db(c_out, 1);
    for (int o = 0; o < c_out; ++o)
    {
        double s = 0.0;
        for (int idx = 0; idx < h_out * w_out; ++idx)
            s += y(o, idx);
        ref_db(o, 0) = s;
    }
    ExpectNearMat(b_before - conv.bias(), ref_db, 1e-9);
}

TEST(Conv2d, NetTypeReportsShapes)
{
    conv_t conv(2, 3, 4, 5, 3, 3, 1, 1, 1, 1);
    const std::string s = conv.net_type();
    EXPECT_NE(s.find("conv2d_net_t"), std::string::npos);
    EXPECT_NE(s.find("in:2x4x5"), std::string::npos);
    EXPECT_NE(s.find("out:3x4x5"), std::string::npos);   // 3x3 + pad 1 保持尺寸
}
