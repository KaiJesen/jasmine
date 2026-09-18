/**
 * Unit tests for conv2d_net_t (2-D convolution implemented as im2col + GEMM).
 *
 * Three things are covered:
 *   1. the forward matches a naive convolution reference point by point (including stride / padding /
 *      dilation / 1x1 channel mixing / the 1-D special case);
 *   2. the backward matches finite differences -- all three gradients (input, weight, bias) are
 *      compared, not just the first and last entries;
 *   3. as a member of `complex_net_t` it takes part in forward / backward / step (occupying no reinit
 *      container slot).
 *
 * The reference values need no weight file and no external tool, so these tests always run in CI.
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
/** Plain SGD, so `weight_after == weight_before - grad` exposes the gradient directly */
using conv_sgd_t = conv2d_net_t<dmat, sgd_t>;

/** Deterministic pseudo-random fill: no global RNG, so results are reproducible */
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

/** Naive reference: six nested loops accumulating straight from the definition -- the only judge of the forward */
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

/** One configuration group: keeps the "construct -> compare forward" boilerplate in one place */
struct conv_case
{
    int c_in, c_out, h, w, kh, kw, sh, sw, ph, pw, dh, dw;
    unsigned seed;
};

} // namespace

TEST(Conv2d, ForwardMatchesReference)
{
    // covers: unit stride without padding, stride=2 + padding, dilation=2, non-square input, and a
    // size large enough to take the BLAS fast path (M*N*K >= 16^3).
    const std::vector<conv_case> cases = {
        {2, 3, 3, 4, 2, 2, 1, 1, 0, 0, 1, 1, 11},
        {2, 3, 5, 6, 3, 3, 2, 2, 1, 1, 1, 1, 12},
        {1, 2, 6, 6, 3, 3, 1, 1, 0, 0, 2, 2, 13},
        {3, 4, 4, 7, 2, 3, 1, 2, 0, 0, 1, 1, 14},
        {4, 5, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 15},   // M*N*K = 5*64*36 > 4096, takes the BLAS path
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
    // a 1x1 convolution has no spatial mixing: y = W * x, exactly the shape of weight_net_t
    const int c_in = 3, c_out = 2, h = 2, w = 3;
    conv_t conv(c_in, c_out, h, w, 1, 1);
    conv.weight() = make_mat(c_out, c_in, 0.5, 21);
    conv.bias() = make_mat(c_out, 1, 0.25, 22);
    const dmat x = make_mat(c_in, h * w, 0.5, 23);

    const dmat y = conv.forward(x);
    ExpectShape(y, c_out, h * w);
    const dmat ref = conv.weight().dot(x) + conv.bias();     // a 1x1 kernel is one channel mixing
    ExpectNearMat(y, ref, 1e-12);
}

TEST(Conv2d, Conv1dIsOneRowSpecialCase)
{
    // sequence convolution: H=1, Kh=1, pad_h=0, a length-L sequence fed as [C_in, 1*L]
    const int c_in = 2, c_out = 3, L = 7, k = 3, stride = 2, pad = 1;
    conv_t conv = conv_t::one_d(c_in, c_out, L, k, stride, pad);
    conv.weight() = make_mat(c_out, c_in * k, 0.5, 31);
    conv.bias() = make_mat(c_out, 1, 0.25, 32);

    const dmat x = make_mat(c_in, L, 0.5, 33);
    const dmat y = conv.forward(x);

    // the convenience entry must not change the underlying configuration: H/Kh must be 1, H_out must be 1
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
    // End-to-end numerical gradient: central differences over every input, weight and bias entry,
    // compared with the analytic gradient from backward. The loss is L = sum 0.5*y^2, so dL/dy = y
    // and the forward output serves as delta. With SGD(lr=1) the parameter difference is the
    // gradient. stride=2 + padding + a non-square input are mixed in on purpose: positions dropped
    // by padding and positions skipped by the stride must contribute **exactly zero** gradient --
    // the place where col2im is easiest to get wrong.
    const int c_in = 2, c_out = 3, h = 4, w = 5, kh = 3, kw = 2;
    const int sh = 2, sw = 1, ph = 1, pw = 0;

    conv_sgd_t conv;
    conv.set_param(c_in, c_out, h, w, kh, kw, sh, sw, ph, pw, 1, 1);
    conv.set_updator(1.0);                       // sgd, lr = 1
    conv.weight() = make_mat(c_out, c_in * kh * kw, 0.5, 41);
    conv.bias() = make_mat(c_out, 1, 0.3, 42);
    const dmat x = make_mat(c_in, h * w, 0.5, 43);

    const dmat y = conv.forward(x);

    // snapshot: backward updates weight and bias in place, so the numerical gradient has to start
    // from the same parameter state
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
    // Unit stride, large kernel: one input pixel is shared by several output windows, so dL/dx must
    // **accumulate** rather than be overwritten. With all-ones weights, zero bias and an all-ones
    // delta the analytic gradient is simply the number of windows the pixel takes part in -- an
    // overlap count that any "write instead of add" implementation fails.
    const int c_in = 1, c_out = 1, h = 3, w = 3, k = 2;
    conv_sgd_t conv(c_in, c_out, h, w, k, k);
    conv.weight() = 1.0;
    conv.bias() = 0.0;

    dmat x(1, h * w);
    x = 1.0;
    conv.forward(x);
    const dmat dx = conv.backward(dmat(c_out, (h - k + 1) * (w - k + 1)) = 1.0);

    // a 2x2 kernel on a 3x3 input: corners once, edge middles twice, centre four times
    ExpectNearMat(dx, dmat(1, 9, {1, 2, 1, 2, 4, 2, 1, 2, 1}), 1e-12);
}

TEST(Conv2d, DegenerateShapesStillWork)
{
    // Degenerating to "one channel + 1x1 input + 1x1 kernel": mat_t treats the weight, bias and
    // im2col all as scalar matrices. This path exists to pin the scalar branch down -- it must
    // neither crash (reshape(1,1) reads (0,0) of an empty matrix) nor compute the gradient wrongly,
    // and a single output channel is very common in real networks.
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
    // C_in=1, no convolution along the rows, width windows non-overlapping by kernel size -- here col
    // is a reinterpretation of the input itself (patchify / ViT patch embedding / non-overlapping
    // 1-D convolution) and the layer has to take the zero-copy view path.
    const int c_out = 4, h = 2, w = 12, kh = 1, kw = 3;
    conv_t conv(/*c_in=*/1, c_out, h, w, kh, kw, /*sh=*/1, /*sw=*/kw, /*ph=*/0, /*pw=*/0);
    EXPECT_TRUE(conv.col_view_enabled());

    // only with a single channel: with several channels memory is channel-major and cannot be folded into one matrix
    conv_t multi(2, c_out, h, w, kh, kw, 1, kw, 0, 0);
    EXPECT_FALSE(multi.col_view_enabled());

    // overlapping windows (stride < kw) do not qualify
    conv_t overlap(1, c_out, h, w, 1, kw, 1, 1, 0, 0);
    EXPECT_FALSE(overlap.col_view_enabled());

    // a kernel height other than 1 (kh > 1) does not qualify either
    conv_t tall(1, c_out, h, w, 2, kw, 1, kw, 0, 0);
    EXPECT_FALSE(tall.col_view_enabled());

    conv.weight() = make_mat(c_out, kw, 0.5, 91);
    conv.bias() = make_mat(c_out, 1, 0.25, 92);
    const dmat x = make_mat(1, h * w, 0.5, 93);

    const dmat y = conv.forward(x);
    const dmat ref = reference_conv(conv.weight(), conv.bias(), x, h, w, kh, kw,
                                    1, kw, 0, 0, 1, 1);
    ExpectNearMat(y, ref, 1e-12);

    // compute it explicitly through the view semantics to confirm which identity is used
    const int n = (h - kh + 1) * ((w - kw) / kw + 1);
    dmat col(kw, n);
    for (int oh = 0; oh < h - kh + 1; ++oh)
        for (int ow = 0; ow < w / kw; ++ow)
            for (int j = 0; j < kw; ++j)
                col(j, oh * (w / kw) + ow) = x(0, oh * w + ow * kw + j);
    dmat ref_col = conv.weight().dot(col) + conv.bias();
    ExpectNearMat(y, ref_col, 1e-12);

    // backward: dW comes from the view as well (delta * col^T = delta * windows)
    conv_sgd_t sgd_conv(1, c_out, h, w, kh, kw, 1, kw, 0, 0);
    sgd_conv.weight() = conv.weight();
    sgd_conv.bias() = conv.bias();
    sgd_conv.set_updator(1.0);
    const dmat w_before = sgd_conv.weight();
    const dmat delta = make_mat(c_out, n, 0.5, 94);
    sgd_conv.forward(x);                       // the view path's backward relies on the cached input
    const dmat dx = sgd_conv.backward(delta);
    ExpectShape(dx, 1, h * w);

    // naive reference gradient
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

    // the view path agrees with the generic path (equivalent geometry, same weights, multi-channel config)
    conv_sgd_t generic(1, c_out, h, w, kh, kw, 1, kw, 0, 0);
    generic.set_param(1, c_out, h, w, kh, kw, 1, kw, 0, 0);
    generic.weight() = conv.weight();
    generic.bias() = conv.bias();
    ExpectNearMat(generic.forward(x), y, 1e-12);
}

TEST(Conv2d, OneDZeroCopyMatchesReference)
{
    // non-overlapping 1-D convolution (H=1): the one_d entry should reach the zero-copy path too
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

    // forward: must be [C_in, H*W]
    EXPECT_THROW(conv.forward(make_mat(3, 9, 0.1, 52)), std::invalid_argument);
    EXPECT_THROW(conv.forward(make_mat(2, 8, 0.1, 53)), std::invalid_argument);

    // backward: must be [C_out, H_out*W_out] = [3, 2*2]
    EXPECT_NO_THROW(conv.backward(make_mat(3, 4, 0.1, 54)));
    EXPECT_THROW(conv.backward(make_mat(2, 4, 0.1, 55)), std::runtime_error);
    EXPECT_THROW(conv.backward(make_mat(3, 5, 0.1, 56)), std::runtime_error);
}

TEST(Conv2d, UnconfiguredLayerFailsFast)
{
    conv_sgd_t conv;
    EXPECT_THROW(conv.forward(make_mat(1, 4, 0.1, 61)), std::runtime_error);
    EXPECT_THROW(conv.backward(make_mat(1, 4, 0.1, 62)), std::runtime_error);
    // usable immediately after set_param (weight and bias are 0)
    conv.set_param(1, 1, 2, 2, 1, 1);
    ExpectNearMat(conv.forward(make_mat(1, 4, 0.5, 63)), dmat(1, 4) = 0.0, 1e-12);
}

TEST(Conv2d, SetParamRejectsInvalidConfig)
{
    conv_sgd_t conv;
    EXPECT_THROW(conv.set_param(0, 1, 3, 3, 1, 1), std::invalid_argument);   // 0 channels
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 0, 1), std::invalid_argument);   // 0 kernel
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 2, 2, 0, 1), std::invalid_argument);  // 0 stride
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 2, 2, 1, 1, -1, 0), std::invalid_argument); // negative pad
    EXPECT_THROW(conv.set_param(1, 1, 3, 3, 4, 4), std::invalid_argument);   // kernel larger than input
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
    EXPECT_GT(norm, 0.0) << "the weights must not be all zero after init_weight";
}

TEST(Conv2d, IsUpdatableButNotReinitDetectedByConcept)
{
    // set_param rather than reinit: this layer's shape comes from its own configuration, so it
    // should not occupy a complex_net_t::reinit container slot
    static_assert(is_updatable_net<conv_t>);
    static_assert(!is_reinitable_net<conv_t>);
    SUCCEED();
}

TEST(Conv2d, ChainInsideComplexNet)
{
    // conv -> relu as one complex_net_t chain: forward yields a shape, backward returns one, step survives
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
    // Sized well beyond the BLAS/blocked GEMM threshold (M*N*K far above it), so both
    // implementation paths must agree: the analytic gradient matches the naive reference.
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

    // the naive reference backward: differentiate the definition directly, accumulating dL/dx per point
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
    EXPECT_NE(s.find("out:3x4x5"), std::string::npos);   // 3x3 + pad 1 preserves the size
}
