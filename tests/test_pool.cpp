/**
 * Unit tests for pool2d_net_t (max / average pooling).
 *
 * Four things are covered:
 *   1. the forward matches a naive reference point by point (stride / padding / non-square kernel /
 *      several channels / the 1-D special case);
 *   2. the semantics aligned with PyTorch -- ties go to the first maximum, average pooling's
 *      count_include_pad, the output sizes and the padding constraint (the golden values come from
 *      torch 2.9's F.max_pool2d / F.avg_pool2d);
 *   3. the backward matches finite differences, and overlapping windows must accumulate;
 *   4. as a "static layer" it takes part in complex_net_t (no updator, no reinit slot).
 */

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_conv_t.hpp"
#include "jas_mat_concepts.hpp"
#include "jas_net_t.hpp"
#include "jas_pool_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

using dmat = mat_t<double>;
using pool_t = pool2d_net_t<dmat>;

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

/** Naive reference: scan the window per channel and per output position, straight from the definition */
dmat reference_pool(const dmat& x, int h, int w, int kh, int kw,
                    int sh, int sw, int ph, int pw, pool_mode mode, bool include_pad)
{
    const int c = x.row_num();
    const int h_out = (h + 2 * ph - kh) / sh + 1;
    const int w_out = (w + 2 * pw - kw) / sw + 1;

    dmat y(c, h_out * w_out);
    for (int ch = 0; ch < c; ++ch)
    {
        for (int oh = 0; oh < h_out; ++oh)
        {
            for (int ow = 0; ow < w_out; ++ow)
            {
                double acc = (mode == pool_mode::max)
                    ? -std::numeric_limits<double>::infinity()
                    : 0.0;
                int valid = 0;
                for (int i = 0; i < kh; ++i)
                {
                    for (int j = 0; j < kw; ++j)
                    {
                        const int ih = oh * sh - ph + i;
                        const int iw = ow * sw - pw + j;
                        if (ih < 0 || ih >= h || iw < 0 || iw >= w)
                            continue;
                        const double v = x(ch, ih * w + iw);
                        if (mode == pool_mode::max)
                            acc = (v > acc) ? v : acc;
                        else
                            acc += v;
                        ++valid;
                    }
                }
                y(ch, oh * w_out + ow) = (mode == pool_mode::max)
                    ? acc
                    : acc / static_cast<double>(include_pad ? kh * kw : valid);
            }
        }
    }
    return y;
}

/** 1..9 laid out as 1 x 1 x 3 x 3; every PyTorch golden value uses it */
dmat ramp_3x3()
{
    return dmat(1, 9, {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

dmat ones(int rows, int cols)
{
    dmat m(rows, cols);
    m = 1.0;
    return m;
}

} // namespace

TEST(Pool2d, MaxForwardMatchesReference)
{
    // covers: non-overlapping, overlapping, padding, non-square kernel, non-square input, channels
    struct c { int h, w, kh, kw, sh, sw, ph, pw, ch; unsigned seed; };
    const std::vector<c> cases = {
        {3, 3, 2, 2, 2, 2, 0, 0, 1, 11},
        {3, 3, 2, 2, 1, 1, 1, 1, 1, 12},
        {5, 7, 3, 2, 2, 1, 1, 0, 2, 13},
        {6, 6, 3, 3, 2, 2, 1, 1, 3, 14},
        {4, 4, 1, 1, 1, 1, 0, 0, 2, 15},   // 1x1 pooling = identity
    };

    for (const auto& t : cases)
    {
        pool_t pool(pool_mode::max, t.h, t.w, t.kh, t.kw, t.sh, t.sw, t.ph, t.pw);
        const dmat x = make_mat(t.ch, t.h * t.w, 0.5, t.seed);
        const dmat y = pool.forward(x);
        ExpectNearMat(y, reference_pool(x, t.h, t.w, t.kh, t.kw, t.sh, t.sw, t.ph, t.pw,
                                        pool_mode::max, true),
                      1e-12);
    }
}

TEST(Pool2d, AverageForwardMatchesReference)
{
    struct c { int h, w, kh, kw, sh, sw, ph, pw, ch; bool include_pad; unsigned seed; };
    const std::vector<c> cases = {
        {3, 3, 2, 2, 2, 2, 0, 0, 1, true, 21},
        {3, 3, 2, 2, 1, 1, 1, 1, 1, true, 22},
        {3, 3, 2, 2, 1, 1, 1, 1, 1, false, 23},
        {5, 7, 3, 2, 2, 1, 1, 0, 2, true, 24},
        {5, 7, 3, 2, 2, 1, 1, 0, 2, false, 25},
        {6, 6, 3, 3, 2, 2, 1, 1, 3, true, 26},
        {6, 6, 3, 3, 2, 2, 1, 1, 3, false, 27},
    };

    for (const auto& t : cases)
    {
        pool_t pool(pool_mode::average, t.h, t.w, t.kh, t.kw, t.sh, t.sw, t.ph, t.pw,
                    t.include_pad);
        const dmat x = make_mat(t.ch, t.h * t.w, 0.5, t.seed);
        const dmat y = pool.forward(x);
        ExpectNearMat(y, reference_pool(x, t.h, t.w, t.kh, t.kw, t.sh, t.sw, t.ph, t.pw,
                                        pool_mode::average, t.include_pad),
                      1e-12);
    }
}

TEST(Pool2d, MatchesPyTorchMaxGoldens)
{
    // golden values from torch 2.9: F.max_pool2d(x, 2, 2) and (2, stride=1, padding=1), with
    // x = arange(1,10).reshape(1,1,3,3); grad = d(sum y)/dx (what backward returns for delta = 1)
    const dmat x = ramp_3x3();

    {
        pool_t pool(pool_mode::max, 3, 3, 2, 2);
        const dmat y = pool.forward(x);
        ExpectNearMat(y, dmat(1, 1, {5.0}), 1e-12);
        const dmat grad = pool.backward(ones(1, 1));
        ExpectNearMat(grad, dmat(1, 9, {0, 0, 0, 0, 1, 0, 0, 0, 0}), 1e-12);
    }

    {
        pool_t pool(pool_mode::max, 3, 3, 2, 2, 1, 1, 1, 1);
        const dmat y = pool.forward(x);
        ExpectNearMat(y, dmat(1, 16, {1, 2, 3, 3,
                                      4, 5, 6, 6,
                                      7, 8, 9, 9,
                                      7, 8, 9, 9}), 1e-12);
        const dmat grad = pool.backward(ones(1, 16));
        ExpectNearMat(grad, dmat(1, 9, {1, 1, 2,
                                        1, 1, 2,
                                        2, 2, 4}), 1e-12);
    }

    {
        // non-square kernel + non-square stride, 3x4 input (arange(1,13))
        const dmat x2(1, 12, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        pool_t pool(pool_mode::max, 3, 4, 2, 3, 2, 1);
        const dmat y = pool.forward(x2);
        ExpectShape(y, 1, 2);
        ExpectNearMat(y, dmat(1, 2, {7.0, 8.0}), 1e-12);
        const dmat grad = pool.backward(ones(1, 2));
        ExpectNearMat(grad, dmat(1, 12, {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}), 1e-12);
    }

    {
        // several channels: they must not interfere with each other
        const dmat x3(2, 9, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18});
        pool_t pool(pool_mode::max, 3, 3, 2, 2);
        ExpectNearMat(pool.forward(x3), dmat(2, 1, {5.0, 14.0}), 1e-12);
    }
}

TEST(Pool2d, MatchesPyTorchAverageGoldens)
{
    const dmat x = ramp_3x3();

    {
        // F.avg_pool2d(x, 2) -- with no padding count_include_pad is irrelevant
        pool_t pool(pool_mode::average, 3, 3, 2, 2);
        ExpectNearMat(pool.forward(x), dmat(1, 1, {3.0}), 1e-12);
        const dmat grad = pool.backward(ones(1, 1));
        ExpectNearMat(grad, dmat(1, 9, {0.25, 0.25, 0, 0.25, 0.25, 0, 0, 0, 0}), 1e-12);
    }

    {
        // F.avg_pool2d(x, 2, stride=1, padding=1): count_include_pad defaults to True, divisor always 4
        pool_t pool(pool_mode::average, 3, 3, 2, 2, 1, 1, 1, 1, true);
        const dmat y = pool.forward(x);
        ExpectNearMat(y, dmat(1, 16, {0.25, 0.75, 1.25, 0.75,
                                      1.25, 3.0, 4.0, 2.25,
                                      2.75, 6.0, 7.0, 3.75,
                                      1.75, 3.75, 4.25, 2.25}), 1e-12);
        // every output contributes 1/4 to each of the 4 window positions, i.e. 1 in total, so each
        // input ends up with exactly 1
        const dmat grad = pool.backward(ones(1, 16));
        ExpectNearMat(grad, dmat(1, 9, {1, 1, 1, 1, 1, 1, 1, 1, 1}), 1e-12);
    }

    {
        // count_include_pad=False: the divisor becomes the number of valid elements in the window
        pool_t pool(pool_mode::average, 3, 3, 2, 2, 1, 1, 1, 1, false);
        const dmat y = pool.forward(x);
        ExpectNearMat(y, dmat(1, 16, {1.0, 1.5, 2.5, 3.0,
                                      2.5, 3.0, 4.0, 4.5,
                                      5.5, 6.0, 7.0, 7.5,
                                      7.0, 7.5, 8.5, 9.0}), 1e-12);
        const dmat grad = pool.backward(ones(1, 16));
        ExpectNearMat(grad, dmat(1, 9, {2.25, 1.5, 2.25,
                                        1.5, 1.0, 1.5,
                                        2.25, 1.5, 2.25}), 1e-12);
    }

    {
        // non-square kernel + non-square stride, 3x4 input: window 2x3, stride 2x1
        const dmat x2(1, 12, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        pool_t pool(pool_mode::average, 3, 4, 2, 3, 2, 1, 0, 0);
        const dmat y = pool.forward(x2);
        ExpectNearMat(y, dmat(1, 2, {4.0, 5.0}), 1e-12);
        const dmat grad = pool.backward(ones(1, 2));
        const double s = 1.0 / 6.0;
        ExpectNearMat(grad, dmat(1, 12, {s, 2 * s, 2 * s, s,
                                         s, 2 * s, 2 * s, s,
                                         0, 0, 0, 0}), 1e-12);
    }
}

TEST(Pool2d, MaxTieRoutesWholeGradientToFirstIndex)
{
    // PyTorch's CPU convention: a tie gives the whole gradient to the **first** maximum (row-major)
    // instead of splitting it. For x = [[1,1],[1,1]], max_pool2d(2) yields grad [1,0,0,0].
    pool_t pool(pool_mode::max, 2, 2, 2, 2);
    const dmat x(1, 4, {1, 1, 1, 1});
    const dmat y = pool.forward(x);
    ExpectNearMat(y, dmat(1, 1, {1.0}), 1e-12);
    const dmat grad = pool.backward(ones(1, 1));
    ExpectNearMat(grad, dmat(1, 4, {1, 0, 0, 0}), 1e-12);

    // same with a partial tie: whichever appears first takes the whole gradient
    pool_t pool2(pool_mode::max, 2, 2, 2, 2);
    const dmat x2(1, 4, {2, 1, 2, 1});
    pool2.forward(x2);
    const dmat grad2 = pool2.backward(ones(1, 1));
    ExpectNearMat(grad2, dmat(1, 4, {1, 0, 0, 0}), 1e-12);
}

TEST(Pool2d, AverageOverlappingWindowsAccumulate)
{
    // All-ones input, 2x2 kernel, stride 1: every pixel gets "windows covering it / 4". On a 3x3,
    // corners sit in 1 window, edge middles in 2 and the centre in 4 -- an implementation that
    // writes instead of adding cannot pass.
    pool_t pool(pool_mode::average, 3, 3, 2, 2, 1, 1, 0, 0);
    const dmat x = ones(1, 9);
    const dmat y = pool.forward(x);
    ExpectNearMat(y, dmat(1, 4, {1, 1, 1, 1}), 1e-12);   // every window mean is 1

    const dmat grad = pool.backward(ones(1, 4));
    ExpectNearMat(grad, dmat(1, 9, {0.25, 0.5, 0.25,
                                    0.5, 1.0, 0.5,
                                    0.25, 0.5, 0.25}), 1e-12);
}

TEST(Pool2d, BackwardMatchesNumericalGradient)
{
    // Numerical gradients: the loss is L = sum 0.5*y^2, hence dL/dy = y and the forward output is
    // used as delta directly. Max pooling deliberately uses distinct random values (ties are not
    // differentiable, so central differences do not apply); average pooling tests both
    // count_include_pad settings.
    struct c { pool_mode mode; bool include_pad; int h, w, kh, kw, sh, sw, ph, pw; unsigned seed; };
    const std::vector<c> cases = {
        {pool_mode::max, true, 4, 5, 3, 2, 2, 1, 1, 0, 41},
        {pool_mode::max, true, 5, 6, 2, 2, 1, 1, 0, 0, 42},
        {pool_mode::average, true, 4, 5, 3, 2, 2, 1, 1, 0, 43},
        {pool_mode::average, false, 4, 5, 3, 2, 2, 1, 1, 0, 44},
        {pool_mode::average, false, 5, 5, 2, 2, 2, 2, 1, 1, 45},
        {pool_mode::average, true, 6, 6, 3, 3, 2, 2, 1, 1, 46},
    };

    for (const auto& t : cases)
    {
        pool_t pool(t.mode, t.h, t.w, t.kh, t.kw, t.sh, t.sw, t.ph, t.pw, t.include_pad);
        const dmat x = make_mat(2, t.h * t.w, 0.5, t.seed);

        const dmat y = pool.forward(x);
        const dmat dx = pool.backward(y);
        ExpectShape(dx, 2, t.h * t.w);

        auto loss_of = [&](const dmat& xp) {
            const dmat o = pool.forward(xp);
            double s = 0.0;
            for (int i = 0; i < o.row_num(); ++i)
                for (int j = 0; j < o.col_num(); ++j)
                    s += 0.5 * o(i, j) * o(i, j);
            return s;
        };

        const double eps = 1e-6;
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int idx = 0; idx < t.h * t.w; ++idx)
            {
                dmat xp = x, xm = x;
                xp(ch, idx) += eps;
                xm(ch, idx) -= eps;
                const double num = (loss_of(xp) - loss_of(xm)) / (2 * eps);
                EXPECT_NEAR(dx(ch, idx), num, 1e-6)
                    << "mode=" << pool_mode_name(t.mode)
                    << " include_pad=" << t.include_pad
                    << " at (" << ch << "," << idx << ")";
            }
        }
    }
}

TEST(Pool2d, StrideDefaultsToKernelSize)
{
    pool_t pool(pool_mode::max, 6, 6, 3, 2);
    EXPECT_EQ(pool.stride_h(), 3);
    EXPECT_EQ(pool.stride_w(), 2);
    EXPECT_EQ(pool.out_h(), (6 - 3) / 3 + 1);      // 2
    EXPECT_EQ(pool.out_w(), (6 - 2) / 2 + 1);      // 3
}

TEST(Pool2d, OneDIsOneRowSpecialCase)
{
    // sequence pooling: H=1, Kh=1, pad_h=0, a length-L sequence fed as [C, 1*L]
    const int L = 7, k = 2, stride = 2;
    pool_t pool = pool_t::one_d(pool_mode::max, L, k, stride);
    EXPECT_EQ(pool.in_h(), 1);
    EXPECT_EQ(pool.kernel_h(), 1);
    EXPECT_EQ(pool.out_h(), 1);

    const dmat x(2, L, {1, 3, 2, 6, 5, 4, 7,
                        7, 1, 2, 3, 4, 5, 6});
    // floor semantics: (7 - 2)/2 + 1 = 3; the leftover element is dropped (as in PyTorch's floor mode)
    const dmat y = pool.forward(x);
    ExpectShape(y, 2, 3);
    ExpectNearMat(y, dmat(2, 3, {3, 6, 5,
                                 7, 3, 5}), 1e-12);
    ExpectNearMat(y, reference_pool(x, 1, L, 1, k, 1, stride, 0, 0, pool_mode::max, true),
                  1e-12);

    // the 1-D form of average pooling goes through the very same implementation
    pool_t avg = pool_t::one_d(pool_mode::average, L, k, stride);
    ExpectNearMat(avg.forward(x),
                  reference_pool(x, 1, L, 1, k, 1, stride, 0, 0, pool_mode::average, true),
                  1e-12);
}

TEST(Pool2d, RejectsBadShapes)
{
    pool_t pool(pool_mode::max, 3, 3, 2, 2);
    pool.forward(make_mat(2, 9, 0.1, 51));

    EXPECT_THROW(pool.forward(make_mat(2, 8, 0.1, 52)), std::invalid_argument);
    EXPECT_THROW(pool.forward(dmat(0, 0)), std::invalid_argument);

    // delta must be [C, H_out*W_out] = [2, 1] (3x3 input, 2x2 kernel, stride 2 -> output 1x1)
    EXPECT_NO_THROW(pool.backward(make_mat(2, 1, 0.1, 53)));
    EXPECT_THROW(pool.backward(make_mat(1, 1, 0.1, 54)), std::runtime_error);
    EXPECT_THROW(pool.backward(make_mat(2, 2, 0.1, 55)), std::runtime_error);
}

TEST(Pool2d, UnconfiguredLayerFailsFast)
{
    pool_t pool;
    EXPECT_THROW(pool.forward(make_mat(1, 4, 0.1, 61)), std::runtime_error);
    EXPECT_THROW(pool.backward(make_mat(1, 4, 0.1, 62)), std::runtime_error);
    // usable immediately after set_param
    pool.set_param(pool_mode::max, 2, 2, 2, 2);
    const dmat x = make_mat(1, 4, 0.5, 63);
    double expect = x(0, 0);
    for (int j = 1; j < 4; ++j)
        expect = std::max(expect, x(0, j));
    ExpectNearMat(pool.forward(x), dmat(1, 1, {expect}), 1e-12);
}

TEST(Pool2d, SetParamRejectsInvalidConfig)
{
    pool_t pool;
    EXPECT_THROW(pool.set_param(pool_mode::max, 3, 3, 0, 1), std::invalid_argument);   // kernel 0
    EXPECT_THROW(pool.set_param(pool_mode::max, 0, 3, 1, 1), std::invalid_argument);   // spatial size 0
    EXPECT_THROW(pool.set_param(pool_mode::max, 3, 3, 2, 2, -1, 1), std::invalid_argument); // negative stride
    EXPECT_THROW(pool.set_param(pool_mode::max, 3, 3, 2, 2, 1, 1, -1, 0), std::invalid_argument); // negative pad
    // pad beyond half the kernel (PyTorch's own rule): 2*2 > 3
    EXPECT_THROW(pool.set_param(pool_mode::max, 5, 5, 3, 3, 1, 1, 2, 2), std::invalid_argument);
    // kernel larger than the padded input: h_out = (2 - 5) / 1 + 1 = -2
    EXPECT_THROW(pool.set_param(pool_mode::max, 2, 2, 5, 5, 1, 1, 0, 0), std::invalid_argument);
    EXPECT_NO_THROW(pool.set_param(pool_mode::max, 5, 5, 3, 3, 1, 1, 1, 1));
}

TEST(Pool2d, IsStaticLayerNotUpdatableNorReinit)
{
    // parameterless layer: it takes part in neither set_updator/set_lr nor a reinit container slot
    static_assert(!is_updatable_net<pool_t>);
    static_assert(!is_reinitable_net<pool_t>);
    SUCCEED();
}

TEST(Pool2d, ChainInsideComplexNet)
{
    // the whole conv -> relu -> maxpool chain: forward yields a shape, backward returns to the input shape, step survives
    using conv_relu_pool_t = complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, sgd_t>
        ::push_back_staticnet<relu_net_t>
        ::push_back_staticnet<pool2d_net_t>
        ::type;

    const int c_in = 2, c_out = 3, h = 4, w = 4;
    conv_relu_pool_t net;
    auto& conv = net.get<0>();
    conv.set_param(c_in, c_out, h, w, 3, 3, 1, 1, 1, 1);
    conv.set_updator(0.01);
    conv.weight() = make_mat(c_out, c_in * 9, 0.5, 71);
    conv.bias() = make_mat(c_out, 1, 0.3, 72);
    net.get<2>().set_param(pool_mode::max, h, w, 2, 2);

    const dmat x = make_mat(c_in, h * w, 0.5, 73);
    const dmat y = net.forward(x);
    ExpectShape(y, c_out, 2 * 2);

    const dmat delta = make_mat(c_out, 4, 0.5, 74);
    const dmat dx = net.backward(delta);
    ExpectShape(dx, c_in, h * w);
    EXPECT_NO_THROW(net.step());
    EXPECT_THROW(net.backward(make_mat(c_out, 3, 0.5, 75)), std::runtime_error);
}

TEST(Pool2d, NetTypeReportsConfig)
{
    pool_t max_pool(pool_mode::max, 4, 6, 3, 3, 2, 2, 1, 1);
    const std::string s = max_pool.net_type();
    EXPECT_NE(s.find("pool2d_net_t"), std::string::npos);
    EXPECT_NE(s.find("max"), std::string::npos);
    EXPECT_NE(s.find("in:4x6"), std::string::npos);
    EXPECT_NE(s.find("kernel:3x3"), std::string::npos);
    EXPECT_EQ(s.find("count_include_pad"), std::string::npos);   // max mode does not print that field

    pool_t avg_pool(pool_mode::average, 4, 6, 3, 3, 2, 2, 1, 1, false);
    const std::string s2 = avg_pool.net_type();
    EXPECT_NE(s2.find("average"), std::string::npos);
    EXPECT_NE(s2.find("count_include_pad:false"), std::string::npos);
}
