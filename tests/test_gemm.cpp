#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include "jas_mat_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_gemm.hpp"
#include "jas_mat_view_t.hpp"

using namespace jasmine;

namespace {

mat_t<double> naive_dot(mat_t<double> const& a, mat_t<double> const& b)
{
    mat_t<double> c(a.row_num(), b.col_num());
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < b.col_num(); ++j)
        {
            double s = 0.0;
            for (int k = 0; k < a.col_num(); ++k)
                s += a(i, k) * b(k, j);
            c(i, j) = s;
        }
    return c;
}

} // namespace

TEST(MatGemm, SquareMatchesNaive)
{
    constexpr int n = 48; // above gemm_fast_threshold (16^3)
    mat_t<double> a(n, n);
    mat_t<double> b(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            a(i, j) = 0.01 * i + 0.02 * j;
            b(i, j) = 0.03 * i - 0.01 * j;
        }

    mat_t<double> got = a.dot(b).clone();
    mat_t<double> exp = naive_dot(a, b);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(got(i, j), exp(i, j), 1e-9) << "i=" << i << " j=" << j;
}

TEST(MatGemm, RectangularAndFloat)
{
    mat_t<float> a(32, 64);
    mat_t<float> b(64, 16);
    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 64; ++j)
            a(i, j) = static_cast<float>(i - j) * 0.1f;
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 16; ++j)
            b(i, j) = static_cast<float>(i + 2 * j) * 0.05f;

    mat_t<float> got = a.dot(b).clone();
    ASSERT_EQ(got.row_num(), 32);
    ASSERT_EQ(got.col_num(), 16);

    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 16; ++j)
        {
            float s = 0.f;
            for (int k = 0; k < 64; ++k)
                s += a(i, k) * b(k, j);
            EXPECT_NEAR(got(i, j), s, 1e-4f);
        }
}

TEST(MatGemm, TransposeOperandMatchesNaive)
{
    mat_t<double> a(40, 24);
    mat_t<double> b(40, 18);
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 24; ++j)
            a(i, j) = 0.1 * i + j;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 18; ++j)
            b(i, j) = 0.2 * i - j;

    // a.t() is 24x40, b is 40x18 → 24x18
    mat_t<double> got = a.t().dot(b).clone();
    mat_t<double> at = a.t().clone();
    mat_t<double> exp = naive_dot(at, b);
    for (int i = 0; i < got.row_num(); ++i)
        for (int j = 0; j < got.col_num(); ++j)
            EXPECT_NEAR(got(i, j), exp(i, j), 1e-9);
}

TEST(MatGemm, SmallFallsBackStillCorrect)
{
    mat_t<double> a(3, 2, {1, 2, 3, 4, 5, 6});
    mat_t<double> b(2, 3, {1, 0, 2, 0, 1, 3});
    mat_t<double> got = a.dot(b).clone();
    EXPECT_NEAR(got(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(got(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(got(0, 2), 8.0, 1e-12);
    EXPECT_NEAR(got(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(got(2, 0), 5.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Single-column (decode) path
//
// Autoregressive decoding is always N == 1, and for operands too large to be
// cache-resident a plain GEMV streams the weight matrix once instead of letting
// BLAS pack it. These tests pin both the dispatch policy and the kernel itself,
// because the golden alignment tests need an external weight file and are
// skipped in CI, so nothing else would catch a regression here.
// ---------------------------------------------------------------------------

TEST(MatGemm, GemvDispatchPolicy)
{
    using detail::gemv_should_apply;
    constexpr long long kFloat = sizeof(float);
    constexpr long long kDouble = sizeof(double);
    // 16 MiB threshold: 4 Mi elements in float, 2 Mi in double.
    const int big = 8192, big_k = 768;   // 6.3 Mi elements -> 25 MB float, 50 MB double
    const int small = 256;               // 0.05 Mi elements -> well under the threshold

    // The decode case: one column, no transpose, operand too big for cache.
    EXPECT_TRUE(gemv_should_apply(big, 1, big_k, kFloat, false));
    EXPECT_TRUE(gemv_should_apply(big, 1, big_k, kDouble, false));

    // N > 1 gets reuse down the output columns, so GEMM blocking earns its keep.
    EXPECT_FALSE(gemv_should_apply(big, 2, big_k, kFloat, false));

    // A transposed left operand is addressed column-wise, which the streaming
    // kernel cannot do; callers materialise it instead.
    EXPECT_FALSE(gemv_should_apply(big, 1, big_k, kFloat, true));

    // Small operands stay on BLAS: measured, the packing traffic is then paid in
    // cache rather than DRAM, and BLAS is faster (see the table in jas_mat_gemm.hpp).
    EXPECT_FALSE(gemv_should_apply(small, 1, small, kFloat, false));
    EXPECT_FALSE(gemv_should_apply(small, 1, small, kDouble, false));

    // The threshold is on total bytes, so the same shape crosses it at different
    // element counts for the two types.
    const int mid = 4096, mid_k = 512;   // 2.1 Mi elements: 8 MB float, 16 MB double
    EXPECT_FALSE(gemv_should_apply(mid, 1, mid_k, kFloat, false));
    EXPECT_TRUE(gemv_should_apply(mid, 1, mid_k, kDouble, false));
}

TEST(MatGemm, GemvKernelMatchesNaive)
{
    // Large enough to take the dispatch path in a.dot(b), and small enough that a
    // naive triple loop in the test is still cheap.
    constexpr int M = 8192;
    constexpr int K = 768;

    mat_t<float> a(M, K);
    mat_t<float> b(K, 1);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            a(i, k) = static_cast<float>((i * 31 + k * 17) % 251) * 0.01f;
    for (int k = 0; k < K; ++k)
        b(k, 0) = static_cast<float>((k * 7) % 97) * 0.02f;

    // Go through the real dispatch so the test covers what inference actually hits.
    const mat_t<float> got = a.dot(b).clone();
    ASSERT_EQ(got.row_num(), M);
    ASSERT_EQ(got.col_num(), 1);

    // float accumulation over 768 terms: compare against double accumulation.
    for (int i = 0; i < M; i += 97) // sample: a full M-row check adds little signal
    {
        double ref = 0.0;
        for (int k = 0; k < K; ++k)
            ref += static_cast<double>(a(i, k)) * static_cast<double>(b(k, 0));
        EXPECT_NEAR(got(i, 0), ref, 1e-2) << "i=" << i;
    }
}

TEST(MatGemm, GemvKernelHandlesBothVectorStrides)
{
    // trans_b with a single column only changes the vector's stride, and the
    // kernel specialises on it. Both instantiations must agree with the naive dot.
    constexpr int M = 257;
    constexpr int K = 64;

    mat_t<float> a(M, K);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            a(i, k) = static_cast<float>(i % 13) - static_cast<float>(k) * 0.25f;

    std::vector<float> b(K), b_strided(2 * K, 0.0f);
    for (int k = 0; k < K; ++k)
    {
        b[k] = static_cast<float>(k % 11) * 0.5f;
        b_strided[2 * k] = b[k]; // stride 2, as if it were a column of a wider matrix
    }

    std::vector<float> c_contig(M, 0.0f), c_strided(M, 0.0f);
    detail::gemv_rowmajor(M, K, a.data(), K, b.data(), 1, c_contig.data(), 1);
    detail::gemv_rowmajor(M, K, a.data(), K, b_strided.data(), 2, c_strided.data(), 1);

    for (int i = 0; i < M; ++i)
    {
        double ref = 0.0;
        for (int k = 0; k < K; ++k)
            ref += static_cast<double>(a(i, k)) * static_cast<double>(b[k]);
        EXPECT_NEAR(c_contig[i], ref, 1e-3) << "contiguous i=" << i;
        EXPECT_NEAR(c_strided[i], ref, 1e-3) << "strided i=" << i;
    }
}
