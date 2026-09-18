/**
 * Unit tests for mat_reshape_view_t (shape views) and for "zero-copy view operands in GEMM".
 *
 * Both properties have to hold at once for this to be meaningful:
 *   1. a view is an **alias** -- it only changes the indexing, never the data, so both writes and
 *      reads through it behave as if applied to the underlying matrix;
 *   2. the view really can be handed to BLAS without a copy -- so besides the numeric comparison the
 *      tests assert that gemm_view() reports a valid descriptor (otherwise they could pass while
 *      quietly materialising).
 */

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_mat_express_t.hpp"
#include "jas_mat_gemm.hpp"
#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

using dmat = mat_t<double>;

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

/** Naive multiply: the only judge of the view path */
dmat naive_dot(const dmat& a, const dmat& b)
{
    dmat c(a.row_num(), b.col_num());
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

TEST(ReshapeView, ReinterpretsShapeWithoutCopying)
{
    dmat m(2, 6, {1, 2, 3, 4, 5, 6,
                  7, 8, 9, 10, 11, 12});
    auto v = m.reshape_view(3, 4);
    ExpectShape(v, 3, 4);

    // element i*4+j of the row-major flattening: the same cell of the same storage
    EXPECT_DOUBLE_EQ(v(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(v(0, 3), 4.0);
    EXPECT_DOUBLE_EQ(v(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(v(2, 3), 12.0);

    // semantically this is the underlying data itself (not a copy)
    for (int i = 0, r = 0; r < 2; ++r)
        for (int c = 0; c < 6; ++c, ++i)
            EXPECT_DOUBLE_EQ(v(i / 4, i % 4), m(r, c));
}

TEST(ReshapeView, WritesAliasBothWays)
{
    dmat m(2, 6);
    m = 0.0;
    auto v = m.reshape_view(3, 4);

    v(1, 2) = 42.0;                       // write through the view
    EXPECT_DOUBLE_EQ(m(1, 0), 42.0);      // flat index 1*4+2 = 6 -> underlying (1,0) (6 elements per row)

    m(0, 5) = 7.0;                        // write the underlying matrix directly
    EXPECT_DOUBLE_EQ(v(1, 1), 7.0);       // the view sees it at once (flat index 5 -> (1,1))
}

TEST(ReshapeView, TransposeIsAlsoAView)
{
    dmat m(2, 6);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 6; ++j)
            m(i, j) = 10 * i + j;

    auto v = m.reshape_view(3, 4);
    auto vt = v.t();
    ExpectShape(vt, 4, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(vt(j, i), v(i, j));

    vt(2, 1) = 123.0;                     // a transposed view writes through as well
    EXPECT_DOUBLE_EQ(v(1, 2), 123.0);
}

TEST(ReshapeView, SizeMismatchThrows)
{
    dmat m(2, 6);
    EXPECT_THROW(m.reshape_view(3, 3), std::invalid_argument);
    EXPECT_THROW(m.reshape_view(13, 1), std::invalid_argument);
    EXPECT_NO_THROW(m.reshape_view(1, 12));
    EXPECT_NO_THROW(m.reshape_view(12, 1));
}

TEST(ReshapeView, ColumnMajorBaseKeepsStorageOrder)
{
    // column-major underlying storage: the reshape view must flatten in column-major order too,
    // otherwise it reads the wrong elements
    dmat m(2, 3, false);                 // row_first = false
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            m(i, j) = 10 * i + j;

    auto v = m.reshape_view(3, 2);
    // column-major flattening order: m(0,0), m(1,0), m(0,1), m(1,1), ...
    EXPECT_DOUBLE_EQ(v(0, 0), m(0, 0));
    EXPECT_DOUBLE_EQ(v(1, 0), m(1, 0));
    EXPECT_DOUBLE_EQ(v(0, 1), m(1, 1));   // 4th element in column-major order = row 1 of column 1
    EXPECT_DOUBLE_EQ(v(2, 1), m(1, 2));
}

TEST(ReshapeView, GemmDescriptorIsExposedForRowMajorStorage)
{
    // Gate check: the view has to expose a "pointer + leading dimension + transpose" descriptor
    // before GEMM can be copy-free
    dmat m(4, 6);
    auto v = m.reshape_view(6, 4);
    const auto dv = v.gemm_view();
    ASSERT_TRUE(dv.valid);
    EXPECT_EQ(dv.ptr, m.data());
    EXPECT_EQ(dv.ld, 4);                 // after the reshape every row holds 4 elements
    EXPECT_FALSE(dv.transposed);

    const auto dvt = v.t().gemm_view();
    ASSERT_TRUE(dvt.valid);
    EXPECT_EQ(dvt.ptr, m.data());
    EXPECT_EQ(dvt.ld, 4);                // when transposed the stored matrix is still (4 x 6) row-major
    EXPECT_TRUE(dvt.transposed);

    // a transposed mat_view_t exposes a descriptor too (so `a.t().dot(b)` no longer materialises)
    const auto dt = m.t().gemm_view();
    ASSERT_TRUE(dt.valid);
    EXPECT_TRUE(dt.transposed);
    EXPECT_EQ(dt.ld, 6);

    // sub-view: the offset is correct and the leading dimension still comes from the base
    const auto ds = m.view(1, 2, 2, 3).gemm_view();
    ASSERT_TRUE(ds.valid);
    EXPECT_EQ(ds.ptr, m.data() + 1 * 6 + 2);
    EXPECT_EQ(ds.ld, 6);
    EXPECT_FALSE(ds.transposed);

    // column-major storage cannot provide a descriptor: GEMM falls back to materialising (unchanged)
    dmat cm(2, 6, false);
    EXPECT_FALSE(cm.gemm_view().valid);
}

TEST(ReshapeView, TransposedOperandsMatchNaive)
{
    // sizes beyond the GEMM fast-path threshold (16^3), so the descriptor + BLAS/blocked branch is used
    // note: mat_t::t()/reshape_view() only have non-const overloads, so no const locals here
    const int M = 40, K = 48, N = 36;
    dmat a = make_mat(M, K, 0.5, 1);
    dmat b = make_mat(K, N, 0.5, 2);
    dmat at_src = make_mat(K, M, 0.5, 3);
    dmat bt_src = make_mat(N, K, 0.5, 4);
    dmat b_kn = make_mat(K, N, 0.5, 5);
    dmat b_mn = make_mat(M, N, 0.5, 6);

    ExpectNearMat(dmat(a.dot(b)), naive_dot(a, b), 1e-12);
    ExpectNearMat(dmat(a.dot(bt_src.t())), naive_dot(a, mat_t<double>(bt_src.t())), 1e-12);
    ExpectNearMat(dmat(at_src.t().dot(b_kn)), naive_dot(mat_t<double>(at_src.t()), b_kn), 1e-12);
    ExpectNearMat(dmat(at_src.t().dot(bt_src.t())),
                  naive_dot(mat_t<double>(at_src.t()), mat_t<double>(bt_src.t())), 1e-12);
    ExpectNearMat(dmat(a.view(0, 0, M, K).t().dot(b_mn)),
                  naive_dot(mat_t<double>(a.t()), b_mn), 1e-12);
}

TEST(ReshapeView, ReshapeOperandMatchesNaive)
{
    const int M = 40, K = 48, N = 36;
    dmat a = make_mat(M, K, 0.5, 11);
    dmat b = make_mat(K, N, 0.5, 12);

    // the identity reinterpretation: matches using the matrix directly
    ExpectNearMat(dmat(a.reshape_view(M, K).dot(b)), naive_dot(a, b), 1e-12);

    // a transposed reshape view: matches materialising first and then multiplying
    auto v = a.reshape_view(K, M);
    ExpectNearMat(dmat(v.t().dot(b)), naive_dot(mat_t<double>(v.t()), b), 1e-12);

    // one 1x(L) row cut into (n, t) and transposed: this is the window matrix of patchify /
    // non-overlapping 1-D convolution
    const int t = 3, n = 20, c_out = 8;
    dmat x = make_mat(1, n * t, 0.5, 13);
    dmat w = make_mat(c_out, t, 0.5, 14);
    dmat col_ref(n, t);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < t; ++i)
            col_ref(j, i) = x(0, j * t + i);
    ExpectNearMat(dmat(w.dot(x.reshape_view(n, t).t())),
                  naive_dot(w, mat_t<double>(col_ref.t())), 1e-12);
}

TEST(ReshapeView, RowSliceCanBeReshaped)
{
    // a single-row slice is contiguous, so it can be reinterpreted as a window matrix without
    // copying -- exactly the entry point of "1-D convolution row by row"
    dmat x(3, 12);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 12; ++j)
            x(i, j) = i * 100 + j;

    auto row_view = x.view(1, 0, 1, 12);        // row 1, the whole row
    auto win = row_view.reshape_view(4, 3);     // (4 x 3)
    ExpectShape(win, 4, 3);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(win(i, j), x(1, i * 3 + j));

    // a partial row (not the full width, and more than one row) has gaps in memory and cannot be flattened
    auto gap_view = x.view(0, 0, 2, 6);
    EXPECT_FALSE(gap_view.densely_packed());
    EXPECT_THROW(gap_view.reshape_view(3, 4), std::invalid_argument);

    // the same holds for a single-column slice
    dmat cm(4, 4, false);
    auto col_view = cm.view(0, 1, 4, 1);
    EXPECT_TRUE(col_view.densely_packed());
    EXPECT_NO_THROW(col_view.reshape_view(2, 2));
}

TEST(ReshapeView, ReshapingATemporaryIsRejectedAtCompileTime)
{
    // A view holds a reference to its source matrix: reshaping a temporary always dangles, so the
    // rvalue overload is deleted. A deleted function is also a hard error inside a requires
    // expression (GCC does not treat it as a substitution failure), so "it must not compile" can
    // only be verified by an actual compile failure -- the command is recorded in TESTING.md 13:
    //   echo 'auto v = dmat(2,6).reshape_view(3,4);' | g++ -std=c++20 -fsyntax-only -I. -x c++ -
    //   → error: use of deleted function ... mat_t<double>::reshape_view(int, int) &&
    static_assert(requires(dmat& m) { m.reshape_view(3, 4); });          // an lvalue is fine
    static_assert(requires(dmat& m) { m.view(0, 0, 1, 12); });           // a named view is fine
    SUCCEED();
}

TEST(ReshapeView, ViewOperandIsNotModified)
{
    // copy-free means GEMM only reads its operands: the source matrix must be untouched afterwards
    const int M = 40, K = 48, N = 36;
    dmat a = make_mat(M, K, 0.5, 21);
    dmat b = make_mat(K, N, 0.5, 22);
    const dmat a_before = a;

    auto v = a.reshape_view(K, M).t();
    const dmat y = v.dot(b);
    ExpectShape(y, M, N);
    ExpectNearMat(a, a_before, 0.0);
}
