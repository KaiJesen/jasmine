/**
 * mat_reshape_view_t（形状视图）与「视图操作数零拷贝进 GEMM」的单测。
 *
 * 两件事必须同时成立才有意义：
 *   1. 视图是**别名**语义——只换索引、不碰数据，写穿、读穿都对；
 *   2. 视图能真的零拷贝交给 BLAS——所以除了数值对拍，还要断言 gemm_view() 给出的
 *      描述符是有效的（否则测试即使通过，也可能是在偷偷物化）。
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

/** 朴素乘法：视图路径的唯一裁判 */
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

    // 行优先展平后的第 i*4+j 个元素：与底层是同一段存储的同一个单元
    EXPECT_DOUBLE_EQ(v(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(v(0, 3), 4.0);
    EXPECT_DOUBLE_EQ(v(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(v(2, 3), 12.0);

    // 语义上就是底层数据本身（不是副本）
    for (int i = 0, r = 0; r < 2; ++r)
        for (int c = 0; c < 6; ++c, ++i)
            EXPECT_DOUBLE_EQ(v(i / 4, i % 4), m(r, c));
}

TEST(ReshapeView, WritesAliasBothWays)
{
    dmat m(2, 6);
    m = 0.0;
    auto v = m.reshape_view(3, 4);

    v(1, 2) = 42.0;                       // 通过视图写
    EXPECT_DOUBLE_EQ(m(1, 0), 42.0);      // 展平下标 1*4+2 = 6 → 底层 (1,0)（每行 6 个元素）

    m(0, 5) = 7.0;                        // 直接写底层
    EXPECT_DOUBLE_EQ(v(1, 1), 7.0);       // 视图立刻可见（展平下标 5 → (1,1)）
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

    vt(2, 1) = 123.0;                     // 转置视图同样写穿
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
    // 列优先存储的底层：reshape 视图也必须按列优先展平，否则会读错元素
    dmat m(2, 3, false);                 // row_first = false
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            m(i, j) = 10 * i + j;

    auto v = m.reshape_view(3, 2);
    // 列优先展平顺序：m(0,0), m(1,0), m(0,1), m(1,1), ...
    EXPECT_DOUBLE_EQ(v(0, 0), m(0, 0));
    EXPECT_DOUBLE_EQ(v(1, 0), m(1, 0));
    EXPECT_DOUBLE_EQ(v(0, 1), m(1, 1));   // 列优先展平的第 4 个元素 = 第 1 列第 1 行
    EXPECT_DOUBLE_EQ(v(2, 1), m(1, 2));
}

TEST(ReshapeView, GemmDescriptorIsExposedForRowMajorStorage)
{
    // 门禁：视图必须能给出「指针 + 前导维 + 转置」描述符，GEMM 才可能零拷贝
    dmat m(4, 6);
    auto v = m.reshape_view(6, 4);
    const auto dv = v.gemm_view();
    ASSERT_TRUE(dv.valid);
    EXPECT_EQ(dv.ptr, m.data());
    EXPECT_EQ(dv.ld, 4);                 // reshape 之后每行 4 个元素
    EXPECT_FALSE(dv.transposed);

    const auto dvt = v.t().gemm_view();
    ASSERT_TRUE(dvt.valid);
    EXPECT_EQ(dvt.ptr, m.data());
    EXPECT_EQ(dvt.ld, 4);                // 转置时存储矩阵仍是 (4 x 6) 行优先
    EXPECT_TRUE(dvt.transposed);

    // mat_view_t 的转置视图同样能给出描述符（`a.t().dot(b)` 不再物化）
    const auto dt = m.t().gemm_view();
    ASSERT_TRUE(dt.valid);
    EXPECT_TRUE(dt.transposed);
    EXPECT_EQ(dt.ld, 6);

    // 子视图：偏移正确、前导维仍用底层的
    const auto ds = m.view(1, 2, 2, 3).gemm_view();
    ASSERT_TRUE(ds.valid);
    EXPECT_EQ(ds.ptr, m.data() + 1 * 6 + 2);
    EXPECT_EQ(ds.ld, 6);
    EXPECT_FALSE(ds.transposed);

    // 列优先存储给不出描述符：GEMM 会退回物化（行为不变）
    dmat cm(2, 6, false);
    EXPECT_FALSE(cm.gemm_view().valid);
}

TEST(ReshapeView, TransposedOperandsMatchNaive)
{
    // 尺寸跨过 GEMM 快路径阈值（16^3），确保走的是描述符 + BLAS/阻塞分支
    // 注意：mat_t::t()/reshape_view() 目前只有非 const 版本，所以这里不能用 const 局部量
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

    // 恒等重解释：与直接用矩阵一致
    ExpectNearMat(dmat(a.reshape_view(M, K).dot(b)), naive_dot(a, b), 1e-12);

    // 转置后的 reshape 视图：与物化后再乘一致
    auto v = a.reshape_view(K, M);
    ExpectNearMat(dmat(v.t().dot(b)), naive_dot(mat_t<double>(v.t()), b), 1e-12);

    // 1x(L) 一行切成 (n, t) 再转置：这就是 patchify / 非重叠一维卷积的窗口矩阵
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
    // 单行切片是连续的，可以零拷贝重解释成窗口矩阵 —— 这正是"按行做一维卷积"的入口
    dmat x(3, 12);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 12; ++j)
            x(i, j) = i * 100 + j;

    auto row_view = x.view(1, 0, 1, 12);        // 第 1 行、整行
    auto win = row_view.reshape_view(4, 3);     // (4 x 3)
    ExpectShape(win, 4, 3);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(win(i, j), x(1, i * 3 + j));

    // 局部行（没有横跨整行、且不止一行）在内存里带空洞：不能线性展平
    auto gap_view = x.view(0, 0, 2, 6);
    EXPECT_FALSE(gap_view.densely_packed());
    EXPECT_THROW(gap_view.reshape_view(3, 4), std::invalid_argument);

    // 单列切片同理
    dmat cm(4, 4, false);
    auto col_view = cm.view(0, 1, 4, 1);
    EXPECT_TRUE(col_view.densely_packed());
    EXPECT_NO_THROW(col_view.reshape_view(2, 2));
}

TEST(ReshapeView, ReshapingATemporaryIsRejectedAtCompileTime)
{
    // 视图持有源矩阵的引用：从临时量取 reshape 视图必然悬垂，右值重载被 delete 拦下。
    // 被 delete 的函数在 requires 表达式里也是硬错误（GCC 不把它当替换失败），
    // 所以"编不过"这件事只能靠编译失败来验证 —— 命令记在 TESTING.md 第 13 节：
    //   echo 'auto v = dmat(2,6).reshape_view(3,4);' | g++ -std=c++20 -fsyntax-only -I. -x c++ -
    //   → error: use of deleted function ... mat_t<double>::reshape_view(int, int) &&
    static_assert(requires(dmat& m) { m.reshape_view(3, 4); });          // 左值可以
    static_assert(requires(dmat& m) { m.view(0, 0, 1, 12); });           // 具名视图可以
    SUCCEED();
}

TEST(ReshapeView, ViewOperandIsNotModified)
{
    // 零拷贝意味着 GEMM 只读操作数：算完之后源矩阵必须原样不动
    const int M = 40, K = 48, N = 36;
    dmat a = make_mat(M, K, 0.5, 21);
    dmat b = make_mat(K, N, 0.5, 22);
    const dmat a_before = a;

    auto v = a.reshape_view(K, M).t();
    const dmat y = v.dot(b);
    ExpectShape(y, M, N);
    ExpectNearMat(a, a_before, 0.0);
}
