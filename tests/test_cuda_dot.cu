#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "test_helpers.hpp"

/**
 * `dot` / `matmul` 的设备分派。
 *
 * 主机端的 `a.dot(b)` 返回一个**惰性**的 `mat_dot_t` 节点，能被并进更大的表达式树、
 * 由 `work()` 逐元素求值。设备端行不通：`mat_dot_t::operator()` 是「每个输出元素自己走
 * 一遍 K 循环」，融进逐元素 kernel 等于把访存复用全丢掉。GEMM 本质上不可融合。
 *
 * 所以设备端的 `.dot()` **立即求值**，返回拥有显存的结果。这里验证的正是这套分派：
 * 形状、转置组合、链式调用、表达式操作数、以及与主机端逐元素对拍。
 *
 * 散热约束同其它 CUDA 测试：矩阵都小、用例前后测温、超 80℃ 跳过。详见 CUDA.md。
 */

using namespace jasmine;

namespace
{

constexpr int kHotCelsius = 80;

int gpu_temperature_c()
{
    // Thermal control is only needed on the fanless P4 development card.
    // A800 and other well-cooled sm_80+ targets should not emit temperature spam.
    if (jasmine::cuda::device_info().compute_capability() != 61)
        return -1;

    FILE* pipe = ::popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null", "r");
    if (pipe == nullptr)
        return -1;

    char buf[64] = {};
    char* got = std::fgets(buf, sizeof(buf), pipe);
    ::pclose(pipe);
    if (got == nullptr)
        return -1;

    return std::atoi(buf);
}

/** 确定性地填一个矩阵；行列非线性变化，避免掩盖转置/次序类错误。 */
template <typename T>
mat_t<T> make_host(int rows, int cols, T base)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + T(0.3) * i - T(0.2) * j + T(0.07) * ((i * 3 + j * 5) % 11);
    return m;
}

void expect_matrices_match(const mat_t<double>& a, const mat_t<double>& b, double rel_tol,
                           const char* what)
{
    ASSERT_EQ(a.row_num(), b.row_num()) << what;
    ASSERT_EQ(a.col_num(), b.col_num()) << what;
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < a.col_num(); ++j)
        {
            const double scale = std::max(1.0, std::abs(b(i, j)));
            ASSERT_NEAR(a(i, j), b(i, j), rel_tol * scale)
                << what << " 在 (" << i << "," << j << ") 不一致";
        }
}

class CudaDotTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const int t = gpu_temperature_c();
        if (t >= kHotCelsius)
            GTEST_SKIP() << "GPU 结温 " << t << "℃ 已达上限 " << kHotCelsius << "℃，跳过";
    }

    void TearDown() override
    {
        const int t = gpu_temperature_c();
        if (t >= 0)
            std::printf("        [GPU %d℃]\n", t);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// 类型契约（零发热，纯编译期）
// ---------------------------------------------------------------------------

TEST(CudaDotContract, DotIsEagerOnDeviceUnlikeTheHostNode)
{
    // 主机端返回惰性节点（用 trait 判，避免写死 receiver 的引用类别）
    using host_result_t = std::remove_cvref_t<
        decltype(std::declval<mat_t<double>&>().dot(std::declval<mat_t<double>&>()))>;
    static_assert(is_mat_dot_v<host_result_t>,
                  "主机端的 .dot() 应当仍返回惰性的 mat_dot_t 节点");

    // 设备端立即求值，返回拥有显存的结果
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(std::declval<dev_mat_t<double>>().dot(
                      std::declval<dev_mat_t<double>>()))>,
                  cuda::dev_matrix_t<double>>,
                  "设备端的 .dot() 必须立即求值成拥有者，而不是节点");

    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(std::declval<cuda::dev_matrix_t<double>&>().dot(
                      std::declval<cuda::dev_matrix_t<double>&>()))>,
                  cuda::dev_matrix_t<double>>);
    SUCCEED();
}

TEST(CudaDotContract, AddingDotDidNotBreakTheLeafContract)
{
    // 加了成员函数之后，叶子仍必须满足「能按值传给 kernel」的全部条件
    static_assert(std::is_trivially_copyable_v<dev_mat_t<double>>);
    static_assert(std::is_trivially_destructible_v<dev_mat_t<double>>);
    static_assert(std::is_standard_layout_v<dev_mat_t<double>>);
    static_assert(is_device_evaluable_v<dev_mat_t<double>>);
    static_assert(is_matrix<dev_mat_t<double>>);
    SUCCEED();
}

TEST(CudaDotContract, DotNodeIsStillRefusedOnDevice)
{
    // 显式写成 false（而不是靠"成员不存在"），免得日后有人顺手补上
    static_assert(!mat_dot_t<dev_mat_t<double>&, dev_mat_t<double>&>::device_evaluable,
                  "mat_dot_t 必须显式声明不可上设备");
    static_assert(!is_device_evaluable_v<mat_dot_t<dev_mat_t<double>&, dev_mat_t<double>&>>);

    // 于是「主机式写法产出的节点」依旧被编译期挡住 —— 这个陷阱要保持可见。
    // 注意这里刻意调用**自由函数** jasmine::dot 而不是成员：成员在设备端立即求值，
    // 不会产出节点；只有自由函数才复刻主机端那种惰性节点。
    auto lhs = dev_mat_t<double>(nullptr, 2, 3);
    auto rhs = dev_mat_t<double>(nullptr, 3, 4);
    auto node = jasmine::dot(lhs, rhs);
    static_assert(is_mat_dot_v<decltype(node)>, "自由函数 dot 应当产出 mat_dot_t 节点");
    static_assert(!is_device_evaluable_v<decltype(node)>,
                  "jasmine::dot 出来的节点不该能进融合 kernel");

    // 而含该节点的表达式也一并被挡
    auto other = dev_mat_t<double>(nullptr, 2, 4);
    auto mixed = node + other;
    static_assert(!is_device_evaluable_v<decltype(mixed)>);
    EXPECT_EQ(mixed.row_num(), 2);
    EXPECT_EQ(mixed.col_num(), 4);
}

// ---------------------------------------------------------------------------
// 基本形状与数值：与主机端 dot 对拍
// ---------------------------------------------------------------------------

TEST_F(CudaDotTest, MatmulMatchesHostDot)
{
    const int M = 12, K = 7, N = 5;   // 三个维度互不相等，次序搞反必然被抓到
    auto a = make_host<double>(M, K, 0.5);
    auto b = make_host<double>(K, N, -0.3);

    cuda::dev_matrix_t<double> da(M, K, a), db(K, N, b);
    auto got = cuda::matmul(da.leaf(), db.leaf()).download();

    ASSERT_EQ(got.row_num(), M);
    ASSERT_EQ(got.col_num(), N);
    expect_matrices_match(got, a.dot(b).clone(), 1e-12, "matmul");
}

TEST_F(CudaDotTest, LeafDotMemberMatchesHost)
{
    const int M = 9, K = 6, N = 11;
    auto a = make_host<double>(M, K, 0.4);
    auto b = make_host<double>(K, N, 0.6);

    cuda::dev_matrix_t<double> da(M, K, a), db(K, N, b);
    auto got = da.leaf().dot(db.leaf()).download();
    expect_matrices_match(got, a.dot(b).clone(), 1e-12, "leaf.dot(leaf)");
}

TEST_F(CudaDotTest, MatrixDotMemberMatchesHost)
{
    const int M = 8, K = 10, N = 6;
    auto a = make_host<double>(M, K, 0.35);
    auto b = make_host<double>(K, N, 0.45);

    cuda::dev_matrix_t<double> da(M, K, a), db(K, N, b);
    auto got = da.dot(db).download();
    expect_matrices_match(got, a.dot(b).clone(), 1e-12, "matrix.dot(matrix)");
}

TEST_F(CudaDotTest, MixedLeafAndOwnerOperands)
{
    // 叶子与拥有者混用：两种重载都要能落到同一条路径
    auto a = make_host<double>(7, 9, 0.3);
    auto b = make_host<double>(9, 4, 0.5);
    cuda::dev_matrix_t<double> da(7, 9, a), db(9, 4, b);

    expect_matrices_match(da.dot(db.leaf()).download(), a.dot(b).clone(), 1e-12, "owner·leaf");
    expect_matrices_match(da.leaf().dot(db).download(), a.dot(b).clone(), 1e-12, "leaf·owner");
}

TEST_F(CudaDotTest, AlphaScalesTheProduct)
{
    auto a = make_host<double>(6, 5, 0.4);
    auto b = make_host<double>(5, 8, 0.3);
    cuda::dev_matrix_t<double> da(6, 5, a), db(5, 8, b);

    auto got = cuda::matmul(da.leaf(), db.leaf(), 0.25).download();
    expect_matrices_match(got, (a.dot(b) * 0.25).clone(), 1e-12, "alpha");
}

// ---------------------------------------------------------------------------
// 转置组合（与主机端一致地靠叶子视图表达）
// ---------------------------------------------------------------------------

TEST_F(CudaDotTest, TransposedOperandsMatchHost)
{
    // 三种转置组合各用一组形状合法的操作数（内维必须真的对得上，
    // 否则 mat_dot_t 的构造校验会先抛，测不到 GEMM 的分支）
    const int M = 6, K = 9, N = 5;

    // Aᵀ·B：A 是 M×K，B 是 M×N → Aᵀ(K×M)·B(M×N) = K×N
    {
        auto a = make_host<double>(M, K, 0.4);
        auto b = make_host<double>(M, N, 0.55);
        cuda::dev_matrix_t<double> da(M, K, a), db(M, N, b);
        expect_matrices_match(cuda::matmul(da.leaf().t(), db.leaf()).download(),
                              a.t().dot(b).clone(), 1e-12, "Aᵀ·B");
    }

    // A·Bᵀ：A 是 M×K，B 是 N×K → A(M×K)·Bᵀ(K×N) = M×N
    {
        auto a = make_host<double>(M, K, 0.4);
        auto b = make_host<double>(N, K, 0.6);
        cuda::dev_matrix_t<double> da(M, K, a), db(N, K, b);
        expect_matrices_match(cuda::matmul(da.leaf(), db.leaf().t()).download(),
                              a.dot(b.t()).clone(), 1e-12, "A·Bᵀ");
    }

    // Aᵀ·Bᵀ：A 是 M×K，B 是 N×M → Aᵀ(K×M)·Bᵀ(M×N) = K×N
    {
        auto a = make_host<double>(M, K, 0.45);
        auto b = make_host<double>(N, M, 0.5);
        cuda::dev_matrix_t<double> da(M, K, a), db(N, M, b);
        expect_matrices_match(cuda::matmul(da.leaf().t(), db.leaf().t()).download(),
                              a.t().dot(b.t()).clone(), 1e-12, "Aᵀ·Bᵀ");
    }
}

TEST_F(CudaDotTest, TransposeDoesNotAlterTheStoredLayout)
{
    // 转置只翻标志位，不改前导维 —— 这正是 cuBLAS 映射正确的前提。
    // 形状不对称（3×7），把"转置后前导维被改成 3"这类错误暴露出来
    auto a = make_host<double>(3, 7, 0.5);
    cuda::dev_matrix_t<double> da(3, 7, a);

    const auto plain = da.leaf();
    const auto tr = da.leaf().t();
    EXPECT_EQ(plain.leading_dim(), 7);
    EXPECT_EQ(tr.leading_dim(), 7) << "转置不该改变存储步长";
    EXPECT_EQ(plain.row_num(), 3);
    EXPECT_EQ(plain.col_num(), 7);
    EXPECT_EQ(tr.row_num(), 7);
    EXPECT_EQ(tr.col_num(), 3);
    EXPECT_EQ(plain.m_data, tr.m_data) << "转置是同一块内存的另一种解释";
}

// ---------------------------------------------------------------------------
// 链式调用与表达式操作数
// ---------------------------------------------------------------------------

TEST_F(CudaDotTest, DotChainsThroughOwners)
{
    // A·B 返回拥有者，所以可以继续 .dot(C) —— 对应主机端的 a.dot(b).dot(c)
    const int M = 5, K = 4, N = 6, P = 3;
    auto a = make_host<double>(M, K, 0.3);
    auto b = make_host<double>(K, N, 0.4);
    auto c = make_host<double>(N, P, 0.5);

    cuda::dev_matrix_t<double> da(M, K, a), db(K, N, b), dc(N, P, c);
    auto got = da.dot(db).dot(dc).download();
    expect_matrices_match(got, a.dot(b).dot(c).clone(), 1e-12, "链式 dot");
}

TEST_F(CudaDotTest, MatmulAcceptsExpressionOperands)
{
    // 任意设备可求值的表达式都能当 GEMM 操作数：非叶子的先融合物化再进 GEMM
    const int M = 6, K = 5, N = 7;
    auto a = make_host<double>(M, K, 0.3);
    auto b = make_host<double>(M, K, 0.4);
    auto c = make_host<double>(K, N, 0.5);

    cuda::dev_matrix_t<double> da(M, K, a), db(M, K, b), dc(K, N, c);

    // (a + b) · c
    auto got = cuda::matmul(da.leaf() + db.leaf(), dc.leaf()).download();
    expect_matrices_match(got, (a + b).dot(c).clone(), 1e-12, "表达式·叶子");

    // a · (c * 2)
    auto got2 = cuda::matmul(da, dc.leaf() * 2.0).download();
    expect_matrices_match(got2, a.dot(c * 2.0).clone(), 1e-12, "拥有者·表达式");
}

TEST_F(CudaDotTest, DotResultComposesBackIntoExpressions)
{
    // 结果可再用 .leaf() 参与表达式运算 —— 这是"分派点可组合"的关键
    const int M = 6, K = 4, N = 5;
    auto a = make_host<double>(M, K, 0.3);
    auto b = make_host<double>(K, N, 0.4);
    auto bias = make_host<double>(M, N, 0.1);

    cuda::dev_matrix_t<double> da(M, K, a), db(K, N, b), dbias(M, N, bias);
    auto prod = da.leaf().dot(db.leaf());

    auto got = cuda::eval_fused_to_host(exp(prod.leaf() / 2.0) + dbias.leaf());
    auto ref = (exp(a.dot(b) / 2.0) + bias).clone();
    expect_matrices_match(got, ref, 1e-12, "GEMM 结果参与融合");
}

// ---------------------------------------------------------------------------
// 错误处理
// ---------------------------------------------------------------------------

TEST_F(CudaDotTest, InnerDimensionMismatchThrows)
{
    cuda::dev_matrix_t<double> a(4, 3), b(5, 2);   // 3 != 5
    EXPECT_THROW(cuda::matmul(a.leaf(), b.leaf()), std::invalid_argument);
    EXPECT_THROW(a.dot(b), std::invalid_argument);
}

TEST_F(CudaDotTest, DimensionCheckAccountsForTransposition)
{
    // 转置会改变内维，所以校验必须在「把转置考虑进去」之后做。
    // a 是 4×3、b 是 4×5：直接用 a 内维是 3 != 4，该抛；
    // 用 aᵀ（3×4）内维就是 4 == 4，合法。两者形状相同的内存，结果却相反 ——
    // 这正好证明校验读的是叶子报出的（已含转置的）形状，而不是原始存储。
    cuda::dev_matrix_t<double> a(4, 3), b(4, 5);

    EXPECT_THROW(cuda::matmul(a.leaf(), b.leaf()), std::invalid_argument);
    EXPECT_NO_THROW(cuda::matmul(a.leaf().t(), b.leaf()));
}

// ---------------------------------------------------------------------------
// float32
// ---------------------------------------------------------------------------

TEST_F(CudaDotTest, Float32MatmulMatchesHost)
{
    const int M = 10, K = 8, N = 12;
    auto a = make_host<float>(M, K, 0.3f);
    auto b = make_host<float>(K, N, 0.4f);

    cuda::dev_matrix_t<float> da(M, K, a), db(K, N, b);
    auto got = da.leaf().dot(db.leaf()).download();
    auto ref = a.dot(b).clone();

    ASSERT_EQ(got.row_num(), ref.row_num());
    ASSERT_EQ(got.col_num(), ref.col_num());
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            ASSERT_NEAR(got(i, j), ref(i, j), 1e-5f * std::max(1.0f, std::abs(ref(i, j))));
}

// ---------------------------------------------------------------------------
// 注意力：用设备端自然语法重写主机端的算式
// ---------------------------------------------------------------------------

TEST_F(CudaDotTest, AttentionScoresUseTheNaturalDotSyntax)
{
    // 主机端 mat_head_gen_t::forward_at 里是 `m_q.t().dot(m_k)`；
    // 设备端应当能写成几乎一样的 `q.leaf().t().dot(k.leaf())`
    const int d_head = 8, seq = 16;
    auto q = make_host<double>(d_head, seq, 0.4);
    auto k = make_host<double>(d_head, seq, 0.5);

    cuda::dev_matrix_t<double> dq(d_head, seq, q), dk(d_head, seq, k);
    auto scores = dq.leaf().t().dot(dk);        // 拥有者·拥有者

    expect_matrices_match(scores.download(), q.t().dot(k).clone(), 1e-12, "Q·Kᵀ 打分");
}

TEST_F(CudaDotTest, FullAttentionForwardReadsLikeTheHostVersion)
{
    // 端到端注意力，全程设备：打分(GEMM) → 缩放+掩码(融合) → softmax(归约) →
    // 加权求和(GEMM)。除输入上传/结果回读外不回主机。
    const int d_head = 8, seq = 12;
    auto q = make_host<double>(d_head, seq, 0.35);
    auto k = make_host<double>(d_head, seq, 0.45);
    auto v = make_host<double>(d_head, seq, 0.3);

    // ---- 主机参考（与 jas_mha_t.hpp 同序）----
    auto scores_h = (q.t().dot(k) / std::sqrt(static_cast<double>(d_head))).clone();
    for (int i = 0; i < seq; ++i)
        for (int j = i + 1; j < seq; ++j)
            scores_h(i, j) = -std::numeric_limits<double>::infinity();
    auto ref = v.dot(hsoftmax(scores_h).t()).clone();

    // ---- 设备 ----
    cuda::dev_matrix_t<double> dq(d_head, seq, q), dk(d_head, seq, k), dv(d_head, seq, v);

    mat_t<double> mask_h(seq, seq);
    mask_h = 0.0;
    for (int i = 0; i < seq; ++i)
        for (int j = i + 1; j < seq; ++j)
            mask_h(i, j) = -std::numeric_limits<double>::infinity();
    cuda::dev_matrix_t<double> dmask(seq, seq, mask_h);

    auto scores = dq.leaf().t().dot(dk);                               // GEMM
    auto weights = cuda::softmax_rows(scores.leaf() / std::sqrt(static_cast<double>(d_head))
                                      + dmask.leaf());                  // 融合 + 归约
    auto out = dv.leaf().dot(weights.leaf().t());                      // GEMM

    expect_matrices_match(out.download(), ref, 1e-11, "注意力前向（自然语法）");
}
