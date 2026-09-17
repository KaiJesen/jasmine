#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

/**
 * 归约 / softmax / 归一化层的设备端测试。
 *
 * 散热约束同 test_cuda_fused.cu：本机是无风扇的 Tesla P4，所以矩阵都很小、
 * 每个用例前后测温、超 80℃ 就 skip。详见 CUDA.md。
 *
 * 一个重要前提：**这些用例全部与主机端实现逐元素对拍**。归约会改变求和顺序
 * （warp shuffle 是树形归约，主机是顺序累加），所以末位 ulp 必然有差异 ——
 * 用相对容差描述这个契约。
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

/** 确定性地填一个矩阵；刻意让各行/各列的量级不同，避免掩盖广播方向搞反的错误。 */
template <typename T>
mat_t<T> make_host(int rows, int cols, T base)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + T(0.01) * ((i * 7 + j * 13) % 41) - T(0.02) * i + T(0.03) * j;
    return m;
}

/** 含负值的填充，用于 softmax 的数值稳定性验证。 */
template <typename T>
mat_t<T> make_host_signed(int rows, int cols, T amp)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = amp * ((i * 5 + j * 11) % 23 - 11);
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

class CudaReduceTest : public ::testing::Test
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
// 广播叶子：类型层面的保证（零发热）
// ---------------------------------------------------------------------------

TEST(CudaReduceContract, BroadcastLeavesSatisfyMatrixConcept)
{
    // 广播叶子是鸭子类型的矩阵：有 row_num/col_num/operator() 就能参与表达式
    static_assert(is_matrix<dev_col_leaf_t<double>>);
    static_assert(is_matrix<dev_row_leaf_t<double>>);
    static_assert(is_caculable<dev_col_leaf_t<double>, dev_mat_t<double>>);
    SUCCEED();
}

TEST(CudaReduceContract, BroadcastLeavesAreDeviceEvaluableAndKernelPassable)
{
    static_assert(is_device_evaluable_v<dev_col_leaf_t<double>>);
    static_assert(is_device_evaluable_v<dev_row_leaf_t<double>>);
    static_assert(std::is_trivially_copyable_v<dev_col_leaf_t<double>>);
    static_assert(std::is_trivially_copyable_v<dev_row_leaf_t<double>>);
    SUCCEED();
}

TEST(CudaReduceContract, BroadcastLeavesDeclareTheirShape)
{
    // col 叶子是 (rows × 1)，row 叶子是 (1 × cols)：
    // 形状如实上报，表达式才能用 device_max 推出正确的并集形状
    dev_col_leaf_t<double> c(nullptr, 5);
    dev_row_leaf_t<double> r(nullptr, 7);
    EXPECT_EQ(c.row_num(), 5);
    EXPECT_EQ(c.col_num(), 1);
    EXPECT_EQ(r.row_num(), 1);
    EXPECT_EQ(r.col_num(), 7);
}

TEST(CudaReduceContract, BroadcastLeavesAreOwnedByValueWhenNamed)
{
    // 与 dev_mat_t 同理：设备端借引用没有意义
    using tree_t = decltype(std::declval<dev_mat_t<double>&>() - std::declval<dev_col_leaf_t<double>&>());
    static_assert(!std::is_reference_v<typename tree_t::lval_storage_type>,
                  "具名广播叶子也必须按值拥有");
    static_assert(tree_t::device_evaluable);
    SUCCEED();
}

TEST(CudaReduceContract, DeviceExpressionNodesAreOwnedByValueEvenAsLvalues)
{
    // 这是一次真实事故的回归哨兵。
    //
    // 归约接口的形参是 `Expr const&`，于是传进来的具名表达式节点是【左值】，
    // 会被按引用借进派生出的子树。树能编译、能拷贝，`is_trivially_copyable` 也为真
    // （含引用成员的类照样平凡可拷贝），但 kernel 参数是按值搬到设备上的 ——
    // 搬过去的是主机栈地址，设备端一解引用就是 cudaErrorIllegalAddress。
    //
    // 规则：只要操作数 device_evaluable，就一律按值拥有。
    auto counter = dev_col_leaf_t<double>(nullptr, 3);
    auto leaf = dev_mat_t<double>(nullptr, 3, 4);
    auto node = leaf * 2.0;                       // 具名 → 左值表达式节点
    using node_t = decltype(node);
    static_assert(node_t::device_evaluable);

    auto derived = node - counter;                // 把左值节点借进新树
    static_assert(std::is_same_v<typename decltype(derived)::lval_storage_type, node_t>,
                  "左值的设备表达式节点必须按值存，而不是存成 node_t const&");
    static_assert(is_self_contained_v<decltype(derived)>,
                  "借了左值节点之后，新树也必须自持");

    auto deep = exp(derived / counter);           // 再串一层
    static_assert(is_self_contained_v<decltype(deep)>);
    static_assert(is_device_evaluable_v<decltype(deep)>);
    static_assert(std::is_trivially_copyable_v<decltype(deep)>);
    EXPECT_EQ(deep.row_num(), 3);
    EXPECT_EQ(deep.col_num(), 4);
}

TEST(CudaReduceContract, HostTreesStillBorrowLvaluesSoNoRegression)
{
    // 上一条改成「按值拥有」只应作用于设备树。主机端必须保持借引用（零拷贝），
    // 否则表达式模板最基本的卖点（不深拷贝大矩阵）就没了。
    mat_t<double> a(2, 2), b(2, 2);
    auto sum_tree = a + b;                         // a、b 都是左值
    using tree_t = decltype(sum_tree);

    static_assert(std::is_same_v<typename tree_t::lval_storage_type, const mat_t<double>&>,
                  "主机左值操作数必须仍然借引用");
    static_assert(!tree_t::device_evaluable,
                  "含 mat_t 的树不应当在设备上求值");
    SUCCEED();
}

TEST(CudaReduceContract, ReductionResultOwnersAreNotThemselvesDeviceLeaves)
{
    // 归约返回的是「拥有缓冲区的向量」（dev_colvec_t / dev_rowvec_t），它不参与表达式 ——
    // 参与表达式的是它现场取出的广播叶子。这条区分很重要：
    // 拥有者管生命周期，叶子管求值，混在一起就会像 dev_matrix_t 那样不能再传进 kernel。
    static_assert(!is_device_evaluable_v<cuda::dev_colvec_t<double>>,
                  "拥有者本身不是设备可求值的叶子");
    static_assert(!is_device_evaluable_v<cuda::dev_rowvec_t<double>>,
                  "拥有者本身不是设备可求值的叶子");
    static_assert(
        is_device_evaluable_v<decltype(std::declval<cuda::dev_colvec_t<double>&>().leaf())>);
    static_assert(
        is_device_evaluable_v<decltype(std::declval<cuda::dev_rowvec_t<double>&>().leaf())>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// 归约：与主机端逐元素对拍
// ---------------------------------------------------------------------------

TEST_F(CudaReduceTest, HsumMatchesHost)
{
    const int rows = 23, cols = 37;   // 非 2 的幂，覆盖不满 warp 的边界
    auto h = make_host<double>(rows, cols, 1.0);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::row_sum(d.leaf()).download();
    ASSERT_EQ(got.row_num(), rows);
    ASSERT_EQ(got.col_num(), 1);
    expect_matrices_match(got, hsum(h).clone(), 1e-12, "hsum");
}

TEST_F(CudaReduceTest, HmaxMatchesHost)
{
    const int rows = 19, cols = 41;
    auto h = make_host_signed<double>(rows, cols, 1.0);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::row_max(d.leaf()).download();
    expect_matrices_match(got, hmax(h).clone(), 1e-12, "hmax");
}

TEST_F(CudaReduceTest, VsumMatchesHost)
{
    // vsum 是逐列归约，块内按 (行 × 列) 二维映射；这里验证列语义没搞反
    const int rows = 31, cols = 17;
    auto h = make_host<double>(rows, cols, 2.0);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::col_sum(d.leaf()).download();
    ASSERT_EQ(got.row_num(), 1);
    ASSERT_EQ(got.col_num(), cols);
    expect_matrices_match(got, vsum(h).clone(), 1e-12, "vsum");
}

TEST_F(CudaReduceTest, VmaxMatchesColumnwiseMax)
{
    // 主机端没有 vmax，所以用手算的逐列最大值对拍
    const int rows = 13, cols = 29;
    auto h = make_host_signed<double>(rows, cols, 1.0);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::col_max(d.leaf()).download();
    ASSERT_EQ(got.col_num(), cols);

    mat_t<double> ref(1, cols);
    for (int j = 0; j < cols; ++j)
    {
        double m = std::numeric_limits<double>::lowest();
        for (int i = 0; i < rows; ++i)
            m = std::max(m, h(i, j));
        ref(0, j) = m;
    }
    expect_matrices_match(got, ref, 1e-12, "vmax");
}

TEST_F(CudaReduceTest, FullSumMatchesHost)
{
    const int rows = 47, cols = 53;
    auto h = make_host<double>(rows, cols, 1.0);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    const double got = cuda::sum_all(d.leaf());
    const double ref = sum(h);
    ASSERT_NEAR(got, ref, 1e-12 * std::max(1.0, std::abs(ref)));
}

TEST_F(CudaReduceTest, FullMaxMatchesHost)
{
    const int rows = 37, cols = 43;
    auto h = make_host_signed<double>(rows, cols, 1.0);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    ASSERT_NEAR(cuda::max_all(d.leaf()), max(h), 1e-12);
}

TEST_F(CudaReduceTest, VmeanMatchesHost)
{
    const int rows = 21, cols = 33;
    auto h = make_host<double>(rows, cols, 1.5);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    expect_matrices_match(cuda::col_mean(d.leaf()).download(), vmean(h).clone(), 1e-12, "vmean");
}

TEST_F(CudaReduceTest, ReductionAxesAreNotInterchanged)
{
    // 形状不对称的矩阵 + 各不相同的行和/列和：把 hsum 和 vsum 搞反必然被抓到
    const int rows = 6, cols = 10;
    mat_t<double> h(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            h(i, j) = static_cast<double>(i + 1) * 100.0 + static_cast<double>(j + 1);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto hs = cuda::row_sum(d.leaf()).download();   // (rows × 1)
    auto vs = cuda::col_sum(d.leaf()).download();   // (1 × cols)

    EXPECT_EQ(hs.row_num(), rows);
    EXPECT_EQ(vs.col_num(), cols);
    expect_matrices_match(hs, hsum(h).clone(), 1e-12, "hsum 轴");
    expect_matrices_match(vs, vsum(h).clone(), 1e-12, "vsum 轴");
}

TEST_F(CudaReduceTest, ReductionFusesTheExpressionAway)
{
    // 归约读的是整棵表达式树，所以这里没有任何中间物化：
    // hsum(exp(x * 2)) 一趟就完成"乘 2 → exp → 求和"
    const int rows = 15, cols = 25;
    auto h = make_host<double>(rows, cols, 0.1);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::row_sum(exp(d.leaf() * 2.0)).download();
    expect_matrices_match(got, hsum(exp(h * 2.0)).clone(), 1e-12, "融合归约");
}

TEST_F(CudaReduceTest, Float32ReductionsMatchHost)
{
    const int rows = 26, cols = 30;
    auto h = make_host<float>(rows, cols, 1.0f);
    cuda::dev_matrix_t<float> d(rows, cols, h);

    auto got = cuda::row_sum(d.leaf()).download();
    auto ref = hsum(h).clone();
    ASSERT_EQ(got.row_num(), ref.row_num());
    for (int i = 0; i < rows; ++i)
        ASSERT_NEAR(got(i, 0), ref(i, 0), 1e-5f * std::max(1.0f, std::abs(ref(i, 0))));
}

TEST_F(CudaReduceTest, SingleRowAndSingleColumnEdges)
{
    {
        auto h = make_host<double>(1, 64, 1.0);
        cuda::dev_matrix_t<double> d(1, 64, h);
        expect_matrices_match(cuda::row_sum(d.leaf()).download(), hsum(h).clone(), 1e-12, "单行 hsum");
        expect_matrices_match(cuda::col_sum(d.leaf()).download(), vsum(h).clone(), 1e-12, "单行 vsum");
    }
    {
        auto h = make_host<double>(64, 1, 1.0);
        cuda::dev_matrix_t<double> d(64, 1, h);
        expect_matrices_match(cuda::row_sum(d.leaf()).download(), hsum(h).clone(), 1e-12, "单列 hsum");
        expect_matrices_match(cuda::col_sum(d.leaf()).download(), vsum(h).clone(), 1e-12, "单列 vsum");
    }
    {
        auto h = make_host<double>(1, 1, 3.0);
        cuda::dev_matrix_t<double> d(1, 1, h);
        auto got = cuda::row_sum(d.leaf()).download();
        ASSERT_EQ(got.row_num(), 1);
        ASSERT_NEAR(got(0, 0), 3.0, 1e-12);
        ASSERT_NEAR(cuda::sum_all(d.leaf()), 3.0, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// 逐行 softmax
// ---------------------------------------------------------------------------

TEST_F(CudaReduceTest, HsoftmaxMatchesHost)
{
    const int rows = 12, cols = 20;
    auto h = make_host_signed<double>(rows, cols, 0.7);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::softmax_rows(d.leaf()).download();
    expect_matrices_match(got, hsoftmax(h).clone(), 1e-12, "hsoftmax");
}

TEST_F(CudaReduceTest, HsoftmaxRowsSumToOne)
{
    const int rows = 17, cols = 23;
    auto h = make_host_signed<double>(rows, cols, 1.3);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::softmax_rows(d.leaf()).download();
    for (int i = 0; i < rows; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < cols; ++j)
            s += got(i, j);
        ASSERT_NEAR(s, 1.0, 1e-12) << "第 " << i << " 行";
    }
}

TEST_F(CudaReduceTest, HsoftmaxIsNumericallyStableForLargeInputs)
{
    // 不减去行最大值的话 exp(1000) 会溢出成 inf；减了之后各行应当归一
    const int rows = 8, cols = 16;
    mat_t<double> h(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            h(i, j) = 1000.0 + static_cast<double>(i * 3 + j);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::softmax_rows(d.leaf()).download();
    for (int i = 0; i < rows; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < cols; ++j)
        {
            ASSERT_TRUE(std::isfinite(got(i, j))) << "第 " << i << "," << j << " 行出现非有限值";
            s += got(i, j);
        }
        ASSERT_NEAR(s, 1.0, 1e-12);
    }
    expect_matrices_match(got, hsoftmax(h).clone(), 1e-12, "大值 hsoftmax");
}

TEST_F(CudaReduceTest, HsoftmaxHandlesCausalMaskMinusInfinity)
{
    // 注意力掩码用 -inf 填上三角：exp(-inf - finite) 必须是 0，
    // 且不得让整行的和变成 nan
    const int n = 6;
    mat_t<double> h = make_host_signed<double>(n, n, 0.5);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            h(i, j) = -std::numeric_limits<double>::infinity();

    cuda::dev_matrix_t<double> d(n, n, h);
    auto got = cuda::softmax_rows(d.leaf()).download();

    for (int i = 0; i < n; ++i)
    {
        // 未来位置必须恰好是 0
        for (int j = i + 1; j < n; ++j)
            ASSERT_EQ(got(i, j), 0.0) << "掩码位置 (" << i << "," << j << ") 应为 0";

        double s = 0.0;
        for (int j = 0; j <= i; ++j)
        {
            ASSERT_TRUE(std::isfinite(got(i, j)));
            s += got(i, j);
        }
        ASSERT_NEAR(s, 1.0, 1e-12) << "第 " << i << " 行";
    }
    expect_matrices_match(got, hsoftmax(h).clone(), 1e-12, "掩码 hsoftmax");
}

TEST_F(CudaReduceTest, HsoftmaxAcceptsAnExpression)
{
    // softmax 的输入可以是表达式，整个输入链都会被融进三趟里
    const int rows = 10, cols = 14;
    auto h = make_host_signed<double>(rows, cols, 0.4);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::softmax_rows(d.leaf() * 2.0).download();
    expect_matrices_match(got, hsoftmax(h * 2.0).clone(), 1e-12, "表达式 hsoftmax");
}

// ---------------------------------------------------------------------------
// softmax 的路径选择：单趟共享内存 / 三趟回退
// ---------------------------------------------------------------------------

namespace
{

/**
 * 覆盖值的作用域守卫。
 *
 * 用 ASSERT_* 的辅助函数会提前 return，不能靠函数末尾手动恢复；一旦残留，
 * 后续用例会莫名其妙地走错路径。析构恢复才可靠。
 */
struct SoftmaxPathScope
{
    SoftmaxPathScope() { cuda::softmax_max_cols_override() = -1; }
    ~SoftmaxPathScope() { cuda::softmax_max_cols_override() = -1; }

    void force_fallback() { cuda::softmax_max_cols_override() = 0; }
};

/**
 * 同一份输入分别走单趟路径与三趟回退路径，都要求与主机 `hsoftmax` 一致。
 *
 * 这是「两条路径不许分叉」的直接表达。必须显式检查**走了哪条路**：
 * 优化的典型失败方式不是算错，而是压根没生效 —— 阈值算错、分支写反，
 * 结果照样正确、测例照样全绿。计数器就是防这个的。
 */
void expect_softmax_both_paths_match_host(const mat_t<double>& h, const char* what,
                                          SoftmaxPathScope& scope)
{
    cuda::dev_matrix_t<double> d(h.row_num(), h.col_num(), h);
    const mat_t<double> ref = hsoftmax(h).clone();
    const int before = cuda::softmax_shared_launch_count();

    cuda::softmax_max_cols_override() = -1;
    auto fast = cuda::softmax_rows(d.leaf()).download();
    ASSERT_EQ(cuda::softmax_shared_launch_count(), before + 1)
        << what << "：行放得下时应当走单趟路径";
    expect_matrices_match(fast, ref, 1e-12, what);

    scope.force_fallback();
    auto slow = cuda::softmax_rows(d.leaf()).download();
    ASSERT_EQ(cuda::softmax_shared_launch_count(), before + 1)
        << what << "：覆盖值生效后不该再走单趟路径";
    expect_matrices_match(slow, ref, 1e-12, what);

    cuda::softmax_max_cols_override() = -1;
}

} // namespace

TEST_F(CudaReduceTest, BothSoftmaxPathsMatchHost)
{
    SoftmaxPathScope scope;

    expect_softmax_both_paths_match_host(make_host_signed<double>(12, 20, 0.7), "普通输入", scope);
    expect_softmax_both_paths_match_host(make_host_signed<double>(1, 300, 0.9), "单行宽", scope);
    expect_softmax_both_paths_match_host(make_host_signed<double>(5, 2, 1.7), "窄行", scope);

    // 大值：不先减最大值就会溢出成 inf
    mat_t<double> big(6, 18);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 18; ++j)
            big(i, j) = 1000.0 + static_cast<double>(i * 3 + j);
    expect_softmax_both_paths_match_host(big, "大值", scope);

    // 因果掩码：-inf 在两条路径上都必须恰好归零
    const int n = 7;
    mat_t<double> masked = make_host_signed<double>(n, n, 0.5);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            masked(i, j) = -std::numeric_limits<double>::infinity();
    expect_softmax_both_paths_match_host(masked, "因果掩码", scope);
}

TEST_F(CudaReduceTest, SoftmaxPathsAgreeOnFloat)
{
    cuda::softmax_max_cols_override() = -1;
    const int rows = 9, cols = 15;
    auto h = make_host_signed<float>(rows, cols, 0.6f);
    cuda::dev_matrix_t<float> d(rows, cols, h);

    const int before = cuda::softmax_shared_launch_count();
    auto fast = cuda::softmax_rows(d.leaf()).download();
    ASSERT_EQ(cuda::softmax_shared_launch_count(), before + 1);

    cuda::softmax_max_cols_override() = 0;
    auto slow = cuda::softmax_rows(d.leaf()).download();
    ASSERT_EQ(cuda::softmax_shared_launch_count(), before + 1);
    cuda::softmax_max_cols_override() = -1;

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            ASSERT_NEAR(fast(i, j), slow(i, j), 1e-6)
                << "float 两条路径在 (" << i << "," << j << ") 不一致";
}

TEST_F(CudaReduceTest, SoftmaxPicksFallbackWhenRowExceedsSharedMemory)
{
    // 不设覆盖，用真实阈值：造一行放不进共享内存的输入，
    // 必须自动回退而不是让 kernel 启动失败
    cuda::softmax_max_cols_override() = -1;
    const int max_cols = cuda::softmax_shared_max_cols<double>();
    ASSERT_GT(max_cols, 0) << "设备报出的动态共享内存上限异常";
    const int cols = max_cols + 1;

    auto h = make_host_signed<double>(1, cols, 0.25);
    cuda::dev_matrix_t<double> d(1, cols, h);

    const int before = cuda::softmax_shared_launch_count();
    auto got = cuda::softmax_rows(d.leaf()).download();
    ASSERT_EQ(cuda::softmax_shared_launch_count(), before)
        << "行放不下时应当自动回退到三趟路径";

    double s = 0.0;
    for (int j = 0; j < cols; ++j)
        s += got(0, j);
    ASSERT_NEAR(s, 1.0, 1e-12);
}

TEST_F(CudaReduceTest, SoftmaxSharedPathHandlesMaxSizedRow)
{
    // 边界：恰好用满共享内存预算的那一行仍要走单趟路径，
    // 同时也就验证了 cudaFuncSetAttribute 的 opt-in 生效
    cuda::softmax_max_cols_override() = -1;
    const int cols = cuda::softmax_shared_max_cols<double>();
    ASSERT_GT(cols, 0);

    auto h = make_host_signed<double>(2, cols, 0.3);
    cuda::dev_matrix_t<double> d(2, cols, h);

    const int before = cuda::softmax_shared_launch_count();
    auto got = cuda::softmax_rows(d.leaf()).download();
    ASSERT_EQ(cuda::softmax_shared_launch_count(), before + 1)
        << "恰好装满共享内存的行应当仍走单趟路径";

    for (int i = 0; i < 2; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < cols; ++j)
            s += got(i, j);
        ASSERT_NEAR(s, 1.0, 1e-12) << "第 " << i << " 行";
    }
}

// ---------------------------------------------------------------------------
// LayerNorm / RMSNorm
// ---------------------------------------------------------------------------

TEST_F(CudaReduceTest, LayerNormMatchesHostNet)
{
    const int d_model = 24, seq = 9;
    auto x = make_host<double>(d_model, seq, 0.3);

    layer_norm_net_t<mat_t<double>, nadam_t> ln;
    ln.set_param(d_model);
    // 用非平凡的 gamma / beta，否则仿射部分等于没测
    for (int i = 0; i < d_model; ++i)
    {
        ln.gama()(i, 0) = 0.5 + 0.1 * i;
        ln.beta()(i, 0) = -0.2 + 0.05 * i;
    }
    auto ref = ln.forward(x);

    cuda::dev_matrix_t<double> dx(d_model, seq, x);
    cuda::dev_colvec_t<double> g(d_model), b(d_model);
    g.buffer().upload(ln.gama().data(), d_model);
    b.buffer().upload(ln.beta().data(), d_model);

    auto got = cuda::layer_norm(dx.leaf(), g, b, 1e-5).download();
    expect_matrices_match(got, ref, 1e-12, "layer_norm");
}

TEST_F(CudaReduceTest, LayerNormNormalizesEachColumn)
{
    // 独立于主机 net 的第二重验证：gamma=1/beta=0 时每列应当零均值、单位方差。
    // 这与 tests/test_layer_norm.cpp 的 PerColumnZeroMeanUnitVariance 是同一个契约 ——
    // "按列"是重点：统计量沿行（特征维）算。
    const int d_model = 16, seq = 5;
    auto x = make_host<double>(d_model, seq, 0.4);
    cuda::dev_matrix_t<double> dx(d_model, seq, x);

    cuda::dev_colvec_t<double> g(d_model), b(d_model);
    std::vector<double> ones(d_model, 1.0), zeros(d_model, 0.0);
    g.buffer().upload(ones.data(), d_model);
    b.buffer().upload(zeros.data(), d_model);

    auto got = cuda::layer_norm(dx.leaf(), g, b, 1e-5).download();

    // 期望值不是「恰好 1.0」：输出方差 = var / (var + eps)，只是 eps 通常小到看不见。
    // 这里输入的列方差只有 ~0.02，1e-5 的 eps 就显出 5e-4 量级了 —— 所以要按公式比，
    // 而不是用一个拍脑袋的绝对容差去掩盖它。
    constexpr double eps = 1e-5;
    for (int j = 0; j < seq; ++j)
    {
        double in_mean = 0.0;
        for (int i = 0; i < d_model; ++i)
            in_mean += x(i, j);
        in_mean /= d_model;

        double in_var = 0.0;
        for (int i = 0; i < d_model; ++i)
        {
            const double dv = x(i, j) - in_mean;
            in_var += dv * dv;
        }
        in_var /= d_model;

        double out_mean = 0.0;
        for (int i = 0; i < d_model; ++i)
            out_mean += got(i, j);
        out_mean /= d_model;
        EXPECT_NEAR(out_mean, 0.0, 1e-9) << "第 " << j << " 列均值";

        double out_var = 0.0;
        for (int i = 0; i < d_model; ++i)
        {
            const double dv = got(i, j) - out_mean;
            out_var += dv * dv;
        }
        out_var /= d_model;
        EXPECT_NEAR(out_var, in_var / (in_var + eps), 1e-9) << "第 " << j << " 列方差";
    }
}

TEST_F(CudaReduceTest, HsoftmaxAcceptsANamedExpression)
{
    // 触发过真实事故的形态：具名（左值）表达式节点作为输入。
    // 修复前这里会抛 cudaErrorIllegalAddress。
    const int rows = 10, cols = 14;
    auto h = make_host_signed<double>(rows, cols, 0.4);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto named = d.leaf() * 2.0;
    auto got = cuda::softmax_rows(named).download();
    expect_matrices_match(got, hsoftmax(h * 2.0).clone(), 1e-12, "具名表达式 softmax");
}

TEST_F(CudaReduceTest, FusedEvalAcceptsANamedExpression)
{
    // 同一个隐患在 eval_fused 上也存在（它同样按值把树搬进 kernel），一并锁住
    const int rows = 9, cols = 13;
    auto h = make_host_signed<double>(rows, cols, 0.5);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto named = exp(d.leaf() * 0.5) + 1.0;
    auto got = cuda::eval_fused_to_host(named);
    expect_matrices_match(got, (exp(h * 0.5) + 1.0).clone(), 1e-12, "具名表达式融合");
}

TEST_F(CudaReduceTest, ReductionAcceptsANamedExpression)
{
    const int rows = 11, cols = 15;
    auto h = make_host<double>(rows, cols, 0.6);
    cuda::dev_matrix_t<double> d(rows, cols, h);

    auto named = d.leaf() * 3.0;
    expect_matrices_match(cuda::row_sum(named).download(), hsum(h * 3.0).clone(), 1e-12,
                          "具名表达式 row_sum");
}

TEST_F(CudaReduceTest, AttentionForwardMatchesHost)
{
    // 端到端：GEMM(cuBLAS) → 逐元素缩放/加掩码 → 逐行 softmax → GEMM，
    // 算式对齐主机端 mat_head_gen_t::forward_at（不含 RoPE）。
    // 这条用例的价值在于证明几个部件能真的拼起来，而不只是各自能跑：
    // cuBLAS 的输出直接喂给融合 kernel，softmax 的结果又直接喂回 cuBLAS，中间无需回主机。
    const int d_head = 8, seq = 12;
    auto q = make_host_signed<double>(d_head, seq, 0.5);
    auto k = make_host_signed<double>(d_head, seq, 0.5);
    auto v = make_host_signed<double>(d_head, seq, 0.4);

    // ---- 主机参考（与 jas_mha_t.hpp 同序）----
    auto scores_h = (q.t().dot(k) / std::sqrt(static_cast<double>(d_head))).clone();
    for (int i = 0; i < seq; ++i)
        for (int j = i + 1; j < seq; ++j)
            scores_h(i, j) = -std::numeric_limits<double>::infinity();
    auto ref = (v.dot(hsoftmax(scores_h).t())).clone();

    // ---- 设备 ----
    cuda::dev_matrix_t<double> dq(d_head, seq, q);
    cuda::dev_matrix_t<double> dk(d_head, seq, k);
    cuda::dev_matrix_t<double> dv(d_head, seq, v);

    // 掩码做成叶子直接加进表达式：0 或 -inf，于是"缩放 + 掩码"融合成一趟
    mat_t<double> mask_h(seq, seq);
    mask_h = 0.0;
    for (int i = 0; i < seq; ++i)
        for (int j = i + 1; j < seq; ++j)
            mask_h(i, j) = -std::numeric_limits<double>::infinity();
    cuda::dev_matrix_t<double> dmask(seq, seq, mask_h);

    // scores = q^T·k，留在设备上
    cuda::dev_matrix_t<double> dscores(seq, seq,
                                       cuda::gemm_to_host(dq.leaf_transposed(), dk.leaf()));

    const double scale = std::sqrt(static_cast<double>(d_head));
    auto weights = cuda::softmax_rows(dscores.leaf() / scale + dmask.leaf());

    auto got = cuda::gemm_to_host(dv.leaf(), weights.leaf_transposed());
    expect_matrices_match(got, ref, 1e-11, "attention 前向");
}

TEST_F(CudaReduceTest, AttentionCausalPropertyHolds)
{
    // 因果性的行为检查（不依赖参考实现）：改动第 t 个 token 之后的内容，
    // 前 t 个位置（含 t）的输出必须一字不变。
    // 这是注意力最容易搞错的语义 —— 掩码方向反了会在这里立刻暴露。
    const int d_head = 6, seq = 8;
    auto q = make_host_signed<double>(d_head, seq, 0.5);
    auto k = make_host_signed<double>(d_head, seq, 0.5);
    auto v = make_host_signed<double>(d_head, seq, 0.4);
    auto v2 = v;
    for (int j = 4; j < seq; ++j)      // 只动 4 及以后的列
        for (int i = 0; i < d_head; ++i)
            v2(i, j) += 3.0;

    mat_t<double> mask_h(seq, seq);
    mask_h = 0.0;
    for (int i = 0; i < seq; ++i)
        for (int j = i + 1; j < seq; ++j)
            mask_h(i, j) = -std::numeric_limits<double>::infinity();

    auto run = [&](const mat_t<double>& vv)
    {
        cuda::dev_matrix_t<double> dq(d_head, seq, q);
        cuda::dev_matrix_t<double> dk(d_head, seq, k);
        cuda::dev_matrix_t<double> dv(d_head, seq, vv);
        cuda::dev_matrix_t<double> dmask(seq, seq, mask_h);
        cuda::dev_matrix_t<double> dscores(seq, seq,
                                           cuda::gemm_to_host(dq.leaf_transposed(), dk.leaf()));
        auto w = cuda::softmax_rows(dscores.leaf() / std::sqrt(static_cast<double>(d_head))
                                    + dmask.leaf());
        return cuda::gemm_to_host(dv.leaf(), w.leaf_transposed());
    };

    auto a = run(v);
    auto b = run(v2);

    for (int i = 0; i < d_head; ++i)
        for (int j = 0; j < 4; ++j)    // 位置 0..3 看不到被改动的 4..7 列
            EXPECT_NEAR(a(i, j), b(i, j), 1e-12) << "(" << i << "," << j << ") 不该受影响";
}

TEST_F(CudaReduceTest, RmsNormMatchesHostNet)
{
    const int d_model = 32, seq = 11;
    auto x = make_host<double>(d_model, seq, 0.2);

    rms_norm_net_t<mat_t<double>, nadam_t> rn;
    rn.set_param(d_model);
    for (int i = 0; i < d_model; ++i)
        rn.gama()(i, 0) = 0.8 + 0.05 * (i % 7);
    auto ref = rn.forward(x);

    cuda::dev_matrix_t<double> dx(d_model, seq, x);
    cuda::dev_colvec_t<double> g(d_model);
    g.buffer().upload(rn.gama().data(), d_model);

    auto got = cuda::rms_norm(dx.leaf(), g, rn.eps()).download();
    expect_matrices_match(got, ref, 1e-12, "rms_norm");
}

TEST_F(CudaReduceTest, RmsNormIsScaleInvariantButNotTranslationInvariant)
{
    // RMSNorm 的定义性质：缩放输入不改变输出（gamma=1 时），平移会改变。
    // 这条能在不依赖任何参考实现的前提下验证公式确实是对的。
    const int d_model = 18, seq = 4;
    auto x = make_host<double>(d_model, seq, 0.6);
    cuda::dev_matrix_t<double> dx(d_model, seq, x);

    cuda::dev_colvec_t<double> g(d_model);
    std::vector<double> ones(d_model, 1.0);
    g.buffer().upload(ones.data(), d_model);

    auto base = cuda::rms_norm(dx.leaf(), g, 0.0).download();
    auto scaled = cuda::rms_norm(dx.leaf() * 3.5, g, 0.0).download();
    expect_matrices_match(scaled, base, 1e-12, "尺度不变性");

    auto shifted = cuda::rms_norm(dx.leaf() + 2.0, g, 0.0).download();
    bool differs = false;
    for (int i = 0; i < d_model && !differs; ++i)
        for (int j = 0; j < seq && !differs; ++j)
            if (std::abs(shifted(i, j) - base(i, j)) > 1e-6)
                differs = true;
    EXPECT_TRUE(differs) << "RMSNorm 应当不是平移不变的（它不减均值）";
}

TEST_F(CudaReduceTest, NormsAcceptExpressions)
{
    // norm 的输入也可以是表达式：x*2 会被融进归一化的那几趟
    const int d_model = 20, seq = 6;
    auto x = make_host<double>(d_model, seq, 0.35);
    cuda::dev_matrix_t<double> dx(d_model, seq, x);

    cuda::dev_colvec_t<double> g(d_model);
    std::vector<double> ones(d_model, 1.0);
    g.buffer().upload(ones.data(), d_model);

    auto from_expr = cuda::rms_norm(dx.leaf() * 2.0, g, 1e-5).download();
    // 用同一份数据在主机端算一遍表达式
    auto hx2 = (x * 2.0).clone();
    cuda::dev_matrix_t<double> d2(d_model, seq, hx2);
    auto from_mat = cuda::rms_norm(d2.leaf(), g, 1e-5).download();

    expect_matrices_match(from_expr, from_mat, 1e-12, "表达式输入");
}
