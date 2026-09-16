#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "test_helpers.hpp"

/**
 * CUDA 后端的测试。
 *
 * ## 散热约束（本机只有一张无风扇的 Tesla P4）
 *
 * 这张卡没有风扇，靠机箱风道散热，长时间满负载会持续升温。所以本文件的测试刻意做成：
 *
 *   - 矩阵都很小（逐元素用例最大 512×512，GEMM 最大 256³），单次 kernel 都是**亚毫秒级**的；
 *   - 每个用例前后都测温，超过 `kHotCelsius` 就直接 skip，而不是硬着头皮跑；
 *   - 真正吃算力的用例（大矩阵、重复迭代）一律**默认关闭**，需要显式
 *     `JASMINE_CUDA_STRESS=1` 才跑。
 *
 * 这样 `ctest` 可以随便跑，不会把卡烤热。
 */

using namespace jasmine;

namespace
{

/** 结温超过这个值就不跑了。P4 的降频阈值在 85℃ 附近，留出余量。 */
constexpr int kHotCelsius = 80;

/** 读 GPU 结温。用 nvidia-smi 而不是 NVML，免得多引一个链接依赖。 */
int gpu_temperature_c()
{
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

bool stress_enabled()
{
    const char* v = std::getenv("JASMINE_CUDA_STRESS");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

/** 确定性地填一个矩阵，避免依赖随机数（两侧必须是同一份输入）。 */
template <typename T>
mat_t<T> make_host(int rows, int cols, T base)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + T(0.5) * i - T(0.25) * j + T(0.125) * (i % 7) - T(0.0625) * (j % 5);
    return m;
}

/**
 * 逐元素比对两个同形矩阵。
 *
 * 用**相对**容差而不是绝对容差：表达式里有 `exp()` 时结果量级会被指数放大
 * （`exp(13)` 就是 7e5），此时一个 ulp 的差异就有 1e-10 量级，
 * 拿 1e-12 的绝对容差去比等于在比"完全相等"，纯属误报。
 *
 * 另外，主机走 glibc 的 `std::exp`，设备走 CUDA 的 `::exp`；两者都是高精度实现
 * 但**不保证逐位一致**，"最后几个比特可能不同"是预期行为，不是 bug。
 * 相对容差正是描述这种契约的正确工具。
 */
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

void expect_matrices_match_f32(const mat_t<float>& a, const mat_t<float>& b, float rel_tol,
                               const char* what)
{
    ASSERT_EQ(a.row_num(), b.row_num()) << what;
    ASSERT_EQ(a.col_num(), b.col_num()) << what;
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < a.col_num(); ++j)
        {
            const float scale = std::max(1.0f, std::abs(b(i, j)));
            ASSERT_NEAR(a(i, j), b(i, j), rel_tol * scale)
                << what << " 在 (" << i << "," << j << ") 不一致";
        }
}

template <typename T>
using dev_matrix_t = cuda::dev_matrix_t<T>;

/**
 * 测温守门夹具。
 *
 * 过热就 skip 而不是 fail —— 目的是别把卡烤坏，不是判定代码错。
 * 同时把温度打出来，方便观察长时间跑测试时的趋势。
 */
class CudaDeviceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const int t = gpu_temperature_c();
        if (t >= kHotCelsius)
        {
            GTEST_SKIP() << "GPU 结温 " << t << "℃ 已达上限 " << kHotCelsius
                         << "℃（本机是无风扇的 Tesla P4），跳过以免继续升温";
        }
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
// 环境与类型契约（不跑 kernel，零发热）
// ---------------------------------------------------------------------------

TEST(CudaEnvironment, DeviceIsVisible)
{
    const auto& info = cuda::device_info();
    std::printf("        设备: %s\n", info.to_string().c_str());
    EXPECT_GT(info.sm_count, 0);
    EXPECT_EQ(info.compute_capability(), 61) << "本机应为 Tesla P4 (Pascal, sm_61)";
}

TEST(CudaEnvironment, DeviceLeafIsAKernelPassablePod)
{
    // 这两个性质是「整棵表达式树能当 kernel 参数递进设备」的前提
    static_assert(std::is_trivially_copyable_v<dev_mat_t<double>>,
                  "设备叶子必须平凡可拷贝，否则不能按值传给 kernel");
    static_assert(std::is_trivially_destructible_v<dev_mat_t<double>>,
                  "设备叶子不能有非平凡析构");
    SUCCEED();
}

TEST(CudaEnvironment, HostTypesAreNotDeviceEvaluable)
{
    // mat_t / mat_view_t 都是主机独占的（new[] 和主机指针），不该被允许上设备
    static_assert(!is_device_evaluable_v<mat_t<double>>,
                  "mat_t 绝不能被认为是设备可求值的");
    static_assert(!is_device_evaluable_v<mat_view_t<mat_t<double>>>,
                  "mat_view_t 绝不能被认为是设备可求值的");
    SUCCEED();
}

TEST(CudaEnvironment, DeviceLeafIsOwnedByValueEvenWhenNamed)
{
    // 设备叶子即使以具名左值出现，也必须被按值拷进树里；
    // 借引用在设备端毫无意义（设备栈上不可能有主机对象的地址）
    using tree_t = decltype(std::declval<dev_mat_t<double>&>() + std::declval<dev_mat_t<double>&>());
    static_assert(!std::is_reference_v<typename tree_t::lval_storage_type>,
                  "具名的设备叶子也必须按值拥有");
    static_assert(std::is_same_v<typename tree_t::lval_storage_type, dev_mat_t<double>>,
                  "设备叶子的存储类型应当是薄壳本身");
    static_assert(tree_t::device_evaluable, "由设备叶子构成的树应当可上设备");
    SUCCEED();
}

TEST(CudaEnvironment, ScalarLeafIsDeviceEvaluable)
{
    // 标量叶子是 POD，主机/设备通用；含标量的表达式同样可上设备
    using tree_t = decltype(std::declval<dev_mat_t<double>&>() * 2.0);
    static_assert(tree_t::device_evaluable, "含标量的表达式应当可上设备");
    SUCCEED();
}

TEST(CudaEnvironment, DotAndSoftmaxAreNotMarkedDeviceEvaluable)
{
    // dot / softmax 需要跨线程协作或两趟扫描，不能逐元素融合，
    // 指示器必须如实反映这一点，否则会被 kernel 静默地算错
    static_assert(!is_device_evaluable_v<mat_dot_t<mat_t<double>, mat_t<double>>>,
                  "mat_dot_t 不能宣称自己是设备可逐元素求值的");
    static_assert(!is_device_evaluable_v<mat_softmax_t<mat_t<double>>>,
                  "mat_softmax_t 不能宣称自己是设备可逐元素求值的");
    SUCCEED();
}

// ---------------------------------------------------------------------------
// 融合逐元素求值
// ---------------------------------------------------------------------------

TEST_F(CudaDeviceTest, SingleLeafCopiesThrough)
{
    const int rows = 8, cols = 8;
    auto h = make_host<double>(rows, cols, 1.0);
    dev_matrix_t<double> d(rows, cols, h);

    auto got = cuda::eval_fused_to_host(d.leaf());
    expect_matrices_match(got, h, 1e-12, "叶子直传");
}

TEST_F(CudaDeviceTest, AddSubMulDivMatchHost)
{
    const int rows = 17, cols = 23;   // 刻意取非 2 的幂，顺带验证边界处理
    auto ha = make_host<double>(rows, cols, 1.5);
    auto hb = make_host<double>(rows, cols, -0.75);

    dev_matrix_t<double> da(rows, cols, ha), db(rows, cols, hb);

    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() + db.leaf()), (ha + hb).clone(), 1e-12, "add");
    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() - db.leaf()), (ha - hb).clone(), 1e-12, "sub");
    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() * db.leaf()), (ha * hb).clone(), 1e-12, "mul");
    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() / db.leaf()), (ha / hb).clone(), 1e-12, "div");
}

TEST_F(CudaDeviceTest, ScalarOperandsMatchHost)
{
    const int rows = 11, cols = 13;
    auto ha = make_host<double>(rows, cols, 2.0);
    dev_matrix_t<double> da(rows, cols, ha);

    // 左标量、右标量、两侧都有，以及整数标量
    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() * 2.0), (ha * 2.0).clone(), 1e-12, "右标量");
    expect_matrices_match(cuda::eval_fused_to_host(2.0 * da.leaf()), (2.0 * ha).clone(), 1e-12, "左标量");
    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() / 2), (ha / 2).clone(), 1e-12, "整型标量");
    expect_matrices_match(cuda::eval_fused_to_host((da.leaf() + 1.0) * 3.0),
                          ((ha + 1.0) * 3.0).clone(), 1e-12, "两侧标量");
}

TEST_F(CudaDeviceTest, ExpAndSigmoidMatchHost)
{
    const int rows = 9, cols = 15;
    auto ha = make_host<double>(rows, cols, 0.0);
    dev_matrix_t<double> da(rows, cols, ha);

    // exp / sigmoid 在设备端必须走 ::exp，这里是那条路径的数值验证
    expect_matrices_match(cuda::eval_fused_to_host(exp(da.leaf())), exp(ha).clone(), 1e-12, "exp");
    expect_matrices_match(cuda::eval_fused_to_host(sigmoid(da.leaf())), sigmoid(ha).clone(), 1e-12,
                          "sigmoid");
}

TEST_F(CudaDeviceTest, DeepChainIsFusedAndMatchesHost)
{
    const int rows = 21, cols = 19;
    auto ha = make_host<double>(rows, cols, 1.0);
    auto hb = make_host<double>(rows, cols, 2.5);
    auto hc = make_host<double>(rows, cols, 0.5);

    dev_matrix_t<double> da(rows, cols, ha), db(rows, cols, hb), dc(rows, cols, hc);

    // 五层链，中间结果一个都不物化
    auto host_ref = exp(((ha + hb) * hc - ha) / (hb + 1.0)).clone();
    auto dev_got = cuda::eval_fused_to_host(
        exp(((da.leaf() + db.leaf()) * dc.leaf() - da.leaf()) / (db.leaf() + 1.0)));
    expect_matrices_match(dev_got, host_ref, 1e-12, "深层链");
}

TEST_F(CudaDeviceTest, TransposedViewWorksElementwise)
{
    const int rows = 12, cols = 20;
    auto ha = make_host<double>(rows, cols, 1.0);
    dev_matrix_t<double> da(rows, cols, ha);

    // 转置视图在逐元素路径上也要正确（缓存里塞的是翻转过的下标）
    expect_matrices_match(cuda::eval_fused_to_host(da.leaf().t() * 2.0),
                          (ha.t() * 2.0).clone(), 1e-12, "转置视图");
}

TEST_F(CudaDeviceTest, ComparisonOperatorsMatchHost)
{
    const int rows = 7, cols = 7;
    auto ha = make_host<double>(rows, cols, 0.0);
    auto hb = make_host<double>(rows, cols, 0.5);
    dev_matrix_t<double> da(rows, cols, ha), db(rows, cols, hb);

    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() > db.leaf()), (ha > hb).clone(), 1e-12, "gt");
    expect_matrices_match(cuda::eval_fused_to_host(da.leaf() < db.leaf()), (ha < hb).clone(), 1e-12, "lt");
}

TEST_F(CudaDeviceTest, Float32WorksToo)
{
    const int rows = 31, cols = 29;
    auto ha = make_host<float>(rows, cols, 1.0f);
    auto hb = make_host<float>(rows, cols, 3.0f);
    dev_matrix_t<float> da(rows, cols, ha), db(rows, cols, hb);

    auto got = cuda::eval_fused_to_host(da.leaf() * db.leaf() + 1.0f);
    auto ref = (ha * hb + 1.0f).clone();
    expect_matrices_match_f32(got, ref, 1e-5f, "float32 逐元素");
}

TEST_F(CudaDeviceTest, ThreadBlockSizesAgreeWithEachOther)
{
    const int rows = 64, cols = 65;   // 4160 个元素，多种分块都覆盖不满
    auto ha = make_host<double>(rows, cols, 1.0);
    auto hb = make_host<double>(rows, cols, 0.25);
    dev_matrix_t<double> da(rows, cols, ha), db(rows, cols, hb);

    auto ref = cuda::eval_fused_to_host(da.leaf() * db.leaf() + 1.0, 256);
    for (int tpb : {32, 64, 128, 256, 512, 1024})
    {
        auto got = cuda::eval_fused_to_host(da.leaf() * db.leaf() + 1.0, tpb);
        expect_matrices_match(got, ref, 1e-12, "不同 block 大小");
    }
}

TEST_F(CudaDeviceTest, SingleElementAndSingleRowEdgeCases)
{
    {
        auto ha = make_host<double>(1, 1, 3.0);
        dev_matrix_t<double> da(1, 1, ha);
        auto got = cuda::eval_fused_to_host(da.leaf() * 2.0);
        ASSERT_EQ(got.row_num(), 1);
        ASSERT_EQ(got.col_num(), 1);
        EXPECT_NEAR(got(0, 0), 6.0, 1e-12);
    }
    {
        auto ha = make_host<double>(1, 40, 1.0);
        dev_matrix_t<double> da(1, 40, ha);
        expect_matrices_match(cuda::eval_fused_to_host(da.leaf() + 1.0), (ha + 1.0).clone(), 1e-12,
                              "单行");
    }
    {
        auto ha = make_host<double>(40, 1, 1.0);
        dev_matrix_t<double> da(40, 1, ha);
        expect_matrices_match(cuda::eval_fused_to_host(da.leaf() + 1.0), (ha + 1.0).clone(), 1e-12,
                              "单列");
    }
}

// ---------------------------------------------------------------------------
// cuBLAS GEMM
// ---------------------------------------------------------------------------

TEST_F(CudaDeviceTest, GemmSquareMatchesHost)
{
    const int n = 48;
    auto ha = make_host<double>(n, n, 0.1);
    auto hb = make_host<double>(n, n, -0.2);
    dev_matrix_t<double> da(n, n, ha), db(n, n, hb);

    auto ref = ha.dot(hb).clone();
    auto got = cuda::gemm_to_host(da.leaf(), db.leaf());
    expect_matrices_match(got, ref, 1e-12, "方阵 GEMM");
}

TEST_F(CudaDeviceTest, GemmNonSquareMatchesHost)
{
    // M、N、K 互不相等，能同时验证 M/N 交换和前导维有没有搞错
    const int M = 37, K = 53, N = 29;
    auto ha = make_host<double>(M, K, 0.05);
    auto hb = make_host<double>(K, N, -0.03);
    dev_matrix_t<double> da(M, K, ha), db(K, N, hb);

    auto ref = ha.dot(hb).clone();
    auto got = cuda::gemm_to_host(da.leaf(), db.leaf());
    ASSERT_EQ(got.row_num(), M);
    ASSERT_EQ(got.col_num(), N);
    expect_matrices_match(got, ref, 1e-12, "非方阵 GEMM");
}

TEST_F(CudaDeviceTest, GemmTransposeAMatchesHost)
{
    const int K = 31, M = 47, N = 23;
    auto ha_stored = make_host<double>(K, M, 0.1);   // 存储为 K×M，转置后当 M×K 用
    auto hb = make_host<double>(K, N, 0.2);
    dev_matrix_t<double> da(K, M, ha_stored), db(K, N, hb);

    auto ref = ha_stored.t().dot(hb).clone();
    auto got = cuda::gemm_to_host(da.leaf().t(), db.leaf());
    ASSERT_EQ(got.row_num(), M);
    ASSERT_EQ(got.col_num(), N);
    expect_matrices_match(got, ref, 1e-12, "转置 A");
}

TEST_F(CudaDeviceTest, GemmTransposeBIsTheAttentionScorePattern)
{
    // S = Q·Kᵀ —— 这正是注意力打分那一步，也是最容易把转置搞反的地方
    const int T = 24, d = 16;
    auto hq = make_host<double>(T, d, 0.1);
    auto hk = make_host<double>(T, d, -0.15);
    dev_matrix_t<double> dq(T, d, hq), dk(T, d, hk);

    auto ref = hq.dot(hk.t()).clone();
    auto got = cuda::gemm_to_host(dq.leaf(), dk.leaf().t());
    ASSERT_EQ(got.row_num(), T);
    ASSERT_EQ(got.col_num(), T);
    expect_matrices_match(got, ref, 1e-12, "Q·Kᵀ");
}

TEST_F(CudaDeviceTest, GemmBothTransposedMatchesHost)
{
    const int K1 = 19, M = 33, K2 = 27, N = 41;
    auto ha = make_host<double>(K1, M, 0.1);   // 转置后是 M×K1
    auto hb = make_host<double>(N, K1, 0.2);   // 转置后是 K1×N —— 内维必须一致
    dev_matrix_t<double> da(K1, M, ha), db(N, K1, hb);

    auto ref = ha.t().dot(hb.t()).clone();
    auto got = cuda::gemm_to_host(da.leaf().t(), db.leaf().t());
    ASSERT_EQ(got.row_num(), M);
    ASSERT_EQ(got.col_num(), N);
    expect_matrices_match(got, ref, 1e-12, "双转置");
    (void)K2;
}

TEST_F(CudaDeviceTest, GemmAccumulatesIntoExistingOutput)
{
    // beta != 0：在已有结果上累加，训练里常用
    const int M = 20, K = 24, N = 18;
    auto ha = make_host<double>(M, K, 0.1);
    auto hb = make_host<double>(K, N, 0.2);
    auto hc = make_host<double>(M, N, 0.3);
    dev_matrix_t<double> da(M, K, ha), db(K, N, hb), dc(M, N, hc);

    cuda::gemm(da.leaf(), db.leaf(), dc.leaf(), 1.0, 1.0);
    cuda::sync();

    auto ref = (ha.dot(hb) + hc).clone();
    expect_matrices_match(dc.download(), ref, 1e-12, "beta 累加");
}

TEST_F(CudaDeviceTest, GemmRejectsShapeMismatch)
{
    auto ha = make_host<double>(4, 6, 0.1);
    auto hb = make_host<double>(5, 7, 0.2);   // 内维 6 != 5
    dev_matrix_t<double> da(4, 6, ha), db(5, 7, hb), out(4, 7);

    EXPECT_THROW(cuda::gemm(da.leaf(), db.leaf(), out.leaf()), std::invalid_argument);
}

TEST_F(CudaDeviceTest, GemmFloat32MatchesHost)
{
    const int M = 26, K = 34, N = 22;
    auto ha = make_host<float>(M, K, 0.05f);
    auto hb = make_host<float>(K, N, 0.03f);
    dev_matrix_t<float> da(M, K, ha), db(K, N, hb);

    auto ref = ha.dot(hb).clone();
    auto got = cuda::gemm_to_host(da.leaf(), db.leaf());
    expect_matrices_match_f32(got, ref, 1e-4f, "float32 GEMM");
}

TEST_F(CudaDeviceTest, FusedElementwiseThenGemmComposesWithHost)
{
    // 真实流水线：先逐元素（融合 kernel），再 GEMM（cuBLAS），两条路拼接
    const int M = 30, K = 18, N = 25;
    auto ha = make_host<double>(M, K, 0.1);
    auto hb = make_host<double>(K, N, 0.2);
    auto hs = make_host<double>(M, K, 1.5);
    dev_matrix_t<double> da(M, K, ha), db(K, N, hb), ds(M, K, hs);

    // (A * S) · B，其中 (A * S) 由融合 kernel 算出
    auto fused = cuda::eval_fused_to_host((da.leaf() * ds.leaf()));
    expect_matrices_match(fused, (ha * hs).clone(), 1e-12, "融合段");

    dev_matrix_t<double> d_fused(M, K, fused);
    auto got = cuda::gemm_to_host(d_fused.leaf(), db.leaf());
    auto ref = (ha * hs).dot(hb).clone();
    expect_matrices_match(got, ref, 1e-12, "融合 + GEMM");
}

// ---------------------------------------------------------------------------
// 光学热：默认关闭的算力型用例
// ---------------------------------------------------------------------------

TEST_F(CudaDeviceTest, LargeMatrixFusionMatchesHostWhenStressEnabled)
{
    if (!stress_enabled())
        GTEST_SKIP() << "默认跳过（会持续占用 GPU 约数秒）。需要时设 JASMINE_CUDA_STRESS=1";

    const int rows = 4096, cols = 4096;
    auto ha = make_host<double>(rows, cols, 1.0);
    auto hb = make_host<double>(rows, cols, 0.5);
    dev_matrix_t<double> da(rows, cols, ha), db(rows, cols, hb);

    auto got = cuda::eval_fused_to_host((da.leaf() + db.leaf()) * 2.0);
    for (int i = 0; i < rows; i += 512)
        for (int j = 0; j < cols; j += 512)
        {
            const double ref = (ha(i, j) + hb(i, j)) * 2.0;
            ASSERT_NEAR(got(i, j), ref, 1e-12 * std::max(1.0, std::abs(ref)));
        }
}

TEST_F(CudaDeviceTest, LargeGemmMatchesHostWhenStressEnabled)
{
    if (!stress_enabled())
        GTEST_SKIP() << "默认跳过（GEMM 是真正吃算力的用例）。需要时设 JASMINE_CUDA_STRESS=1";

    // 256³ 对 P4 来说仍是很短的一瞬，但比其他用例重，单独归到压力组
    const int n = 256;
    auto ha = make_host<float>(n, n, 0.01f);
    auto hb = make_host<float>(n, n, 0.02f);
    dev_matrix_t<float> da(n, n, ha), db(n, n, hb);

    auto ref = ha.dot(hb).clone();
    auto got = cuda::gemm_to_host(da.leaf(), db.leaf());
    expect_matrices_match_f32(got, ref, 1e-3f, "大 GEMM");
}
