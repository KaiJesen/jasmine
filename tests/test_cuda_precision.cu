#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_precision.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"

/**
 * 混精度（`bf16` / `fp16` 存储、`fp32` 累加）的测试。
 *
 * 散热约束同其它 CUDA 测试：矩阵刻意取小、每个用例前后测温、超 80℃ 就 skip。
 *
 * ## 这里在证明什么
 *
 * 降精度最容易滑向"把容差放宽到 1e-2，然后什么都不验"。这组测例拒绝那条路，
 * 改成**证明误差只来自它该来的地方**：
 *
 *  1. **误差模型本身要对**：`u = 2^-(m+1)` 里的 `m` 写错一个比特就是错一倍，
 *     所以先拿"实测的最大舍入误差"去校准模型（`PrecisionModelMatchesMeasuredRounding`）。
 *  2. **乘积精确 + 累加 fp32**：拿"先把操作数舍入、再用 double 算"当参考，
 *     容差取 `accumulation_error_bound`（`K · u_acc`，量级 1e-6）。容差这么大是有
 *     依据的，而不是"试出来能过"。**如果 cuBLAS 用 16 位累加，这条会以量级之差失败** ——
 *     这就是它值得写下来的理由。
 *  3. **量化确实发生了**：与"不舍入"的参考比，误差必须**明显大于** `K · u_acc`。
 *     否则说明降精度压根没生效（某处悄悄提升了精度），而第 1、2 条照样会过。
 *  4. **两种精度的差异符合模型预测**：`bf16` 的 `u` 是 `fp16` 的 8 倍（尾数 7 位 vs 10 位，
 *     差 3 位），实测差异就该在那个量级 —— 模型能预测格式之间的差别，才叫模型。
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

    FILE* pipe =
        ::popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null", "r");
    if (pipe == nullptr)
        return -1;

    char buf[64] = {};
    char* got = std::fgets(buf, sizeof(buf), pipe);
    ::pclose(pipe);
    if (got == nullptr)
        return -1;

    return std::atoi(buf);
}

/** 定种子的均匀填充，取值避开 0 与极值（相对误差的定义要求 `x != 0`）。 */
mat_t<double> make_host(int rows, int cols, unsigned seed, double lo = 0.5, double hi = 1.5)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    mat_t<double> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = dist(rng);
    return m;
}

/** `C = A·B`（double，全精度参考）。 */
mat_t<double> ref_double(const mat_t<double>& a, const mat_t<double>& b)
{
    mat_t<double> c(a.row_num(), b.col_num());
    for (int i = 0; i < c.row_num(); ++i)
        for (int j = 0; j < c.col_num(); ++j)
        {
            double acc = 0;
            for (int k = 0; k < a.col_num(); ++k)
                acc += a(i, k) * b(k, j);
            c(i, j) = acc;
        }
    return c;
}

/** 逐元素量化到 `T` 再提升回 double —— "操作数先被舍入"的那份参考。 */
template <typename T>
mat_t<double> quantize_matrix(const mat_t<double>& m)
{
    mat_t<double> out(m.row_num(), m.col_num());
    for (int i = 0; i < m.row_num(); ++i)
        for (int j = 0; j < m.col_num(); ++j)
            out(i, j) = static_cast<double>(cuda::quantize<T>(m(i, j)));
    return out;
}

/** 最大**相对**偏差，参考值本身太小时跳过（相对误差在那个尺度上没有意义）。 */
double max_relative_deviation(const mat_t<double>& got, const mat_t<double>& ref)
{
    double worst = 0;
    for (int i = 0; i < got.row_num(); ++i)
        for (int j = 0; j < got.col_num(); ++j)
        {
            const double scale = std::abs(ref(i, j));
            if (scale < 1e-6)
                continue;
            worst = std::max(worst, std::abs(got(i, j) - ref(i, j)) / scale);
        }
    return worst;
}

void report(const char* tag, double measured, double bound)
{
    std::fprintf(stderr, "[混精度] %-28s 实测 %.3e / 上界 %.3e（占 %.1f%%）\n", tag, measured, bound,
                 100.0 * measured / bound);
}

} // namespace

class CudaPrecisionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const int t = gpu_temperature_c();
        if (t >= kHotCelsius)
            GTEST_SKIP() << "GPU 结温 " << t << "℃ 已达上限 " << kHotCelsius << "℃，跳过";
        if (t >= 0)
            std::fprintf(stderr, "[测温] 测试开始前 GPU %d℃\n", t);
    }

    void TearDown() override
    {
        const int t = gpu_temperature_c();
        if (t >= 0)
            std::fprintf(stderr, "[测温] 测试结束后 GPU %d℃\n", t);
        cuda::sync();
    }
};

// ===========================================================================
// 误差模型：先证明模型本身是对的
// ===========================================================================

/**
 * `u = 2^-(mantissa_bits + 1)` 必须同时满足两件事：**成立**（实测不超）与**紧**（实测算得到）。
 *
 * 只证"不超"是不够的：把一个明显偏大的数当上界，它永远不会失败，也就永远没用。
 * 所以这里还要求实测量达到上界的 1/8 以上 —— 就近舍入的最坏情况就是 `u`，
 * 均匀采样足够多的点必然逼近它。这一条同时校正了"位数填错"（差一倍就会越界）。
 */
TEST_F(CudaPrecisionTest, PrecisionModelMatchesMeasuredRounding)
{
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(1.0, 2.0);  // 尾数全随机的一档

    auto measure = [&](auto tag) {
        using T = decltype(tag);
        const double u = cuda::precision_traits<T>::unit_roundoff();
        double worst = 0;
        for (int i = 0; i < 200000; ++i)
        {
            const double x = dist(rng);
            const double q = static_cast<double>(cuda::quantize<T>(x));
            worst = std::max(worst, std::abs(q - x) / x);
        }
        report("u(实测 vs 模型)", worst, u);
        EXPECT_LE(worst, u) << "实测舍入误差超过了模型上界（模型偏小）";
        EXPECT_GT(worst, u / 8.0) << "模型上界离实测太远（模型偏大，等于没约束）";
    };

    measure(cuda::bf16_t{});
    measure(cuda::fp16_t{});

    // 常量本身也钉一下：差一个比特就是错一倍，这类"安静的错误"值得写死
    EXPECT_DOUBLE_EQ(cuda::precision_traits<cuda::bf16_t>::unit_roundoff(), 1.0 / 256.0);
    EXPECT_DOUBLE_EQ(cuda::precision_traits<cuda::fp16_t>::unit_roundoff(), 1.0 / 2048.0);
    EXPECT_DOUBLE_EQ(cuda::precision_traits<float>::unit_roundoff(), std::ldexp(1.0, -24));
}

/**
 * 设备端 cast 必须与主机端的 `narrow<T>` **逐位相同**。
 *
 * 这不是"差不多就行"：转换成两套实现（主机一次、设备一次）时，最容易出现的是
 * 舍入方向不同（截断 vs 就近），而那会让"先量化再上传"和"上传后再量化"得到
 * 两份不同的权重 —— 误差不大，但对拍时永远差一点，非常难查。
 */
TEST_F(CudaPrecisionTest, DeviceCastMatchesHostNarrowingBitForBit)
{
    const int rows = 7, cols = 11;
    // 混入负值与量级悬殊的值：负数的"就近舍入"与正数对称性不同，是指数位出错的常见盲区
    const mat_t<double> src = make_host(rows, cols, 3, -3.0, 3.0);

    cuda::dev_matrix_t<double> dev_src(rows, cols, src);

    auto check = [&](auto tag) {
        using T = decltype(tag);
        const cuda::dev_matrix_t<T> narrowed = cuda::cast<T>(dev_src.const_leaf());
        const mat_t<double> got = cuda::to_host_double(narrowed);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
                const T want = cuda::narrow<T>(src(i, j));
                // 逐位比较：把两边都提升回 double，量化过的值在 double 里是精确的
                EXPECT_EQ(got(i, j), static_cast<double>(cuda::widen(want)))
                    << "cast 与主机 narrow 在第 (" << i << "," << j << ") 位不一致";
            }
    };

    check(cuda::bf16_t{});
    check(cuda::fp16_t{});
    check(float{});
}

/**
 * 混合精度**不做隐式类型提升**：`matmul(float 叶子, bf16 叶子)` 必须是编译错误。
 *
 * 这条用 `static_assert` 表达 —— 它要守的正是"编译期挡住"这个性质本身。
 * 若哪天有人给 `matmul` 加了一条宽松的重载，这里会先编译失败。
 */
template <typename A, typename B>
concept can_matmul = requires(A a, B b) { cuda::matmul(a, b); };

static_assert(can_matmul<dev_mat_t<float>, dev_mat_t<float>>);
static_assert(can_matmul<dev_mat_t<cuda::bf16_t>, dev_mat_t<cuda::bf16_t>>);
static_assert(!can_matmul<dev_mat_t<float>, dev_mat_t<cuda::bf16_t>>,
              "混精度必须显式转换：gemm 的两边类型不同就该在编译期被挡住，"
              "而不是变成 cuBLAS 的一个含糊状态码");

// ===========================================================================
// GEMM：乘积精确、累加 fp32
// ===========================================================================

/**
 * 降精度操作数 + **fp32 输出**：与"先舍入、再用 double 算"的参考只差 `K · u_acc`。
 *
 * 这是整组测例里最有信息量的一条：
 *   - 若 cuBLAS 用 16 位累加 → 误差是 `K · 4e-3` 量级，差 5 个数量级，必然失败；
 *   - 若乘积不是精确的（比如先降到 fp16 再乘）→ 同样差好几个数量级；
 *   - 只有"乘积精确 + fp32 累加"才能落在这个容差里。
 *
 * 容差不是试出来的：`accumulation_error_bound<T>(K)` 就是模型算出的那一项。
 */
TEST_F(CudaPrecisionTest, ReducedInputFp32OutputAccumulatesInFp32)
{
    constexpr int M = 6, K = 48, N = 5;

    const mat_t<double> a = make_host(M, K, 21);
    const mat_t<double> b = make_host(K, N, 22);

    auto run = [&](auto tag) {
        using T = decltype(tag);

        const mat_t<double> aq = quantize_matrix<T>(a);
        const mat_t<double> bq = quantize_matrix<T>(b);

        cuda::dev_matrix_t<T> da = cuda::from_host_double<T>(a);
        cuda::dev_matrix_t<T> db = cuda::from_host_double<T>(b);
        cuda::dev_matrix_t<float> dc(M, N);
        dc.buffer().zero();

        // 输出显式给 float：把"结果那次舍入"从链路里拿掉，只留量化与累加
        cuda::gemm(da.const_leaf(), db.const_leaf(), dc.leaf());

        const mat_t<double> got = cuda::to_host_double(dc);

        const double dev_rounded = max_relative_deviation(got, ref_double(aq, bq));
        const double dev_exact = max_relative_deviation(got, ref_double(a, b));

        const double acc_bound = cuda::accumulation_error_bound<T>(K);
        const double quant_bound = cuda::quantization_error_bound<T>();

        report("累加误差（vs 舍入参考）", dev_rounded, acc_bound);
        report("量化误差（vs 精确参考）", dev_exact, quant_bound + acc_bound);

        // 乘积精确 + 累加 fp32 ⇒ 与"舍入参考"的差异只能是累加舍入那一项
        EXPECT_LE(dev_rounded, 4.0 * acc_bound)
            << "与舍入参考的偏差远大于 fp32 累加误差 —— 累加精度或乘积精度不是 fp32 精确乘";

        // 量化那一项必须真的在（否则降精度没生效）
        EXPECT_GT(dev_exact, acc_bound) << "误差小到看不出量化 —— 操作数可能压根没被降精度";
        EXPECT_LE(dev_exact, 4.0 * (quant_bound + acc_bound));
    };

    run(cuda::bf16_t{});
    run(cuda::fp16_t{});
}

/**
 * 输出也是降精度时，多出来的那一层就是**结果自身的舍入**（`u_op` 量级）。
 *
 * 把这一条与上一条并排看，误差的每一个来源都落在明处：操作数各一次、结果一次、
 * 累加 `K` 次。写测试的意义就在这 —— 换一种输出类型，容差该变多少是可以算的，
 * 而不是"看着调"。
 */
TEST_F(CudaPrecisionTest, ReducedOutputAddsItsOwnRounding)
{
    constexpr int M = 6, K = 48, N = 5;

    const mat_t<double> a = make_host(M, K, 31);
    const mat_t<double> b = make_host(K, N, 32);

    auto run = [&](auto tag) {
        using T = decltype(tag);

        cuda::dev_matrix_t<T> da = cuda::from_host_double<T>(a);
        cuda::dev_matrix_t<T> db = cuda::from_host_double<T>(b);
        cuda::dev_matrix_t<T> dc(M, N);
        cuda::gemm(da.const_leaf(), db.const_leaf(), dc.leaf());

        // 基准是**同一个 gemm 的 fp32 输出**，而不是"再量化一次的数学参考"。
        // 后者会把结果舍入也做一遍，两次舍入互相抵消 —— 测出来的只剩累加误差，
        // 于是"输出类型到底生没生效"根本没被验证。两份输出的差别恰好就是那一次舍入。
        cuda::dev_matrix_t<float> dc_f32(M, N);
        dc_f32.buffer().zero();
        cuda::gemm(da.const_leaf(), db.const_leaf(), dc_f32.leaf());

        const mat_t<double> got = cuda::to_host_double(dc);
        const mat_t<double> fine = cuda::to_host_double(dc_f32);

        const double dev = max_relative_deviation(got, fine);
        const double u_out = cuda::precision_traits<T>::unit_roundoff();
        const double bound = u_out + cuda::accumulation_error_bound<T>(K);

        report("结果舍入（vs fp32 输出）", dev, bound);
        EXPECT_LE(dev, 4.0 * bound);
        // 结果那一次舍入必须真的存在（否则说明输出其实还是 fp32）
        EXPECT_GT(dev, 0.1 * u_out) << "输出看不出舍入 —— 输出类型可能没生效";
    };

    run(cuda::bf16_t{});
    run(cuda::fp16_t{});
}

/**
 * 两种格式的实测误差之比应当落在模型预测的比值附近（`bf16` 的 `u` 是 `fp16` 的 8 倍）。
 *
 * 这条是"模型可预测"的直接检验：如果只是"两种都过得去"，它对两种格式的差别毫无发言权。
 */
TEST_F(CudaPrecisionTest, Bf16IsEightTimesLooserThanFp16InTheory)
{
    // 2^-8 / 2^-11 = 8：bf16 尾数 7 位、fp16 尾数 10 位，差 3 位
    EXPECT_DOUBLE_EQ(cuda::precision_traits<cuda::bf16_t>::unit_roundoff()
                         / cuda::precision_traits<cuda::fp16_t>::unit_roundoff(),
                     8.0);

    // 取多个种子、每个种子比一大片元素，然后拿**所有元素的最大值**比。
    // 单一种子下"最大误差"离它的上界还差得远，两个格式的比较就是在比抽样噪声；
    // 样本多了以后两个格式的 max 都逼近各自的 `c·u`，比值才会收敛到模型给的 8。
    constexpr int M = 8, K = 64, N = 8;

    auto measure_error = [&](auto tag) {
        using T = decltype(tag);
        double worst = 0;
        for (unsigned seed = 41; seed < 45; ++seed)
        {
            const mat_t<double> a = make_host(M, K, seed);
            const mat_t<double> b = make_host(K, N, seed + 100);
            cuda::dev_matrix_t<T> da = cuda::from_host_double<T>(a);
            cuda::dev_matrix_t<T> db = cuda::from_host_double<T>(b);
            cuda::dev_matrix_t<float> dc(M, N);
            dc.buffer().zero();
            cuda::gemm(da.const_leaf(), db.const_leaf(), dc.leaf());
            worst = std::max(worst, max_relative_deviation(cuda::to_host_double(dc), ref_double(a, b)));
        }
        return worst;
    };

    const double e_bf16 = measure_error(cuda::bf16_t{});
    const double e_fp16 = measure_error(cuda::fp16_t{});
    std::fprintf(stderr, "[混精度] 同一算例：bf16 误差 %.3e / fp16 误差 %.3e（比 %.1f，模型比 8）\n",
                 e_bf16, e_fp16, e_bf16 / e_fp16);

    // 实测是有舍入的随机量，比值不会正好 8；只要落在模型量级里（同一数量级上下），
    // 就说明"两种格式的尾数位数差别真的体现在结果里"，而不是两者都走同一条路径。
    EXPECT_GT(e_bf16 / e_fp16, 4.0) << "bf16 的误差没有明显比 fp16 大 —— 两者的尾数位数可能没生效";
    EXPECT_LT(e_bf16 / e_fp16, 16.0) << "bf16 的误差比模型预测的还大一倍 —— 界的方向反了或模型写错了";
}

/**
 * `beta != 0` 与 `alpha != 1` 在降精度路径上也要对（GemmEx 的标量是按**累加类型**传的，
 * 写错类型编译器不报，只会得到一个莫名其妙的 status 或者静默错值）。
 */
TEST_F(CudaPrecisionTest, ReducedGemmHonoursAlphaBeta)
{
    constexpr int M = 4, K = 16, N = 3;

    const mat_t<double> a = make_host(M, K, 51, 0.5, 1.5);
    const mat_t<double> b = make_host(K, N, 52, 0.5, 1.5);
    const mat_t<double> c0 = make_host(M, N, 53, 0.5, 1.5);

    const double alpha = 0.75, beta = 1.25;

    const mat_t<double> aq = quantize_matrix<cuda::bf16_t>(a);
    const mat_t<double> bq = quantize_matrix<cuda::bf16_t>(b);
    mat_t<double> want = ref_double(aq, bq);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            want(i, j) = alpha * want(i, j) + beta * c0(i, j);

    cuda::dev_matrix_t<cuda::bf16_t> da = cuda::from_host_double<cuda::bf16_t>(a);
    cuda::dev_matrix_t<cuda::bf16_t> db = cuda::from_host_double<cuda::bf16_t>(b);
    cuda::dev_matrix_t<float> dc(M, N, c0);  // 初值直接决定 beta 项
    cuda::gemm(da.const_leaf(), db.const_leaf(), dc.leaf(), static_cast<float>(alpha),
               static_cast<float>(beta));

    const mat_t<double> got = cuda::to_host_double(dc);
    const double dev = max_relative_deviation(got, want);
    // alpha/beta 自身也会被收窄到 float（模型里没算这一项，所以这里单独留一档余量）
    report("alpha/beta 组合", dev, cuda::accumulation_error_bound<cuda::bf16_t>(K));
    EXPECT_LE(dev, 1e-6) << "alpha/beta 没有按累加类型正确传入";
}

// ===========================================================================
// 收益与代价
// ===========================================================================

/**
 * 降精度的收益是**显存与带宽**，这一条把它量化出来。
 *
 * 同时把"本机是否原生"打出来：Pascal 上 cuBLAS 的 16 位 GEMM 是软件回退，
 * 所以这里省的是显存**不是时间**。不写清楚的话，很容易拿测试机的结论去推目标机。
 */
TEST_F(CudaPrecisionTest, HalvesStorageButNotNativeOnPascal)
{
    const int rows = 1024, cols = 1024;
    const std::size_t f32_bytes = static_cast<std::size_t>(rows) * cols * sizeof(float);
    const std::size_t bf16_bytes = static_cast<std::size_t>(rows) * cols * sizeof(cuda::bf16_t);
    const std::size_t fp16_bytes = static_cast<std::size_t>(rows) * cols * sizeof(cuda::fp16_t);

    EXPECT_EQ(bf16_bytes, f32_bytes / 2);
    EXPECT_EQ(fp16_bytes, f32_bytes / 2);

    // 实际分配出来的字节数也要对（不是只有 sizeof 对）
    cuda::dev_matrix_t<cuda::bf16_t> m(rows, cols);
    EXPECT_EQ(m.buffer().size() * sizeof(cuda::bf16_t), f32_bytes / 2);

    const bool native = cuda::reduced_precision_is_native();
    std::fprintf(stderr,
                 "[混精度] 1024×1024：fp32 %.1f MiB → bf16/fp16 %.1f MiB；"
                 "本机降精度 GEMM %s（算力 %d）\n",
                 f32_bytes / 1048576.0, bf16_bytes / 1048576.0,
                 native ? "有张量核支撑" : "是软件回退：省显存不省时间",
                 cuda::device_info().compute_capability());

    if (!native)
    {
        // P4（sm_61）：cuBLAS 仍然给出正确结果（上面几组测例已经证了），但没有硬件加速
        EXPECT_LT(cuda::device_info().compute_capability(), 80u);
    }
}
