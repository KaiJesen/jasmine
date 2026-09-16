/**
 * CUDA 后端演示：同一份表达式模板，在 CPU 和 GPU 上求值，并对比"融合"与"分步物化"。
 *
 * 这是对那个问题 —— "这个项目里的模板表达式能不能用到 CUDA 上" —— 的一个可运行回答。
 *
 * 用法：
 *   ./cuda_fused_demo [迭代次数]      # 默认 20，刻意给小，避免无风扇的 P4 升温
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_mat_t.hpp"

using namespace jasmine;

namespace
{

constexpr int kRows = 1024;
constexpr int kCols = 1024;

/**
 * 演示数据刻意限幅在小范围内。
 *
 * 如果按 (0.5*i - 0.25*j) 那种斜率填，1024 阶时元素值会到 512 附近，
 * 表达式里的 `exp()` 直接溢出成 inf，`inf - inf = nan`，
 * 误差对比就完全失去意义了 —— 那是数据选得不好，不是后端算错了。
 */
mat_t<double> make_host(int rows, int cols, double base)
{
    mat_t<double> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + 0.001 * ((i * 7 + j * 13) % 41) - 0.02;
    return m;
}

/** 最大绝对误差；顺便把非有限值挑出来单独报，免得 inf/nan 悄悄混进结论。 */
struct diff_report
{
    double max_abs = 0.0;
    int non_finite = 0;
};

diff_report max_abs_diff(const mat_t<double>& a, const mat_t<double>& b)
{
    diff_report r;
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < a.col_num(); ++j)
        {
            const double d = std::abs(a(i, j) - b(i, j));
            if (!std::isfinite(d))
            {
                ++r.non_finite;
                continue;
            }
            r.max_abs = std::max(r.max_abs, d);
        }
    return r;
}

void print_diff(const mat_t<double>& dev, const mat_t<double>& host)
{
    const diff_report r = max_abs_diff(dev, host);
    std::printf("  最大绝对误差: %.3e\n", r.max_abs);
    if (r.non_finite > 0)
        std::printf("  ⚠ %d 个元素是非有限值（inf/nan）—— 通常是测试数据让 exp() 溢出了，"
                    "不是后端算错\n",
                    r.non_finite);
}

void print_temperature()
{
    FILE* pipe = ::popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null", "r");
    if (pipe == nullptr)
        return;
    char buf[64] = {};
    char* got = std::fgets(buf, sizeof(buf), pipe);
    ::pclose(pipe);
    if (got != nullptr)
        std::printf("\n[GPU 结温 %d℃]\n", std::atoi(buf));
}

/** 用 CUDA event 给一段 GPU 工作计时（毫秒）。 */
template <typename Fn>
double time_ms(Fn&& fn, int iterations)
{
    cudaEvent_t start, stop;
    JAS_CUDA_CHECK(cudaEventCreate(&start));
    JAS_CUDA_CHECK(cudaEventCreate(&stop));

    fn();               // 预热：把首次的上下文/内存分配开销排除掉
    cuda::sync();

    JAS_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i)
        fn();
    JAS_CUDA_CHECK(cudaEventRecord(stop));
    JAS_CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    JAS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iterations;
}

} // namespace

int main(int argc, char** argv)
{
    int iterations = 20;
    if (argc > 1)
        iterations = std::atoi(argv[1]);
    if (iterations <= 0)
        iterations = 1;

    // ---------------------------------------------------------------------
    std::printf("=== 设备信息 ===\n%s\n", cuda::device_info().to_string().c_str());

    auto ha = make_host(kRows, kCols, 1.0);
    auto hb = make_host(kRows, kCols, -0.5);
    auto hc = make_host(kRows, kCols, 0.25);

    cuda::dev_matrix_t<double> da(kRows, kCols, ha);
    cuda::dev_matrix_t<double> db(kRows, kCols, hb);
    cuda::dev_matrix_t<double> dc(kRows, kCols, hc);

    // ---------------------------------------------------------------------
    // 同一套写法，两个后端：表达式源码一字不改，只是叶子类型不同
    // ---------------------------------------------------------------------
    std::printf("\n=== 同一份表达式，两个后端 ===\n");
    std::printf("表达式: exp((a + b) * c - a) / (b + 1)\n");

    auto host_tree = exp(((ha + hb) * hc - ha) / (hb + 1.0));
    auto host_result = host_tree.clone();

    auto dev_tree = exp(((da.leaf() + db.leaf()) * dc.leaf() - da.leaf()) / (db.leaf() + 1.0));
    auto dev_result = cuda::eval_fused_to_host(dev_tree);

    std::printf("  host   : %d×%d，逐元素递归求值\n", host_result.row_num(), host_result.col_num());
    std::printf("  device : %d×%d，一次 kernel launch 融合求完整条链\n", dev_result.row_num(),
                dev_result.col_num());
    print_diff(dev_result, host_result);

    // ---------------------------------------------------------------------
    // 融合 vs 分步物化
    // ---------------------------------------------------------------------
    std::printf("\n=== 融合 vs 分步物化（%d×%d，%d 次）===\n", kRows, kCols, iterations);

    cuda::dev_buf_t<double> fused_out;
    const double fused_ms =
        time_ms([&] { cuda::eval_fused((da.leaf() + db.leaf()) * dc.leaf(), fused_out); }, iterations);

    // 分步：先物化 a+b 到一块显存，再拿它乘 c。多一次 8 MB 写 + 8 MB 读。
    cuda::dev_buf_t<double> tmp;
    cuda::dev_buf_t<double> staged_out;
    const double staged_ms = time_ms(
        [&] {
            cuda::eval_fused(da.leaf() + db.leaf(), tmp);              // launch 1：写 tmp
            dev_mat_t<double> tmp_leaf(tmp.data(), kRows, kCols);
            cuda::eval_fused(tmp_leaf * dc.leaf(), staged_out);        // launch 2：读 tmp
        },
        iterations);

    std::printf("  融合（1 次 launch，零中间物化）      : %8.3f ms\n", fused_ms);
    std::printf("  分步（2 次 launch + 一次显存往返）   : %8.3f ms\n", staged_ms);
    std::printf("  加速比                               : %8.2fx\n", staged_ms / fused_ms);
    std::printf("  （省下的正是中间结果那 8 MB 的写 + 读；P4 是访存受限的）\n");

    // ---------------------------------------------------------------------
    // GEMM：注意力打分
    // ---------------------------------------------------------------------
    std::printf("\n=== cuBLAS GEMM：注意力打分 S = Q·Kᵀ ===\n");
    const int seq = 256, dim = 64;
    auto hq = make_host(seq, dim, 0.1);
    auto hk = make_host(seq, dim, -0.15);
    cuda::dev_matrix_t<double> dq(seq, dim, hq), dk(seq, dim, hk);

    auto s_dev = cuda::gemm_to_host(dq.leaf(), dk.leaf().t());
    auto s_host = hq.dot(hk.t()).clone();
    std::printf("  Q: %d×%d，K: %d×%d → S: %d×%d\n", seq, dim, seq, dim, s_dev.row_num(),
                s_dev.col_num());
    print_diff(s_dev, s_host);

    print_temperature();
    return 0;
}
