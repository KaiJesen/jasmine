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
#include <limits>
#include <string>
#include <vector>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_kv_cache.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_rope.hpp"
#include "jas_mat_t.hpp"
#include "jas_net_t.hpp"
#include "jas_RoPE_t.hpp"
#include "jas_updator_t.hpp"

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

    // ---------------------------------------------------------------------
    // .dot() 的设备分派：立即落成 cuBLAS，结果可再参与表达式
    // ---------------------------------------------------------------------
    std::printf("\n=== .dot() 设备分派（与主机端几乎同形的写法）===\n");
    {
        const int dh = 32, sq = 24;
        auto hq2 = make_host(dh, sq, 0.13);
        auto hk2 = make_host(dh, sq, 0.17);
        auto hv2 = make_host(dh, sq, 0.11);

        cuda::dev_matrix_t<double> dq2(dh, sq, hq2), dk2(dh, sq, hk2), dv2(dh, sq, hv2);

        // 主机端 m_head_gen_t::forward_at 里是 m_q.t().dot(m_k) / sqrt(d) 再 softmax 再 dot
        auto scores_h = (hq2.t().dot(hk2) / std::sqrt(static_cast<double>(dh))).clone();
        for (int i = 0; i < sq; ++i)
            for (int j = i + 1; j < sq; ++j)
                scores_h(i, j) = -std::numeric_limits<double>::infinity();
        auto attn_h = hsoftmax(scores_h);
        auto out_h = (hv2.dot(attn_h.t())).clone();

        mat_t<double> mask_h(sq, sq);
        mask_h = 0.0;
        for (int i = 0; i < sq; ++i)
            for (int j = i + 1; j < sq; ++j)
                mask_h(i, j) = -std::numeric_limits<double>::infinity();
        cuda::dev_matrix_t<double> dmask(sq, sq, mask_h);

        // 设备端：三个 GEMM 分派点 + 两个融合/归约送算，全部同形
        auto scores = dq2.leaf().t().dot(dk2);                        // GEMM：打分
        auto weights = cuda::softmax_rows(scores.leaf() / std::sqrt(static_cast<double>(dh))
                                          + dmask.leaf());            // 融合 + 掩码 + softmax
        auto out = dv2.leaf().dot(weights.leaf().t());                // GEMM：加权求和
        std::printf("  q.t().dot(k) → %d×%d，softmax，再 v.dot(w.t()) → %d×%d\n", scores.row_num(),
                    scores.col_num(), out.row_num(), out.col_num());
        print_diff(out.download(), out_h);

        // GEMM 的结果是「拥有者」，能继续链式 dot，也能用 .leaf() 回到表达式世界
        auto chained = scores.dot(scores.leaf().t());                 // 结果·结果ᵀ
        auto composed = cuda::eval_fused_to_host(chained.leaf() / static_cast<double>(sq));
        std::printf("  链式 s.dot(s.t()) → %d×%d；再 .leaf() 参与融合 → %d×%d\n",
                    chained.row_num(), chained.col_num(), composed.row_num(), composed.col_num());
    }

    // ---------------------------------------------------------------------
    // 归约：逐行 softmax（把 mask 直接加进表达式）
    // ---------------------------------------------------------------------
    std::printf("\n=== 逐行 softmax（因果掩码 + 数值稳定）===\n");
    const int t = 128;
    auto hs = make_host(t, t, 0.05);
    cuda::dev_matrix_t<double> dscores(t, t, hs);

    // 掩码做成叶子：0 或 -inf，于是「缩放 + 掩码」在融合途中一次完成
    mat_t<double> mask_h(t, t);
    mask_h = 0.0;
    for (int i = 0; i < t; ++i)
        for (int j = i + 1; j < t; ++j)
            mask_h(i, j) = -std::numeric_limits<double>::infinity();
    cuda::dev_matrix_t<double> dmask(t, t, mask_h);

    auto w_dev = cuda::softmax_rows(dscores.leaf() * 0.125 + dmask.leaf()).download();

    auto masked_h = (hs * 0.125).clone();
    for (int i = 0; i < t; ++i)
        for (int j = i + 1; j < t; ++j)
            masked_h(i, j) = -std::numeric_limits<double>::infinity();
    auto w_host = hsoftmax(masked_h);

    print_diff(w_dev, w_host);
    double worst_row_sum_err = 0.0;
    for (int i = 0; i < t; ++i)
    {
        double s = 0.0;
        for (int j = 0; j <= i; ++j)
            s += w_dev(i, j);
        worst_row_sum_err = std::max(worst_row_sum_err, std::abs(s - 1.0));
    }
    std::printf("  每行概率和与 1 的最大偏差: %.3e\n", worst_row_sum_err);

    // ---------------------------------------------------------------------
    // 归一化层：RMSNorm / LayerNorm
    // ---------------------------------------------------------------------
    std::printf("\n=== 归一化层（统计量沿行方向算，对齐 jas_net_t.hpp）===\n");
    const int d_model = 256, seq2 = 16;
    auto hx = make_host(d_model, seq2, 0.2);
    cuda::dev_matrix_t<double> dx(d_model, seq2, hx);

    // gamma 是 (d_model × 1) 的列向量，沿列广播 —— 与主机端 m_gama 的形状一致
    cuda::dev_colvec_t<double> gamma(d_model);
    rms_norm_net_t<mat_t<double>, nadam_t> rn;
    rn.set_param(d_model);
    gamma.buffer().upload(rn.gama().data(), d_model);

    auto rms_dev = cuda::rms_norm(dx.leaf(), gamma, rn.eps()).download();
    auto rms_host = rn.forward(hx);
    std::printf("  RMSNorm（%d×%d）\n", d_model, seq2);
    print_diff(rms_dev, rms_host);

    cuda::dev_colvec_t<double> ln_gamma(d_model), ln_beta(d_model);
    layer_norm_net_t<mat_t<double>, nadam_t> ln;
    ln.set_param(d_model);
    ln_gamma.buffer().upload(ln.gama().data(), d_model);
    ln_beta.buffer().upload(ln.beta().data(), d_model);

    auto ln_dev = cuda::layer_norm(dx.leaf(), ln_gamma, ln_beta).download();
    auto ln_host = ln.forward(hx);
    std::printf("  LayerNorm（%d×%d）\n", d_model, seq2);
    print_diff(ln_dev, ln_host);

    // ---------------------------------------------------------------------
    // 多轮 decode：设备端 KV cache + 逐头 RoPE
    // ---------------------------------------------------------------------
    std::printf("\n=== 多轮 decode（设备 KV cache + 逐头 RoPE + GQA）===\n");
    {
        const int num_heads = 4;
        const int num_kv_heads = 2; // GQA：每 2 个 Q 头共享 1 个 KV 头
        const int d_head = 32;
        const int steps = 16;
        const int d_kv = num_kv_heads * d_head;

        cuda::dev_kv_caches_t<double> caches;
        // 容量刻意留出余量：这样 keys() 的"逻辑列数"与"存储前导维"就看得出不同
        caches.configure(num_heads, num_kv_heads, d_head, steps * 3);

        // RoPE 作用在 d_head 上（不是打包后的 d_kv）
        cuda::dev_rope_t<double> rope(d_head);
        rope.reserve(steps + 2);
        RoPE_net_t<mat_t<double>> host_rope(d_head);

        // 逐头 RoPE：打包成 (num_heads * d_head) 的行按头切成**行子块**（零拷贝），
        // 各自旋转到目标矩阵的对应行段。row_slice 就是为这个场景加的。
        auto rope_heads = [&](const dev_mat_t<double>& src, cuda::dev_matrix_t<double>& dst,
                              int heads, int pos) {
            for (int g = 0; g < heads; ++g)
                rope.rotate_into(row_slice(src, g * d_head, d_head),
                                 row_slice(dst.leaf(), g * d_head, d_head), pos);
        };
        // 主机侧同一操作的参考实现
        auto host_rope_heads = [&](const mat_t<double>& src, int heads, int pos) {
            mat_t<double> dst(src.row_num(), src.col_num());
            for (int g = 0; g < heads; ++g)
            {
                mat_t<double> head(d_head, src.col_num());
                for (int i = 0; i < d_head; ++i)
                    for (int j = 0; j < src.col_num(); ++j)
                        head(i, j) = src(g * d_head + i, j);
                auto rotated = host_rope.forward_at(head, pos);
                for (int i = 0; i < d_head; ++i)
                    for (int j = 0; j < src.col_num(); ++j)
                        dst(g * d_head + i, j) = rotated(i, j);
            }
            return dst;
        };

        // 主机侧照旧累积整块 K/V，用来对拍
        std::vector<mat_t<double>> host_k;
        std::vector<mat_t<double>> host_v;

        double worst_k = 0.0;
        double worst = 0.0;
        for (int s = 0; s < steps; ++s)
        {
            mat_t<double> k = make_host(d_kv, 1, 0.1 + 0.01 * s);
            mat_t<double> v = make_host(d_kv, 1, -0.05 + 0.005 * s);
            host_k.push_back(k);
            host_v.push_back(v);

            cuda::dev_matrix_t<double> dk(d_kv, 1, k), dv(d_kv, 1, v);

            // 契约：进 KV cache 的 K **必须是旋转过的**，位置是该 token 的绝对位置
            cuda::dev_matrix_t<double> k_rot(d_kv, 1);
            rope_heads(dk.const_leaf(), k_rot, num_kv_heads, s);
            worst_k = std::max(worst_k, max_abs_diff(k_rot.download(),
                                                     host_rope_heads(k, num_kv_heads, s))
                                                   .max_abs);

            // 唯一的写入入口：一批 token 一次写进**所有** KV 头。
            // 所以 GQA 下"多个 Q 头各自 append 同一个 KV 头"这种写法压根表达不出来。
            caches.append_all(k_rot.const_leaf(), dv);

            for (int h = 0; h < num_heads; ++h)
            {
                mat_t<double> q = make_host(d_head, 1, 0.03 * (h + 1) + 0.001 * s);
                cuda::dev_matrix_t<double> dq(d_head, 1, q);

                // Q 同样要旋转，位置是当前步
                auto q_rot = rope.forward_at(dq.const_leaf(), s);
                // Q→KV 头的映射由容器内部完成，调用方不必自己算分组
                mat_t<double> got = caches.attend_q_head(h, q_rot.const_leaf()).download();

                const int g = caches.kv_head_of(h);
                mat_t<double> k_g(d_head, s + 1), v_g(d_head, s + 1);
                for (int j = 0; j <= s; ++j)
                {
                    // cache 里存的是**旋转后**的 K：位置 j 的 token 用位置 j 旋转
                    mat_t<double> k_head(d_head, 1);
                    for (int i = 0; i < d_head; ++i)
                        k_head(i, 0) = host_k[static_cast<std::size_t>(j)](g * d_head + i, 0);
                    auto k_used = host_rope.forward_at(k_head, j);
                    for (int i = 0; i < d_head; ++i)
                    {
                        k_g(i, j) = k_used(i, 0);
                        v_g(i, j) = host_v[static_cast<std::size_t>(j)](g * d_head + i, 0);
                    }
                }

                mat_t<double> q_used = host_rope.forward_at(q, s);
                mat_t<double> scores = q_used.t().dot(k_g) / std::sqrt(static_cast<double>(d_head));
                mat_t<double> weights = hsoftmax(scores);
                mat_t<double> want = v_g.dot(weights.t());

                worst = std::max(worst, max_abs_diff(got, want).max_abs);
            }
        }

        std::printf("  %d 步 × %d 个 Q 头 = %d 次注意力，全部与主机端实现逐元素对拍\n", steps,
                    num_heads, steps * num_heads);
        std::printf("  RoPE（逐头，旋转后的 K）最大绝对误差: %.3e\n", worst_k);
        std::printf("  注意力输出最大绝对误差: %.3e\n", worst);

        const cuda::dev_kv_cache_t<double>& head0 = caches.cache_of_kv_head(0);
        std::printf("  KV 头 0：容量 %d 步、已用 %d 步\n", head0.capacity(), head0.length());

        // 零拷贝视图：逻辑列数是"已用长度"，存储前导维是"容量"，两者不等
        const dev_mat_t<double> keys = head0.keys();
        std::printf("  keys() 视图: %d×%d，存储前导维 %d（= 容量，不是已用长度）\n",
                    keys.row_num(), keys.col_num(), keys.leading_dim());
        std::printf("  → 视图直接指向 cache 缓冲区，每步都零拷贝；这是 dev_mat_t 把逻辑列数\n"
                    "     与前导维分开的直接好处（cuBLAS 支持 ld > cols，所以直接喂给 GEMM 即可）\n");
    }

    print_temperature();
    return 0;
}
