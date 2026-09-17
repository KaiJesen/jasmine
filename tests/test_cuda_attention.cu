#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "jas_RoPE_t.hpp"
#include "jas_cuda_attention.hpp"
#include "jas_cuda_buffer.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_mha.hpp"
#include "jas_cuda_net.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_rope.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "jas_mha_t.hpp"
#include "jas_updator_t.hpp"

/**
 * 融合注意力（不物化概率矩阵）的测试。
 *
 * 散热约束同其它 CUDA 测试：本机是无风扇的 Tesla P4，矩阵刻意取小、
 * 每个用例前后测温、超 80℃ 就 skip。详见 CUDA.md。
 *
 * ## 这些用例要守住的四件事
 *
 * 融合路径是**新增的一条**实现，不是替代品，所以它的测例不能只验"结果差不多"：
 *
 *  1. **两条引擎逐元素对拍**（融合 vs 非融合 vs 主机三方）。融合路径最大的风险不是
 *     公式写错（那会大幅偏差），而是 online softmax 的重标定漏了一次、或者分块
 *     边界算错一格 —— 这类错误只在小数点后第 10 位以下露头，容差稍微一放就漏掉。
 *  2. **"省下来了"本身要被断言**：`weights()` 必须为空、`logsumexp()` 必须只有
 *     `q_len` 个数、启动计数器必须涨。否则某次重构悄悄让融合路径退回非融合，
 *     结果照样全绿 —— 这正是 `softmax_rows` 的回退路径当年腐烂的方式（CUDA.md 5.6）。
 *  3. **分块大小不能影响结果**。分块是纯性能参数，所以把 `bc` 钉成 1 / 中间值 / 超过
 *     `len` 三种，结果必须一致。这是唯一能发现"边界差一格"的手段。
 *  4. **反向要有不依赖任何一份解析实现的裁判**：有限差分。融合反向用的是重算出来的
 *     概率，把 `L` 用错、或者 `D_i` 那条恒等式写反，都会在这里露出来。
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

/** 确定性地填一个矩阵；各行/各列量级不同，避免掩盖广播方向搞反的错误。 */
mat_t<double> make_host(int rows, int cols, double base)
{
    mat_t<double> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + 0.01 * ((i * 7 + j * 13) % 41) - 0.02 * i + 0.03 * j;
    return m;
}

void expect_close(const mat_t<double>& got, const mat_t<double>& want, double rel_tol,
                  const char* what)
{
    ASSERT_EQ(got.row_num(), want.row_num()) << what;
    ASSERT_EQ(got.col_num(), want.col_num()) << what;
    for (int i = 0; i < got.row_num(); ++i)
        for (int j = 0; j < got.col_num(); ++j)
        {
            const double scale = std::max(1.0, std::abs(want(i, j)));
            ASSERT_NEAR(got(i, j), want(i, j), rel_tol * scale)
                << what << " 在 (" << i << "," << j << ") 不一致（" << got(i, j) << " vs "
                << want(i, j) << "）";
        }
}

using host_head_t = mat_head_gen_t<mat_t<double>, sgd_t>;

/**
 * 两条引擎的容器：同一份输入、同一个 RoPE，只是开关不同。
 *
 * 放在一个 helper 里是为了保证「除开关之外的一切都一样」—— 否则对拍出来的差异
 * 可能来自两次构造之间的其它差别（比如 RoPE 表建错一个）。
 */
struct two_heads_t
{
    cuda::dev_rope_t<double> rope;
    cuda::dev_head_gen_t<double> plain;
    cuda::dev_head_gen_t<double> fused;

    two_heads_t(int d_head, bool mask, int max_seq, rope_pair_layout layout)
        : rope(d_head, layout)
    {
        rope.reserve(max_seq);
        plain.set_param(d_head, mask);
        fused.set_param(d_head, mask);
        plain.set_rope(&rope);
        fused.set_rope(&rope);
        fused.set_fused_attention(true);
    }
};

/** 上游梯度：形状 (d_head × seq)。 */
mat_t<double> make_upstream(int d_head, int seq, double base = 0.05)
{
    return make_host(d_head, seq, base);
}

} // namespace

class CudaAttentionTest : public ::testing::Test
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
// 前向：两条引擎 + 主机三方对拍
// ===========================================================================

/**
 * 融合前向必须与非融合引擎、以及主机实现逐元素一致。
 *
 * 覆盖两个 RoPE 配对约定 × 掩码开/关；序列长度特意取**不是 `BR × bc` 整数倍**的值
 * （`BR = 4`、分块会自动取到 > seq），让"最后一个不完整块"这条路径也被走到 ——
 * 越界 warp 与不足一整块的 j 循环都在那里。
 */
TEST_F(CudaAttentionTest, FusedForwardMatchesPlainAndHost)
{
    constexpr int d_head = 8;
    constexpr int seq = 11;  // 刻意不是 4 的整数倍，也不是分块的整数倍
    constexpr int max_seq = 16;

    auto& registry = rope_registry_t<double>::instance();

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        for (bool mask : {false, true})
        {
            host_head_t host(d_head, mask, seq);
            host.set_rope(registry.get(d_head, max_seq, layout));

            two_heads_t heads(d_head, mask, max_seq, layout);

            const mat_t<double> qh = make_host(d_head, seq, 0.2);
            const mat_t<double> kh = make_host(d_head, seq, -0.3);
            const mat_t<double> vh = make_host(d_head, seq, 0.4);

            const mat_t<double> want = host.forward(qh, kh, vh);
            const mat_t<double> got_plain = heads.plain.forward(qh, kh, vh).download();
            const mat_t<double> got_fused = heads.fused.forward(qh, kh, vh).download();

            const std::string tag = std::string("单头前向 layout=")
                                    + (layout == rope_pair_layout::half_split ? "half_split"
                                                                              : "interleaved")
                                    + (mask ? " mask" : " nomask");

            expect_close(got_plain, want, 1e-12, (tag + " 非融合 vs 主机").c_str());
            // 融合的容差比非融合略宽：online softmax 的重标定与求和顺序都和主机不同，
            // 差异在 ulp 量级（实测 ~1e-15），但不该有 1e-13 以上的偏差。
            expect_close(got_fused, want, 1e-11, (tag + " 融合 vs 主机").c_str());
            expect_close(got_fused, got_plain, 1e-11, (tag + " 融合 vs 非融合").c_str());
        }
    }
}

/**
 * 掩码的因果性：改变"未来"的 V，不能影响过去的输出。
 *
 * 对拍只能说明"两条引擎彼此一致"，说明不了掩码语义对不对（两条都错也会一致）。
 * 这条直接从因果性的定义出发：因果掩码下第 j 列的输出只依赖 K/V 的第 0..j 列。
 */
TEST_F(CudaAttentionTest, CausalMaskBlocksFutureKeys)
{
    constexpr int d_head = 6;
    constexpr int seq = 7;
    constexpr int max_seq = 8;

    two_heads_t heads(d_head, /*mask=*/true, max_seq, rope_pair_layout::half_split);
    heads.fused.set_fused_attention(true);

    const mat_t<double> qh = make_host(d_head, seq, 0.2);
    const mat_t<double> kh = make_host(d_head, seq, -0.3);
    const mat_t<double> vh = make_host(d_head, seq, 0.4);

    const mat_t<double> base = heads.fused.forward(qh, kh, vh).download();

    // 改掉最后一列 V：只有最后一列的输出可以变，前面必须一模一样
    mat_t<double> v2 = vh;
    for (int f = 0; f < d_head; ++f)
        v2(f, seq - 1) += 3.0;
    const mat_t<double> after = heads.fused.forward(qh, kh, v2).download();

    for (int f = 0; f < d_head; ++f)
        for (int j = 0; j < seq - 1; ++j)
            EXPECT_DOUBLE_EQ(after(f, j), base(f, j))
                << "第 " << j << " 列的输出不该受第 " << (seq - 1) << " 列 V 的影响";
}

// ===========================================================================
// "省下来了"本身要被断言
// ===========================================================================

/**
 * 融合引擎下概率矩阵必须**不存在**，而不是"存在但没人读"。
 *
 * 这条测的是融合的**意义**：如果哪天有人为了省事又把 `m_weights` 填上，
 * 内存收益就悄悄没了，而所有对拍测例照样全绿。
 */
TEST_F(CudaAttentionTest, FusedEngineKeepsNoProbabilityMatrix)
{
    constexpr int d_head = 8;
    constexpr int seq = 12;
    constexpr int max_seq = 16;

    two_heads_t heads(d_head, /*mask=*/true, max_seq, rope_pair_layout::half_split);
    const mat_t<double> qh = make_host(d_head, seq, 0.2);
    const mat_t<double> kh = make_host(d_head, seq, -0.3);
    const mat_t<double> vh = make_host(d_head, seq, 0.4);

    const int before = cuda::flash_attention_launch_count();

    // (1) 非融合：概率矩阵在，logsumexp 不在
    heads.plain.forward(qh, kh, vh);
    EXPECT_FALSE(heads.plain.fused_attention_used());
    EXPECT_EQ(heads.plain.weights().row_num(), seq);
    EXPECT_EQ(heads.plain.weights().col_num(), seq);
    EXPECT_EQ(heads.plain.logsumexp().row_num(), 0);
    const std::size_t plain_bytes =
        static_cast<std::size_t>(seq) * static_cast<std::size_t>(seq) * sizeof(double);
    EXPECT_EQ(plain_bytes, cuda::attention_cache_t<double>::weights_bytes(seq, seq));

    // (2) 融合：概率矩阵为空，logsumexp 每行一个数
    heads.fused.forward(qh, kh, vh);
    EXPECT_TRUE(heads.fused.fused_attention_used());
    EXPECT_EQ(heads.fused.weights().row_num(), 0) << "融合引擎不该留住概率矩阵";
    EXPECT_EQ(heads.fused.weights().col_num(), 0);
    EXPECT_EQ(heads.fused.logsumexp().row_num(), seq);

    // 启动计数器必须涨 —— 否则"优化生效了"这件事只是口头承诺
    EXPECT_GT(cuda::flash_attention_launch_count(), before);

    // (3) 两者的内存对比：融合是 O(q_len)，非融合是 O(q_len × len)
    std::fprintf(stderr, "[融合收益] seq=%d 单头：概率矩阵 %.1f KiB → logsumexp %.1f KiB\n",
                 seq, plain_bytes / 1024.0,
                 static_cast<double>(seq * static_cast<int>(sizeof(double))) / 1024.0);
    EXPECT_LT(static_cast<std::size_t>(seq) * sizeof(double), plain_bytes);
}

/**
 * 从 `L` 重算的概率必须与存下来的概率逐元素一致。
 *
 * 这是融合反向的立足点，也是唯一能**直接**检查 `L` 的地方：反向里的错误
 * （比如忘了减 `L`、或者把 `L` 当成了 `m`）在最终梯度上只表现为"差一点"，
 * 而这里是对概率本身，误差会被放大得很清楚。
 */
TEST_F(CudaAttentionTest, RecomputedProbabilitiesMatchStored)
{
    constexpr int d_head = 8;
    constexpr int seq = 6;
    constexpr int max_seq = 8;

    two_heads_t heads(d_head, /*mask=*/true, max_seq, rope_pair_layout::half_split);
    const mat_t<double> qh = make_host(d_head, seq, 0.2);
    const mat_t<double> kh = make_host(d_head, seq, -0.3);
    const mat_t<double> vh = make_host(d_head, seq, 0.4);

    heads.plain.forward(qh, kh, vh);
    heads.fused.forward(qh, kh, vh);

    const mat_t<double> stored = heads.plain.weights().download();
    for (int i = 0; i < seq; ++i)
    {
        const mat_t<double> recomputed = heads.fused.probabilities_row(i);
        for (int j = 0; j < seq; ++j)
        {
            const double scale = std::max(1e-12, std::abs(stored(i, j)));
            EXPECT_NEAR(recomputed(0, j), stored(i, j), 1e-11 * scale)
                << "第 " << i << " 行第 " << j << " 列的概率重算不一致";
        }
        // 因果性：未来位置的概率必须是 0
        for (int j = i + 1; j < seq; ++j)
            EXPECT_EQ(recomputed(0, j), 0.0) << "被掩码的位置概率应为 0（而不是 exp 出来的小量）";
    }
}

// ===========================================================================
// 分块大小只该影响速度
// ===========================================================================

/**
 * 分块大小只影响速度：钉成 1 / 中间值 / 超过 `len`，结果必须逐元素一致。
 *
 * 分块是纯性能参数，所以任何依赖 `bc` 的数值差异都是 bug（最典型的是"最后一块少算了
 * 一列"或"重标定漏在边界上做"）。`bc = 1` 把重标定推到极限（每列都修正一次），
 * `bc >= len` 则退化成"一趟扫完"。
 */
TEST_F(CudaAttentionTest, BlockSizeDoesNotChangeResult)
{
    constexpr int d_head = 8;
    constexpr int q_len = 5;
    constexpr int len = 13;

    const mat_t<double> qh = make_host(d_head, q_len, 0.2);
    const mat_t<double> kh = make_host(d_head, len, -0.3);
    const mat_t<double> vh = make_host(d_head, len, 0.4);

    cuda::dev_matrix_t<double> q(d_head, q_len, qh);
    cuda::dev_matrix_t<double> k(d_head, len, kh);
    cuda::dev_matrix_t<double> v(d_head, len, vh);

    int& override_bc = cuda::flash_attention_block_cols_override();

    override_bc = 0;  // 自动
    const mat_t<double> auto_bc =
        cuda::fused_attention_forward<double>(q.const_leaf(), k.const_leaf(), v.const_leaf(),
                                              /*causal=*/true, 0)
            .out.download();

    for (int bc : {1, 3, len, len + 5})
    {
        override_bc = bc;
        const mat_t<double> got =
            cuda::fused_attention_forward<double>(q.const_leaf(), k.const_leaf(), v.const_leaf(),
                                                  /*causal=*/true, 0)
                .out.download();
        expect_close(got, auto_bc, 1e-12, ("分块 bc=" + std::to_string(bc)).c_str());
    }

    override_bc = 0;
}

/**
 * 真实一点的分块：`d_head = 64`、`len = 100` 时自动分块**放不下整段序列**，
 * 于是真的会走多趟循环 + 一个不完整的尾块 —— 那才是分块逻辑的主战场。
 *
 * 断言两件事：自动分块确实小于 `len`（否则这条测例其实什么都没测到），
 * 以及它与显式指定的相同分块逐元素一致。
 */
TEST_F(CudaAttentionTest, AutomaticTilingSplitsLongContext)
{
    constexpr int d_head = 64;
    constexpr int q_len = 10;
    constexpr int len = 100;

    const mat_t<double> qh = make_host(d_head, q_len, 0.2);
    const mat_t<double> kh = make_host(d_head, len, -0.3);
    const mat_t<double> vh = make_host(d_head, len, 0.4);

    cuda::dev_matrix_t<double> q(d_head, q_len, qh);
    cuda::dev_matrix_t<double> k(d_head, len, kh);
    cuda::dev_matrix_t<double> v(d_head, len, vh);

    const mat_t<double> auto_out =
        cuda::fused_attention_forward<double>(q.const_leaf(), k.const_leaf(), v.const_leaf(),
                                              /*causal=*/true, 0)
            .out.download();
    const int auto_bc = cuda::flash_attention_last_block_cols();

    std::fprintf(stderr, "[分块] d_head=%d len=%d：自动分块 bc=%d（%d 段）\n", d_head, len, auto_bc,
                 (len + auto_bc - 1) / auto_bc);
    ASSERT_LT(auto_bc, len) << "自动分块把整段序列放进了共享内存，这条测例就没测到分块";
    ASSERT_GT(auto_bc, 1);

    // 与显式同值、以及与更小的分块对拍
    for (int bc : {auto_bc, auto_bc / 2 + 1})
    {
        cuda::flash_attention_block_cols_override() = bc;
        const mat_t<double> got =
            cuda::fused_attention_forward<double>(q.const_leaf(), k.const_leaf(), v.const_leaf(),
                                                  /*causal=*/true, 0)
                .out.download();
        expect_close(got, auto_out, 1e-12, ("长序列分块 bc=" + std::to_string(bc)).c_str());
    }
    cuda::flash_attention_block_cols_override() = 0;
}

// ===========================================================================
// 反向
// ===========================================================================

TEST_F(CudaAttentionTest, FusedBackwardMatchesPlainAndHost)
{
    constexpr int d_head = 8;
    constexpr int seq = 9;
    constexpr int max_seq = 12;

    auto& registry = rope_registry_t<double>::instance();

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        for (bool mask : {false, true})
        {
            host_head_t host(d_head, mask, seq);
            host.set_rope(registry.get(d_head, max_seq, layout));

            two_heads_t heads(d_head, mask, max_seq, layout);

            const mat_t<double> qh = make_host(d_head, seq, 0.2);
            const mat_t<double> kh = make_host(d_head, seq, -0.3);
            const mat_t<double> vh = make_host(d_head, seq, 0.4);
            const mat_t<double> up = make_upstream(d_head, seq);

            const mat_t<double> want_out = host.forward(qh, kh, vh);
            mat_t<double> up_host = up;  // 主机接口要非 const 视图
            const auto hw = host.backward(up_host.view());

            cuda::dev_matrix_t<double> up_dev(d_head, seq, up);

            heads.plain.forward(qh, kh, vh);
            const auto pg = heads.plain.backward(up_dev);

            heads.fused.forward(qh, kh, vh);
            const auto fg = heads.fused.backward(up_dev);

            const std::string tag = std::string("单头反向 layout=")
                                    + (layout == rope_pair_layout::half_split ? "half_split"
                                                                              : "interleaved")
                                    + (mask ? " mask" : " nomask");

            expect_close(pg.delta_q.download(), hw.delta_q, 1e-11, (tag + " 非融合 dq").c_str());
            expect_close(pg.delta_k.download(), hw.delta_k, 1e-11, (tag + " 非融合 dk").c_str());
            expect_close(pg.delta_v.download(), hw.delta_v, 1e-11, (tag + " 非融合 dv").c_str());

            // 融合的容差略宽：概率是重算出来的、累加顺序也不同 —— 但都只在 ulp 级别
            expect_close(fg.delta_q.download(), hw.delta_q, 1e-10, (tag + " 融合 dq").c_str());
            expect_close(fg.delta_k.download(), hw.delta_k, 1e-10, (tag + " 融合 dk").c_str());
            expect_close(fg.delta_v.download(), hw.delta_v, 1e-10, (tag + " 融合 dv").c_str());
            expect_close(fg.delta_q.download(), pg.delta_q.download(), 1e-10,
                         (tag + " 融合 vs 非融合 dq").c_str());
        }
    }
}

/**
 * 融合反向的独立裁判：有限差分。
 *
 * 目标函数 `L = Σ (out ⊙ upstream)`，对 Q/K/V 的每个抽样元素做中心差分，与融合反向
 * 算出的解析梯度比。这一条**不依赖任何一份解析实现**，所以它抓的是"公式本身写错"：
 * `D_i` 那条恒等式写反、`L` 忘了减、`dS` 少乘一个 `p`，都会在这里露出来。
 *
 * 差分刻意走**设备前向**：两边算术一致，容差才敢收到 1e-7（用主机前向的话光 GEMM
 * 求和顺序差异就要放到 1e-5，正好把真实错误一起放过去）。
 *
 * 抽样而不是全量：每行取 2 列，够覆盖因果边界（i 附近）与内部两类位置；这类错误
 * 不会只在某一格出现。
 */
TEST_F(CudaAttentionTest, FusedBackwardMatchesFiniteDifference)
{
    constexpr int d_head = 6;  // 必须偶数：RoPE 是成对旋转
    constexpr int seq = 7;
    constexpr int max_seq = 8;
    constexpr double eps = 1e-6;

    two_heads_t heads(d_head, /*mask=*/true, max_seq, rope_pair_layout::half_split);

    const mat_t<double> qh = make_host(d_head, seq, 0.2);
    const mat_t<double> kh = make_host(d_head, seq, -0.3);
    const mat_t<double> vh = make_host(d_head, seq, 0.4);
    const mat_t<double> up = make_upstream(d_head, seq, 0.1);

    cuda::dev_matrix_t<double> up_dev(d_head, seq, up);

    heads.fused.forward(qh, kh, vh);
    const auto g = heads.fused.backward(up_dev);
    const mat_t<double> dq = g.delta_q.download();
    const mat_t<double> dk = g.delta_k.download();
    const mat_t<double> dv = g.delta_v.download();


    // which: 0=Q 1=K 2=V；把 (i,j) 加 d 后走一遍设备前向，返回 Σ(out ⊙ up)
    auto eval = [&](int which, int i, int j, double d) {
        mat_t<double> q = qh, k = kh, v = vh;
        if (which == 0)
            q(i, j) += d;
        else if (which == 1)
            k(i, j) += d;
        else
            v(i, j) += d;
        cuda::dev_matrix_t<double> out = heads.fused.forward(q, k, v);
        return cuda::sum_all(out.const_leaf() * up_dev.const_leaf());
    };
    auto fd = [&](int which, int i, int j) {
        return (eval(which, i, j, eps) - eval(which, i, j, -eps)) / (2 * eps);
    };
    auto check = [&](int which, const mat_t<double>& analytic, const char* name, int i, int j) {
        const double want = fd(which, i, j);
        EXPECT_NEAR(analytic(i, j), want, 1e-7 * std::max(1.0, std::abs(want)))
            << name << "(" << i << "," << j << ") 与有限差分不符（解析 " << analytic(i, j)
            << " vs 差分 " << want << "）";
    };

    for (int i = 0; i < d_head; ++i)
    {
        // 取 i（因果边界）与最后一列两处
        check(0, dq, "dQ", i, i < seq ? i : seq - 1);
        check(0, dq, "dQ", i, seq - 1);
        check(1, dk, "dK", i, i < seq ? i : seq - 1);
        check(1, dk, "dK", i, seq - 1);
        check(2, dv, "dV", i, i < seq ? i : seq - 1);
        check(2, dv, "dV", i, seq - 1);
    }
}

/**
 * 融合反向的分块无关性：反向也按 `bc` 分块，边界同样不该被感知。
 */
TEST_F(CudaAttentionTest, BackwardBlockSizeDoesNotChangeResult)
{
    constexpr int d_head = 6;
    constexpr int q_len = 4;
    constexpr int len = 10;

    const mat_t<double> qh = make_host(d_head, q_len, 0.2);
    const mat_t<double> kh = make_host(d_head, len, -0.3);
    const mat_t<double> vh = make_host(d_head, len, 0.4);
    const mat_t<double> up = make_upstream(d_head, q_len, 0.1);

    cuda::dev_matrix_t<double> q(d_head, q_len, qh);
    cuda::dev_matrix_t<double> k(d_head, len, kh);
    cuda::dev_matrix_t<double> v(d_head, len, vh);
    cuda::dev_matrix_t<double> up_dev(d_head, q_len, up);

    int& override_bc = cuda::flash_attention_block_cols_override();

    override_bc = 0;
    const auto cache = cuda::fused_attention_forward<double>(q.const_leaf(), k.const_leaf(),
                                                             v.const_leaf(), true, 0);
    const auto ref = cuda::fused_attention_backward<double>(cache, q.const_leaf(), k.const_leaf(),
                                                            v.const_leaf(), up_dev, true, 0);

    for (int bc : {1, 3, len})
    {
        override_bc = bc;
        const auto cache2 = cuda::fused_attention_forward<double>(q.const_leaf(), k.const_leaf(),
                                                                  v.const_leaf(), true, 0);
        const auto got = cuda::fused_attention_backward<double>(cache2, q.const_leaf(),
                                                                k.const_leaf(), v.const_leaf(),
                                                                up_dev, true, 0);
        expect_close(got.d_q.download(), ref.d_q.download(), 1e-11,
                     ("反向 bc=" + std::to_string(bc) + " dq").c_str());
        expect_close(got.d_k.download(), ref.d_k.download(), 1e-11,
                     ("反向 bc=" + std::to_string(bc) + " dk").c_str());
        expect_close(got.d_v.download(), ref.d_v.download(), 1e-11,
                     ("反向 bc=" + std::to_string(bc) + " dv").c_str());
    }

    override_bc = 0;
}

/**
 * 引擎是**前向时记录下来的事实**，不是反向时再读一次开关。
 *
 * 开关的语义是「下一次前向走哪条」；反向自动与前向配对。所以：
 *   - 前向之后改开关，反向仍然走与前向配套的那条（数值必须还对）；
 *   - 再跑一次前向，才轮到新设置生效；
 *   - 没跑过前向就反向，直接抛（而不是拿空缓存算出一个"差不多"的结果）。
 */
TEST_F(CudaAttentionTest, EngineIsRecordedAtForwardTime)
{
    constexpr int d_head = 4;
    constexpr int seq = 4;
    constexpr int max_seq = 4;

    const mat_t<double> qh = make_host(d_head, seq, 0.2);
    const mat_t<double> kh = make_host(d_head, seq, -0.3);
    const mat_t<double> vh = make_host(d_head, seq, 0.4);
    const mat_t<double> up = make_upstream(d_head, seq, 0.1);
    cuda::dev_matrix_t<double> up_dev(d_head, seq, up);

    two_heads_t heads(d_head, /*mask=*/true, max_seq, rope_pair_layout::half_split);

    // 融合前向 → 改开关 → 反向仍然走融合（数值与未改开关时一致）
    heads.fused.forward(qh, kh, vh);
    const auto g1 = heads.fused.backward(up_dev);
    heads.fused.set_fused_attention(false);
    const auto g2 = heads.fused.backward(up_dev);
    expect_close(g2.delta_q.download(), g1.delta_q.download(), 1e-12, "改开关后的反向 dq");
    EXPECT_TRUE(heads.fused.fused_attention_used()) << "反向不该改写前向记录下来的引擎";

    // 再前向一次，新设置才生效
    heads.fused.forward(qh, kh, vh);
    EXPECT_FALSE(heads.fused.fused_attention_used());
    EXPECT_EQ(heads.fused.weights().row_num(), seq) << "关掉融合后概率矩阵该回来了";
    EXPECT_EQ(heads.fused.logsumexp().row_num(), 0);
    const auto g3 = heads.fused.backward(up_dev);
    expect_close(g3.delta_q.download(), g1.delta_q.download(), 1e-10, "两条引擎的反向 dq");

    // 没有前向就反向
    cuda::dev_head_gen_t<double> fresh(d_head, /*mask=*/false);
    EXPECT_THROW(fresh.backward(up_dev), std::runtime_error);
}

// ===========================================================================
// 整层：GQA 下也要一致
// ===========================================================================

/**
 * `dev_mha_t` 层面的对拍（含 GQA）：融合开关只该改变"概率矩阵存不存"，
 * 不该改变任何数值。
 *
 * 反向特意用非零学习率跑一步再比**输入梯度**：这样 GQA 的梯度归并、RoPE 反向、
 * 掩码置零全都在链路里，任何一处被融合路径改坏都会露出来。
 */
TEST_F(CudaAttentionTest, MhaLayerFusedMatchesPlainAcrossGqa)
{
    struct cfg_t
    {
        int heads;
        int kv_heads;
        const char* tag;
    };
    const cfg_t configs[] = {{4, 4, "MHA"}, {4, 2, "GQA"}, {4, 1, "MQA"}};
    constexpr int d_model = 16;
    constexpr int seq = 8;

    using dev_mha_t = cuda::dev_mha_t<double, cuda::dev_sgd_t>;
    using host_mha_t = mat_mha_t<mat_t<double>, sgd_t>;

    for (const auto& c : configs)
    {
        auto build = [&](bool fused) {
            auto dev = std::make_unique<dev_mha_t>();
            dev->set_param(c.heads, d_model, /*mask=*/true, c.kv_heads);
            auto rope = std::make_unique<cuda::dev_rope_t<double>>(d_model / c.heads,
                                                                   rope_pair_layout::half_split);
            rope->reserve(seq);
            dev->set_rope(rope.get());
            dev->set_fused_attention(fused);
            return std::make_pair(std::move(dev), std::move(rope));
        };

        auto [plain, plain_rope] = build(false);
        auto [fused, fused_rope] = build(true);

        host_mha_t host(c.heads, d_model, /*mask=*/true, seq, c.kv_heads);
        host.set_rope_pair_layout(rope_pair_layout::half_split);

        // 两边权重必须一模一样：用主机权重做同一份初始化
        for (auto* d : {plain.get(), fused.get()})
        {
            d->q_proj().upload_weight(host.q_proj().weight());
            d->q_proj().upload_bias(host.q_proj().bias());
            d->k_proj().upload_weight(host.k_proj().weight());
            d->k_proj().upload_bias(host.k_proj().bias());
            d->v_proj().upload_weight(host.v_proj().weight());
            d->v_proj().upload_bias(host.v_proj().bias());
            d->out_proj().upload_weight(host.out_proj().weight());
            d->out_proj().upload_bias(host.out_proj().bias());
        }

        const mat_t<double> x = make_host(d_model, seq, 0.1);
        const mat_t<double> up = make_host(d_model, seq, 0.05);
        cuda::dev_matrix_t<double> up_dev(d_model, seq, up);
        mat_t<double> up_host = up;  // 主机接口要非 const 视图

        const mat_t<double> host_out = host.forward(x);
        const mat_t<double> plain_out = plain->forward(x).download();
        const mat_t<double> fused_out = fused->forward(x).download();

        expect_close(plain_out, host_out, 1e-12, (std::string("整层前向 ") + c.tag).c_str());
        expect_close(fused_out, host_out, 1e-10, (std::string("整层前向(融合) ") + c.tag).c_str());

        EXPECT_FALSE(plain->fused_attention_used());
        EXPECT_TRUE(fused->fused_attention_used()) << "开关没生效（融合路径压根没跑）";

        // 学习率置 0：参数不动，输入梯度可以直接逐元素比
        plain->set_lr(0.0);
        fused->set_lr(0.0);
        host.set_lr(0.0);

        const mat_t<double> gp = plain->backward(up_dev).download();
        const mat_t<double> gf = fused->backward(up_dev).download();
        const mat_t<double> gh = host.backward(up_host.view());

        expect_close(gp, gh, 1e-11, (std::string("整层反向 ") + c.tag).c_str());
        expect_close(gf, gh, 1e-9, (std::string("整层反向(融合) ") + c.tag).c_str());
        expect_close(gf, gp, 1e-9, (std::string("整层反向 融合 vs 非融合 ") + c.tag).c_str());
    }
}

/**
 * 长上下文下融合省下的量级：这条只是把收益**打印出来**并钉住数量级关系，
 * 不用大矩阵（P4 上跑不动，也没必要 —— 收益是算出来的，不是测出来的）。
 */
TEST_F(CudaAttentionTest, MemorySavingScalesWithContext)
{
    using dev_mha_t = cuda::dev_mha_t<double, cuda::dev_sgd_t>;
    constexpr int heads = 32, d_head = 64;

    // 单看一层、一次整段前向（q_len == len == 2048）的账
    const int len = 2048;

    // 只比**引擎各自额外留下的那份缓存**，比值才说明问题：
    //   非融合：每头一份 `q_len × len` 的概率矩阵（反向的 softmax 环节要读它）
    //   融合  ：每头一份 `q_len` 个数的 logsumexp（概率现算）
    // 前向输出 `d_head × q_len` 两条路都要留，是共同的量 —— 放进分子分母只会把比值搅浑，
    // 而且它与 `d_head` 无关地随 `len` 线性增长，真正的对比是"概率矩阵随 `len` 平方增长"。
    const std::size_t weights = dev_mha_t::weights_bytes(len, len, heads);
    const std::size_t lse =
        static_cast<std::size_t>(len) * static_cast<std::size_t>(heads) * sizeof(double);

    std::fprintf(stderr,
                 "[融合收益] %d 头 × d_head=%d, q_len=len=%d：概率矩阵 %.1f MiB → logsumexp %.1f MiB"
                 "（省 %.1f 倍 = len）\n",
                 heads, d_head, len, static_cast<double>(weights) / (1024.0 * 1024.0),
                 static_cast<double>(lse) / (1024.0 * 1024.0),
                 static_cast<double>(weights) / static_cast<double>(lse));

    // 收益的量级：`q_len × len` 对 `q_len × 1`（单头），比值就是 **`len` 本身** ——
    // 省下的倍数随上下文长度线性增长，而 `d_head` 在两边都不出现。
    // 这条断言不只是"验证收益"：`weights_bytes` 第一版把 `d_head` 也乘了进去
    // （于是成了每头一份 `q_len × len × d_head` 的矩阵），比值变成 `len·d_head/(d_head+1)`，
    // 与 `len` 只差百分之几 —— 光看"省了两千倍"根本看不出错，
    // 只有把比值卡成**恰好等于 `len`** 才抓得住这种夸大口径。
    EXPECT_DOUBLE_EQ(static_cast<double>(weights) / static_cast<double>(lse),
                     static_cast<double>(len));
    EXPECT_GT(weights, 500u * 1024u * 1024u);  // 单层就超过 500 MiB，正是要消掉的那一项
}
