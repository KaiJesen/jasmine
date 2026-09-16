#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_kv_cache.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_rope.hpp"
#include "jas_kv_cache_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_net_t.hpp"
#include "jas_RoPE_t.hpp"
#include "test_helpers.hpp"

/**
 * 设备端 RoPE 的测试。
 *
 * 散热约束同其它 CUDA 测例：本机是无风扇的 Tesla P4，矩阵刻意取小、
 * 每个用例前后测温、超 80℃ 就 skip。详见 CUDA.md。
 *
 * **对拍基准一律是主机端 `RoPE_net_t`**，而不是另写一份公式 —— 这样测的是
 * 「设备端与主机端口径一致」，而不是「两处实现都照着同一份我写的公式抄对了」。
 * 覆盖两种配对约定（interleaved / half_split）与「列 j 用绝对位置 start_pos + j」
 * 这条容易写错的口径。
 */

using namespace jasmine;
using namespace jasmine::cuda;

namespace
{

constexpr int kHotCelsius = 80;

int gpu_temperature_c()
{
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

template <typename T>
mat_t<T> make_host(int rows, int cols, T base)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + T(0.1) * static_cast<T>((i * 7 + j * 13) % 29) - T(0.2) * i
                      + T(0.3) * j;
    return m;
}

void expect_matrices_match(const mat_t<double>& got, const mat_t<double>& want, double rel_tol,
                           const char* what)
{
    ASSERT_EQ(got.row_num(), want.row_num()) << what;
    ASSERT_EQ(got.col_num(), want.col_num()) << what;
    for (int i = 0; i < want.row_num(); ++i)
        for (int j = 0; j < want.col_num(); ++j)
        {
            const double scale = std::max(1.0, std::abs(want(i, j)));
            ASSERT_NEAR(got(i, j), want(i, j), rel_tol * scale)
                << what << " 在 (" << i << "," << j << ") 不一致";
        }
}

/** 主机端 `keys()` / `values()` 返回的是视图，没有到 `mat_t` 的隐式转换。 */
template <typename View>
mat_t<double> to_host_mat(const View& src)
{
    mat_t<double> out(src.row_num(), src.col_num());
    for (int i = 0; i < out.row_num(); ++i)
        for (int j = 0; j < out.col_num(); ++j)
            out(i, j) = static_cast<double>(src(i, j));
    return out;
}

/** 主机端参考：直接用项目既有的 RoPE_net_t，保证测的是口径一致而不是公式复述。 */
mat_t<double> host_rope(const mat_t<double>& x, int d, int start_pos, rope_pair_layout layout)
{
    RoPE_net_t<mat_t<double>> net(d);
    net.set_pair_layout(layout);
    return net.forward_at(x, start_pos);
}

class CudaRopeTest : public ::testing::Test
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

/** 两种配对约定 × 若干 (d, seq, start_pos) 组合，逐元素与主机对拍。 */
void check_rope_against_host(int d, int seq, int start_pos, rope_pair_layout layout)
{
    auto h = make_host<double>(d, seq, 0.5);
    dev_matrix_t<double> d_in(d, seq, h);

    dev_rope_t<double> rope(d, layout);
    auto got = rope.forward_at(d_in.leaf(), start_pos).download();

    expect_matrices_match(got, host_rope(h, d, start_pos, layout), 1e-12,
                          rope_pair_layout_name(layout));
}

} // namespace

// ---------------------------------------------------------------------------
// 行子块切片：类型与地址层面的保证（零发热，用主机缓冲验算地址）
// ---------------------------------------------------------------------------

TEST(CudaRowSliceContract, AddressArithmeticIsZeroCopy)
{
    // 每个元素填自己的下标：于是「读到的值」就等价于「算出的地址」
    double buf[24] = {};
    for (int k = 0; k < 24; ++k)
        buf[k] = static_cast<double>(k);

    const dev_mat_t<double> parent = make_dev_leaf(buf, 4, 4); // 步长 4

    const dev_mat_t<double> s = row_slice(parent, 1, 2);
    EXPECT_EQ(s.m_data, buf + 4) << "应当只挪指针，不拷贝";
    EXPECT_EQ(s.row_num(), 2);
    EXPECT_EQ(s.col_num(), 4);
    EXPECT_EQ(s.leading_dim(), 4); // 存储步长不动
    EXPECT_DOUBLE_EQ(s(0, 0), buf[4]);
    EXPECT_DOUBLE_EQ(s(1, 3), buf[11]);

    // 转置视图下逻辑行对应存储的列，偏移退化成 row0 个元素：
    // tr(r, c) == buf[c * 4 + r]，切片后 (r', c) == buf[c * 4 + 1 + r']
    const dev_mat_t<double> tr = parent.t();
    const dev_mat_t<double> ts = row_slice(tr, 1, 2);
    EXPECT_EQ(ts.m_data, buf + 1);
    EXPECT_EQ(ts.row_num(), 2);
    EXPECT_EQ(ts.col_num(), 4);
    EXPECT_DOUBLE_EQ(ts(0, 0), buf[1]);
    EXPECT_DOUBLE_EQ(ts(1, 2), buf[10]);
}

// ---------------------------------------------------------------------------
// 与主机 RoPE_net_t 对拍
// ---------------------------------------------------------------------------

TEST_F(CudaRopeTest, InterleavedMatchesHost)
{
    check_rope_against_host(8, 5, 0, rope_pair_layout::interleaved);
    check_rope_against_host(16, 1, 0, rope_pair_layout::interleaved);
}

TEST_F(CudaRopeTest, HalfSplitMatchesHost)
{
    // 位置 0 处两种约定都是恒等变换，所以必须用多列输入才有区分度
    check_rope_against_host(8, 5, 0, rope_pair_layout::half_split);
    check_rope_against_host(32, 3, 0, rope_pair_layout::half_split);
}

TEST_F(CudaRopeTest, MatchesHostAtNonZeroStartPosition)
{
    // 「列 j 用绝对位置 start_pos + j」是 decode 路径的命门：每步只喂一列，
    // 位置必须来自 cache 长度而不是列下标
    check_rope_against_host(16, 4, 7, rope_pair_layout::interleaved);
    check_rope_against_host(16, 4, 7, rope_pair_layout::half_split);
    check_rope_against_host(8, 1, 13, rope_pair_layout::interleaved);
    check_rope_against_host(8, 1, 13, rope_pair_layout::half_split);
}

TEST_F(CudaRopeTest, MatchesHostForLargerMatrix)
{
    check_rope_against_host(64, 24, 0, rope_pair_layout::interleaved);
    check_rope_against_host(64, 24, 0, rope_pair_layout::half_split);
}

TEST_F(CudaRopeTest, MatchesHostForFloat)
{
    constexpr int d = 16, seq = 6;
    auto h = make_host<float>(d, seq, 0.5f);
    dev_matrix_t<float> d_in(d, seq, h);

    for (rope_pair_layout layout :
         {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        dev_rope_t<float> rope(d, layout);
        auto got = rope.forward_at(d_in.leaf(), 2).download();

        RoPE_net_t<mat_t<float>> net(d);
        net.set_pair_layout(layout);
        auto want = net.forward_at(h, 2);

        for (int i = 0; i < d; ++i)
            for (int j = 0; j < seq; ++j)
            {
                const float scale = std::max(1.0f, std::abs(want(i, j)));
                ASSERT_NEAR(got(i, j), want(i, j), 1e-5f * scale)
                    << rope_pair_layout_name(layout) << " 在 (" << i << "," << j << ") 不一致";
            }
    }
}

TEST_F(CudaRopeTest, OnlyColumnZeroIsIdentityAtPositionZero)
{
    /**
     * 「列 j 的绝对位置是 start_pos + j」意味着 `start_pos = 0` 时**只有第 0 列**的旋转角
     * 是 0（恒等），第 j 列的绝对位置就是 j。整片矩阵并不是恒等变换 ——
     * 这一点最容易想当然，单独钉一下。
     */
    constexpr int d = 12, seq = 4;
    auto h = make_host<double>(d, seq, 0.7);

    for (rope_pair_layout layout :
         {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        dev_matrix_t<double> d_in(d, seq, h);
        dev_rope_t<double> rope(d, layout);
        auto got = rope.forward_at(d_in.leaf(), 0).download();

        for (int i = 0; i < d; ++i)
            ASSERT_DOUBLE_EQ(got(i, 0), h(i, 0))
                << rope_pair_layout_name(layout) << " 第 0 列（绝对位置 0）应恒等，行 " << i;

        bool any_differ = false;
        for (int i = 0; i < d && !any_differ; ++i)
            for (int j = 1; j < seq && !any_differ; ++j)
                any_differ = (got(i, j) != h(i, j));
        ASSERT_TRUE(any_differ) << rope_pair_layout_name(layout)
                                << " 第 1..seq-1 列绝对位置非 0，不该是恒等变换";
    }
}

TEST_F(CudaRopeTest, RotationPreservesPairNorm)
{
    // 旋转是正交变换：每一对特征构成的向量长度必须不变。
    // 这条不依赖主机参考，能独立抓出「下标映射写错」这类错误
    constexpr int d = 16, seq = 5;
    auto h = make_host<double>(d, seq, 0.9);
    dev_matrix_t<double> d_in(d, seq, h);

    for (rope_pair_layout layout :
         {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        dev_rope_t<double> rope(d, layout);
        auto got = rope.forward_at(d_in.leaf(), 3).download();

        const int half = d / 2;
        for (int i = 0; i < half; ++i)
            for (int j = 0; j < seq; ++j)
            {
                const int r0 = (layout == rope_pair_layout::half_split) ? i : 2 * i;
                const int r1 = (layout == rope_pair_layout::half_split) ? (i + half) : (2 * i + 1);
                const double before = h(r0, j) * h(r0, j) + h(r1, j) * h(r1, j);
                const double after = got(r0, j) * got(r0, j) + got(r1, j) * got(r1, j);
                ASSERT_NEAR(after, before, 1e-12 * std::max(1.0, before))
                    << rope_pair_layout_name(layout) << " 第 " << i << " 对 / 列 " << j;
            }
    }
}

// ---------------------------------------------------------------------------
// 输入形态与融合
// ---------------------------------------------------------------------------

TEST_F(CudaRopeTest, AcceptsTransposedInput)
{
    // 叶子可以是转置视图：kernel 通过 operator() 取元素，理应自动正确
    constexpr int d = 8, seq = 6;
    auto h = make_host<double>(d, seq, 0.4);
    dev_matrix_t<double> d_owner(seq, d, h.t());

    dev_rope_t<double> rope(d);
    auto got = rope.forward_at(d_owner.leaf().t(), 2).download();

    expect_matrices_match(got, host_rope(h, d, 2, rope_pair_layout::interleaved), 1e-12,
                          "转置输入");
}

TEST_F(CudaRopeTest, AcceptsAnExpression)
{
    // 旋转应当能直接融进上游表达式，不必先落一个临时矩阵
    constexpr int d = 8, seq = 5;
    auto h = make_host<double>(d, seq, 0.4);
    dev_matrix_t<double> d_in(d, seq, h);

    dev_rope_t<double> rope(d);
    auto got = rope.forward_at(d_in.leaf() * 2.0 + 1.0, 1).download();

    expect_matrices_match(got, host_rope(h * 2.0 + 1.0, d, 1, rope_pair_layout::interleaved),
                          1e-12, "表达式输入");
}

TEST_F(CudaRopeTest, InplaceMatchesOutOfPlace)
{
    constexpr int d = 16, seq = 7;
    auto h = make_host<double>(d, seq, 0.6);

    dev_matrix_t<double> a(d, seq, h);
    dev_matrix_t<double> b(d, seq, h);

    dev_rope_t<double> rope(d);
    auto fresh = rope.forward_at(a.leaf(), 4).download();

    auto before = b.download();
    rope.forward_inplace(b, 4);
    auto inplace = b.download();

    for (int i = 0; i < d; ++i)
        for (int j = 0; j < seq; ++j)
            ASSERT_DOUBLE_EQ(inplace(i, j), fresh(i, j)) << "在 (" << i << "," << j << ")";
    (void)before;
}

TEST_F(CudaRopeTest, ForwardDefaultsToPositionZero)
{
    constexpr int d = 8, seq = 3;
    auto h = make_host<double>(d, seq, 0.5);
    dev_matrix_t<double> d_in(d, seq, h);

    dev_rope_t<double> rope(d);
    auto a = rope.forward(d_in.leaf()).download();
    auto b = rope.forward_at(d_in.leaf(), 0).download();

    expect_matrices_match(a, b, 0.0, "forward 应等价于 forward_at(0)");
}

TEST_F(CudaRopeTest, PerHeadRopeOnPackedQkvMatchesHost)
{
    /**
     * 真实用法：QKV 打包成 `(num_heads * d_head × seq)`，RoPE **逐头**作用在 `d_head` 上。
     * 用 `row_slice` 把头切出来，各自 `rotate_into` 到目标矩阵的对应行段 —— 全程零拷贝。
     *
     * 主机端用同一个 `RoPE_net_t` 逐头旋转作为基准。
     */
    constexpr int num_heads = 3;
    constexpr int d_head = 8;
    constexpr int seq = 5;
    constexpr int packed = num_heads * d_head;
    constexpr int pos = 3;

    auto h = make_host<double>(packed, seq, 0.4);
    dev_matrix_t<double> d_in(packed, seq, h);

    dev_rope_t<double> rope(d_head);
    dev_matrix_t<double> d_out(packed, seq);
    for (int g = 0; g < num_heads; ++g)
        rope.rotate_into(row_slice(d_in.const_leaf(), g * d_head, d_head),
                         row_slice(d_out.leaf(), g * d_head, d_head), pos);

    RoPE_net_t<mat_t<double>> host_net(d_head);
    mat_t<double> want(packed, seq);
    for (int g = 0; g < num_heads; ++g)
    {
        mat_t<double> head(d_head, seq);
        for (int i = 0; i < d_head; ++i)
            for (int j = 0; j < seq; ++j)
                head(i, j) = h(g * d_head + i, j);
        auto rotated = host_net.forward_at(head, pos);
        for (int i = 0; i < d_head; ++i)
            for (int j = 0; j < seq; ++j)
                want(g * d_head + i, j) = rotated(i, j);
    }

    expect_matrices_match(d_out.download(), want, 1e-12, "逐头 RoPE（打包 QKV）");
}

// ---------------------------------------------------------------------------
// 表容量与模式
// ---------------------------------------------------------------------------

TEST_F(CudaRopeTest, AutoGrowsTableInDynamicMode)
{
    constexpr int d = 8;
    auto h = make_host<double>(d, 3, 0.5);
    dev_matrix_t<double> d_in(d, 3, h);

    dev_rope_t<double> rope(d);
    ASSERT_EQ(rope.capacity(), 0);

    // 先跑位置 0..2，再跑位置 100..102：中间没有调 reserve，
    // 表必须自己长大，且两种情况下位置口径都不能漂
    auto early = rope.forward_at(d_in.leaf(), 0).download();
    auto later = rope.forward_at(d_in.leaf(), 100).download();

    ASSERT_GT(rope.capacity(), 103);
    expect_matrices_match(early, host_rope(h, d, 0, rope_pair_layout::interleaved), 1e-12,
                          "扩容前的旋转");
    expect_matrices_match(later, host_rope(h, d, 100, rope_pair_layout::interleaved), 1e-12,
                          "扩容后的旋转");
}

TEST_F(CudaRopeTest, ReserveMakesStaticModeUsable)
{
    constexpr int d = 8, max_seq = 32;
    auto h = make_host<double>(d, 4, 0.5);
    dev_matrix_t<double> d_in(d, 4, h);

    dev_rope_t<double> rope(d);
    rope.set_cache_mode(rope_cache_mode::static_fixed);
    rope.reserve(max_seq);
    ASSERT_EQ(rope.capacity(), max_seq);

    auto got = rope.forward_at(d_in.leaf(), max_seq - 4).download();
    expect_matrices_match(got, host_rope(h, d, max_seq - 4, rope_pair_layout::interleaved), 1e-12,
                          "static 模式旋转");
}

TEST_F(CudaRopeTest, StaticModeRefusesToGrowBeyondReserve)
{
    constexpr int d = 8;
    auto h = make_host<double>(d, 4, 0.5);
    dev_matrix_t<double> d_in(d, 4, h);

    dev_rope_t<double> rope(d);
    rope.set_cache_mode(rope_cache_mode::static_fixed);
    rope.reserve(8);

    // 位置 5..8 需要下标 8，超出 reserve(8) 的 [0,8)
    EXPECT_THROW(rope.forward_at(d_in.leaf(), 5), std::runtime_error);
}

TEST_F(CudaRopeTest, SetParamInvalidatesTable)
{
    constexpr int d = 8;
    auto h = make_host<double>(d, 4, 0.5);
    dev_matrix_t<double> d_in(d, 4, h);

    dev_rope_t<double> rope(d);
    rope.reserve(16);
    ASSERT_GT(rope.capacity(), 0);

    rope.set_d(16);
    ASSERT_EQ(rope.capacity(), 0) << "换维度后旧表必须作废";

    dev_matrix_t<double> d_in16(16, 4, make_host<double>(16, 4, 0.5));
    auto got = rope.forward_at(d_in16.leaf(), 3).download();
    expect_matrices_match(got, host_rope(make_host<double>(16, 4, 0.5), 16, 3,
                                         rope_pair_layout::interleaved),
                          1e-12, "换维度后的旋转");
}

// ---------------------------------------------------------------------------
// 参数校验
// ---------------------------------------------------------------------------

TEST_F(CudaRopeTest, RejectsOddDimension)
{
    EXPECT_THROW(dev_rope_t<double> rope(7), std::invalid_argument);
    EXPECT_THROW(dev_rope_t<double> rope(0), std::invalid_argument);
    EXPECT_THROW(dev_rope_t<double> rope(-4), std::invalid_argument);
}

TEST_F(CudaRopeTest, RejectsDimensionMismatch)
{
    dev_rope_t<double> rope(8);
    dev_matrix_t<double> wrong(16, 4, make_host<double>(16, 4, 0.5));
    EXPECT_THROW(rope.forward_at(wrong.leaf(), 0), std::invalid_argument);
}

TEST_F(CudaRopeTest, RejectsNegativeStartPosition)
{
    constexpr int d = 8;
    dev_rope_t<double> rope(d);
    dev_matrix_t<double> d_in(d, 2, make_host<double>(d, 2, 0.5));
    EXPECT_THROW(rope.forward_at(d_in.leaf(), -1), std::invalid_argument);
}

TEST_F(CudaRopeTest, RejectsCallWithoutSetParam)
{
    dev_rope_t<double> rope; // 没设维度
    dev_matrix_t<double> d_in(8, 2, make_host<double>(8, 2, 0.5));
    // d == 0，先是「还没 set_param」再是形状不符，两者都是运行时错误
    EXPECT_THROW(rope.forward_at(d_in.leaf(), 0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 与 KV cache 合起来：这才是 RoPE 存在的理由
// ---------------------------------------------------------------------------

TEST_F(CudaRopeTest, DecodeLoopWithKvCacheMatchesHost)
{
    /**
     * 端到端：每一步把新 token 投影出的 k/v 先做 RoPE（**K 旋转后再 append**，
     * 这是 KV cache 的契约），再追加进 cache，然后 attend。
     *
     * 基准是主机端 RoPE_net_t + 主机端 kv_cache_t + 手写注意力公式 ——
     * 这条链路上任何一环的位置口径错了，末端的输出都对不上。
     */
    constexpr int d = 8;
    constexpr int steps = 6;
    constexpr int cap = steps + 2;

    dev_rope_t<double> rope(d);
    rope.reserve(cap);
    RoPE_net_t<mat_t<double>> host_rope_net(d);

    dev_kv_cache_t<double> dev_cache;
    dev_cache.reserve(d, cap);
    kv_cache_t<double> host_cache;
    host_cache.reserve(d, cap);

    for (int s = 0; s < steps; ++s)
    {
        auto k_h = make_host<double>(d, 1, 0.1 * (s + 1));
        auto v_h = make_host<double>(d, 1, -0.05 * (s + 1));
        dev_matrix_t<double> dk(d, 1, k_h);
        dev_matrix_t<double> dv(d, 1, v_h);

        // 契约：进 cache 的 K 必须是旋转过的，位置 = 当前 cache 长度
        auto k_rot = rope.forward_at(dk.leaf(), dev_cache.length());
        dev_cache.append(k_rot.leaf(), dv.leaf());
        host_cache.append(host_rope_net.forward_at(k_h, s), v_h);

        ASSERT_EQ(dev_cache.length(), host_cache.length()) << "第 " << s << " 步长度不一致";

        // 容器内容对拍：keys() 是零拷贝视图，先物化回紧凑布局再比
        dev_matrix_t<double> keys_packed(d, dev_cache.length());
        eval_fused(dev_cache.keys(), keys_packed.buffer());
        cuda::sync();
        expect_matrices_match(keys_packed.download(), to_host_mat(host_cache.keys()), 1e-12,
                              ("第 " + std::to_string(s) + " 步 cache 中的旋转后 K").c_str());
    }

    // 最后一步的注意力输出：q 也要旋转，位置是当前步
    const int last = steps - 1;
    auto q_h = make_host<double>(d, 1, 1.1);
    dev_matrix_t<double> dq(d, 1, q_h);
    auto q_rot = rope.forward_at(dq.leaf(), last);

    auto got = attend_cached(q_rot.const_leaf(), dev_cache);
    cuda::sync();

    auto host_keys = to_host_mat(host_cache.keys());
    auto host_values = to_host_mat(host_cache.values());
    auto q_host = host_rope_net.forward_at(q_h, last);

    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    auto q_copy = q_host;
    auto scores_h = mat_t<double>(q_copy.t().dot(host_keys) * scale);
    auto weights_h = hsoftmax(scores_h);
    auto want = mat_t<double>(host_values.dot(weights_h.t()));

    expect_matrices_match(got.download(), want, 1e-10, "带 RoPE 的 decode 注意力");
}
