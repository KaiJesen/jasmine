#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_kv_cache.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_kv_cache_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "test_helpers.hpp"

/**
 * 设备端 KV cache 的测试。
 *
 * 散热约束同其它 CUDA 测试：本机是无风扇的 Tesla P4，矩阵刻意取小、
 * 每个用例前后测温、超 80℃ 就 skip。详见 CUDA.md。
 *
 * 对拍基准有两个，分别验证两件事：
 *
 *  1. **容器语义**：以主机端 `kv_cache_t` 为参照 —— 同一串 append 之后
 *     `keys()` / `values()` 的长度与内容必须逐元素一致。这直接验证零拷贝
 *     子视图（前导维 > 列数）真的读对了位置。
 *  2. **注意力数值**：以手写的主机公式为参照，覆盖 decode（无掩码）、
 *     prefill（因果掩码）、GQA（多个 Q 头共享一个 KV 头）。
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

/** 确定性地填一个矩阵；让各行/各列量级不同，避免掩盖行列搞反的错误。 */
template <typename T>
mat_t<T> make_host(int rows, int cols, T base)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + T(0.01) * static_cast<T>((i * 7 + j * 13) % 41) - T(0.02) * i
                      + T(0.03) * j;
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

/**
 * 把零拷贝视图物化成紧凑矩阵再回读。
 *
 * 这本身就是一条关键测试：`eval_fused` 走的是叶子的 `operator()`，
 * 若它错用了 col_num() 而不是 leading_dim() 当行步长，读到的就是错位元素。
 */
mat_t<double> pack_to_host(const dev_mat_t<double>& leaf)
{
    dev_matrix_t<double> out(leaf.row_num(), leaf.col_num());
    eval_fused(leaf, out.buffer());
    cuda::sync();
    return out.download();
}

/** 主机参考：与设备 attend_cached 完全同一个公式。 */
mat_t<double> host_attend(const mat_t<double>& q, const mat_t<double>& k, const mat_t<double>& v,
                          const mat_t<double>* mask = nullptr)
{
    const double scale = 1.0 / std::sqrt(static_cast<double>(q.row_num()));
    // 主机端 mat_t::t() 没有 const 版本，先拷一份再取转置
    mat_t<double> q_copy = q;
    mat_t<double> qt = q_copy.t();
    mat_t<double> scores = (mask != nullptr) ? mat_t<double>(qt.dot(k) * scale + *mask)
                                             : mat_t<double>(qt.dot(k) * scale);
    mat_t<double> weights = hsoftmax(scores);
    mat_t<double> wt = weights.t();
    mat_t<double> out = v.dot(wt);
    return out;
}

/** `mat_view_t` 没有到 `mat_t` 的隐式转换，要显式搬一份。 */
template <typename View>
mat_t<double> to_host_mat(const View& src)
{
    mat_t<double> out(src.row_num(), src.col_num());
    for (int i = 0; i < out.row_num(); ++i)
        for (int j = 0; j < out.col_num(); ++j)
            out(i, j) = static_cast<double>(src(i, j));
    return out;
}

/** 取 src 的 [row0, row0+rows) 行、前 cols 列（模拟按 KV 头切片）。 */
mat_t<double> slice_rows(const mat_t<double>& src, int row0, int rows, int cols)
{
    mat_t<double> out(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out(i, j) = src(row0 + i, j);
    return out;
}

/** 因果掩码：query 第 i 列（绝对位置 q_pos+i）可见 key 第 0..q_pos+i 列。 */
mat_t<double> make_causal_mask(int q_len, int kv_len, int q_pos)
{
    mat_t<double> m(q_len, kv_len);
    for (int i = 0; i < q_len; ++i)
        for (int j = 0; j < kv_len; ++j)
            m(i, j) = (j <= q_pos + i) ? 0.0 : -std::numeric_limits<double>::infinity();
    return m;
}

/**
 * 把 b 写到 acc 的第 col0 列起。
 *
 * 刻意**不**自增 col0 —— 早先的版本自增，于是 `put(k); put(v)` 共用一个游标时
 * V 会被整体写偏一列，看起来像设备端算错，实际是测试自己错位。
 */
void put_cols(mat_t<double>& acc, int col0, const mat_t<double>& b)
{
    for (int i = 0; i < b.row_num(); ++i)
        for (int j = 0; j < b.col_num(); ++j)
            acc(i, col0 + j) = b(i, j);
}

class CudaKvCacheTest : public ::testing::Test
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
// 前导维：类型层面的保证（零发热）
// ---------------------------------------------------------------------------

TEST(CudaLeadingDimContract, SeparateFromLogicalCols)
{
    // 每个元素填自己的下标：于是"读到的值"就等价于"算出的地址"，断言失败时也好读
    double buf[16] = {};
    for (int k = 0; k < 16; ++k)
        buf[k] = static_cast<double>(k);

    const dev_mat_t<double> strided = make_dev_leaf_strided(buf, 2, 3, 8);

    EXPECT_EQ(strided.row_num(), 2);
    EXPECT_EQ(strided.col_num(), 3);     // 逻辑列数
    EXPECT_EQ(strided.leading_dim(), 8); // 存储步长 —— 两者刻意不等

    // 行优先、步长 8：元素 (i, j) 落在 i*8 + j
    EXPECT_DOUBLE_EQ(strided(0, 0), buf[0]);
    EXPECT_DOUBLE_EQ(strided(0, 2), buf[2]);
    EXPECT_DOUBLE_EQ(strided(1, 0), buf[8]);
    EXPECT_DOUBLE_EQ(strided(1, 2), buf[10]);
    EXPECT_NE(strided(1, 2), buf[5]) << "若错用 col_num() 当步长，这里会读到 buf[5]";

    // view() 只改逻辑形状，指针与步长都不动（零拷贝子视图的全部内容）
    const dev_mat_t<double> sub = strided.view(2, 1);
    EXPECT_EQ(sub.row_num(), 2);
    EXPECT_EQ(sub.col_num(), 1);
    EXPECT_EQ(sub.leading_dim(), 8);
    EXPECT_EQ(sub.m_data, strided.m_data);
    EXPECT_DOUBLE_EQ(sub(1, 0), buf[8]);

    // 转置只翻标志位，步长不变
    EXPECT_EQ(strided.t().leading_dim(), 8);
    EXPECT_EQ(strided.t().row_num(), 3);
    EXPECT_EQ(strided.t().col_num(), 2);
}

TEST(CudaLeadingDimContract, LeafStaysTriviallyCopyable)
{
    static_assert(std::is_trivially_copyable_v<dev_mat_t<double>>);
    static_assert(is_device_evaluable_v<dev_mat_t<double>>);
    static_assert(is_self_contained_v<dev_mat_t<double>>);
    SUCCEED();
}

TEST(CudaLeadingDimContract, DefaultLeadingDimEqualsCols)
{
    double buf[6] = {};
    const dev_mat_t<double> packed = make_dev_leaf(buf, 2, 3);
    EXPECT_EQ(packed.leading_dim(), packed.col_num());

    const dev_mat_t<double> flipped = dev_mat_t<double>(buf, 2, 3, true);
    EXPECT_EQ(flipped.leading_dim(), 3);
    EXPECT_TRUE(flipped.transposed());
}

// ---------------------------------------------------------------------------
// 容器语义：与主机端 kv_cache_t 对拍
// ---------------------------------------------------------------------------

TEST_F(CudaKvCacheTest, MatchesHostContainerContents)
{
    const int d = 6;
    const int cap = 10;

    kv_cache_t<double> host;
    host.reserve(d, cap);

    dev_kv_cache_t<double> dev;
    dev.reserve(d, cap);

    ASSERT_EQ(dev.dim(), d);
    ASSERT_EQ(dev.capacity(), cap);
    ASSERT_EQ(dev.length(), 0);
    ASSERT_TRUE(dev.empty());

    int pos = 0;
    for (const int n_new : {1, 3, 2})
    {
        mat_t<double> k = make_host<double>(d, n_new, 0.1 * (pos + 1));
        mat_t<double> v = make_host<double>(d, n_new, -0.05 * (pos + 1));
        host.append(k, v);

        dev_matrix_t<double> dk(d, n_new, k);
        dev_matrix_t<double> dv(d, n_new, v);
        dev.append(dk, dv);

        pos += n_new;
        ASSERT_EQ(dev.length(), host.length()) << "长度应与主机端一致";
        EXPECT_EQ(dev.length(), pos);

        // 零拷贝视图必须与主机端 view 的内容一致 —— 这直接验证前导维被用对了
        mat_t<double> host_keys = to_host_mat(host.keys());
        mat_t<double> host_values = to_host_mat(host.values());
        expect_matrices_match(pack_to_host(dev.keys()), host_keys, 1e-12, "keys 内容");
        expect_matrices_match(pack_to_host(dev.values()), host_values, 1e-12, "values 内容");
    }
}

TEST_F(CudaKvCacheTest, KeysViewIsZeroCopyWithCapacityStride)
{
    const int d = 4;
    const int cap = 12;

    dev_kv_cache_t<double> dev;
    dev.reserve(d, cap);

    mat_t<double> k1 = make_host<double>(d, 3, 0.5);
    mat_t<double> v1 = make_host<double>(d, 3, 0.25);
    dev_matrix_t<double> dk1(d, 3, k1);
    dev_matrix_t<double> dv1(d, 3, v1);
    dev.append(dk1, dv1);

    const dev_mat_t<double> keys = dev.keys();
    EXPECT_EQ(keys.row_num(), d);
    EXPECT_EQ(keys.col_num(), 3);
    // 关键：视图的前导维是**容量**而不是已用长度 —— 零拷贝就是这么来的
    EXPECT_EQ(keys.leading_dim(), cap);

    // 未触发扩容，指针应当稳定（薄壳指向同一块缓冲区）
    const double* before = keys.m_data;
    mat_t<double> k2 = make_host<double>(d, 2, 0.75);
    mat_t<double> v2 = make_host<double>(d, 2, 0.4);
    dev_matrix_t<double> dk2(d, 2, k2);
    dev_matrix_t<double> dv2(d, 2, v2);
    dev.append(dk2, dv2);

    const dev_mat_t<double> keys_after = dev.keys();
    EXPECT_EQ(keys_after.m_data, before) << "reserve 之后不应重新分配";
    EXPECT_EQ(keys_after.col_num(), 5);
    EXPECT_EQ(keys_after.leading_dim(), cap);

    // 再取一次的视图内容 == 两次 append 的拼接
    mat_t<double> expected(d, 5);
    put_cols(expected, 0, k1);
    put_cols(expected, 3, k2);
    expect_matrices_match(pack_to_host(keys_after), expected, 1e-12, "拼接后的 keys");
}

TEST_F(CudaKvCacheTest, DynamicGrowthPreservesContents)
{
    // 不 reserve，靠 dynamic 模式自己翻倍扩容；跨过最小扩容步长才会真的重分配
    const int d = 3;

    kv_cache_t<double> host;
    dev_kv_cache_t<double> dev;

    for (int step = 0; step < 40; ++step)
    {
        mat_t<double> k = make_host<double>(d, 2, 0.01 * (step + 1));
        mat_t<double> v = make_host<double>(d, 2, 0.02 * (step + 1));
        host.append(k, v);

        dev_matrix_t<double> dk(d, 2, k);
        dev_matrix_t<double> dv(d, 2, v);
        dev.append(dk, dv);
    }

    ASSERT_EQ(dev.length(), host.length());
    EXPECT_EQ(dev.length(), 80);
    EXPECT_GE(dev.capacity(), 80);

    // 扩容要搬老数据，这里验证搬完之后一个元素都没错位
    mat_t<double> host_keys = to_host_mat(host.keys());
    mat_t<double> host_values = to_host_mat(host.values());
    expect_matrices_match(pack_to_host(dev.keys()), host_keys, 1e-12, "扩容后的 keys");
    expect_matrices_match(pack_to_host(dev.values()), host_values, 1e-12, "扩容后的 values");
}

TEST_F(CudaKvCacheTest, StaticFixedRefusesToGrow)
{
    dev_kv_cache_t<double> dev;
    dev.set_mode(kv_cache_mode::static_fixed);
    dev.reserve(4, 3);

    mat_t<double> k = make_host<double>(4, 3, 0.5);
    mat_t<double> v = make_host<double>(4, 3, 0.25);
    dev_matrix_t<double> dk(4, 3, k);
    dev_matrix_t<double> dv(4, 3, v);
    dev.append(dk, dv);
    EXPECT_EQ(dev.length(), 3);

    EXPECT_THROW(dev.append(dk, dv), std::runtime_error);
    EXPECT_EQ(dev.length(), 3) << "抛异常后长度不应变化";

    // clear 之后内存还在，可以接着用
    dev.clear();
    EXPECT_EQ(dev.length(), 0);
    dev.append(dk, dv);
    EXPECT_EQ(dev.length(), 3);
}

TEST_F(CudaKvCacheTest, RejectsBadShapes)
{
    dev_kv_cache_t<double> dev;
    dev.reserve(4, 8);

    EXPECT_THROW(dev.keys(), std::runtime_error) << "空缓存取视图应报错";

    mat_t<double> k = make_host<double>(4, 2, 0.5);
    mat_t<double> v = make_host<double>(4, 3, 0.25); // token 数不一致
    dev_matrix_t<double> dk(4, 2, k);
    dev_matrix_t<double> dv(4, 3, v);
    EXPECT_THROW(dev.append(dk, dv), std::runtime_error);

    mat_t<double> k_wrong = make_host<double>(5, 2, 0.5); // 行数与 head_dim 不符
    mat_t<double> v_wrong = make_host<double>(5, 2, 0.25);
    dev_matrix_t<double> dk2(5, 2, k_wrong);
    dev_matrix_t<double> dv2(5, 2, v_wrong);
    EXPECT_THROW(dev.append(dk2, dv2), std::runtime_error);
}

TEST_F(CudaKvCacheTest, ReserveRejectsBadSizes)
{
    dev_kv_cache_t<double> dev;
    EXPECT_THROW(dev.reserve(0, 8), std::invalid_argument);
    EXPECT_THROW(dev.reserve(4, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// attention：decode / prefill
// ---------------------------------------------------------------------------

TEST_F(CudaKvCacheTest, DecodeStepsMatchHost)
{
    const int d = 8;
    const int steps = 12;
    const int cap = 16;

    kv_cache_t<double> host_cache;
    host_cache.reserve(d, cap);

    dev_kv_cache_t<double> dev_cache;
    dev_cache.reserve(d, cap);

    for (int s = 0; s < steps; ++s)
    {
        mat_t<double> k = make_host<double>(d, 1, 0.1 * (s + 1));
        mat_t<double> v = make_host<double>(d, 1, -0.05 * (s + 1));

        host_cache.append(k, v);
        dev_matrix_t<double> dk(d, 1, k);
        dev_matrix_t<double> dv(d, 1, v);
        dev_cache.append(dk, dv);

        mat_t<double> q = make_host<double>(d, 1, 0.03 * (s + 1));

        // decode 单列天然不需要掩码：没有"未来"可看
        mat_t<double> host_keys = to_host_mat(host_cache.keys());
        mat_t<double> host_values = to_host_mat(host_cache.values());
        const mat_t<double> expected = host_attend(q, host_keys, host_values);

        dev_matrix_t<double> dq(d, 1, q);
        dev_matrix_t<double> got = cuda::attend_cached(dq.const_leaf(), dev_cache);
        cuda::sync();

        expect_matrices_match(got.download(), expected, 1e-10,
                              ("第 " + std::to_string(s) + " 步 decode 输出").c_str());
    }
}

TEST_F(CudaKvCacheTest, MaskedPrefillMatchesHost)
{
    const int d = 6;
    const int cap = 12;
    const int q_len = 4;
    const int ctx = 5;

    kv_cache_t<double> host_cache;
    host_cache.reserve(d, cap);
    dev_kv_cache_t<double> dev_cache;
    dev_cache.reserve(d, cap);

    // 先来一段"上下文"，再让 q_len 个 token 一次性做 prefill
    mat_t<double> ctx_k = make_host<double>(d, ctx, 0.2);
    mat_t<double> ctx_v = make_host<double>(d, ctx, -0.1);
    host_cache.append(ctx_k, ctx_v);
    dev_matrix_t<double> dck(d, ctx, ctx_k);
    dev_matrix_t<double> dcv(d, ctx, ctx_v);
    dev_cache.append(dck, dcv);

    mat_t<double> new_k = make_host<double>(d, q_len, 0.35);
    mat_t<double> new_v = make_host<double>(d, q_len, -0.15);
    host_cache.append(new_k, new_v);
    dev_matrix_t<double> dnk(d, q_len, new_k);
    dev_matrix_t<double> dnv(d, q_len, new_v);
    dev_cache.append(dnk, dnv);

    mat_t<double> q = make_host<double>(d, q_len, 0.05);

    // q 的首列绝对位置是 ctx（前 ctx 个时间步已在缓存里）
    mat_t<double> mask = make_causal_mask(q_len, ctx + q_len, ctx);
    mat_t<double> host_keys = to_host_mat(host_cache.keys());
    mat_t<double> host_values = to_host_mat(host_cache.values());
    const mat_t<double> expected = host_attend(q, host_keys, host_values, &mask);

    dev_matrix_t<double> dq(d, q_len, q);
    dev_matrix_t<double> dmask(q_len, ctx + q_len, mask);
    dev_matrix_t<double> got =
        cuda::attend_cached(dq.const_leaf(), dev_cache, dmask.const_leaf());
    cuda::sync();

    expect_matrices_match(got.download(), expected, 1e-10, "prefill（因果掩码）输出");

    // 掩码形状不对必须报错，而不是静默算错
    dev_matrix_t<double> bad_mask(q_len, 7);
    EXPECT_THROW(cuda::attend_cached(dq.const_leaf(), dev_cache, bad_mask.const_leaf()),
                 std::invalid_argument);
}

TEST_F(CudaKvCacheTest, RejectsHeadDimMismatch)
{
    dev_kv_cache_t<double> dev;
    dev.reserve(4, 8);

    mat_t<double> k = make_host<double>(4, 2, 0.5);
    mat_t<double> v = make_host<double>(4, 2, 0.25);
    dev_matrix_t<double> dk(4, 2, k);
    dev_matrix_t<double> dv(4, 2, v);
    dev.append(dk, dv);

    mat_t<double> q_bad = make_host<double>(5, 1, 0.3); // 行数 != head_dim
    dev_matrix_t<double> dq_bad(5, 1, q_bad);
    EXPECT_THROW(cuda::attend_cached(dq_bad.const_leaf(), dev), std::invalid_argument);

    dev_kv_cache_t<double> empty_cache;
    empty_cache.reserve(4, 8);
    mat_t<double> q_ok = make_host<double>(4, 1, 0.3);
    dev_matrix_t<double> dq(4, 1, q_ok);
    EXPECT_THROW(cuda::attend_cached(dq.const_leaf(), empty_cache), std::runtime_error);
}

// ---------------------------------------------------------------------------
// GQA：多 KV 头容器
// ---------------------------------------------------------------------------

TEST_F(CudaKvCacheTest, GqaAppendAllKeepsHeadsAlignedAndMatchesHost)
{
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int d_head = 4;
    const int group = num_heads / num_kv_heads;
    const int d_kv = num_kv_heads * d_head;
    const int cap = 10;

    dev_kv_caches_t<double> caches;
    caches.configure(num_heads, num_kv_heads, d_head, cap);

    EXPECT_EQ(caches.num_heads(), num_heads);
    EXPECT_EQ(caches.num_kv_heads(), num_kv_heads);
    EXPECT_EQ(caches.group_size(), group);
    EXPECT_EQ(caches.head_dim(), d_head);
    EXPECT_EQ(caches.kv_head_of(0), 0);
    EXPECT_EQ(caches.kv_head_of(1), 0);
    EXPECT_EQ(caches.kv_head_of(2), 1);
    EXPECT_EQ(caches.kv_head_of(3), 1);

    // 主机侧的整块 K/V 当参照
    mat_t<double> k_full(d_kv, cap);
    mat_t<double> v_full(d_kv, cap);
    int pos = 0;

    for (const int n_new : {2, 3, 1})
    {
        mat_t<double> k = make_host<double>(d_kv, n_new, 0.1 * (pos + 1));
        mat_t<double> v = make_host<double>(d_kv, n_new, -0.07 * (pos + 1));
        put_cols(k_full, pos, k);
        put_cols(v_full, pos, v);
        pos += n_new;
        const int total = pos;

        dev_matrix_t<double> dk(d_kv, n_new, k);
        dev_matrix_t<double> dv(d_kv, n_new, v);
        caches.append_all(dk, dv);

        // append_all 一次写完所有 KV 头，长度天然对齐 —— 不存在"某个头被写两次"
        EXPECT_EQ(caches.length(), total);
        for (int g = 0; g < num_kv_heads; ++g)
            EXPECT_EQ(caches.cache_of_kv_head(g).length(), total)
                << "KV 头 " << g << " 的长度被写歪了";

        // 各 KV 头里的内容也要对得上（按行区间切片）
        for (int g = 0; g < num_kv_heads; ++g)
        {
            mat_t<double> expected_k = slice_rows(k_full, g * d_head, d_head, total);
            mat_t<double> got_k = pack_to_host(caches.cache_of_kv_head(g).keys());
            expect_matrices_match(got_k, expected_k, 1e-12, "KV 头内容");
        }
    }

    const int total = pos;

    // 每个 Q 头都要和主机端对应 KV 头的注意力一致
    for (int h = 0; h < num_heads; ++h)
    {
        const int g = caches.kv_head_of(h);
        mat_t<double> q = make_host<double>(d_head, 1, 0.05 * (h + 1));

        mat_t<double> k_g = slice_rows(k_full, g * d_head, d_head, total);
        mat_t<double> v_g = slice_rows(v_full, g * d_head, d_head, total);
        const mat_t<double> expected = host_attend(q, k_g, v_g);

        dev_matrix_t<double> dq(d_head, 1, q);
        dev_matrix_t<double> got = caches.attend_q_head(h, dq.const_leaf());
        cuda::sync();

        expect_matrices_match(got.download(), expected, 1e-10,
                              ("Q 头 " + std::to_string(h) + " 的输出").c_str());
    }

    // 共享同一个 KV 头的两个 Q 头，输出必须来自**同一份** K/V ——
    // 这正是主机端那个"每头各自 append"的事故会破坏的不变量
    EXPECT_EQ(caches.kv_head_of(0), caches.kv_head_of(1));
    EXPECT_EQ(caches.cache_of_kv_head(0).length(), total);
}

TEST_F(CudaKvCacheTest, GqaRejectsBadConfiguration)
{
    dev_kv_caches_t<double> caches;

    // num_heads 必须能被 num_kv_heads 整除
    EXPECT_THROW(caches.configure(4, 3, 4, 8), std::invalid_argument);
    EXPECT_THROW(caches.configure(0, 1, 4, 8), std::invalid_argument);
    EXPECT_THROW(caches.configure(4, 2, 0, 8), std::invalid_argument);

    caches.configure(4, 2, 4, 8);
    EXPECT_THROW(caches.kv_head_of(4), std::out_of_range);
    EXPECT_THROW(caches.cache_of_kv_head(2), std::out_of_range);

    // 行数必须是 num_kv_heads * d_head = 8
    dev_matrix_t<double> k_bad(9, 1);
    dev_matrix_t<double> v_bad(9, 1);
    EXPECT_THROW(caches.append_all(k_bad, v_bad), std::runtime_error);

    // K/V 的 token 数要一致
    dev_matrix_t<double> k2(8, 2);
    dev_matrix_t<double> v2(8, 3);
    EXPECT_THROW(caches.append_all(k2, v2), std::runtime_error);

    EXPECT_EQ(caches.length(), 0) << "失败的 append 不应改变长度";
}

TEST_F(CudaKvCacheTest, UnconfiguredContainerRefusesToWork)
{
    dev_kv_caches_t<double> caches;
    EXPECT_THROW(caches.reserve_all(8), std::runtime_error);

    dev_matrix_t<double> k(4, 1);
    dev_matrix_t<double> v(4, 1);
    EXPECT_THROW(caches.append_all(k, v), std::runtime_error);
    EXPECT_EQ(caches.length(), 0);
    EXPECT_TRUE(caches.empty());
}

// ---------------------------------------------------------------------------
// 单精度
// ---------------------------------------------------------------------------

TEST_F(CudaKvCacheTest, FloatDecodeMatchesHost)
{
    const int d = 8;
    const int steps = 6;

    dev_kv_cache_t<float> dev;
    dev.reserve(d, steps);

    mat_t<double> k_acc(d, steps);
    mat_t<double> v_acc(d, steps);

    for (int s = 0; s < steps; ++s)
    {
        mat_t<float> k(d, 1);
        mat_t<float> v(d, 1);
        mat_t<float> q(d, 1);
        for (int i = 0; i < d; ++i)
        {
            k(i, 0) = 0.1f * static_cast<float>((i * 3 + s) % 7) - 0.2f;
            v(i, 0) = 0.05f * static_cast<float>((i * 5 + s) % 11) - 0.1f;
            q(i, 0) = 0.02f * static_cast<float>(i + 1);
        }
        put_cols(k_acc, s, mat_t<double>(k));
        put_cols(v_acc, s, mat_t<double>(v));

        dev_matrix_t<float> dk(d, 1, k);
        dev_matrix_t<float> dv(d, 1, v);
        dev.append(dk, dv);

        dev_matrix_t<float> dq(d, 1, q);
        dev_matrix_t<float> got = cuda::attend_cached(dq.const_leaf(), dev);
        cuda::sync();

        // 主机参考用 double 累加：容差按单精度本身的量级给 1e-5
        mat_t<double> q_host(d, 1);
        for (int i = 0; i < d; ++i)
            q_host(i, 0) = static_cast<double>(q(i, 0));
        mat_t<double> k_view = slice_rows(k_acc, 0, d, s + 1);
        mat_t<double> v_view = slice_rows(v_acc, 0, d, s + 1);
        const mat_t<double> expected = host_attend(q_host, k_view, v_view);

        const mat_t<float> out = got.download();
        ASSERT_EQ(out.row_num(), d);
        ASSERT_EQ(out.col_num(), 1);
        for (int i = 0; i < d; ++i)
        {
            EXPECT_TRUE(std::isfinite(out(i, 0))) << "输出出现非有限值";
            const double scale = std::max(1.0, std::abs(expected(i, 0)));
            EXPECT_NEAR(static_cast<double>(out(i, 0)), expected(i, 0), 1e-5 * scale)
                << "第 " << s << " 步 decode 的第 " << i << " 行";
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM 精度模式：跨机器可复现的前提
// ---------------------------------------------------------------------------

TEST_F(CudaKvCacheTest, GemmMathModeDefaultsToPrecise)
{
    EXPECT_EQ(cuda::gemm_math_mode(), cuda::gemm_math::precise)
        << "默认必须是真 FP32，否则 float 结果会随机器（P4 vs Ampere）漂移";

    cuda::set_gemm_math(cuda::gemm_math::precise);
    EXPECT_EQ(cuda::gemm_math_mode(), cuda::gemm_math::precise);

    if (cuda::tf32_supported())
    {
        cuda::set_gemm_math(cuda::gemm_math::tf32);
        EXPECT_EQ(cuda::gemm_math_mode(), cuda::gemm_math::tf32);
        cuda::set_gemm_math(cuda::gemm_math::precise);
    }
    else
    {
        // P4（sm_61）没有 TF32 张量核：必须**报错**而不是静默降级，
        // 否则会在测试机上"验证过"一个在目标机上根本没生效的加速
        EXPECT_THROW(cuda::set_gemm_math(cuda::gemm_math::tf32), std::runtime_error);
        EXPECT_EQ(cuda::gemm_math_mode(), cuda::gemm_math::precise);
    }
}
