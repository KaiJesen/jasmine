#ifndef __JAS_CUDA_KV_CACHE_HPP__
#define __JAS_CUDA_KV_CACHE_HPP__

/**
 * 设备端 KV cache：多轮 decode 的 attention 状态。
 *
 * ## 与主机端 kv_cache_t 的关系
 *
 * 职责切分完全一致 —— K 必须**已经做过 RoPE** 再交进来（主机端 `kv_cache_t::append`
 * 的注释也是这么写的：「调用方保证已做 RoPE」）。所以设备端 RoPE 是**另一件事**，
 * 不混在这里。
 *
 * 差别在两处：
 *
 *  1. **零拷贝视图**：主机端 `keys()` 走 `mat_t::view(0, 0, d, len)`，设备端靠
 *     `dev_mat_t` 新增的独立前导维（`m_ld`）做同一件事：缓冲区按 `cap` 分配，
 *     只对外暴露前 `len` 列，**不做任何拷贝**。这正是 KV cache 存在的意义所在
 *     —— 每个 decode step 都把已用部分拷成紧凑缓冲的话，等于把 attention 的访存
 *     瓶颈换成拷贝瓶颈。
 *
 *  2. **扩容是翻倍而不是固定步长**：主机端 `grow_to` 是 `m_cap + 64`，
 *     累计到 cap 要拷贝 O(cap²) 个元素。设备上这个拷贝要走显存带宽，改成翻倍后
 *     摊销 O(cap)。（小模型主机上无所谓，设备端值得改。）
 *
 * ## GQA：把「重复 append」从注释变成写不出来的代码
 *
 * 主机端 `jas_mha_t.hpp` 里有一段重要的事故记录：
 *
 *   GQA 的 KV cache 必须「每个 KV 头 append 一次」。共享同一个 KV 头的多个 Q 头
 *   如果各自 append，同一份 K/V 会被写入两次：形状不报错、单步看似可用，
 *   但 cache 长度与内容全错，到多轮对话才炸。
 *
 * 这里不靠注释，而是让 API 形状本身排除这种写法：多 KV 头容器**只提供
 * `append_all()`** —— 一次给**所有** KV 头追加同一批 token。压根没有
 * 「按单个 KV 头append」的入口，也就没有「同一个头 append 两次」的可能。
 * 按 KV 头的访问只有 `const` 的只读形式，给 Q 头读。
 *
 * ## 前置条件
 *
 * 使用前必须 `reserve()`（或依赖 dynamic 模式的自动扩容）。**扩容会重新分配缓冲区，
 * 先前取出的 `keys()` / `values()` 叶子随即悬空** —— 和 `dev_matrix_t::leaf()` 一样，
 * 薄壳不是稳定的。decode 循环里请先 reserve 好容量。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_kv_cache_t.hpp" // 复用主机端的 kv_cache_mode，一个概念一个名字
#include "jas_mat_express_t.hpp"

namespace jasmine {
namespace cuda {

namespace detail {

/**
 * KV cache 的 K/V 只接受**叶子**或**拥有者**，不接受任意表达式。
 *
 * 不是偷懒：append 的形状契约是 (d × n_new)，而表达式可以是任意形状的中间结果，
 * 放进来只会把「形状对不对」这件事推后到更难定位的地方。需要先把表达式算出来的话，
 * 显式 `matmul(...)` 或 `eval_fused` 落成一个 `dev_matrix_t` 再 append，意图更清楚。
 */
template <typename T, typename X>
dev_mat_t<T> read_leaf_of(const X& x)
{
    using XT = std::remove_cvref_t<X>;
    if constexpr (std::is_same_v<XT, dev_mat_t<T>>)
    {
        return x;
    }
    else if constexpr (is_dev_matrix_t<XT>::value)
    {
        return x.const_leaf();
    }
    else
    {
        static_assert(!std::is_same_v<XT, XT>,
                      "KV cache 的 K/V 只能是 dev_mat_t 或 dev_matrix_t；"
                      "表达式请先物化成 dev_matrix_t");
    }
}

/**
 * 把 src 的第 [src_row0, src_row0+rows) 行、第 0..src.col_num()-1 列，
 * 写到 dst 的 [dst_col0, ...) 列（行优先，dst 前导维 dst_ld）。
 *
 * 一个网格同时铺开「行」和「列」两个方向，所以一批 token 只有一次 launch。
 * 用 `src(r, c)` 而不是裸指针，是为了让转置视图也能正确工作（数组下标落在 src_row0 之后）。
 */
template <typename T>
__global__ void append_columns_kernel(dev_mat_t<T> src, int src_row0, int rows, T* dst, int dst_ld,
                                      int dst_col0)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= rows)
        return;
    dst[static_cast<std::ptrdiff_t>(i) * dst_ld + dst_col0 + j] = src(src_row0 + i, j);
}

/** 扩容时把已用的 n_cols 列按新的前导维搬过去。两边的步长不同，所以是二维拷贝。 */
template <typename T>
void copy_columns(const dev_buf_t<T>& src, int src_ld, dev_buf_t<T>& dst, int dst_ld, int rows,
                  int n_cols)
{
    if (n_cols <= 0 || rows <= 0)
        return;
    JAS_CUDA_CHECK(cudaMemcpy2D(dst.data(), static_cast<std::size_t>(dst_ld) * sizeof(T),
                                src.data(), static_cast<std::size_t>(src_ld) * sizeof(T),
                                static_cast<std::size_t>(n_cols) * sizeof(T),
                                static_cast<std::size_t>(rows), cudaMemcpyDeviceToDevice));
}

} // namespace detail

// 前向声明：单头 cache 需要把它声明为友元，好让多 KV 头容器按头切开写入。
template <typename T>
class dev_kv_caches_t;

/**
 * 单头 KV cache：K/V 均为 (d × cap) 行优先，列 = 时间步。
 *
 * 只做「存」和「零拷贝地取出来」，不做 attention —— attention 见文件末尾的
 * `attend_cached`，它只依赖 `keys()` / `values()`。
 */
template <typename T>
class dev_kv_cache_t
{
public:
    dev_kv_cache_t() = default;

    void set_mode(kv_cache_mode mode) { m_mode = mode; }
    kv_cache_mode mode() const { return m_mode; }

    int dim() const { return m_d; }
    int length() const { return m_len; }
    int capacity() const { return m_cap; }
    bool empty() const { return m_len == 0; }

    /**
     * 预留 [d × max_seq]。长度清零，与主机端 `kv_cache_t::reserve` 一致。
     *
     * decode 前调一次就够：之后 append 不再分配，取出的叶子也就一直稳定。
     */
    void reserve(int d, int max_seq)
    {
        if (d <= 0 || max_seq <= 0)
            throw std::invalid_argument("dev_kv_cache reserve: 尺寸必须为正");
        m_d = d;
        m_cap = max_seq;
        m_len = 0;
        const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(max_seq);
        m_k.allocate(n);
        m_v.allocate(n);
    }

    /** 逻辑长度清零，**内存保留**（下一轮对话复用同一块缓冲区）。 */
    void clear() { m_len = 0; }

    /**
     * 追加一批新 token 的 K/V。
     *
     * `k` / `v` 形状均为 (d × n_new) 的逻辑行列（转置视图按逻辑坐标解释）。
     * **调用方必须已经给 K 做过 RoPE** —— 与主机端同一契约。
     */
    template <typename K, typename V>
    void append(const K& k, const V& v)
    {
        const dev_mat_t<T> kl = detail::read_leaf_of<T>(k);
        const dev_mat_t<T> vl = detail::read_leaf_of<T>(v);
        append_leaf(kl, vl, 0, kl.row_num());
    }

    /**
     * 零拷贝只读视图：形状 (d × len)，**存储前导维仍是 cap**。
     *
     * 这就是 `dev_mat_t` 要把逻辑列数与前导维分开的原因：返回的叶子直接指向
     * cache 的缓冲区，`len` 涨了只需再取一次，不需要搬数据。
     *
     * 语义上只读（`dev_mat_t` 的指针是可写的，因为 GEMM 的输出要写它）；
     * 请只把它喂给 GEMM / 表达式这类只读用法。
     */
    dev_mat_t<T> keys() const
    {
        if (m_len == 0)
            throw std::runtime_error("dev_kv_cache keys: 缓存为空");
        return dev_mat_t<T>(const_cast<T*>(m_k.data()), m_d, m_len, m_cap, false);
    }

    dev_mat_t<T> values() const
    {
        if (m_len == 0)
            throw std::runtime_error("dev_kv_cache values: 缓存为空");
        return dev_mat_t<T>(const_cast<T*>(m_v.data()), m_d, m_len, m_cap, false);
    }

private:
    friend class dev_kv_caches_t<T>;

    static constexpr int kMinGrow = 64;

    /**
     * 从 k/v 的第 src_row0 行起取 rows 行追加进去。
     * 多 KV 头容器就是靠这个参数把一整块 (n_kv_heads × d_head, n_new) 按头切开的。
     */
    void append_leaf(const dev_mat_t<T>& k, const dev_mat_t<T>& v, int src_row0, int rows)
    {
        if (k.row_num() * k.col_num() != v.row_num() * v.col_num())
            throw std::runtime_error("dev_kv_cache append: K/V 形状不一致");
        if (rows <= 0 || k.col_num() <= 0)
            return;
        if (!k.valid() || !v.valid())
            throw std::invalid_argument("dev_kv_cache append: 有操作数还是空叶子");

        if (m_d == 0)
            reserve(rows, std::max(k.col_num(), kMinGrow));
        if (rows != m_d)
            throw std::runtime_error("dev_kv_cache append: 维度不匹配，cache 是 "
                                     + std::to_string(m_d) + " 行，append 进来 "
                                     + std::to_string(rows) + " 行");

        const int n_new = k.col_num();
        grow_to(m_len + n_new);

        constexpr int kThreads = 256;
        const dim3 grid((rows + kThreads - 1) / kThreads, n_new);
        detail::append_columns_kernel<T><<<grid, kThreads>>>(k, src_row0, rows, m_k.data(), m_cap,
                                                             m_len);
        detail::append_columns_kernel<T><<<grid, kThreads>>>(v, src_row0, rows, m_v.data(), m_cap,
                                                             m_len);
        JAS_CUDA_CHECK(cudaGetLastError());
        m_len += n_new;
    }

    void grow_to(int need)
    {
        if (need <= m_cap)
            return;
        if (m_mode == kv_cache_mode::static_fixed)
            throw std::runtime_error(
                "dev_kv_cache: static_fixed 容量不足；请用更大的 max_seq 调 reserve()");

        // 翻倍，而不是主机端那样 +64：累计拷贝从 O(cap²) 降到摊销 O(cap)。
        // 一次翻倍也顺手覆盖了 prefill（n_new 直接大于当前 cap）的情况。
        const int new_cap = std::max(need, std::max(m_cap * 2, kMinGrow));
        const int old_cap = m_cap;

        dev_buf_t<T> nk(static_cast<std::size_t>(m_d) * static_cast<std::size_t>(new_cap));
        dev_buf_t<T> nv(static_cast<std::size_t>(m_d) * static_cast<std::size_t>(new_cap));
        if (m_len > 0)
        {
            detail::copy_columns(m_k, old_cap, nk, new_cap, m_d, m_len);
            detail::copy_columns(m_v, old_cap, nv, new_cap, m_d, m_len);
        }
        m_k = std::move(nk);
        m_v = std::move(nv);
        m_cap = new_cap;
    }

    dev_buf_t<T> m_k;
    dev_buf_t<T> m_v;
    int m_d = 0;
    int m_len = 0;
    int m_cap = 0;
    kv_cache_mode m_mode = kv_cache_mode::dynamic;
};

// ---------------------------------------------------------------------------
// 多 KV 头容器
// ---------------------------------------------------------------------------

/**
 * 一整层 attention 的 KV cache：`num_kv_heads` 份单头 cache + GQA 的 Q→KV 头映射。
 *
 * **唯一的写入入口是 `append_all()`**，它一次给所有 KV 头追加同一批 token。
 * 这是刻意设计的：主机端那种「每个 Q 头各自 append 一次」的写法在这里根本表达不出来，
 * 于是 GQA 重复写入那个只在多轮对话后才暴露的 bug 从根上不存在。
 * 按 KV 头的访问只给 `const` 只读形式（`cache_of_kv_head`），供 Q 头读。
 */
template <typename T>
class dev_kv_caches_t
{
public:
    /**
     * 配置维度。`num_heads`（Q 头数）与 `num_kv_heads` 决定 group_size：
     * 连续 `group_size` 个 Q 头共享一个 KV 头。
     * 约束：`num_heads % num_kv_heads == 0`（GQA/MQA，num_kv_heads == num_heads 即经典 MHA）。
     */
    void configure(int num_heads, int num_kv_heads, int d_head, int max_seq = 0)
    {
        if (num_heads <= 0 || num_kv_heads <= 0 || d_head <= 0)
            throw std::invalid_argument("dev_kv_caches configure: 尺寸必须为正");
        if (num_heads % num_kv_heads != 0)
            throw std::invalid_argument("dev_kv_caches configure: num_heads("
                                        + std::to_string(num_heads) + ") 必须能被 num_kv_heads("
                                        + std::to_string(num_kv_heads) + ") 整除");

        m_num_heads = num_heads;
        m_num_kv_heads = num_kv_heads;
        m_d_head = d_head;
        m_group_size = num_heads / num_kv_heads;
        // 用 resize 而不是 assign(fill)：dev_kv_cache_t 持有 dev_buf_t，只可移动、不可拷贝
        m_caches.clear();
        m_caches.resize(static_cast<std::size_t>(num_kv_heads));
        for (auto& c : m_caches)
            c.set_mode(m_mode);
        if (max_seq > 0)
            reserve_all(max_seq);
    }

    /** 给每个 KV 头预留 [d_head × max_seq]。decode 前调一次，之后 append 不再分配。 */
    void reserve_all(int max_seq)
    {
        ensure_configured();
        for (auto& c : m_caches)
            c.reserve(m_d_head, max_seq);
    }

    void clear_all()
    {
        for (auto& c : m_caches)
            c.clear();
    }

    void set_mode(kv_cache_mode mode)
    {
        m_mode = mode;
        for (auto& c : m_caches)
            c.set_mode(mode);
    }

    int num_heads() const { return m_num_heads; }
    int num_kv_heads() const { return m_num_kv_heads; }
    int group_size() const { return m_group_size; }
    int head_dim() const { return m_d_head; }
    bool empty() const { return m_caches.empty() || m_caches.front().empty(); }

    /** 各 KV 头长度**必然一致**（append_all 保证），所以取第一个即可。 */
    int length() const { return m_caches.empty() ? 0 : m_caches.front().length(); }

    /** Q 头 → 它该读哪个 KV 头（连续 group_size 个 Q 头共享一个）。 */
    int kv_head_of(int q_head) const
    {
        if (q_head < 0 || q_head >= m_num_heads)
            throw std::out_of_range("dev_kv_caches: Q 头下标 " + std::to_string(q_head)
                                    + " 越界");
        return q_head / m_group_size;
    }

    /**
     * 追加一批新 token 到**所有** KV 头。
     *
     * `k_full` / `v_full` 形状为 (num_kv_heads × d_head, n_new)，第 g 个 KV 头取
     * 行区间 `[g*d_head, (g+1)*d_head)`。K 必须已经做过 RoPE。
     *
     * 这是容器唯一的写入入口 —— 没有「按单个 KV 头 append」的版本，
     * 所以 GQA 下重复写入同一份 K/V 是不存在的失败模式。
     */
    template <typename K, typename V>
    void append_all(const K& k_full, const V& v_full)
    {
        ensure_configured();
        const dev_mat_t<T> kl = detail::read_leaf_of<T>(k_full);
        const dev_mat_t<T> vl = detail::read_leaf_of<T>(v_full);

        const int expect_rows = m_num_kv_heads * m_d_head;
        if (kl.row_num() != expect_rows || vl.row_num() != expect_rows)
            throw std::runtime_error(
                "dev_kv_caches append_all: 行数应为 num_kv_heads * d_head = "
                + std::to_string(expect_rows) + "，实际 K=" + std::to_string(kl.row_num())
                + " V=" + std::to_string(vl.row_num()));
        if (kl.col_num() != vl.col_num())
            throw std::runtime_error("dev_kv_caches append_all: K/V 的 token 数不一致");
        if (kl.col_num() <= 0)
            return;

        for (int g = 0; g < m_num_kv_heads; ++g)
            m_caches[static_cast<std::size_t>(g)].append_leaf(kl, vl, g * m_d_head, m_d_head);
    }

    /** 只读访问：给 Q 头读它共享的那份 cache。写入请走 append_all。 */
    const dev_kv_cache_t<T>& cache_of_kv_head(int kv_head) const
    {
        if (kv_head < 0 || kv_head >= m_num_kv_heads)
            throw std::out_of_range("dev_kv_caches: KV 头下标 " + std::to_string(kv_head)
                                    + " 越界");
        return m_caches[static_cast<std::size_t>(kv_head)];
    }

    /** Q 头 `q_head` 的 attention：内部完成 Q→KV 头映射，调用方不必自己算分组。 */
    dev_matrix_t<T> attend_q_head(int q_head, const dev_mat_t<T>& q) const
    {
        return attend_cached(q, cache_of_kv_head(kv_head_of(q_head)));
    }

    dev_matrix_t<T> attend_q_head(int q_head, const dev_mat_t<T>& q, const dev_mat_t<T>& mask) const
    {
        return attend_cached(q, cache_of_kv_head(kv_head_of(q_head)), mask);
    }

private:
    void ensure_configured() const
    {
        if (m_caches.empty())
            throw std::runtime_error("dev_kv_caches: 还没 configure()");
    }

    std::vector<dev_kv_cache_t<T>> m_caches;
    int m_num_heads = 0;
    int m_num_kv_heads = 0;
    int m_d_head = 0;
    int m_group_size = 0;
    kv_cache_mode m_mode = kv_cache_mode::dynamic;
};

// ---------------------------------------------------------------------------
// attention：只看 keys()/values()，与「怎么存的」解耦
// ---------------------------------------------------------------------------

/**
 * 单头 attention：`q`（d_head × q_len）对 cache 里全部 len 个时间步做加权求和。
 *
 *   scores  = qᵀ · K / sqrt(d_head)
 *   weights = softmax_rows(scores)
 *   out     = V · weightsᵀ           → (d_head × q_len)
 *
 * 1/sqrt(d_head) 直接折进 GEMM 的 `alpha`，省掉一趟逐元素缩放。
 *
 * 不传掩码的版本对 decode 是**天然正确**的：q_len == 1 时不存在"未来"，
 * 所以单 token 解码不需要任何掩码。
 */
template <typename T>
dev_matrix_t<T> attend_cached(const dev_mat_t<T>& q, const dev_kv_cache_t<T>& cache)
{
    using ::jasmine::detail::device_sqrt;

    if (cache.length() == 0)
        throw std::runtime_error("attend_cached: KV cache 为空");
    if (q.row_num() != cache.dim())
        throw std::invalid_argument("attend_cached: q 的行数 " + std::to_string(q.row_num())
                                    + " 与 cache 的 head_dim " + std::to_string(cache.dim())
                                    + " 不一致");

    const T scale = T(1) / device_sqrt(static_cast<T>(cache.dim()));
    dev_matrix_t<T> scores = matmul(q.t(), cache.keys(), scale);
    dev_matrix_t<T> weights = softmax_rows(scores.leaf());
    return matmul(cache.values(), weights.leaf().t());
}

/**
 * 带掩码的版本，给 prefill（q_len > 1）用。
 *
 * `mask` 形状为 (q_len × len)，合法位置填 0、屏蔽位置填 `-inf`
 * （与主机端 causal mask 同一约定）。掩码由调用方构造 —— 因为它的语义
 * 取决于 q 的首列绝对位置（`q_pos`），而这属于调用方的上下文。
 */
template <typename T>
dev_matrix_t<T> attend_cached(const dev_mat_t<T>& q, const dev_kv_cache_t<T>& cache,
                              const dev_mat_t<T>& mask)
{
    using ::jasmine::detail::device_sqrt;

    if (cache.length() == 0)
        throw std::runtime_error("attend_cached: KV cache 为空");
    if (q.row_num() != cache.dim())
        throw std::invalid_argument("attend_cached: q 的行数 " + std::to_string(q.row_num())
                                    + " 与 cache 的 head_dim " + std::to_string(cache.dim())
                                    + " 不一致");
    if (mask.row_num() != q.col_num() || mask.col_num() != cache.length())
        throw std::invalid_argument("attend_cached: mask 形状应为 (q_len, len) = ("
                                    + std::to_string(q.col_num()) + ", "
                                    + std::to_string(cache.length()) + ")");

    const T scale = T(1) / device_sqrt(static_cast<T>(cache.dim()));
    dev_matrix_t<T> scores = matmul(q.t(), cache.keys(), scale);
    dev_matrix_t<T> weights = softmax_rows(scores.leaf() + mask);
    return matmul(cache.values(), weights.leaf().t());
}

} // namespace cuda
} // namespace jasmine

#endif
