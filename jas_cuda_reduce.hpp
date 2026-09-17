#ifndef __JAS_CUDA_REDUCE_HPP__
#define __JAS_CUDA_REDUCE_HPP__

/**
 * 设备端归约，以及建立在归约之上的 softmax / LayerNorm / RMSNorm。
 *
 * ## 为什么归约是必须单独做的（而不是靠融合 kernel 硬算）
 *
 * 融合逐元素 kernel 的模型是"一个线程算一个输出元素"，它要求每个输出只依赖**同位置**的输入。
 * 归约不满足这个条件：它要把一整行/一整列汇总成一个数，必须跨线程协作。
 * 所以归约走**块内归约**（warp shuffle + 共享内存），与表达式融合正交地组合：
 * 归约 kernel 读的是**整棵表达式树**，于是 `hsum(exp(x - m))` 里 exp 被融进了归约那一趟，
 * 指数结果一个都不物化。
 *
 * ## 轴的方向（与主机端严格对齐）
 *
 * 这套代码库的约定是 **特征维在行方向、序列/批次在列方向**，所以两个轴都有用武之地：
 *
 *   - `hsum` / `hmax`：**逐行**归约，结果 (rows × 1)。注意力 softmax 走这条
 *     （`attn_scores` 是 (seq × seq)，softmax 沿 j = key 方向）。
 *   - `vsum` / `vmax`：**逐列**归约，结果 (1 × cols)。LayerNorm / RMSNorm 走这条
 *     （输入 (d_model × T)，沿 d_model 求统计量）。
 *
 * 命名沿用主机端的 `h*` / `v*`（见 jas_mat_express_t.hpp 的 hsum / vsum），
 * 这样两边的语义不会产生歧义。
 *
 * ## 广播叶子
 *
 * 归约结果是"一行"或"一列"，要参与 (rows × cols) 的逐元素运算必须能广播。
 * 主机端靠 `mat_t::operator()` 里的取模实现广播；设备端**刻意不用取模**：
 * 那会让每个线程重算一遍模运算，还会挡住合并访存分析。
 * 取而代之的是两种专用叶子，把"忽略哪个下标"变成类型信息：
 *
 *   - `dev_col_leaf_t`：(rows × 1)，`operator()(r, c)` 忽略 `c` → 沿列广播
 *   - `dev_row_leaf_t`：(1 × cols)，`operator()(r, c)` 忽略 `r` → 沿行广播
 *
 * 这不只是性能考量。若拿一个 (rows × 1) 的普通 `dev_mat_t` 去参与 (rows × cols) 表达式，
 * `operator()(r, c)` 会按 `data[r * 1 + c]` 寻址，c > 0 时直接**越界读**。
 * 用忽略下标、无越界可能的专用类型，这个错误在类型层面就写不出来了。
 */

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_mat_express_t.hpp"

namespace jasmine {

/**
 * (rows × 1) 的列向量叶子，沿列广播。
 *
 * 存的是 `const T*`：它是纯粹的只读广播句柄，没有任何人会从它写回
 * （GEMM 的输出叶子另走 `dev_mat_t`，那里才需要可写指针）。
 * 用 const 指针顺带让"从 const 的拥有者取叶子"变得自然。
 */
template <typename T>
struct dev_col_leaf_t
{
    using ele_type = T;

    const T* m_data = nullptr;
    int m_rows = 0;

    static constexpr bool device_evaluable = true;

    dev_col_leaf_t() = default;
    JAS_HD dev_col_leaf_t(const T* data, int rows) : m_data(data), m_rows(rows) {}

    JAS_HD int row_num() const { return m_rows; }
    JAS_HD int col_num() const { return 1; }

    /** 忽略列下标 —— 这就是"广播"。没有取模，也没有越界可能。 */
    JAS_HD T operator()(int r, int) const { return m_data[r]; }
};

/** (1 × cols) 的行向量叶子，沿行广播。 */
template <typename T>
struct dev_row_leaf_t
{
    using ele_type = T;

    const T* m_data = nullptr;
    int m_cols = 0;

    static constexpr bool device_evaluable = true;

    dev_row_leaf_t() = default;
    JAS_HD dev_row_leaf_t(const T* data, int cols) : m_data(data), m_cols(cols) {}

    JAS_HD int row_num() const { return 1; }
    JAS_HD int col_num() const { return m_cols; }

    JAS_HD T operator()(int, int c) const { return m_data[c]; }
};

// 设备叶子按值拥有；这两种广播叶子同理（见 jas_cuda_leaf.hpp 的说明）
template <typename T>
inline constexpr bool operand_owned_by_value<dev_col_leaf_t<T>> = true;
template <typename T>
inline constexpr bool operand_owned_by_value<dev_row_leaf_t<T>> = true;

namespace cuda {

/** 拥有 (rows × 1) 缓冲区的列向量。归约结果的宿主，负责内存生死。 */
template <typename T>
class dev_colvec_t
{
public:
    dev_colvec_t() = default;
    explicit dev_colvec_t(int rows) : m_buf(rows > 0 ? static_cast<std::size_t>(rows) : 0), m_rows(rows) {}

    /** 广播叶子：拿去参与 (rows × cols) 的表达式。 */
    dev_col_leaf_t<T> leaf() const { return dev_col_leaf_t<T>(m_buf.data(), m_rows); }

    /**
     * (rows × 1) 的**普通**薄壳，不做广播。
     *
     * 更新器这类逐元素接口要的是「第 i 个元素对第 i 个元素」，
     * 广播叶子在那里是错的（它把行下标当成了列下标来用）。
     *
     * 与 `dev_matrix_t::const_leaf()` 同一个套路：`dev_mat_t` 天生带可写指针
     * （GEMM 的输出要写它），所以从 const 对象里取不出它。这里用一次显式
     * `const_cast` 把「取一个叶子」这件事表达出来；**拿到它只应喂给逐元素接口**，
     * 真正的写路径走 `buffer()`。
     */
    dev_mat_t<T> flat_leaf() const
    {
        return dev_mat_t<T>(const_cast<T*>(m_buf.data()), m_rows, 1);
    }

    void upload(const T* host, std::size_t n) { m_buf.upload(host, n); }

    /** 回读成 (rows × 1) 的主机矩阵，便于和主机端结果对拍。 */
    mat_t<T> download() const
    {
        mat_t<T> host(m_rows, 1);
        if (!m_buf.empty())
            m_buf.download(host.data(), m_buf.size());
        return host;
    }

    dev_buf_t<T>& buffer() { return m_buf; }
    const dev_buf_t<T>& buffer() const { return m_buf; }

    int size() const { return m_rows; }
    bool empty() const { return m_rows == 0; }

private:
    dev_buf_t<T> m_buf;
    int m_rows = 0;
};

/** 拥有 (1 × cols) 缓冲区的行向量。 */
template <typename T>
class dev_rowvec_t
{
public:
    dev_rowvec_t() = default;
    explicit dev_rowvec_t(int cols) : m_buf(cols > 0 ? static_cast<std::size_t>(cols) : 0), m_cols(cols) {}

    dev_row_leaf_t<T> leaf() const { return dev_row_leaf_t<T>(m_buf.data(), m_cols); }

    /** (1 × cols) 的**普通**薄壳，不做广播；理由同 `dev_colvec_t::flat_leaf`。 */
    dev_mat_t<T> flat_leaf() const
    {
        return dev_mat_t<T>(const_cast<T*>(m_buf.data()), 1, m_cols);
    }

    void upload(const T* host, std::size_t n) { m_buf.upload(host, n); }

    mat_t<T> download() const
    {
        mat_t<T> host(1, m_cols);
        if (!m_buf.empty())
            m_buf.download(host.data(), m_buf.size());
        return host;
    }

    dev_buf_t<T>& buffer() { return m_buf; }
    const dev_buf_t<T>& buffer() const { return m_buf; }

    int size() const { return m_cols; }
    bool empty() const { return m_cols == 0; }

private:
    dev_buf_t<T> m_buf;
    int m_cols = 0;
};

namespace detail {

/** 归约用的加法与取大。都是无状态仿函数，可直接按值传给 kernel。 */
struct add_op
{
    template <typename T>
    JAS_HD T operator()(T a, T b) const { return a + b; }
};

struct max_op
{
    template <typename T>
    JAS_HD T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename T>
T identity_add() { return T(0); }

/**
 * 取大的单位元用 `lowest()`，与主机端 `max()` 的初值完全一致。
 *
 * 注意它对 `-inf` 的行为：`op(lowest, -inf)` 返回 `lowest`（因为 `lowest > -inf`），
 * 也就是说"整行全是 -inf"时会得到 `lowest` 而不是 `-inf`。
 * 这与主机端逐字相同（主机也是 `ret = lowest(); if (v > ret) ret = v;`），
 * 所以两边一致；而因果掩码下每行至少有一个合法位置，这种情况实际不可达。
 */
template <typename T>
T identity_max() { return std::numeric_limits<T>::lowest(); }

/**
 * 块内归约：先 warp shuffle，再跨 warp 走共享内存。
 * 只有 `threadIdx.x == 0` 的返回值有意义。
 *
 * `smem` 由调用方声明并传入（至少 32 个元素）。不在函数内部声明 `__shared__`
 * 是因为该函数可能被同一个 kernel 调用多次，函数内静态共享数组会互相踩。
 *
 * 调用方必须保证 `blockDim.x` 是 32 的整数倍：shuffle 用的是全掩码
 * `0xffffffff`，不满一个 warp 时掩码与实际活跃线程不符。启动配置里已强制。
 */
template <typename T, typename Op>
JAS_DEV T block_reduce(T acc, T init, Op op, T* smem)
{
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc = op(acc, __shfl_down_sync(0xffffffffu, acc, off));

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0)
        smem[warp] = acc;
    __syncthreads();

    if (warp == 0)
    {
        const int nwarps = (blockDim.x + 31) >> 5;
        acc = (lane < nwarps) ? smem[lane] : init;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            acc = op(acc, __shfl_down_sync(0xffffffffu, acc, off));
    }
    return acc;
}

/**
 * 块内归约的**广播**版本：返回值在块内**每个**线程上都有效。
 *
 * `block_reduce` 只保证 `threadIdx.x == 0` 的返回值有意义（最终结果留在 warp 0 的
 * lane 0），而单趟 softmax 里每个线程都要用行最大值和指数和来算自己那几个元素，
 * 所以必须再经一次共享内存广播出去。
 *
 * 结尾那次 `__syncthreads()` 不是多余的：广播写完 `smem[0]` 并同步之后，
 * 各线程**读** `smem[0]` 的动作并不受那次同步保护 —— 快的线程可能已经跑到下一轮
 * 归约、把 `smem[0]` 覆盖掉了（`block_reduce` 第一件事就是 `smem[warp] = acc`，
 * 其中 warp 0 写的正是 `smem[0]`），而慢的线程还没读。多这一次同步才真正安全。
 * 同一个 kernel 里连续做 max 与 sum 两次归约时，这条是必需的。
 */
template <typename T, typename Op>
JAS_DEV T block_reduce_all(T acc, T init, Op op, T* smem)
{
    acc = block_reduce(acc, init, op, smem);
    if (threadIdx.x == 0)
        smem[0] = acc;
    __syncthreads();
    const T result = smem[0];
    __syncthreads();
    return result;
}

/**
 * 单趟逐行 softmax：整行留在共享内存里，max / exp+求和 / 归一化全在片上做。
 *
 * 与三趟版本相比（三趟 = `row_max` → `row_sum(exp(...))` → 归一化，各自一次全局读）：
 *
 *   - 全局访存：读 1 遍 + 写 1 遍（三趟是读 3 遍 + 写 1 遍）；
 *   - `exp` 次数：每个元素 **1 次**（三趟是 2 次，第 2、3 趟各算一遍）；
 *   - 表达式树只求值一遍。
 *
 * 代价是整行必须放得进共享内存。放不下时由调用方回退到三趟版本 ——
 * 那版没有行长上限，两者结果按相对容差一致。
 *
 * 数值结构**刻意与三趟版本保持一致**（先减最大值再求指数和，逐线程跨步累加后树形归约），
 * 所以两条路径的结果不会因为实现不同而分叉。
 */
template <typename Expr, typename T>
__global__ void softmax_rows_shared_kernel(Expr expr, T* __restrict__ out, int cols, T max_init,
                                           T sum_init)
{
    const int row = blockIdx.x;
    extern __shared__ char smem_raw[];
    T* row_cache = reinterpret_cast<T*>(smem_raw);
    T* scratch = row_cache + cols; // block_reduce 的跨 warp 暂存，至少 32 个元素

    // 表达式树只求值这一遍；后面全在片上读
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        row_cache[j] = static_cast<T>(expr(row, j));
    // 下面每个线程要读别人写的 row_cache，必须先同步
    __syncthreads();

    // 单位元由主机算好传进来（同 row_reduce_kernel 的做法）：
    // std::numeric_limits 是 constexpr 主机函数，在设备端直接用会告警
    T local_max = max_init;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_max = ::jasmine::detail::device_max(local_max, row_cache[j]);
    const T m = block_reduce_all(local_max, max_init, max_op{}, scratch);

    // exp 结果直接写回 row_cache：省掉第二次 exp，也省掉一遍全局读
    T local_sum = sum_init;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        const T e = ::jasmine::detail::device_exp(row_cache[j] - m);
        row_cache[j] = e;
        local_sum += e;
    }
    const T s = block_reduce_all(local_sum, sum_init, add_op{}, scratch);

    const T inv = T(1) / s;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        out[static_cast<std::ptrdiff_t>(row) * cols + j] = row_cache[j] * inv;
}

/** 逐行归约：一个 block 负责一行，块内并行扫过该行的所有列。 */
template <typename Expr, typename T, typename Op>
__global__ void row_reduce_kernel(Expr expr, T* out, T init, Op op)
{
    const int row = blockIdx.x;
    const int cols = expr.col_num();

    T acc = init;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        acc = op(acc, static_cast<T>(expr(row, j)));

    __shared__ T smem[32];
    acc = block_reduce(acc, init, op, smem);

    if (threadIdx.x == 0)
        out[row] = acc;
}

/**
 * 逐列归约：一个 block 负责 `BlockCols` 列，块内用 (行分块 × 列) 的二维映射。
 *
 * 二维映射是为了**访存合并**：想让连续线程读到连续地址，读的就必须是同一行里的相邻列，
 * 所以线程的 x 维对应列、y 维对应行。每个线程只扫自己那一列的一部分行，
 * 最后再跨 y 把部分和加起来。
 *
 * 若改成"一个线程负责一整列"，连续线程就会跨行访问（步长 = cols），完全无法合并。
 */
template <typename Expr, typename T, typename Op, int BlockRows, int BlockCols>
__global__ void col_reduce_kernel(Expr expr, T* out, int rows, T init, Op op)
{
    const int col = blockIdx.x * BlockCols + threadIdx.x;
    const int ty = threadIdx.y;
    const int cols = expr.col_num();

    T acc = init;
    if (col < cols)
    {
        for (int i = ty; i < rows; i += BlockRows)
            acc = op(acc, static_cast<T>(expr(i, col)));
    }

    __shared__ T smem[BlockRows][BlockCols];
    smem[ty][threadIdx.x] = acc;
    // 所有线程都必须到这里：越界线程写的是单位元，不影响结果
    __syncthreads();

    // 由 y = 0 那一行把 BlockRows 个部分和串起来。BlockRows 很小（默认 8），
    // 这点串行开销远小于多起一轮块内归约的成本。
    if (ty == 0 && col < cols)
    {
        T total = smem[0][threadIdx.x];
        for (int r = 1; r < BlockRows; ++r)
            total = op(total, smem[r][threadIdx.x]);
        out[col] = total;
    }
}

/** 全归约第一阶段：每个 block 算一段连续区间的部分和。 */
template <typename Expr, typename T, typename Op>
__global__ void partial_reduce_kernel(Expr expr, T* partials, int cols, int total, T init, Op op)
{
    T acc = init;
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride)
        acc = op(acc, static_cast<T>(expr(idx / cols, idx % cols)));

    __shared__ T smem[32];
    acc = block_reduce(acc, init, op, smem);

    if (threadIdx.x == 0)
        partials[blockIdx.x] = acc;
}

/** 全归约第二阶段：把各 block 的部分和收成一个。 */
template <typename T, typename Op>
__global__ void final_reduce_kernel(const T* partials, int n, T* out, T init, Op op)
{
    T acc = init;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        acc = op(acc, partials[i]);

    __shared__ T smem[32];
    acc = block_reduce(acc, init, op, smem);

    if (threadIdx.x == 0)
        out[0] = acc;
}

/** 逐元素变换：把小向量原地过一遍仿函数（用于 sqrt 之类的收尾）。 */
template <typename T, typename Fn>
__global__ void transform_kernel(T* data, int n, Fn fn)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = fn(data[idx]);
}

/** 归约前的公共检查。 */
template <typename Expr>
void assert_reducible()
{
    assert_device_ready<Expr>();
}

inline int round_up_to_warp(int n)
{
    const int warp = 32;
    return ((n + warp - 1) / warp) * warp;
}

/** 逐行归约的公共实现。 */
template <typename Expr, typename Op>
dev_colvec_t<typename Expr::ele_type> row_reduce(const Expr& expr,
                                                typename Expr::ele_type init, Op op)
{
    using T = typename Expr::ele_type;
    assert_reducible<Expr>();

    const int rows = expr.row_num();
    dev_colvec_t<T> out(rows);
    if (rows == 0 || expr.col_num() == 0)
        return out;

    const int tpb = round_up_to_warp(256);
    row_reduce_kernel<Expr, T, Op><<<rows, tpb>>>(expr, out.buffer().data(), init, op);
    JAS_CUDA_CHECK(cudaGetLastError());
    return out;
}

/** 逐列归约的公共实现。 */
template <typename Expr, typename Op>
dev_rowvec_t<typename Expr::ele_type> col_reduce(const Expr& expr,
                                                typename Expr::ele_type init, Op op)
{
    using T = typename Expr::ele_type;
    assert_reducible<Expr>();

    constexpr int kBlockRows = 8;
    constexpr int kBlockCols = 32;      // 与 warpSize 一致，使一个 warp 正好覆盖一整行

    const int rows = expr.row_num();
    const int cols = expr.col_num();
    dev_rowvec_t<T> out(cols);
    if (rows == 0 || cols == 0)
        return out;

    // rows == 0 时下面每个线程的部分和都是 init；这里已提前返回，不会误把 init 当结果写出去。
    const int blocks = (cols + kBlockCols - 1) / kBlockCols;
    dim3 threads(kBlockCols, kBlockRows);
    col_reduce_kernel<Expr, T, Op, kBlockRows, kBlockCols>
        <<<blocks, threads>>>(expr, out.buffer().data(), rows, init, op);
    JAS_CUDA_CHECK(cudaGetLastError());
    return out;
}

/**
 * 全归约的公共实现。
 *
 * 分两阶段（部分和 → 收尾）而不是单 block 网格步长：单 block 只能用到 1 个 SM，
 * 对 20 个 SM 的 P4 是巨大浪费。两阶段既用满 SM，又保持**结果确定**
 * （分块固定、归约树固定，不依赖原子加的顺序）。
 */
template <typename Expr, typename Op>
typename Expr::ele_type full_reduce(const Expr& expr, typename Expr::ele_type init, Op op)
{
    using T = typename Expr::ele_type;
    assert_reducible<Expr>();

    const int rows = expr.row_num();
    const int cols = expr.col_num();
    if (rows == 0 || cols == 0)
        return init;

    // rows * cols 用 int 表示，超过 2^31 的矩阵需要改成长整型（当前用不到）
    const int total = rows * cols;

    const int tpb = round_up_to_warp(256);
    const int max_blocks = 64;
    const int blocks = std::min(max_blocks, (total + tpb - 1) / tpb);

    dev_buf_t<T> partials(blocks);
    partial_reduce_kernel<Expr, T, Op>
        <<<blocks, tpb>>>(expr, partials.data(), cols, total, init, op);
    JAS_CUDA_CHECK(cudaGetLastError());

    dev_buf_t<T> result(1);
    final_reduce_kernel<T, Op><<<1, 64>>>(partials.data(), blocks, result.data(), init, op);
    JAS_CUDA_CHECK(cudaGetLastError());

    T host_value{};
    result.download(&host_value, 1);
    return host_value;
}

/** 小向量上的逐元素变换（原地）。 */
template <typename T, typename Fn>
void transform_buffer(dev_buf_t<T>& buf, Fn fn)
{
    const int n = static_cast<int>(buf.size());
    if (n == 0)
        return;

    const int blocks = (n + 127) / 128;
    transform_kernel<T, Fn><<<blocks, 128>>>(buf.data(), n, fn);
    JAS_CUDA_CHECK(cudaGetLastError());
}

/** `sqrt(scale * v + eps)`，用于 RMSNorm / LayerNorm 的收尾。 */
template <typename T>
struct sqrt_scaled_eps
{
    T scale;
    T eps;

    // 注意写成 ::jasmine::detail 而不是 detail：本函数位于 jasmine::cuda::detail，
    // 非限定的 `detail` 会先命中 jasmine::cuda::detail（即它自己），从而找不到 device_sqrt
    JAS_HD T operator()(T v) const
    {
        return ::jasmine::detail::device_sqrt(v * scale + eps);
    }
};

/** `v * scale`，用于把和折算成均值。 */
template <typename T>
struct scale_op
{
    T scale;

    JAS_HD T operator()(T v) const { return v * scale; }
};

} // namespace detail

// ---------------------------------------------------------------------------
// 归约
// ---------------------------------------------------------------------------

/*
 * 命名说明（重要）：这里的入口**刻意不叫** `hsum` / `vsum` / `hsoftmax`，
 * 尽管主机端正是那些名字。
 *
 * 原因是 ADL：设备叶子 `dev_mat_t` 和主机端表达式都住在命名空间 `jasmine`，
 * 于是任何非限定的 `vsum(expr)` 都会把 `jasmine::vsum` 一并拉进候选集。
 * 实测中主机端那个重载被**静默**选中（主机 `vsum` 返回 `mat_t`，随后报出
 * "mat_t has no member leaf" 这种风马牛不相及的错），而不是给出重载歧义 ——
 * 这类错误极难定位。用互不相同的名字从根上消除隐患，
 * 顺带 `row_*` / `col_*` 也比 `h*` / `v*` 直白。
 */

/** 逐行求和 → (rows × 1)。语义同主机端 `hsum`。 */
template <typename Expr>
dev_colvec_t<typename Expr::ele_type> row_sum(const Expr& expr)
{
    using T = typename Expr::ele_type;
    return detail::row_reduce(expr, detail::identity_add<T>(), detail::add_op{});
}

/** 逐行取大 → (rows × 1)。语义同主机端 `hmax`。 */
template <typename Expr>
dev_colvec_t<typename Expr::ele_type> row_max(const Expr& expr)
{
    using T = typename Expr::ele_type;
    return detail::row_reduce(expr, detail::identity_max<T>(), detail::max_op{});
}

/** 逐列求和 → (1 × cols)。语义同主机端 `vsum`。 */
template <typename Expr>
dev_rowvec_t<typename Expr::ele_type> col_sum(const Expr& expr)
{
    using T = typename Expr::ele_type;
    return detail::col_reduce(expr, detail::identity_add<T>(), detail::add_op{});
}

/** 逐列取大 → (1 × cols)。主机端没有对应函数，但逐行那套的对称版本。 */
template <typename Expr>
dev_rowvec_t<typename Expr::ele_type> col_max(const Expr& expr)
{
    using T = typename Expr::ele_type;
    return detail::col_reduce(expr, detail::identity_max<T>(), detail::max_op{});
}

/** 全矩阵求和 → 主机标量。语义同主机端 `sum`。 */
template <typename Expr>
typename Expr::ele_type sum_all(const Expr& expr)
{
    using T = typename Expr::ele_type;
    return detail::full_reduce(expr, detail::identity_add<T>(), detail::add_op{});
}

/** 全矩阵取大 → 主机标量。语义同主机端 `max`。 */
template <typename Expr>
typename Expr::ele_type max_all(const Expr& expr)
{
    using T = typename Expr::ele_type;
    return detail::full_reduce(expr, detail::identity_max<T>(), detail::max_op{});
}

/** 逐列均值 → (1 × cols)。语义同主机端 `vmean`：统计量沿**行**方向算。 */
template <typename Expr>
dev_rowvec_t<typename Expr::ele_type> col_mean(const Expr& expr)
{
    using T = typename Expr::ele_type;
    auto s = col_sum(expr);
    const T inv_n = T(1) / static_cast<T>(expr.row_num());
    // 小向量，直接乘 1/n：除法在这里比乘法贵，且 1/n 只算一次
    detail::transform_buffer(s.buffer(), detail::scale_op<T>{inv_n});
    return s;
}

// ---------------------------------------------------------------------------
// 小向量上的原地变换
// ---------------------------------------------------------------------------

/** 对拥有缓冲区的小向量做原地逐元素变换；`fn` 必须是设备可调用的标量函数。 */
template <typename T, typename Fn>
void transform_vector(dev_colvec_t<T>& v, Fn fn)
{
    detail::transform_buffer(v.buffer(), fn);
}

template <typename T, typename Fn>
void transform_vector(dev_rowvec_t<T>& v, Fn fn)
{
    detail::transform_buffer(v.buffer(), fn);
}

/**
 * 仅供测试：「能走单趟路径的最大列数」的覆盖值。默认 -1 表示按真实共享内存容量判断。
 *
 * 存在的理由：单趟路径上线后，所有用小矩阵的老测例都会走快路径，三趟回退路径就**没人测了**。
 * 有了这个开关，同一份输入可以两边都跑一遍再对拍，回退路径不会因为「很少被选中」而悄悄腐烂。
 */
inline int& softmax_max_cols_override()
{
    static int v = -1;
    return v;
}

/**
 * 行驻留共享内存路径能处理的最大列数。
 *
 * 等于「共享内存预算 ÷ sizeof(T)」再扣掉 block_reduce 的暂存（32 个元素）。
 * 超出这个长度就必须走三趟回退路径 —— 硬启动会直接失败。
 */
template <typename T>
inline int softmax_shared_max_cols()
{
    if (softmax_max_cols_override() >= 0)
        return softmax_max_cols_override();

    constexpr int kScratch = 32;
    const int budget = max_dynamic_shared_bytes();
    const int cols = budget / static_cast<int>(sizeof(T)) - kScratch;
    return cols > 0 ? cols : 0;
}

/**
 * 追踪共享内存路径实际被用了几次。
 *
 * 纯粹是可观测性：优化最容易出的问题不是算错，而是**根本没生效**（阈值算错、分支写反），
 * 而结果依然正确、测试依然全绿。有了这个计数，测例才能断言「该走快路径时确实走了」。
 */
inline int& softmax_shared_launch_count()
{
    static int n = 0;
    return n;
}

/** 共享内存单趟逐行 softmax；`x` 的整行会落在共享内存里，结果写进 `out`。 */
template <typename Expr>
void softmax_rows_shared(const Expr& x, dev_matrix_t<typename Expr::ele_type>& out)
{
    using T = typename Expr::ele_type;

    detail::assert_device_ready<Expr>();

    const int rows = x.row_num();
    const int cols = x.col_num();
    const std::size_t smem_bytes = (static_cast<std::size_t>(cols) + 32) * sizeof(T);

    constexpr int kThreads = 256;

    // cudaFuncSetAttribute 要按「每个 kernel 实例化」调用，且只在需要更大共享内存时
    // 才有必要。函数内静态量正好做到这点；超过 48 KiB 的块有上限，故上面先判过长度。
    static std::size_t configured = 0;
    if (smem_bytes > configured)
    {
        JAS_CUDA_CHECK(cudaFuncSetAttribute(detail::softmax_rows_shared_kernel<Expr, T>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem_bytes)));
        configured = smem_bytes;
    }

    detail::softmax_rows_shared_kernel<Expr, T><<<rows, kThreads, smem_bytes>>>(
        x, out.buffer().data(), cols, detail::identity_max<T>(), detail::identity_add<T>());
    JAS_CUDA_CHECK(cudaGetLastError());
    ++softmax_shared_launch_count();
}

// ---------------------------------------------------------------------------
// 逐行 softmax（注意力用）
// ---------------------------------------------------------------------------

/**
 * 逐行 softmax，数值稳定。语义与主机端 `hsoftmax` 一致：
 *
 *   m = hmax(x)                 // 逐行最大值，做数值稳定
 *   s = hsum(exp(x - m))        // 逐行指数和
 *   out = exp(x - m) / s
 *
 * 两条实现路径，按「整行能否放得进共享内存」自动选择：
 *
 *  1. **单趟（默认路径）**：整行留在共享内存里，max / exp+求和 / 归一化全在片上做。
 *     全局访存读 1 遍 + 写 1 遍，`exp` 每元素 1 次。
 *  2. **三趟（回退）**：`row_max` → `row_sum(exp(...))` → 归一化，每趟融合但有各自的一次
 *     全局读，`exp` 每元素 2 次。行长超过共享内存时使用。
 *
 * 两条路径的数值结构相同（先减最大值再求指数和），结果按相对容差一致。
 *
 * `x` 里允许出现 `-inf`（因果掩码），`exp(-inf - finite) = 0` 行为正确。
 * 整行全为 `-inf` 时两条路径都得到 `0/0 = nan` —— 与主机端 `hsoftmax` 逐字相同，
 * 而因果掩码下每行至少有一个合法位置，这种情况不可达。
 */
template <typename Expr>
dev_matrix_t<typename Expr::ele_type> softmax_rows(const Expr& x)
{
    using T = typename Expr::ele_type;

    const int rows = x.row_num();
    const int cols = x.col_num();
    dev_matrix_t<T> out(rows, cols);
    if (rows == 0 || cols == 0)
        return out;

    if (cols <= softmax_shared_max_cols<T>())
    {
        softmax_rows_shared(x, out);
        return out;
    }

    auto m = row_max(x);
    auto s = row_sum(exp(x - m.leaf()));
    eval_fused(exp(x - m.leaf()) / s.leaf(), out.buffer());
    return out;
}

// ---------------------------------------------------------------------------
// 归一化层（与 jas_net_t.hpp 的 layer_norm_net_t / rms_norm_net_t 对齐）
// ---------------------------------------------------------------------------

/**
 * LayerNorm 反向需要的中间量。前向必须把它们留下来 —— 反向公式里
 * `hx`（归一化后、**不含** gamma 的值）与 `std`（每列的 sqrt(var+eps)）都出现了。
 *
 * 主机端把这两个量作为成员存在 `layer_norm_net_t` 里（`m_hx` / `m_std`），
 * 这里显式绑成一个结构体，让「前向留下什么给反向」这件事在调用处可见，
 * 而不是藏在隐式成员状态里。
 */
template <typename T>
struct layer_norm_cache_t
{
    dev_matrix_t<T> hx;   // (rows × cols)，= (x - mean) / std
    dev_rowvec_t<T> std;  // (1 × cols)，= sqrt(vmean((x-mean)^2) + eps)
};

/** RMSNorm 反向需要的中间量（没有 mean，所以只有 hx 与 rms）。 */
template <typename T>
struct rms_norm_cache_t
{
    dev_matrix_t<T> hx;  // (rows × cols)，= x / rms
    dev_rowvec_t<T> rms; // (1 × cols)，= sqrt(vmean(x²) + eps)
};

/**
 * 按列标准化（统计量沿**行**方向算），等价于 `layer_norm_net_t`：
 *
 *   mean = vmean(x)
 *   var  = vmean((x - mean)^2) + eps
 *   out  = gamma * (x - mean) / sqrt(var) + beta
 *
 * `gamma` / `beta` 形状为 (rows × 1)，沿列广播（主机端的 `m_gama` 就是 [d_model, 1]）。
 *
 * 传了 `cache` 就顺便留下反向所需的 `hx` / `std`（多一趟全矩阵读写），
 * 不传则保持原来的单趟融合 —— 只做推理时不该为反向付钱。
 */
template <typename Expr>
dev_matrix_t<typename Expr::ele_type> layer_norm(
    const Expr& x, const dev_colvec_t<typename Expr::ele_type>& gamma,
    const dev_colvec_t<typename Expr::ele_type>& beta,
    typename Expr::ele_type eps = typename Expr::ele_type(1e-5),
    layer_norm_cache_t<typename Expr::ele_type>* cache = nullptr)
{
    using T = typename Expr::ele_type;

    const int rows = x.row_num();
    const int cols = x.col_num();
    dev_matrix_t<T> out(rows, cols);
    if (rows == 0 || cols == 0)
        return out;

    auto mean = col_mean(x);

    // centered 要参与两次归约/运算，先物化一次，避免同一棵子树被重复求值
    dev_matrix_t<T> centered(rows, cols);
    eval_fused(x - mean.leaf(), centered.buffer());

    auto var = col_mean(centered.leaf() * centered.leaf());
    detail::transform_buffer(var.buffer(), detail::sqrt_scaled_eps<T>{T(1), eps});

    if (cache == nullptr)
    {
        eval_fused(centered.leaf() / var.leaf() * gamma.leaf() + beta.leaf(), out.buffer());
        return out;
    }

    // 留下反向所需的量：hx 先落一遍，out 再由 hx 导出（多一趟，换取反向可用）
    cache->hx.allocate(rows, cols);
    eval_fused(centered.leaf() / var.leaf(), cache->hx.buffer());
    cache->std = std::move(var);
    eval_fused(cache->hx.const_leaf() * gamma.leaf() + beta.leaf(), out.buffer());
    return out;
}

/**
 * LayerNorm 的反向，逐字对应主机 `layer_norm_net_t::backward`：
 *
 *   dgamma = hsum(delta ⊙ hx)              // 沿序列维（列）累加
 *   dbeta  = hsum(delta)
 *   m      = rows                          // 注意是**行数**（= d_model），与 vmean 同口径
 *   dx_norm = delta ⊙ gamma
 *   sum_dx_norm        = vsum(dx_norm)              // 沿特征维（行）归约，广播回列
 *   sum_dx_norm_x_hx   = vsum(dx_norm ⊙ hx)
 *   dx = (dx_norm·m − sum_dx_norm − hx·sum_dx_norm_x_hx) / m / std
 *
 * `dgamma` / `dbeta` 传 nullptr 就跳过（某些位置 gamma 不参与训练）。
 */
template <typename Delta>
dev_matrix_t<typename Delta::ele_type> layer_norm_backward(
    const Delta& delta, const dev_matrix_t<typename Delta::ele_type>& hx,
    const dev_rowvec_t<typename Delta::ele_type>& std,
    const dev_colvec_t<typename Delta::ele_type>& gamma,
    dev_colvec_t<typename Delta::ele_type>* dgamma = nullptr,
    dev_colvec_t<typename Delta::ele_type>* dbeta = nullptr)
{
    using T = typename Delta::ele_type;

    const int rows = delta.row_num();
    const int cols = delta.col_num();
    dev_matrix_t<T> out(rows, cols);
    if (rows == 0 || cols == 0)
        return out;

    if (dgamma != nullptr)
        *dgamma = row_sum(delta * hx.const_leaf());
    if (dbeta != nullptr)
        *dbeta = row_sum(delta);

    const T m = static_cast<T>(rows);

    dev_matrix_t<T> dx_norm(rows, cols);
    eval_fused(delta * gamma.leaf(), dx_norm.buffer());

    auto sum_dx_norm = col_sum(dx_norm.leaf());
    auto sum_dx_norm_x_hx = col_sum(dx_norm.leaf() * hx.const_leaf());

    eval_fused((dx_norm.leaf() * m - sum_dx_norm.leaf() - hx.const_leaf() * sum_dx_norm_x_hx.leaf())
                   / m / std.leaf(),
               out.buffer());
    return out;
}

/**
 * 按列做 RMSNorm，等价于 `rms_norm_net_t`：
 *
 *   ms  = vmean(x^2) + eps
 *   out = gamma * x / sqrt(ms)
 *
 * 与 LayerNorm 的唯一实质差别是**不减均值**，也没有 beta。
 */
template <typename Expr>
dev_matrix_t<typename Expr::ele_type> rms_norm(
    const Expr& x, const dev_colvec_t<typename Expr::ele_type>& gamma,
    typename Expr::ele_type eps = typename Expr::ele_type(1e-5),
    rms_norm_cache_t<typename Expr::ele_type>* cache = nullptr)
{
    using T = typename Expr::ele_type;

    const int rows = x.row_num();
    const int cols = x.col_num();
    dev_matrix_t<T> out(rows, cols);
    if (rows == 0 || cols == 0)
        return out;

    auto ms = col_sum(x * x);
    const T inv_n = T(1) / static_cast<T>(rows);
    detail::transform_buffer(ms.buffer(), detail::sqrt_scaled_eps<T>{inv_n, eps});

    if (cache == nullptr)
    {
        eval_fused(x / ms.leaf() * gamma.leaf(), out.buffer());
        return out;
    }

    cache->hx.allocate(rows, cols);
    eval_fused(x / ms.leaf(), cache->hx.buffer());
    cache->rms = std::move(ms);
    eval_fused(cache->hx.const_leaf() * gamma.leaf(), out.buffer());
    return out;
}

/**
 * RMSNorm 的反向，逐字对应主机 `rms_norm_net_t::backward`。
 *
 * 比 LayerNorm **少一项**（没有 `sum(dx_norm)`，因为不减均值）：
 *
 *   dgamma = hsum(delta ⊙ hx)
 *   dx = (dx_norm·m − hx·vsum(dx_norm ⊙ hx)) / m / rms
 *
 * 主机源码里那段推导值得一并记住：`∂L/∂x_i = (g_i − hx_i·mean(g ⊙ hx)) / r`，
 * 与上式同值 —— 上式多乘的 `m` 被 `vsum` 与 `mean` 的 `1/m` 抵消了。
 */
template <typename Delta>
dev_matrix_t<typename Delta::ele_type> rms_norm_backward(
    const Delta& delta, const dev_matrix_t<typename Delta::ele_type>& hx,
    const dev_rowvec_t<typename Delta::ele_type>& rms,
    const dev_colvec_t<typename Delta::ele_type>& gamma,
    dev_colvec_t<typename Delta::ele_type>* dgamma = nullptr)
{
    using T = typename Delta::ele_type;

    const int rows = delta.row_num();
    const int cols = delta.col_num();
    dev_matrix_t<T> out(rows, cols);
    if (rows == 0 || cols == 0)
        return out;

    if (dgamma != nullptr)
        *dgamma = row_sum(delta * hx.const_leaf());

    const T m = static_cast<T>(rows);

    dev_matrix_t<T> dx_norm(rows, cols);
    eval_fused(delta * gamma.leaf(), dx_norm.buffer());

    auto sum_dx_norm_x_hx = col_sum(dx_norm.leaf() * hx.const_leaf());

    eval_fused((dx_norm.leaf() * m - hx.const_leaf() * sum_dx_norm_x_hx.leaf()) / m / rms.leaf(),
               out.buffer());
    return out;
}

/**
 * 逐行 softmax 的反向，对应主机 `hsoftmax_net_t::backward`：
 *
 *   dx = w ⊙ (dy − row_sum(w ⊙ dy))
 *
 * `w` 是**前向输出的概率矩阵**，必须由调用方留着 —— 主机端也是这么做的
 * （`hsoftmax_net_t::m_output` 是 public 成员，`mat_head_gen_t::backward` 直接读它）。
 *
 * 那个行和项就是「softmax 的归一化约束」在反向里的体现：每个元素都受整行影响，
 * 所以梯度里要减掉整行的加权和。
 */
template <typename W, typename Delta>
dev_matrix_t<typename W::ele_type> softmax_backward(const W& w, const Delta& delta)
{
    using T = typename W::ele_type;

    const int rows = w.row_num();
    const int cols = w.col_num();
    dev_matrix_t<T> out(rows, cols);
    if (rows == 0 || cols == 0)
        return out;

    auto s = row_sum(w * delta);
    eval_fused(w * (delta - s.leaf()), out.buffer());
    return out;
}

} // namespace cuda
} // namespace jasmine

#endif
