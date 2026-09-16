#ifndef __JAS_CUDA_ROPE_HPP__
#define __JAS_CUDA_ROPE_HPP__

/**
 * 设备端 RoPE（旋转位置编码）。
 *
 * ## 与主机端 RoPE_net_t 的关系
 *
 * 数学定义、两种配对约定（interleaved / half_split）、以及「列 j 用绝对位置
 * start_pos + j」这套口径，全部与 `jas_RoPE_t.hpp` 保持一致，且**与主机端逐元素对拍**。
 *
 * 两处实现上的差别：
 *
 *  1. **不再靠 2×2 小矩阵乘**。主机端把每对特征做成一个 2×2 旋转矩阵、
 *     再用 `dot` 作用到 `(2 × 1)` 视图上 —— 那是为了复用矩阵设施。设备端一个线程
 *     直接读写两个元素就够了，走 GEMM 反而荒谬。
 *
 *  2. **half_split 不再搬数据**。主机端的做法是「行重排 → interleaved 旋转 → 搬回去」，
 *     因为文档里已经论证过两者只差一个特征行置换、而 θ_i 的定义完全相同。
 *     设备端把这个置换直接折成 kernel 里的**行下标映射**（`r0 = i`、`r1 = i + half`），
 *     一趟走完，省掉两次全量搬运。
 *
 * ## 为什么表在主机上算
 *
 * cos/sin 表只依赖「位置 × 维度对」，与数据无关，而且 `reserve()` 时算一次就长期复用。
 * 在主机上用 libm 的 `std::pow` / `std::cos` 算好再上传，比在设备端实现 `pow` 更省心，
 * 也更容易和主机端对齐 —— 反正是一次性成本。
 *
 * 表用 **double** 计算再落成 `T`：`θ_i = m / 10000^(2i/d)` 的精度直接决定旋转角的精度，
 * 用 double 算常量是白拿的准确度（主机端 `RoPE_net_t<double>` 算的就是 double）。
 *
 * ## 前置
 *
 * 使用前需要 `reserve(max_seq_len)`（或依赖 dynamic 模式的自动扩容）。
 * **扩容会重新分配表，但不会影响任何已算出的结果** —— 表是只读常量，不像 KV cache 的
 * 叶子那样会被调用方长期持有（本类也不对外暴露表的叶子）。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_RoPE_t.hpp" // 复用 rope_pair_layout / rope_cache_mode，一个概念一个名字
#include "jas_mat_express_t.hpp"

namespace jasmine {
namespace cuda {

namespace detail {

/**
 * 旋转 kernel：一个线程负责「一对特征 × 一个位置」的两个元素。
 *
 * 线程下标按**位置在内、特征对在外**排布（`j` 连续），这样相邻线程读写的
 * `x(r0, j)` / `out[r0 * ld + j]` 落在连续地址上，访存是合并的。
 * 表的读取（`cos_tab[m * half + i]`，`m` 随 `j` 变）则是跨步的 —— 但表很小、
 * 只读、被所有行复用，缓存得住；拿它换主数据的合并访存是划算的。
 *
 * `X` 可以是任意**设备可求值**的表达式，于是旋转能直接融进上游的投影/加法链，
 * 不必先落一个临时矩阵。
 *
 * `Inverse == true` 时做的是**转置旋转**（反向）：把 2×2 旋转矩阵换成它的转置。
 * 旋转矩阵是正交的，所以 `R⁻¹ = Rᵀ` —— 反向不是"除以什么"，而是**同一个旋转、
 * 反方向转**。这也解释了主机端 `interleaved_backward` 为什么只是把 `rope_mat` 换成
 * `rope_mat.t()`：整个反向里唯一的改动就是这一点。
 *
 *   | c -s | | a |   =   | c·a - s·b |        | c  s | | g |   =   |  c·g + s·h |
 *   | s  c | | b |       | s·a + c·b |        |-s  c | | h |       | -s·g + c·h |
 */
template <typename X, typename T, bool HalfSplit, bool Inverse>
__global__ void rope_rotate_kernel(X x, T* __restrict__ out, const T* __restrict__ cos_tab,
                                   const T* __restrict__ sin_tab, int start_pos, int half,
                                   int seq_len, int ld_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half * seq_len)
        return;

    const int i = idx / seq_len;          // 特征对
    const int j = idx - i * seq_len;      // 位置（线程内连续）
    const int m = start_pos + j;

    const T c = cos_tab[m * half + i];
    const T s = sin_tab[m * half + i];

    // 两种配对约定只差「第 i 对是哪两行」，θ_i 完全相同 —— 所以表是同一张
    const int r0 = HalfSplit ? i : 2 * i;
    const int r1 = HalfSplit ? (i + half) : (2 * i + 1);

    const T a = static_cast<T>(x(r0, j));
    const T b = static_cast<T>(x(r1, j));

    if constexpr (Inverse)
    {
        out[static_cast<std::ptrdiff_t>(r0) * ld_out + j] = c * a + s * b;
        out[static_cast<std::ptrdiff_t>(r1) * ld_out + j] = -s * a + c * b;
    }
    else
    {
        out[static_cast<std::ptrdiff_t>(r0) * ld_out + j] = c * a - s * b;
        out[static_cast<std::ptrdiff_t>(r1) * ld_out + j] = s * a + c * b;
    }
}

} // namespace detail

/**
 * 设备端 RoPE：持有一张 `(max_seq × d/2)` 的 cos / sin 表，外加配对约定。
 *
 * 用法（decode 单步）：
 *
 *   cuda::dev_rope_t<double> rope(d_head, rope_pair_layout::half_split);
 *   rope.reserve(max_seq_len);
 *   auto k_rot = rope.forward_at(dk.leaf(), pos);      // (d_head × 1)
 *   caches.append_all(k_rot.leaf(), dv);               // 契约：append 进来的 K 已旋转
 */
template <typename T>
class dev_rope_t
{
public:
    dev_rope_t() = default;

    explicit dev_rope_t(int d, rope_pair_layout layout = rope_pair_layout::interleaved)
    {
        set_param(d, layout);
    }

    /** 设置维度与配对约定。改了 d 就作废已备好的表。 */
    void set_param(int d, rope_pair_layout layout = rope_pair_layout::interleaved)
    {
        if (d <= 0 || d % 2 != 0)
            throw std::invalid_argument("dev_rope: d 必须是正偶数，实际 "
                                        + std::to_string(d));
        m_d = d;
        m_half = d / 2;
        m_layout = layout;
        m_cos.release();
        m_sin.release();
        m_capacity = 0;
    }

    void set_d(int d) { set_param(d, m_layout); }
    int dim() const { return m_d; }

    void set_pair_layout(rope_pair_layout layout)
    {
        m_layout = layout; // 只影响下标映射，表不用重建
    }
    rope_pair_layout pair_layout() const { return m_layout; }

    void set_cache_mode(rope_cache_mode mode) { m_mode = mode; }
    rope_cache_mode cache_mode() const { return m_mode; }

    /** 已备好表的最大位置数。 */
    int capacity() const { return m_capacity; }

    /**
     * 预分配并填满 [0, max_seq_len) 的表。
     * 只增不缩；static_fixed 模式下这是唯一的扩容途径。
     */
    void reserve(int max_seq_len)
    {
        if (m_d <= 0)
            throw std::runtime_error("dev_rope: 请先 set_param(d)");
        if (max_seq_len <= 0)
            throw std::invalid_argument("dev_rope reserve: max_seq_len 必须为正");
        if (max_seq_len <= m_capacity)
            return;
        build_tables(max_seq_len);
    }

    /**
     * 把 `x` 在绝对位置 `start_pos .. start_pos + x.col_num() - 1` 上旋转，结果写进 `out`。
     *
     * `x` 可以是叶子、拥有者或任意设备可求值的表达式（表达式会被融进这一趟）。
     * `out` 必须是**未转置**的薄壳；形状需与 `x` 一致。
     */
    template <typename X>
    void rotate_into(const X& x, const dev_mat_t<T>& out, int start_pos)
    {
        rotate_impl<false>(x, out, start_pos, "rotate_into");
    }

    /** 旋转并返回一个**拥有显存**的新矩阵。语义同主机端 `RoPE_net_t::forward_at`。 */
    template <typename X>
    dev_matrix_t<T> forward_at(const X& x, int start_pos = 0)
    {
        check_input(x.row_num(), x.col_num(), start_pos);
        dev_matrix_t<T> out(x.row_num(), x.col_num());
        if (x.col_num() > 0)
            rotate_into(x, out.leaf(), start_pos);
        return out;
    }

    /** 同上，位置默认从 0 起（语义同主机端 `forward`）。 */
    template <typename X>
    dev_matrix_t<T> forward(const X& x)
    {
        return forward_at(x, 0);
    }

    /**
     * 原地旋转一个拥有者矩阵。
     *
     * 别名安全：每个线程只读自己那一对元素、再写回同一对，没有别的线程会碰它们
     * （这也是 kernel 不需要任何同步的原因）。decode 路径上省掉一次分配。
     */
    void forward_inplace(dev_matrix_t<T>& m, int start_pos = 0)
    {
        if (m.col_num() > 0)
            rotate_into(m.leaf(), m.leaf(), start_pos);
    }

    // -----------------------------------------------------------------------
    // 反向
    // -----------------------------------------------------------------------

    /**
     * 把上游梯度 `delta` 按转置旋转写进 `out`（**原地安全**，同 `rotate_into`）。
     *
     * `start_pos` 必须与对应的前向一致 —— 反向用的是同一张表、同一批角度，
     * 只是把 R 换成 Rᵀ。
     */
    template <typename X>
    void backward_into(const X& delta, const dev_mat_t<T>& out, int start_pos)
    {
        rotate_impl<true>(delta, out, start_pos, "backward_into");
    }

    template <typename X>
    dev_matrix_t<T> backward_at(const X& delta, int start_pos)
    {
        check_input(delta.row_num(), delta.col_num(), start_pos);
        dev_matrix_t<T> out(delta.row_num(), delta.col_num());
        if (delta.col_num() > 0)
            backward_into(delta, out.leaf(), start_pos);
        return out;
    }

    /**
     * 语义与主机端 `RoPE_net_t::backward` 一致。
     *
     * 注意主机端那个接口**没有** `start_pos` 参数：它把列下标 j 直接当成绝对位置
     * （即隐含 `start_pos == 0`）。设备端把这一点显式化 —— 默认值 0 与主机行为完全对齐，
     * 需要从别的起点开始反向时也不会只能干瞪眼。
     */
    template <typename X>
    dev_matrix_t<T> backward(const X& delta)
    {
        return backward_at(delta, 0);
    }

    /** 原地反向旋转。 */
    void backward_inplace(dev_matrix_t<T>& m, int start_pos = 0)
    {
        if (m.col_num() > 0)
            backward_into(m.leaf(), m.leaf(), start_pos);
    }

private:
    /** 正向与反向的唯一差别是 `Inverse`，其余（校验、分块、表查找）完全共用。 */
    template <bool Inverse, typename X>
    void rotate_impl(const X& x, const dev_mat_t<T>& out, int start_pos, const char* who)
    {
        const int seq = x.col_num();
        check_input(x.row_num(), seq, start_pos);

        if (seq <= 0)
            return;
        if (out.row_num() != m_d || out.col_num() != seq)
            throw std::invalid_argument(std::string("dev_rope ") + who + ": out 形状应为 ("
                                        + std::to_string(m_d) + ", " + std::to_string(seq) + ")");
        if (out.transposed())
            throw std::invalid_argument(std::string("dev_rope ") + who + ": out 不能是转置视图");
        if (!out.valid())
            throw std::invalid_argument(std::string("dev_rope ") + who + ": out 是空叶子");

        detail::assert_device_ready<X>();

        const int total = m_half * seq;
        constexpr int kThreads = 256;
        const int blocks = (total + kThreads - 1) / kThreads;
        const int ld_out = out.leading_dim();

        if (m_layout == rope_pair_layout::half_split)
        {
            if constexpr (Inverse)
                detail::rope_rotate_kernel<X, T, true, true><<<blocks, kThreads>>>(
                    x, out.m_data, m_cos.data(), m_sin.data(), start_pos, m_half, seq, ld_out);
            else
                detail::rope_rotate_kernel<X, T, true, false><<<blocks, kThreads>>>(
                    x, out.m_data, m_cos.data(), m_sin.data(), start_pos, m_half, seq, ld_out);
        }
        else
        {
            if constexpr (Inverse)
                detail::rope_rotate_kernel<X, T, false, true><<<blocks, kThreads>>>(
                    x, out.m_data, m_cos.data(), m_sin.data(), start_pos, m_half, seq, ld_out);
            else
                detail::rope_rotate_kernel<X, T, false, false><<<blocks, kThreads>>>(
                    x, out.m_data, m_cos.data(), m_sin.data(), start_pos, m_half, seq, ld_out);
        }
        JAS_CUDA_CHECK(cudaGetLastError());
    }

    void check_input(int rows, int seq, int start_pos)
    {
        if (m_d <= 0)
            throw std::runtime_error("dev_rope: 还没 set_param(d)");
        if (rows != m_d)
            throw std::invalid_argument("dev_rope: 输入行数 " + std::to_string(rows)
                                        + " 与 d " + std::to_string(m_d) + " 不一致");
        if (start_pos < 0)
            throw std::invalid_argument("dev_rope: start_pos 必须 >= 0");
        if (seq <= 0)
            return;
        ensure_capacity(start_pos + seq);
    }

    void ensure_capacity(int need)
    {
        if (need <= m_capacity)
            return;
        if (m_mode == rope_cache_mode::static_fixed)
            throw std::runtime_error(
                "dev_rope: static_fixed 容量不足；请用更大的 max_seq_len 调 reserve()");
        // 表只依赖位置，扩容是纯重算；翻倍以摊薄重复计算的代价
        build_tables(std::max(need, std::max(m_capacity * 2, 64)));
    }

    void build_tables(int max_seq)
    {
        const std::size_t half = static_cast<std::size_t>(m_half);
        const std::size_t n = static_cast<std::size_t>(max_seq) * half;

        std::vector<T> cos_h(n);
        std::vector<T> sin_h(n);
        for (int m = 0; m < max_seq; ++m)
        {
            for (int i = 0; i < m_half; ++i)
            {
                // θ_i = m / 10000^(2i/d) —— 与主机端 mat_RoPE_t::init 逐字一致
                const double angle =
                    static_cast<double>(m)
                    / std::pow(10000.0, static_cast<double>(2 * i) / static_cast<double>(m_d));
                const std::size_t at = static_cast<std::size_t>(m) * half
                                       + static_cast<std::size_t>(i);
                cos_h[at] = static_cast<T>(std::cos(angle));
                sin_h[at] = static_cast<T>(std::sin(angle));
            }
        }

        m_cos.allocate(n);
        m_sin.allocate(n);
        m_cos.upload(cos_h.data(), n);
        m_sin.upload(sin_h.data(), n);
        m_capacity = max_seq;
    }

    dev_buf_t<T> m_cos;   // (max_seq × half)，行优先：cos_tab[m * half + i]
    dev_buf_t<T> m_sin;   // 同上
    int m_d = 0;
    int m_half = 0;
    int m_capacity = 0;
    rope_pair_layout m_layout = rope_pair_layout::interleaved;
    rope_cache_mode m_mode = rope_cache_mode::dynamic;
};

} // namespace cuda
} // namespace jasmine

#endif
