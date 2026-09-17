#ifndef __JAS_CUDA_ATTENTION_HPP__
#define __JAS_CUDA_ATTENTION_HPP__

/**
 * 融合注意力（Flash Attention 那一类）：**不物化概率矩阵**的注意力。
 *
 * ## 它解决的是哪一个开销
 *
 * 在此之前，注意力一直是三步：
 *
 *   scores  = Qᵀ·K / √d             → (q_len × len)
 *   weights = softmax_rows(scores)  → (q_len × len)   ← **这一步要落一遍显存**
 *   out     = V · weightsᵀ
 *
 * 那个 `(q_len × len)` 的概率矩阵是纯中间产物。它是 softmax 的输出（反向要用），
 * 除此之外没有任何理由存在于显存里。它的代价是 `q_len × len × sizeof(T)` **每个头**，
 * 而且**与 `d_head` 无关** —— 也就是说上下文越长，它相对模型本身越不成比例。
 * TinyLlama 的规模（32 头 × 2048² × 4B）单层就是 512 MB，而同期 Q/K/V 只有十几 MB。
 *
 * 融合的做法是**把 softmax 与 `·V` 合并成一趟**：概率算出来立刻乘进 `V` 的加权和，
 * 中间不落盘。代价是**必须用 online softmax** —— 一趟扫过去时还不知道整行最大值，
 * 只能边算边修正。这正是 CUDA.md 5.6 里「online-softmax 真正不可替代的场合是
 * fused attention」那句话的兑现。
 *
 * ## online softmax 到底在修什么
 *
 * 普通 softmax 要先知道整行最大值 `m` 才能算 `exp(s − m)`（减去它是为了不溢出）。
 * 一趟扫描时 `m` 会逐步变大，早先算出的 `exp(s − m_old)` 就"过期"了：
 *
 *   m_new = max(m_old, max_j∈tile s_j)          ← 本块里出现了更大的值
 *   corr  = exp(m_old − m_new) ≤ 1              ← 老结果整体等比缩小
 *   acc   = acc · corr                          ← 输出累加器同步缩小
 *   l     = l   · corr + Σ_j exp(s_j − m_new)   ← 归一化因子同步缩小
 *
 * 关键在于**修正量与 `s` 无关**：它只依赖两个最大值，所以对已经累加进去的整条
 * `Σ p_j v_j` 做一次统一缩放就够了，不必回头重算。这就是它能一趟走完的原因。
 *
 * 收尾 `out = acc / l`，并留下 `L = m + log l` 供反向用。
 *
 * ## 反向：概率重算，而不是概率留存
 *
 * 反向需要的概率 `p = exp(s − L)` 可以从**每行一个数**的 `L` 精确重算 ——
 * `s` 就是 `Qᵀ·K/√d`（Q/K 本来就留着），减 `L` 再取指数即可。于是：
 *
 * | | 前向要为反向留下的东西（每个头） |
 * | --- | --- |
 * | 非融合 | `weights`：`q_len × len` 个数 |
 * | **融合** | `logsumexp`：`q_len` 个数 |
 *
 * 这才是融合的**全部收益**：省掉一个与上下文长度平方成正比的中间量。
 * 反向的其余部分逐字对应主机 `mat_head_gen_t::backward`，只是把两次 GEMM
 * 拆进了块循环。唯一"需要想一下"的是 `D_i = Σ_f dO(f,i)·O(f,i)` 那条恒等式 ——
 * 它把「对整行 p 加权求和」换成用前向已经算出的 `O` 兜一下，于是不必回头再扫整行。
 *
 * ## 这一版没有做什么
 *
 * 说清楚，免得把「不物化」当成「最优」：
 *
 *  1. **query 方向没有分块复用。** K/V 按 `bc` 列分块常驻共享内存（这一半是 flash 的
 *     精髓：K/V 每个块只从全局读一遍），但一个 block 只处理 `BR` 个 query 行、
 *     行与行之间不共享 K/V 的读。真正的 flash attention 会做 `Br × Bc` 双向分块，
 *     让 Q 也常驻并复用。
 *  2. **dK/dV 走 `atomicAdd`**，没有做「反向也分块、块内先归约再原子加一次」。
 *  3. 没有 warp 级流水（`cp.async` / 双缓冲）。
 *  4. 共享内存按 48 KiB 的默认上限算，没走 `cudaFuncSetAttribute` 的 opt-in。
 *
 * 所以定位是「把算法结构摆正、把数值钉住、把内存收益量化出来」，
 * 与 `softmax_rows` 当初「单趟共享内存 + 三趟回退」的定位一致：
 * **先让快路径存在且可验证，再谈榨干它**。代价这一侧也写明了：融合拿 memory
 * 换的是 K/V 的重复读（每个 query 行都要过一遍 K/V），在 `len` 远大于 `d_head` 时划算。
 *
 * ## 与已有路径的关系
 *
 * 两条路径**都在**，而且默认仍然走非融合的 —— 这条是新增的加速路径，不是替代品。
 * `dev_head_gen_t` 上有个开关（`set_fused_attention`），测试把两边逐元素对拍，
 * 免得融合路径因为「只在特定规模下被选中」而悄悄腐烂（`softmax_rows` 的三趟回退
 * 路径就吃过这个亏，见 CUDA.md 5.6）。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_compat.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_net.hpp"

namespace jasmine {
namespace cuda {

namespace detail {

/** 一个 block 里并排处理几个 query 行（每行一个 warp）。 */
inline constexpr int kAttnRowsPerBlock = 4;
/** warp 宽度；`BR` 与它相乘就是 block 的线程数。 */
inline constexpr int kAttnWarp = 32;

/** warp 内全体取最大值，结果**广播给所有 lane**。 */
template <typename T>
JAS_DEV T warp_all_max(T v)
{
    const unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1)
        v = ::jasmine::detail::device_max(v, __shfl_down_sync(mask, v, off));
    return __shfl_sync(mask, v, 0);
}

/**
 * warp 内全体求和，结果**广播给所有 lane**。
 *
 * 广播是必需的，不是顺手：修正完累加器之后每个 lane 都要用同一个 `l_new` 去算
 * 下一轮的 `l`，只有 lane 0 知道就会各算各的。
 */
template <typename T>
JAS_DEV T warp_all_sum(T v)
{
    const unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(mask, v, off);
    return __shfl_sync(mask, v, 0);
}

/** 因果掩码：第 i 个 query（绝对位置 `i + offset`）能不能看到第 j 个 key。 */
JAS_INLINE_HD bool attn_key_visible(int i, int j, bool causal, int offset)
{
    return !causal || j <= i + offset;
}

/**
 * 共享内存布局：算一次，前后向共用，免得两处各写一份偏移量而悄悄错位。
 *
 * `k_stride` 刻意取 `d + 1`：K/V 分块按 `(bc × d)` 存放，若行跨度正好是 32 的倍数，
 * 同一个 warp 里不同 lane 读不同行的同一列就会落进同一个 bank（32 路冲突）。
 * 多加一列把跨度变成 `d + 1`，冲突就没了。这一条只影响速度不影响数值 ——
 * 也正因为不影响数值，写错了不会被任何测例发现，所以放在这里显式说明。
 *
 * 前向要 m/l（每行两个数）；反向要 dO 的一列、D_i，以及一个与 p 并存的 dS 数组。
 * 两种模式只在多出来的那几个数组上不同，所以用一个 `backward` 开关打包。
 */
struct attn_smem_layout
{
    int d = 0;
    int bc = 0;
    int br = 0;
    int k_stride = 0;

    int q_off = 0;    // br × d   —— Q 分块
    int k_off = 0;    // bc × ks  —— K 分块
    int v_off = 0;    // bc × ks  —— V 分块
    int p_off = 0;    // br × bc  —— 前向：概率 p；反向：概率 p（要与 dS 并存）
    int ds_off = 0;   // br × bc  —— 仅反向：dS
    int out_off = 0;  // br × d   —— 前向：输出累加器；反向：dQ 累加器
    int do_off = 0;   // br × d   —— 仅反向：dO 的第 i 列（每个 warp 一段）
    int di_off = 0;   // br       —— 仅反向：D_i = Σ_f dO(f,i)·O(f,i)
    int m_off = 0;    // br       —— 仅前向：运行最大值
    int l_off = 0;    // br       —— 仅前向：运行指数和
    int total = 0;

    /** 纯整数运算，主机与设备都能算 —— kernel 里要重建同一份布局。 */
    JAS_HD static attn_smem_layout make(int d, int bc, int br, bool backward)
    {
        attn_smem_layout L;
        L.d = d;
        L.bc = bc;
        L.br = br;
        L.k_stride = d + 1;

        int at = 0;
        L.q_off = at;
        at += br * d;
        L.k_off = at;
        at += bc * L.k_stride;
        L.v_off = at;
        at += bc * L.k_stride;
        L.p_off = at;
        at += br * bc;
        if (backward)
        {
            L.ds_off = at;
            at += br * bc;
        }
        L.out_off = at;
        at += br * d;
        if (backward)
        {
            L.do_off = at;
            at += br * d;
            L.di_off = at;
            at += br;
        }
        else
        {
            L.m_off = at;
            at += br;
            L.l_off = at;
            at += br;
        }
        L.total = at;
        return L;
    }

    JAS_HD static attn_smem_layout forward_layout(int d, int bc, int br)
    {
        return make(d, bc, br, /*backward=*/false);
    }

    JAS_HD static attn_smem_layout backward_layout(int d, int bc, int br)
    {
        return make(d, bc, br, /*backward=*/true);
    }
};

/**
 * 前向：一趟扫过 key 分块，online softmax 边修正边累加。
 *
 * 形状约定与 `dev_head_gen_t::attend_impl` 完全一致：`q` 是 `(d × q_len)`、
 * `k`/`v` 是 `(d × len)`、输出是 `(d × q_len)`。scores 用**逐元素除法**除以 `√d`
 * （不是乘 `1/√d`），理由见 `jas_cuda_mha.hpp` 文件头：与主机逐元素对齐优先于省一趟缩放。
 *
 * 线程/数据映射：
 *   - 一个 warp 负责一个 query 行（块内 `warp` 号 → 行号）；
 *   - 打分时 lane 负责 key：`j = lane, lane+32, ...`，每个 lane 算一个完整内积；
 *   - 加权求和时 lane 负责特征：`f = lane, lane+32, ...`，对固定的 `f` 只在 j 方向求和，
 *     所以**不需要跨 lane 归约**。
 *
 * 越界的 warp（`i >= q_len`，只有最后一个 block 可能出现）不参与计算，但**必须**继续
 * 参与分块加载与 `__syncthreads` —— 提前 return 会让同块的其他 warp 卡在屏障上。
 */
template <typename T>
__global__ void fused_attention_fwd_kernel(dev_mat_t<T> q, dev_mat_t<T> k, dev_mat_t<T> v,
                                           T* __restrict__ out, T* __restrict__ lse, int bc,
                                           bool causal, int offset, T scale)
{
    using ::jasmine::detail::device_exp;
    using ::jasmine::detail::device_log;
    using ::jasmine::detail::device_max;

    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    const int d = q.row_num();
    const int q_len = q.col_num();
    const int len = k.col_num();

    const attn_smem_layout L = attn_smem_layout::forward_layout(d, bc, kAttnRowsPerBlock);
    T* s_q = smem + L.q_off;
    T* s_k = smem + L.k_off;
    T* s_v = smem + L.v_off;
    T* s_p = smem + L.p_off;
    T* s_out = smem + L.out_off;
    T* s_m = smem + L.m_off;
    T* s_l = smem + L.l_off;

    const T neg_inf = -std::numeric_limits<T>::infinity();

    const int lane = threadIdx.x % kAttnWarp;
    const int warp = threadIdx.x / kAttnWarp;
    const int i = blockIdx.x * kAttnRowsPerBlock + warp;  // 本 warp 负责的 query 行
    const bool active = (i < q_len);

    // ---- Q 分块落地 ----
    // 按「行号变化最慢」遍历，于是连续线程读的是相邻的列 —— 那才是 q 里连续的方向
    //（q 是 d × q_len 行优先，列方向步长为 1）。
    for (int idx = threadIdx.x; idx < kAttnRowsPerBlock * d; idx += blockDim.x)
    {
        const int row = idx / d;  // 块内第几个 query 行
        const int f = idx - row * d;
        const int col = blockIdx.x * kAttnRowsPerBlock + row;
        s_q[row * d + f] = (col < q_len) ? static_cast<T>(q(f, col)) : T(0);
    }

    if (active)
    {
        for (int f = lane; f < d; f += kAttnWarp)
            s_out[warp * d + f] = T(0);
        if (lane == 0)
        {
            s_m[warp] = neg_inf;
            s_l[warp] = T(0);
        }
    }
    __syncthreads();

    for (int base = 0; base < len; base += bc)
    {
        const int tile = (base + bc <= len) ? bc : (len - base);

        // 分成 (j × d) 存：后面「lane 读自己那个 j 的整行」与「lane 读自己那个 f 的整列」
        // 两种用法都要连续
        for (int idx = threadIdx.x; idx < tile * d; idx += blockDim.x)
        {
            const int j = idx / d;
            const int f = idx - j * d;
            s_k[j * L.k_stride + f] = static_cast<T>(k(f, base + j));
            s_v[j * L.k_stride + f] = static_cast<T>(v(f, base + j));
        }
        __syncthreads();

        if (active)
        {
            // ---- 打分：本 lane 负责的那几个 key，各算一个完整内积 ----
            T tile_max = neg_inf;
            for (int j = lane; j < tile; j += kAttnWarp)
            {
                T s = T(0);
                for (int f = 0; f < d; ++f)
                    s += s_q[warp * d + f] * s_k[j * L.k_stride + f];
                s /= scale;
                if (!attn_key_visible(i, base + j, causal, offset))
                    s = neg_inf;
                s_p[warp * bc + j] = s;
                tile_max = device_max(tile_max, s);
            }

            const T m_old = s_m[warp];
            const T m_new = device_max(m_old, warp_all_max(tile_max));

            // 修正量：老结果整体等比缩小。`m_old == -inf` 表示「此前一个合法位置都没有」，
            // 此时 `exp(-inf - m_new)` 是 nan，必须显式取 0（同一个坑在 softmax 的
            // 全 -inf 行里也踩过，见 jas_cuda_reduce.hpp）。
            const T corr = (m_old == neg_inf) ? T(0) : device_exp(m_old - m_new);

            // 输出累加器与归一化因子一起缩小。每个 lane 只碰自己负责的 f，不需要通信。
            for (int f = lane; f < d; f += kAttnWarp)
                s_out[warp * d + f] *= corr;

            // 上一轮的指数和**只由 lane 0 带过来**：`warp_all_sum` 会把每个 lane 的值都加起来，
            // 若人人都从 `s_l` 起累，那份和就被加了 32 遍。
            // 单块（bc >= len）时它是 0，所以这个 bug 只在**真的分了多块**时才现形 ——
            // 这也正是"分块大小必须对拍"这条测例存在的理由。
            T local_sum = (lane == 0) ? s_l[warp] * corr : T(0);
            for (int j = lane; j < tile; j += kAttnWarp)
            {
                const T p = (m_new == neg_inf) ? T(0) : device_exp(s_p[warp * bc + j] - m_new);
                s_p[warp * bc + j] = p;
                local_sum += p;
            }
            const T l_new = warp_all_sum(local_sum);

            // ---- 加权求和：每个 lane 负责 d 里的一段特征，扫过整块的 j ----
            for (int f = lane; f < d; f += kAttnWarp)
            {
                T acc = T(0);
                for (int j = 0; j < tile; ++j)
                    acc += s_p[warp * bc + j] * s_v[j * L.k_stride + f];
                s_out[warp * d + f] += acc;
            }

            if (lane == 0)
            {
                s_m[warp] = m_new;
                s_l[warp] = l_new;
            }
        }
        // 下一轮会覆盖 s_k / s_v，必须等所有 warp 用完
        __syncthreads();
    }

    if (active)
    {
        __syncwarp();  // s_m / s_l 是 lane 0 写的，同 warp 内先同步再读
        const T m = s_m[warp];
        const T l = s_l[warp];
        const T inv = (l > T(0)) ? (T(1) / l) : T(0);
        for (int f = lane; f < d; f += kAttnWarp)
            out[static_cast<std::ptrdiff_t>(f) * q_len + i] = s_out[warp * d + f] * inv;
        if (lane == 0)
            lse[i] = (l > T(0)) ? (m + device_log(l)) : neg_inf;
    }
}

/**
 * 反向：从 `L = logsumexp` 重算概率，逐块累加 dQ/dK/dV。
 *
 *   p     = exp(s − L)                  ← 重算，不读存下来的概率矩阵
 *   dP    = Σ_f dO(f,i)·V(f,j)
 *   D_i   = Σ_f dO(f,i)·O(f,i)          ← 恒等式 Σ_j p·dP = dO·O，不必重扫整行
 *   dS    = p ⊙ (dP − D)
 *   dV   += p ⊗ dO
 *   dQ   += dS ⊗ K / √d
 *   dK   += dS ⊗ Q / √d
 *
 * dQ 的累加留在共享内存（同一 query 行的所有块贡献都落在同一个 block 内），
 * dK/dV 则跨 block 累加（同一列 key 被多个 query 行用到），所以走 `atomicAdd`。
 */
template <typename T>
__global__ void fused_attention_bwd_kernel(dev_mat_t<T> q, dev_mat_t<T> k, dev_mat_t<T> v,
                                           const T* __restrict__ out, const T* __restrict__ lse,
                                           dev_mat_t<T> d_out, dev_mat_t<T> d_q, dev_mat_t<T> d_k,
                                           dev_mat_t<T> d_v, int bc, bool causal, int offset,
                                           T scale)
{
    using ::jasmine::detail::device_exp;

    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    const int d = q.row_num();
    const int q_len = q.col_num();
    const int len = k.col_num();

    const attn_smem_layout L = attn_smem_layout::backward_layout(d, bc, kAttnRowsPerBlock);
    T* s_q = smem + L.q_off;
    T* s_k = smem + L.k_off;
    T* s_v = smem + L.v_off;
    T* s_p = smem + L.p_off;
    T* s_ds = smem + L.ds_off;
    T* s_dq = smem + L.out_off;
    T* s_do = smem + L.do_off;
    T* s_di = smem + L.di_off;

    const T neg_inf = -std::numeric_limits<T>::infinity();

    const int lane = threadIdx.x % kAttnWarp;
    const int warp = threadIdx.x / kAttnWarp;
    const int i = blockIdx.x * kAttnRowsPerBlock + warp;
    const bool active = (i < q_len);

    T* do_row = s_do + warp * d;  // 本 warp 私有的 dO 列暂存

    for (int idx = threadIdx.x; idx < kAttnRowsPerBlock * d; idx += blockDim.x)
    {
        const int row = idx / d;
        const int f = idx - row * d;
        const int col = blockIdx.x * kAttnRowsPerBlock + row;
        s_q[row * d + f] = (col < q_len) ? static_cast<T>(q(f, col)) : T(0);
    }

    if (active)
    {
        T* my_dq = s_dq + warp * d;
        for (int f = lane; f < d; f += kAttnWarp)
            my_dq[f] = T(0);
        for (int f = lane; f < d; f += kAttnWarp)
            do_row[f] = static_cast<T>(d_out(f, i));
        __syncwarp();

        // D_i = Σ_f dO(f,i)·O(f,i)：只与 query 行有关，进块循环前算一次
        T local_d = T(0);
        for (int f = lane; f < d; f += kAttnWarp)
            local_d += do_row[f] * out[static_cast<std::ptrdiff_t>(f) * q_len + i];
        const T di = warp_all_sum(local_d);
        if (lane == 0)
            s_di[warp] = di;
    }
    __syncthreads();

    for (int base = 0; base < len; base += bc)
    {
        const int tile = (base + bc <= len) ? bc : (len - base);

        for (int idx = threadIdx.x; idx < tile * d; idx += blockDim.x)
        {
            const int j = idx / d;
            const int f = idx - j * d;
            s_k[j * L.k_stride + f] = static_cast<T>(k(f, base + j));
            s_v[j * L.k_stride + f] = static_cast<T>(v(f, base + j));
        }
        __syncthreads();

        if (active)
        {
            const T lse_i = lse[i];
            const T di = s_di[warp];
            T* my_ds = s_ds + warp * bc;
            T* my_p = s_p + warp * bc;

            // ---- 每个 lane 负责若干个 key：重算 s 与 p，再算 dP、dS ----
            for (int j = lane; j < tile; j += kAttnWarp)
            {
                T s = T(0);
                for (int f = 0; f < d; ++f)
                    s += s_q[warp * d + f] * s_k[j * L.k_stride + f];
                s /= scale;

                const bool visible = attn_key_visible(i, base + j, causal, offset);
                // 被屏蔽的位置 p = 0，于是 dS 也自然为 0 —— 与主机端
                // 「前向填 -inf、反向把 delta_scores 清零」是同一件事的两面。
                // lse 为 -inf（整行被屏蔽）时 exp(s + inf) 会变 inf，所以显式挡住。
                const T p = (visible && lse_i != neg_inf) ? device_exp(s - lse_i) : T(0);
                my_p[j] = p;

                T dp = T(0);
                if (p != T(0))
                    for (int f = 0; f < d; ++f)
                        dp += do_row[f] * s_v[j * L.k_stride + f];
                my_ds[j] = p * (dp - di);
            }
            __syncwarp();

            // ---- dQ：按 f 分段，扫过整块的 j ----
            T* my_dq = s_dq + warp * d;
            for (int f = lane; f < d; f += kAttnWarp)
            {
                T acc = T(0);
                for (int j = 0; j < tile; ++j)
                    acc += my_ds[j] * s_k[j * L.k_stride + f];
                my_dq[f] += acc;
            }

            // ---- dK / dV：按 (f, j) 写出，跨 block 累加 ----
            for (int f = lane; f < d; f += kAttnWarp)
            {
                const T qf = s_q[warp * d + f];
                const T dof = do_row[f];
                T* dk_row = d_k.m_data + static_cast<std::ptrdiff_t>(f) * d_k.leading_dim();
                T* dv_row = d_v.m_data + static_cast<std::ptrdiff_t>(f) * d_v.leading_dim();
                for (int j = 0; j < tile; ++j)
                {
                    const T ds = my_ds[j];
                    const T p = my_p[j];
                    // 被掩码的位置两个量都是 0，不必去撞原子操作
                    if (ds == T(0) && p == T(0))
                        continue;
                    // dK 要跨 block 累加，除以 √d 只能逐项做 —— 跨块的和没法在最后统一除。
                    // 与主机「先 matmul 再 divide_inplace」相比只差一个舍入顺序（ulp 级）。
                    if (ds != T(0))
                        atomicAdd(dk_row + base + j, ds * qf / scale);
                    if (p != T(0))
                        atomicAdd(dv_row + base + j, p * dof);
                }
            }
        }
        __syncthreads();
    }

    if (active)
    {
        // dQ 只归本 block 所有，直接写。除以 √d 放在**求和之后**一次做，
        // 与主机端「先 matmul 再 divide_inplace」的顺序一致。
        __syncwarp();
        T* my_dq = s_dq + warp * d;
        T* dq_col = d_q.m_data + static_cast<std::ptrdiff_t>(i);
        for (int f = lane; f < d; f += kAttnWarp)
            dq_col[static_cast<std::ptrdiff_t>(f) * d_q.leading_dim()] = my_dq[f] / scale;
    }
}

/**
 * 选一个能塞进共享内存的 key 分块大小。
 *
 * 分块越大，K/V 从全局内存读的遍数越少；但共享内存是硬上限（这里按 48 KiB 算，
 * 没有走 `cudaFuncSetAttribute` 的 opt-in）。一块都放不下就报错，而不是静默算错 ——
 * 分块大小只影响速度、不影响结果，所以这里选择把「放不下」这种事说出来。
 */
template <typename T>
int choose_block_cols(int d, int br, int want)
{
    const std::size_t budget = 48u * 1024u;
    // 与 bc 无关的部分：q(br×d) + out(br×d) + do(br×d) + m/l 或 di(br×2) + 一点点余量
    const std::size_t fixed =
        static_cast<std::size_t>(3 * br * d + 2 * br + 8) * sizeof(T);
    if (fixed >= budget || d <= 0 || br <= 0)
        return 0;
    // 每多一列 key：K、V 各一行（带 pad），外加 p 与 dS 各一格（反向那套更大，按大的算）
    const std::size_t per_col = static_cast<std::size_t>(2 * (d + 1) + 2 * br) * sizeof(T);
    const std::size_t room = budget - fixed;
    const int room_cols = static_cast<int>(room / per_col);
    int bc = (want < room_cols) ? want : room_cols;
    return bc > 0 ? bc : 0;
}

} // namespace detail

/**
 * 融合注意力的前向产物：**每行一个数**的 logsumexp，加输出本身。
 *
 * 与非融合路径的差别一眼可见：那边要把 `(q_len × len)` 的概率矩阵一直留到反向
 * （`dev_head_gen_t::m_weights`），这里只有 `q_len` 个数。
 */
template <typename T>
struct attention_cache_t
{
    dev_matrix_t<T> out;        // (d_head × q_len)
    dev_matrix_t<T> logsumexp;  // (q_len × 1)

    std::size_t bytes() const
    {
        return static_cast<std::size_t>(out.row_num()) * static_cast<std::size_t>(out.col_num())
                   * sizeof(T)
               + static_cast<std::size_t>(logsumexp.row_num()) * sizeof(T);
    }

    /** 非融合路径在这个形状下要额外留住的字节数：`q_len × len × sizeof(T)`。 */
    static std::size_t weights_bytes(int q_len, int len)
    {
        return static_cast<std::size_t>(q_len) * static_cast<std::size_t>(len) * sizeof(T);
    }
};

/** 反向的三块梯度，与 `dev_head_gen_t::bwd_pack_t` 同形。 */
template <typename T>
struct attention_grad_t
{
    dev_matrix_t<T> d_q;  // (d_head × q_len)
    dev_matrix_t<T> d_k;  // (d_head × len)
    dev_matrix_t<T> d_v;  // (d_head × len)
};

/**
 * 追踪融合路径实际被启动了几次。
 *
 * 与 `softmax_shared_launch_count` 同一个理由：优化最容易出的问题不是算错，而是
 * **根本没生效**（开关没传下去、分支写反），此时结果照样正确、测例照样全绿。
 */
inline int& flash_attention_launch_count()
{
    static int n = 0;
    return n;
}

/** 把 key 分块大小钉成某个值（<=0 表示自动）。测试用它跑到退化路径上（bc = 1、bc = len）。 */
inline int& flash_attention_block_cols_override()
{
    static int v = 0;
    return v;
}

/**
 * 上一次启动**实际**用的 key 分块大小。
 *
 * 与 `softmax_shared_launch_count` 同一个用途：优化的典型失败方式不是算错，而是
 * 压根没生效（分块被算成"一整行"、或者被上限压成 1 列），此时结果照旧正确。
 * 有了这个数，测例才能断言「分块真的把序列切成了多段」。
 */
inline int& flash_attention_last_block_cols()
{
    static int v = 0;
    return v;
}

/**
 * 融合注意力前向。形状与语义与 `dev_head_gen_t::attend_impl` 一致：
 * `q` 是 `(d × q_len)`、`k`/`v` 是 `(d × len)`，返回 `(d × q_len)`。
 *
 * `causal` 决定是否按 `j <= i + offset` 屏蔽；与主机一致：`q_len == 1` 时不存在未来，
 * 掩码是恒等操作，所以调用方只在 `q_len > 1` 时才传 `causal = true`
 * （这一判断在 `dev_head_gen_t` 里，与它传给非融合路径的 `mask_offset` 同一个判断）。
 */
template <typename T>
attention_cache_t<T> fused_attention_forward(const dev_mat_t<T>& q, const dev_mat_t<T>& k,
                                             const dev_mat_t<T>& v, bool causal, int offset)
{
    using ::jasmine::detail::device_sqrt;

    if (q.row_num() <= 0 || q.col_num() <= 0)
        throw std::invalid_argument("fused_attention_forward: q 形状为空");
    if (k.row_num() != q.row_num() || v.row_num() != q.row_num())
        throw std::invalid_argument("fused_attention_forward: Q/K/V 的 d_head 必须一致");
    if (v.col_num() != k.col_num())
        throw std::invalid_argument("fused_attention_forward: K/V 的长度必须一致");

    const int d = q.row_num();
    const int q_len = q.col_num();
    const int len = k.col_num();

    attention_cache_t<T> cache;
    cache.out.allocate(d, q_len);
    cache.logsumexp.allocate(q_len, 1);
    if (len == 0)
        return cache;

    const int br = detail::kAttnRowsPerBlock;
    int bc = flash_attention_block_cols_override();
    if (bc <= 0)
        bc = detail::choose_block_cols<T>(d, br, len);
    if (bc <= 0)
        throw std::runtime_error("fused_attention_forward: d_head = " + std::to_string(d)
                                 + " 太大，一个 key 分块都放不进共享内存");
    if (bc > len)
        bc = len;

    const detail::attn_smem_layout L = detail::attn_smem_layout::forward_layout(d, bc, br);
    const std::size_t smem = static_cast<std::size_t>(L.total) * sizeof(T);

    const int blocks = (q_len + br - 1) / br;
    const int threads = br * detail::kAttnWarp;

    detail::fused_attention_fwd_kernel<T><<<blocks, threads, smem>>>(
        q, k, v, cache.out.buffer().data(), cache.logsumexp.buffer().data(), bc, causal, offset,
        device_sqrt(static_cast<T>(d)));
    JAS_CUDA_CHECK(cudaGetLastError());
    ++flash_attention_launch_count();
    flash_attention_last_block_cols() = bc;
    return cache;
}

/**
 * 融合注意力反向。
 *
 * 输入：前向的 `cache`（只要 `logsumexp` 与 `out`）、Q/K/V（前向留下的那几份即可，
 * 与主机 `mat_head_gen_t` 的契约一致）、以及上游梯度 `d_out`（`d × q_len`）。
 *
 * `d_out` 可以是任意可求值的表达式或主机矩阵（走共用的 `detail::materialize_input`）；
 * 块循环里要反复读它的同一列，所以先物化成拥有者，免得反复重算同一棵树。
 */
template <typename T, typename Delta>
attention_grad_t<T> fused_attention_backward(const attention_cache_t<T>& cache,
                                             const dev_mat_t<T>& q, const dev_mat_t<T>& k,
                                             const dev_mat_t<T>& v, const Delta& d_out, bool causal,
                                             int offset)
{
    using ::jasmine::detail::device_sqrt;

    const int d = q.row_num();
    const int q_len = q.col_num();
    const int len = k.col_num();

    if (cache.logsumexp.row_num() != q_len)
        throw std::invalid_argument("fused_attention_backward: logsumexp 的列数与前向的 q_len "
                                    "不一致（forward 必须先被调用）");
    if (d_out.row_num() != d || d_out.col_num() != q_len)
        throw std::invalid_argument("fused_attention_backward: d_out 形状应为 (d_head × q_len)");

    attention_grad_t<T> g;
    g.d_q.allocate(d, q_len);
    g.d_k.allocate(d, len);
    g.d_v.allocate(d, len);
    if (len == 0 || q_len == 0)
        return g;

    g.d_q.buffer().zero();
    g.d_k.buffer().zero();
    g.d_v.buffer().zero();

    dev_matrix_t<T> dout;
    detail::materialize_input(d_out, dout);

    const int br = detail::kAttnRowsPerBlock;
    int bc = flash_attention_block_cols_override();
    if (bc <= 0)
        bc = detail::choose_block_cols<T>(d, br, len);
    if (bc <= 0)
        throw std::runtime_error("fused_attention_backward: d_head 太大，共享内存放不下一个分块");
    if (bc > len)
        bc = len;

    const detail::attn_smem_layout L = detail::attn_smem_layout::backward_layout(d, bc, br);
    const std::size_t smem = static_cast<std::size_t>(L.total) * sizeof(T);

    const int blocks = (q_len + br - 1) / br;
    const int threads = br * detail::kAttnWarp;

    detail::fused_attention_bwd_kernel<T><<<blocks, threads, smem>>>(
        q, k, v, cache.out.const_leaf().m_data, cache.logsumexp.const_leaf().m_data,
        dout.const_leaf(), g.d_q.leaf(), g.d_k.leaf(), g.d_v.leaf(), bc, causal, offset,
        device_sqrt(static_cast<T>(d)));
    JAS_CUDA_CHECK(cudaGetLastError());
    return g;
}

} // namespace cuda
} // namespace jasmine

#endif
