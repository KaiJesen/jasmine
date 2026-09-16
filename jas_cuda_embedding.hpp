#ifndef __JAS_CUDA_EMBEDDING_HPP__
#define __JAS_CUDA_EMBEDDING_HPP__

/**
 * 设备端 embedding：`embedding_net_t` 的 GPU 对应物。
 *
 * 权重布局与主机**完全一致**（`d_model × vocab`，第 `id` 列就是该 token 的向量），
 * 前向是逐列 gather。这一步没什么可融合的 —— 它本来就是纯数据搬运，
 * 而设备端的内存带宽正是它的主场（主机端那种逐元素双重循环在这里变成一次 launch）。
 *
 * ## 反向为什么和主机长得不一样
 *
 * 主机 `embedding_net_t::backward` 的写法是：
 *
 *   grad_w = zeros(d_model, vocab);          // 每步**新分配**一张全尺寸梯度
 *   for t, for i: grad_w(i, ids[t]) += delta(i, t);
 *   updator.update(grad_w, weight);
 *   return 空矩阵（离散 id 不回传梯度）
 *
 * 小词表没问题，TinyLlama 那种规模（2048 × 32000 × 8B）光是这张梯度就是 **524 MB**，
 * 而且每步都要分配 + 清零 —— 这是主机实现的已知代价。设备端做两件事把它压下来：
 *
 *   1. **梯度缓冲跨步复用**：`allocate` 只在尺寸变化时真正 `cudaMalloc`，
 *      每步只 `zero()` 一次（一次 `cudaMemset`，走带宽，很快）。
 *   2. 累加用 `atomicAdd`：同一个 token 在一段里出现多次时，各次出现都往同一列累加。
 *
 * 但**形状仍然是全尺寸的** —— 这一步没有变魔术：它保留了「一个通用更新器 + 一张
 * 稠密梯度」这套主机语义，代价是显存 O(d_model × vocab)。真正的解法是把更新也变成
 * 稀疏的（只碰出现过的 id，按 `(i, id)` 索引动量），那要三个优化器各加一套 scatter
 * 入口，属于「训练吞吐」那一轮的活。见 CUDA.md 第 9.5 节的记载。
 *
 * ## 输入 id 的合法性
 *
 * 主机在 `forward` 里逐个检查 `0 <= id < vocab`。设备端如果在 kernel 里检查，
 * 报告错误要一次 D2H 同步，不如把检查放在**前向的入口**上做：id 只有 `1 × T` 个
 * （T 是最长序列，几百到几千），代价是 O(T)。换来的是「越界 id 不可能
 * 变成越界访存」这条硬保证 —— 静默读错内存比多一次同步糟糕得多。
 *
 * 检查在哪一侧做，取决于 id 是主机矩阵还是设备矩阵（`materialize_ids` 的分流）：
 * 主机矩阵（tokenizer 出来的那一类）**上传前就直接查**，连下载都省了；
 * 设备端数据则要下载回主机查一遍。两条路都不让越界 id 走到 kernel 里。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

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
#include "jas_cuda_net.hpp"
#include "jas_cuda_updator.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_mat_t.hpp"

namespace jasmine {
namespace cuda {

namespace detail {

/**
 * 逐列 gather：`out(:, t) = weight(:, ids[t])`。
 *
 * 线程下标按**输出**的行优先序（`idx = i * T + t`）：这样写出去的是连续地址
 * （合并写），读进来的 `weight(i, id_t)` 是散的 —— 但那是 gather 的本质，
 * 除非把权重转置成 `(vocab × d_model)` 存储。转置的好处留给将来（`lm_head`
 * 本来就是这个方向，`wte` 不是；两者绑定时可以共享一份）。
 */
template <typename T>
__global__ void embedding_gather_kernel(const T* __restrict__ weight, int vocab,
                                        const T* __restrict__ ids, T* __restrict__ out, int seq,
                                        int d_model)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_model * seq)
        return;
    const int i = idx / seq;
    const int t = idx - i * seq;
    const int id = static_cast<int>(ids[t]);
    out[idx] = weight[static_cast<std::ptrdiff_t>(i) * vocab + id];
}

/** 反向：把 delta 的每一列散射累加到 token 对应的那一列上。 */
template <typename T>
__global__ void embedding_scatter_kernel(const T* __restrict__ delta, const T* __restrict__ ids,
                                         T* __restrict__ grad, int vocab, int seq, int d_model)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_model * seq)
        return;
    const int i = idx / seq;
    const int t = idx - i * seq;
    const int id = static_cast<int>(ids[t]);
    atomicAdd(&grad[static_cast<std::ptrdiff_t>(i) * vocab + id], delta[idx]);
}

} // namespace detail

/**
 * 离散 token → 稠密向量。权重 `(d_model × vocab)`，与主机同布局、同语义。
 *
 * 与主机一样，`backward` **不对 id 回传梯度**（离散变量没有梯度），
 * 返回一个 `(1 × T)` 的空占位矩阵。
 */
template <typename T, template <typename> class updator_type>
class dev_embedding_t
{
public:
    using ele_type = T;

    dev_embedding_t() = default;
    dev_embedding_t(int vocab, int d_model) { set_param(vocab, d_model); }

    /** 分配权重（不初始化数值；用 `init_weight` 或 `upload_weight` 填）。 */
    void set_param(int vocab, int d_model)
    {
        if (vocab <= 0 || d_model <= 0)
            throw std::invalid_argument("dev_embedding_t: 尺寸必须为正");
        m_vocab = vocab;
        m_d_model = d_model;
        m_weight.allocate(d_model, vocab);
        m_ids.allocate(1, 0);
    }

    /** 与主机 `reinit({vocab, d_model})` 同义。 */
    void reinit(const std::vector<int>& container)
    {
        if (container.size() < 2)
            throw std::invalid_argument("dev_embedding_t::reinit: 需要 {vocab, d_model}");
        set_param(container[0], container[1]);
    }

    void upload_weight(const mat_t<T>& host)
    {
        if (host.row_num() != m_d_model || host.col_num() != m_vocab)
            throw std::invalid_argument("dev_embedding_t::upload_weight: 形状应为 (d_model × vocab)");
        m_weight.allocate(m_d_model, m_vocab);
        m_weight.upload(host);
    }

    mat_t<T> weight_to_host() const { return m_weight.download(); }

    template <typename init_type>
    void init_weight()
    {
        detail::init_device_matrix<init_type, T>(m_weight, m_d_model, m_vocab);
    }

    /**
     * `ids` 形状 `1 × T`（元素是 token id，存于 `T`，与主机同一约定）。
     * 返回 `(d_model × T)`。
     *
     * 接受两类输入，分流方式和其它层一致（见 `jas_cuda_net.hpp` 里那段
     * 「`mat_t` 与 `dev_matrix_t` 参与表达式的不对称」）：
     *
     *   - **主机 `mat_t`** —— token id 本来就是从主机上的 tokenizer 来的，
     *     所以这是最常见的一种。走「直接上传 + 在主机矩阵上做越界检查」，
     *     连那次下载都省了。
     *   - 设备叶子 / 拥有者 / 表达式 —— 走共用的 `materialize_input`，检查需要下载回来。
     */
    template <typename Ids>
    dev_matrix_t<T> forward(const Ids& ids)
    {
        if (m_d_model <= 0)
            throw std::runtime_error("dev_embedding_t::forward: 还没 set_param");
        if (ids.row_num() != 1)
            throw std::invalid_argument("dev_embedding_t::forward: ids 必须是 1 × T");

        materialize_ids(ids);
        const int seq = m_ids.col_num();
        dev_matrix_t<T> out(m_d_model, seq);
        if (seq <= 0)
            return out;

        const int total = m_d_model * seq;
        constexpr int kThreads = 256;
        const int blocks = (total + kThreads - 1) / kThreads;
        detail::embedding_gather_kernel<T><<<blocks, kThreads>>>(m_weight.const_leaf().m_data,
                                                                 m_vocab,
                                                                 m_ids.const_leaf().m_data,
                                                                 out.buffer().data(), seq, m_d_model);
        JAS_CUDA_CHECK(cudaGetLastError());
        return out;
    }

    template <typename Ids>
    dev_matrix_t<T> forward_one(const Ids& ids)
    {
        return forward(ids);
    }

    /**
     * 反向：`grad_w(d_model × vocab)` 就地累加，然后交给更新器。
     *
     * 返回值是 `(1 × T)` 的占位（与主机一致：离散 id 不回传梯度）。**调用方不应依赖它**。
     */
    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        if (delta.row_num() != m_d_model || delta.col_num() != m_ids.col_num())
            throw std::runtime_error("dev_embedding_t::backward: delta 形状应为 (d_model × T)");

        const int seq = m_ids.col_num();
        dev_matrix_t<T> placeholder(1, seq);
        if (seq <= 0)
            return placeholder;

        // 梯度缓冲跨步复用：allocate 在尺寸不变时是空操作，这里只是首次分配
        m_d_weight.allocate(m_d_model, m_vocab);
        m_d_weight.buffer().zero();

        const int total = m_d_model * seq;
        constexpr int kThreads = 256;
        const int blocks = (total + kThreads - 1) / kThreads;
        detail::embedding_scatter_kernel<T><<<blocks, kThreads>>>(delta.const_leaf().m_data,
                                                                  m_ids.const_leaf().m_data,
                                                                  m_d_weight.buffer().data(),
                                                                  m_vocab, seq, m_d_model);
        JAS_CUDA_CHECK(cudaGetLastError());

        m_updator.update(m_d_weight.const_leaf(), m_weight.buffer());
        return placeholder;
    }

    void step() { m_updator.step(); }
    void set_lr(T lr) { m_updator.set_lr(lr); }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_updator.set(std::forward<arg_types>(args)...);
    }

    dev_matrix_t<T>& weight() { return m_weight; }
    const dev_matrix_t<T>& weight() const { return m_weight; }
    dev_matrix_t<T>& cached_ids() { return m_ids; }
    dev_matrix_t<T>& cached_grad() { return m_d_weight; }
    int vocab_size() const { return m_vocab; }
    int d_model() const { return m_d_model; }

    /**
     * 稠密梯度缓冲的字节数。
     *
     * 这是本层的**显存代价**，调用方（或测试）应当知道它有多大：
     * `d_model × vocab × sizeof(T)`。要换成稀疏更新看文件头那段说明。
     */
    std::size_t gradient_bytes() const
    {
        return static_cast<std::size_t>(m_d_model) * static_cast<std::size_t>(m_vocab) * sizeof(T);
    }

    std::string net_type(int indent = 0) const
    {
        return std::string(static_cast<std::size_t>(indent) * 4, ' ') + "dev_embedding_t(vocab:"
               + std::to_string(m_vocab) + ", d_model:" + std::to_string(m_d_model) + ")";
    }

private:
    /** id 合法性检查：id 只有 1×T 个，检查一遍比在 kernel 里 OOB 划算。 */
    void check_ids(const dev_matrix_t<T>& ids, int seq)
    {
        std::vector<T> host(static_cast<std::size_t>(seq));
        ids.buffer().download(host.data(), host.size());
        check_ids(host.data(), seq);
    }

    void check_ids(const T* host, int seq) const
    {
        for (int t = 0; t < seq; ++t)
        {
            const int id = static_cast<int>(host[static_cast<std::size_t>(t)]);
            if (id < 0 || id >= m_vocab)
                throw std::out_of_range("dev_embedding_t: token id " + std::to_string(id)
                                        + " 越界（vocab = " + std::to_string(m_vocab) + "）");
        }
    }

    /** 分流交给共用的 `materialize_input`，本层只多加一道「先查越界再上传」。 */
    template <typename Ids>
    void materialize_ids(const Ids& ids)
    {
        if constexpr (std::is_same_v<std::remove_cvref_t<Ids>, mat_t<T>>)
        {
            check_ids(ids.data(), ids.col_num());  // 就地报错，连上传都省了
            detail::materialize_input(ids, m_ids);
        }
        else
        {
            detail::materialize_input(ids, m_ids);
            check_ids(m_ids, m_ids.col_num());
        }
    }

    dev_matrix_t<T> m_weight;    // (d_model × vocab)
    dev_matrix_t<T> m_ids;       // (1 × T)，前向缓存
    dev_matrix_t<T> m_d_weight;  // (d_model × vocab)，跨步复用的稠密梯度
    updator_type<T> m_updator;
    int m_vocab = 0;
    int m_d_model = 0;
};

} // namespace cuda
} // namespace jasmine

#endif
