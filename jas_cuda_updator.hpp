#ifndef __JAS_CUDA_UPDATOR_HPP__
#define __JAS_CUDA_UPDATOR_HPP__

/**
 * 设备端参数更新器：`jas_updator_t.hpp` 里 sgd / adam / nadam 的 GPU 对应物。
 *
 * ## 一个重要的结构性差别
 *
 * 主机端更新器是**纯数值对象**：`update(grad, mat)` 直接对 `mat_t` 做算术。
 * 设备端不是 —— 它的算术跑在 GPU 上，所以更新器必然是一个**主机侧对象**，
 * 手里握着显存缓冲（adam 的 `m_m` / `m_v`），`update` 是一次 kernel 启动。
 *
 * 这跟 `dev_matrix_t` 的定位是一样的：主机侧是「拥有者 + 启动器」，设备侧只有数据。
 * 也正因如此，更新器**不需要**能把对象本身传进 kernel，所以它不是 POD，
 * 也不需要满足任何设备端的类型约束。
 *
 * ## 为什么是专用 kernel 而不是 `eval_fused`
 *
 * `eval_fused` 把结果写进**新分配的**缓冲，而参数更新是**原地**写回已有参数
 * （权重缓冲的地址不能变，否则之前取出的叶子全部悬空，KV cache 那套教训）。
 * 所以这里自己开 kernel，就地读写参数与动量。
 *
 * 顺带一个好处：整套更新（一阶矩、二阶矩、偏差修正、参数写入）能合成**一个**
 * kernel、一趟显存读写。主机端那套写法会物化 `m_hat`、`v_hat` 等一串临时矩阵，
 * 在设备上那都是多余的显存往返 —— 而更新对每个参数都发生在每个 step 上。
 *
 * ## 与主机端的语义契约
 *
 * 公式**逐字**照抄主机实现（含偏差修正的更新时机、nadam 的系数组合），
 * 所以同一串梯度喂给两边会得到逐元素相同的结果。测例就是这个对拍。
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_compat.hpp"
#include "jas_cuda_leaf.hpp"

namespace jasmine {
namespace cuda {
namespace detail {

/** 每块线程数；参数更新是纯访存受限的一趟，块大小对结果无影响。 */
constexpr int kUpdateThreads = 256;

inline int update_blocks(int n)
{
    return (n + kUpdateThreads - 1) / kUpdateThreads;
}

/** 一维索引守卫；所有更新 kernel 共用。 */
JAS_DEV inline bool update_in_range(int i, int n)
{
    return i < n;
}

/** `p -= lr * g` */
template <typename T>
__global__ void sgd_update_kernel(const T* __restrict__ g, T* __restrict__ p, int n, T lr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (update_in_range(i, n))
        p[i] = p[i] - lr * g[i];
}

/**
 * Adam：一趟算完一阶矩、二阶矩、偏差修正与参数写入。
 *
 * `bc1` / `bc2` 是主机算好的 `1 - beta^t`（主机实现就是在 update 里累乘这两个量），
 * 传进来而不是在设备端用 pow 重算，既省设备端的一个幂运算，也让「与主机同序」这件事
 * 在代码上直接可见 —— 主机是先累乘 beta_t 再取 1-beta_t 做修正。
 */
template <typename T>
__global__ void adam_update_kernel(const T* __restrict__ g, T* __restrict__ p,
                                   T* __restrict__ m, T* __restrict__ v, int n, T lr, T b1, T b2,
                                   T bc1, T bc2, T eps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!update_in_range(i, n))
        return;

    const T gi = g[i];
    const T mi = b1 * m[i] + (T(1) - b1) * gi;
    const T vi = b2 * v[i] + (T(1) - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;

    const T m_hat = mi / bc1;
    const T v_hat = vi / bc2;
    p[i] = p[i] - lr * m_hat / (::jasmine::detail::device_sqrt(v_hat) + eps);
}

/**
 * NAdam：与 Adam 的差别只在梯度那一项换成「momentum 与当前梯度的组合」。
 *
 * 主机端写作
 *     mtv_n = lr * ( b1*m_hat/(1 - b1_t*b1) + (1-b1)/(1-b1_t) * g )
 * 所以 `b1t` 必须原样传进来（`1 - b1_t*b1` 里的 `b1_t` 无法由 `bc1` 反推）。
 */
template <typename T>
__global__ void nadam_update_kernel(const T* __restrict__ g, T* __restrict__ p,
                                    T* __restrict__ m, T* __restrict__ v, int n, T lr, T b1, T b2,
                                    T b1t, T b2t, T eps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!update_in_range(i, n))
        return;

    const T gi = g[i];
    const T mi = b1 * m[i] + (T(1) - b1) * gi;
    const T vi = b2 * v[i] + (T(1) - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;

    const T bc1 = T(1) - b1t;
    const T bc2 = T(1) - b2t;
    const T m_hat = mi / bc1;
    const T v_hat = vi / bc2;

    const T mtv_n = lr * (b1 * m_hat / (T(1) - b1t * b1) + (T(1) - b1) / bc1 * gi);
    p[i] = p[i] - mtv_n / (::jasmine::detail::device_sqrt(v_hat) + eps);
}

/** 梯度累积：`cache = cache * (1 - 1/(n+1)) + g/(n+1)`，与主机 `cache_updator_t` 同式。 */
template <typename T>
__global__ void accumulate_kernel(const T* __restrict__ g, T* __restrict__ cache, int n, T inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (update_in_range(i, n))
        cache[i] = cache[i] * (T(1) - inv) + g[i] * inv;
}

inline void check_update_shapes(int grad_count, const void* param_data, std::size_t param_size,
                                const char* who)
{
    if (param_data == nullptr)
        throw std::invalid_argument(std::string(who) + ": 参数缓冲区是空的");
    if (static_cast<std::size_t>(grad_count) != param_size)
        throw std::invalid_argument(std::string(who) + ": 梯度元素数 " + std::to_string(grad_count)
                                    + " 与参数元素数 " + std::to_string(param_size) + " 不一致");
}

} // namespace detail

/**
 * 随机梯度下降：`p -= lr * g`。
 *
 * 无状态，所以除了学习率什么都没有 —— 与主机 `sgd_t` 一样。
 */
template <typename T>
class dev_sgd_t
{
public:
    explicit dev_sgd_t(T learning_rate = T(0.001)) : m_lr(learning_rate) {}

    void set(T learning_rate = T(0.001)) { m_lr = learning_rate; }
    void set_lr(T lr) { m_lr = lr; }
    T learning_rate() const { return m_lr; }

    /**
     * 原地更新参数。`grad` 是薄壳，`param` 是参数自己的缓冲区。
     *
     * 传缓冲区而不是 `dev_matrix_t`，是为了同时覆盖两类参数：
     * 权重是 (out × in) 的矩阵，而 bias / gamma / beta 在主机端是 [d, 1] 的列向量
     * （设备端存在 `dev_colvec_t` 里）。两者的共同点只有「一段连续显存 + 元素个数」。
     */
    void update(const dev_mat_t<T>& grad, dev_buf_t<T>& param)
    {
        const int n = grad.row_num() * grad.col_num();
        detail::check_update_shapes(n, param.data(), param.size(), "dev_sgd_t::update");
        if (n == 0)
            return;

        detail::sgd_update_kernel<T><<<detail::update_blocks(n), detail::kUpdateThreads>>>(
            grad.m_data, param.data(), n, m_lr);
        JAS_CUDA_CHECK(cudaGetLastError());
    }

    /** 无延迟更新，与主机 `sgd_t` 一致。 */
    void step() {}

private:
    T m_lr;
};

/**
 * Adam。动量缓冲按第一次梯度惰性分配 —— 与主机 `adam_t` 的 `m_m.valid() == false` 分支同义。
 */
template <typename T>
class dev_adam_t
{
public:
    dev_adam_t(T learning_rate = T(0.001), T beta1 = T(0.9), T beta2 = T(0.999),
               T epsilon = T(1e-8))
        : m_lr(learning_rate), m_b1(beta1), m_b2(beta2), m_eps(epsilon)
    {
    }

    void set(T learning_rate = T(0.001), T beta1 = T(0.9), T beta2 = T(0.999),
             T epsilon = T(1e-8))
    {
        m_lr = learning_rate;
        m_b1 = beta1;
        m_b2 = beta2;
        m_eps = epsilon;
        // 与主机 set() 一致：重置时间步与累积的 beta^t、丢掉动量
        m_t = 0;
        m_b1_t = T(1);
        m_b2_t = T(1);
        m_m.release();
        m_v.release();
    }

    void set_lr(T lr) { m_lr = lr; }

    void update(const dev_mat_t<T>& grad, dev_buf_t<T>& param)
    {
        const int n = grad.row_num() * grad.col_num();
        detail::check_update_shapes(n, param.data(), param.size(), "dev_adam_t::update");
        if (n == 0)
            return;

        if (m_m.size() != static_cast<std::size_t>(n))
        {
            m_m.release();
            m_v.release();
            m_m.allocate(static_cast<std::size_t>(n));
            m_v.allocate(static_cast<std::size_t>(n));
            JAS_CUDA_CHECK(cudaMemset(m_m.data(), 0, m_m.size() * sizeof(T)));
            JAS_CUDA_CHECK(cudaMemset(m_v.data(), 0, m_v.size() * sizeof(T)));
        }

        ++m_t;
        m_b1_t *= m_b1;
        m_b2_t *= m_b2;

        detail::adam_update_kernel<T><<<detail::update_blocks(n), detail::kUpdateThreads>>>(
            grad.m_data, param.data(), m_m.data(), m_v.data(), n, m_lr, m_b1, m_b2,
            T(1) - m_b1_t, T(1) - m_b2_t, m_eps);
        JAS_CUDA_CHECK(cudaGetLastError());
    }

    void step() {}

private:
    dev_buf_t<T> m_m;
    dev_buf_t<T> m_v;
    T m_lr;
    T m_b1;
    T m_b2;
    T m_eps;
    T m_b1_t = T(1);
    T m_b2_t = T(1);
    int m_t = 0;
};

/**
 * NAdam：本项目的默认优化器（`jas_transformer_t.hpp` 里 `base_upr_tpl` 就是它）。
 * 因此设备端必须有一个，否则「把训练搬到 GPU」无从谈起。
 */
template <typename T>
class dev_nadam_t
{
public:
    dev_nadam_t(T learning_rate = T(0.002), T beta1 = T(0.9), T beta2 = T(0.999),
                T epsilon = T(1e-8))
        : m_lr(learning_rate), m_b1(beta1), m_b2(beta2), m_eps(epsilon)
    {
    }

    void set(T learning_rate = T(0.002), T beta1 = T(0.9), T beta2 = T(0.999),
             T epsilon = T(1e-8))
    {
        m_lr = learning_rate;
        m_b1 = beta1;
        m_b2 = beta2;
        m_eps = epsilon;
        m_t = 0;
        m_b1_t = T(1);
        m_b2_t = T(1);
        m_m.release();
        m_v.release();
    }

    void set_lr(T lr) { m_lr = lr; }

    void update(const dev_mat_t<T>& grad, dev_buf_t<T>& param)
    {
        const int n = grad.row_num() * grad.col_num();
        detail::check_update_shapes(n, param.data(), param.size(), "dev_nadam_t::update");
        if (n == 0)
            return;

        if (m_m.size() != static_cast<std::size_t>(n))
        {
            m_m.release();
            m_v.release();
            m_m.allocate(static_cast<std::size_t>(n));
            m_v.allocate(static_cast<std::size_t>(n));
            JAS_CUDA_CHECK(cudaMemset(m_m.data(), 0, m_m.size() * sizeof(T)));
            JAS_CUDA_CHECK(cudaMemset(m_v.data(), 0, m_v.size() * sizeof(T)));
        }

        ++m_t;
        m_b1_t *= m_b1;
        m_b2_t *= m_b2;

        detail::nadam_update_kernel<T><<<detail::update_blocks(n), detail::kUpdateThreads>>>(
            grad.m_data, param.data(), m_m.data(), m_v.data(), n, m_lr, m_b1, m_b2, m_b1_t, m_b2_t,
            m_eps);
        JAS_CUDA_CHECK(cudaGetLastError());
    }

    void step() {}

private:
    dev_buf_t<T> m_m;
    dev_buf_t<T> m_v;
    T m_lr;
    T m_b1;
    T m_b2;
    T m_eps;
    T m_b1_t = T(1);
    T m_b2_t = T(1);
    int m_t = 0;
};

/**
 * 梯度累积：把多个 micro-batch 的梯度先平均起来，`step()` 时才真正更新参数。
 *
 * 主机端 `cache_updator_t` 记的是**指向参数的主机指针**；设备端对应地记的是
 * 「参数缓冲区 + 属于哪个层」——这里简化为调用方在 `step()` 时提供参数缓冲区，
 * 因为设备端参数的地址在扩容/realloc 后可能变，缓存一个裸指针反而是负担。
 */
template <typename T, template <typename> class updator_type>
class dev_cache_updator_t
{
public:
    explicit dev_cache_updator_t(T learning_rate = T(0.001)) : m_updator(learning_rate) {}

    void set_lr(T lr) { m_updator.set_lr(lr); }
    updator_type<T>& inner() { return m_updator; }

    /** 累积一份梯度；形状由第一份梯度确定。 */
    void update(const dev_mat_t<T>& grad, dev_buf_t<T>& param)
    {
        const int n = grad.row_num() * grad.col_num();
        detail::check_update_shapes(n, param.data(), param.size(),
                                       "dev_cache_updator_t::update");
        if (n == 0)
            return;

        if (m_cache.size() != static_cast<std::size_t>(n))
        {
            m_cache.release();
            m_cache.allocate(static_cast<std::size_t>(n));
            JAS_CUDA_CHECK(cudaMemset(m_cache.data(), 0, m_cache.size() * sizeof(T)));
            m_count = 0;
        }

        ++m_count;
        const T inv = T(1) / static_cast<T>(m_count);
        detail::accumulate_kernel<T><<<detail::update_blocks(n), detail::kUpdateThreads>>>(
            grad.m_data, m_cache.data(), n, inv);
        JAS_CUDA_CHECK(cudaGetLastError());
    }

    /** 把累积的梯度交给内层更新器，然后清空累积。 */
    void step(dev_buf_t<T>& param)
    {
        if (m_cache.empty() || m_count == 0)
            return;

        const int n = static_cast<int>(m_cache.size());
        dev_mat_t<T> cache_leaf(m_cache.data(), n, 1);
        m_updator.update(cache_leaf, param);

        JAS_CUDA_CHECK(cudaMemset(m_cache.data(), 0, m_cache.size() * sizeof(T)));
        m_count = 0;
    }

    void step() { m_updator.step(); }

    /** 当前累积了几份梯度。 */
    int count() const { return m_count; }

private:
    updator_type<T> m_updator;
    dev_buf_t<T> m_cache;
    int m_count = 0;
};

} // namespace cuda
} // namespace jasmine

#endif
