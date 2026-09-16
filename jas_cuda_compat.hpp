#ifndef __JAS_CUDA_COMPAT_HPP__
#define __JAS_CUDA_COMPAT_HPP__

/**
 * CUDA 兼容层：让同一份代码在没有 CUDA 的机器上也能原样编译。
 *
 * 这里只放「主机/设备都要能用」的最小设施，不引入任何 CUDA 运行时依赖，
 * 所以纯 CPU 构建（没装 nvcc）include 这个头也不会有任何副作用。
 */

#include <cmath>
#include <type_traits>

#ifdef __CUDACC__
/**
 * 标注「主机和设备都能调用」。
 *
 * 表达式模板里的 row_num / col_num / operator() / work 全部靠它变为设备可调用，
 * 于是同一个 work() 既能在 CPU 上逐元素求值，也能被塞进 kernel 里在每个线程求值 ——
 * 这就是表达式模板能上 GPU 的关键：算子逻辑与内存布局本来就是解耦的。
 */
#define JAS_HD __host__ __device__
#define JAS_INLINE_HD __host__ __device__ inline
/**
 * 只能由设备调用的函数。
 *
 * 块内归约用到的 `__syncthreads()` 和 `warpSize` 都是设备独有的，
 * 标成 `__host__ __device__` 会被 nvcc 直接拒绝。
 */
#define JAS_DEV __device__
#else
#define JAS_HD
#define JAS_INLINE_HD inline
#define JAS_DEV
#endif

namespace jasmine {
namespace detail {

/**
 * 标量 exp。
 *
 * `std::exp` 不是 `__device__` 函数，直接从设备代码调用会编译失败；
 * CUDA 在设备端提供的是全局命名空间的 `::exp`。编译设备代码时 `__CUDA_ARCH__` 有定义，
 * 正好用来区分这两个世界。
 */
JAS_INLINE_HD double device_exp(double x)
{
#if defined(__CUDA_ARCH__)
    return ::exp(x);
#else
    return std::exp(x);
#endif
}

JAS_INLINE_HD float device_exp(float x)
{
#if defined(__CUDA_ARCH__)
    return ::expf(x);
#else
    return std::exp(x);
#endif
}

/**
 * 设备安全的二值取大。
 *
 * 不能用 `std::max`：它不是 `__device__` 函数。这里手写一个两行的版本，
 * 语义与 `std::max` 一致（相等时取第一个）。
 */
template <typename T>
JAS_INLINE_HD T device_max(T a, T b)
{
    return a > b ? a : b;
}

/** 标量开方。同 device_exp：设备端必须用全局命名空间的 `::sqrt` / `::sqrtf`。 */
JAS_INLINE_HD double device_sqrt(double x)
{
#if defined(__CUDA_ARCH__)
    return ::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

JAS_INLINE_HD float device_sqrt(float x)
{
#if defined(__CUDA_ARCH__)
    return ::sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

} // namespace detail

/**
 * 这个类型能不能在设备上求值？
 *
 * 判据是「有没有 `device_evaluable` 这个静态常量成员」，而不是白名单：
 * 表达式节点自己是模板，没法在别处逐个登记。让指示器沿树传播即可 ——
 * 叶子（`dev_mat_t` / `scalar_leaf_t`）写 `true`，节点从操作数继承判断，
 * `mat_t` 这类主机独占的类型连成员都没有，自然就是 false。
 *
 * 启动 kernel 前用 `static_assert` 挡一道，能把「拿主机表达式直接上设备」
 * 这种错误在编译期就问出来，而不是运行时段错误。
 */
template <typename T, typename = void>
struct is_device_evaluable : std::false_type {};

template <typename T>
struct is_device_evaluable<
    T, std::void_t<decltype(std::remove_cvref_t<T>::device_evaluable)>>
    : std::bool_constant<std::remove_cvref_t<T>::device_evaluable> {};

template <typename T>
inline constexpr bool is_device_evaluable_v = is_device_evaluable<T>::value;

} // namespace jasmine

#endif
