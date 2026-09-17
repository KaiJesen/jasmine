#ifndef __JAS_CUDA_PRECISION_HPP__
#define __JAS_CUDA_PRECISION_HPP__

/**
 * 混精度：`bf16` / `fp16` 作为**存储**类型，`fp32` 作为**累加**精度，以及这件事的误差模型。
 *
 * ## 为什么要把"误差模型"和类型一起写进代码
 *
 * 本项目全部测试的基准是「与主机参考逐元素对齐」，容差在 1e-12 ~ 1e-10 量级。
 * 降精度一上来，这个基准就被打穿了：`bf16` 只有 8 位有效数字，相对误差 ~4e-3。
 * 面对这种量级的差异，最容易做也最没价值的事是**把容差调到 1e-2** —— 那样任何真实
 * 错误（转置写反、漏加一项）都会一起被放过去，测试等于没有。
 *
 * 正确的做法是**把误差算出来**：降精度链路里的误差来自两处，而且两处都有闭式上界
 *
 *   1. **操作数量化**：`a` 与 `b` 各自被舍入一次，相对误差各 ≤ `u_op`
 *   2. **累加舍入**：`K` 项相加，相对误差 ≤ `K · u_acc`（`u_acc` 是累加精度的那一半 ulp）
 *
 * 于是测试可以断言的是**有预测力的东西**：
 *
 *   - 与"先把操作数舍入、再用 double 算"的参考比 → 只该看到 `K · u_acc` 量级的差异，
 *     也就是说**乘积是精确的、累加是 fp32 的**；
 *   - 与不经过舍入的参考比 → 差异必须落在 `2·u_op + K·u_acc` 之内，而且**明显大于**
 *     `K·u_acc`（否则说明"降精度"压根没生效，操作数还是全精度）。
 *
 * 第二条尤其重要：它同时排除了"误差没进来"（优化没生效）和"误差大得没道理"（实现错了）
 * 两种失败。两条一起就把这条路钉死了。
 *
 * ## 为什么乘积是精确的
 *
 * 这不是"cuBLAS 精度高"，而是一条可以手算的事实：`bf16` 有 8 位有效数字、`fp16` 有 11 位，
 * 两个这样的数相乘的尾数分别需要 16 位 / 22 位，而 `fp32` 的尾数有 24 位 ——
 * **乘积一个比特都不丢**。所以「降精度输入 + fp32 累加」的 GEMM 与"把操作数换成它们的
 * 舍入值、再用实数算"是同一个和式，差别只剩下加法的顺序与舍入。
 *
 * （`ReducedInputFp32OutputAccumulatesInFp32` 测的就是这件事：与"先把操作数舍入、
 * 再用 `double` 算"的参考比，差异只该落在 `K · u_acc` 之内。若 cuBLAS 对 16 位输入
 * 用了 16 位累加、或者乘积丢过比特，这条会以**量级之差**失败 —— 这条断言的价值不来自
 * 容差，而来自"容差是这个模型的推论"。）
 *
 * 这也解释了为什么**累加类型不能跟着操作数一起降**：`bf16` 里加 1000 个 `bf16` 数，
 * 相对误差会到 `1000 × 4e-3` 的量级，等于白算。
 *
 * ## 本机（Pascal）上它值多少
 *
 * 说清楚，免得把"支持"当成"加速"：`sm_61` 没有张量核，cuBLAS 的 `GemmEx(16F, COMPUTE_32F)`
 * 在这是**软件回退**（`probe` 确认可用、结果正确，但没有专用硬件）。所以在 P4 上：
 *
 *   - **省的是显存与带宽**（操作数直接砍半，`K` 方向的读流量跟着砍半）；
 *   - **不省时间**，甚至可能更慢。
 *
 * 目标机（Ampere+）才有 `bf16`/`fp16` 张量核，那时这条路径同时省显存与时间。
 * `reduced_precision_is_native()` 把这个区别暴露出来，免得拿测试机上的结论去推目标机。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_compat.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_mat_t.hpp"

namespace jasmine {
namespace cuda {

// ---------------------------------------------------------------------------
// 类型与误差模型
// ---------------------------------------------------------------------------

using bf16_t = __nv_bfloat16;  // 8 位有效数字、8 位指数（范围与 float 同量级）
using fp16_t = __half;         // 11 位有效数字、5 位指数（范围小，容易溢出）

/**
 * 元素类型的精度参数。
 *
 * `mantissa_bits` 是**存储**的尾数位数（不含隐含的 1）。之所以强调"存储"，
 * 是因为 `unit_roundoff` 必须按它来算：
 *
 *   u = 2^-(mantissa_bits + 1)      ← 半个 ulp（就近舍入的最大相对误差）
 *
 * float: 2^-24 ≈ 5.96e-8，bf16: 2^-8 ≈ 3.9e-3，fp16: 2^-11 ≈ 4.9e-4。
 * 差一个比特就是差一倍，所以这里把定义写死，不让调用方自己推。
 *
 * `accumulator` 是**在这个类型上做运算时应当用的累加类型**：降精度一律提升到 `float`。
 */
template <typename T>
struct precision_traits
{
    static constexpr bool reduced = false;
    static constexpr int mantissa_bits = std::numeric_limits<T>::digits - 1;
    static constexpr int exponent_bits = 0;  // 不为降精度类型特意填，用不到
    using accumulator = T;

    static constexpr double unit_roundoff()
    {
        return std::ldexp(1.0, -(mantissa_bits + 1));
    }
};

template <>
struct precision_traits<float>
{
    static constexpr bool reduced = false;
    static constexpr int mantissa_bits = 23;
    static constexpr int exponent_bits = 8;
    using accumulator = float;
    static constexpr double unit_roundoff() { return 5.9604644775390625e-08; }  // 2^-24
};

template <>
struct precision_traits<double>
{
    static constexpr bool reduced = false;
    static constexpr int mantissa_bits = 52;
    static constexpr int exponent_bits = 11;
    using accumulator = double;
    static constexpr double unit_roundoff() { return 1.1102230246251565e-16; }  // 2^-53
};

template <>
struct precision_traits<__half>
{
    static constexpr bool reduced = true;
    static constexpr int mantissa_bits = 10;
    static constexpr int exponent_bits = 5;
    using accumulator = float;
    static constexpr double unit_roundoff() { return 4.8828125e-04; }  // 2^-11
};

template <>
struct precision_traits<__nv_bfloat16>
{
    static constexpr bool reduced = true;
    static constexpr int mantissa_bits = 7;
    static constexpr int exponent_bits = 8;
    using accumulator = float;
    static constexpr double unit_roundoff() { return 3.90625e-03; }  // 2^-8
};

template <typename T>
inline constexpr bool is_reduced_precision_v = precision_traits<T>::reduced;

/** 某个类型上的运算该用哪个精度累加。降精度一律 `float`，这是策略而不是默认值。 */
template <typename T>
using accumulator_t = typename precision_traits<T>::accumulator;

/**
 * 一次长度 `K` 的内积里，**累加**那一半的相对误差上界：`K · u_acc`。
 *
 * 与"操作数怎么来的"无关，所以它是"乘积精确"这条论断的直接推论：拿它当容差去比
 * "先舍入再用 double 算"的参考，就是在检验累加精度到底是不是 fp32。
 */
template <typename operand_t>
constexpr double accumulation_error_bound(int k)
{
    using acc_t = accumulator_t<operand_t>;
    return static_cast<double>(k) * precision_traits<acc_t>::unit_roundoff();
}

/**
 * **操作数量化**那一半：两个操作数各舍入一次 ⇒ 乘积的相对误差 ≤ `2 · u_op`。
 *
 * 这个数与 `K` 无关 —— 也就是说它与累加误差的量级差异不是"大一点"，
 * 而是**差若干个数量级**（`bf16` 上 7.8e-3 对 6e-8）。测试正是靠这个量级差
 * 判断"降精度到底生效了没有"。
 */
template <typename operand_t>
constexpr double quantization_error_bound()
{
    return 2.0 * precision_traits<operand_t>::unit_roundoff();
}

/** 两者相加：长度 `K` 的内积在降精度下的总相对误差上界。 */
template <typename operand_t>
constexpr double inner_product_error_bound(int k)
{
    return quantization_error_bound<operand_t>() + accumulation_error_bound<operand_t>(k);
}

/** 把元素类型映射到 cuBLAS 的 `cudaDataType`（`gemm` 的降精度特化要用）。 */
template <typename T>
constexpr cudaDataType cublas_data_type_of()
{
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, double>)
        return CUDA_R_64F;
    else if constexpr (std::is_same_v<T, __half>)
        return CUDA_R_16F;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return CUDA_R_16BF;
    else
        static_assert(std::is_same_v<T, T>, "没有为这个类型定义 cuBLAS 数据类型");
}

// ---------------------------------------------------------------------------
// 标量转换
// ---------------------------------------------------------------------------

/** 把浮点标量**收窄**成 `Dst`（就近舍入）。源类型先提升到 `float` 再收窄。 */
template <typename Dst, typename Src>
JAS_INLINE_HD Dst narrow(Src x)
{
    if constexpr (std::is_same_v<Dst, Src>)
    {
        return x;
    }
    else if constexpr (std::is_same_v<Dst, __half>)
    {
        return __float2half(static_cast<float>(x));
    }
    else if constexpr (std::is_same_v<Dst, __nv_bfloat16>)
    {
        return __float2bfloat16(static_cast<float>(x));
    }
    else
    {
        return static_cast<Dst>(x);
    }
}

/** 把标量**提升**到它的累加精度（`bf16`/`fp16` → `float`，`float`/`double` 原样）。 */
template <typename T>
JAS_INLINE_HD accumulator_t<T> widen(T x)
{
    if constexpr (std::is_same_v<T, __half>)
    {
        return __half2float(x);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return __bfloat162float(x);
    }
    else
    {
        return x;
    }
}

/** 就近舍入到降精度后再提升回来 —— 也就是"量化一次"的效果，测试里用来算参考值。 */
template <typename T>
JAS_INLINE_HD float quantize(double x)
{
    return widen(narrow<T>(static_cast<float>(x)));
}

// ---------------------------------------------------------------------------
// 矩阵级转换
// ---------------------------------------------------------------------------

namespace detail {

/**
 * 逐元素转换：源表达式按**它自己的精度**求值，再收窄到 `Dst` 写出去。
 *
 * 与 `eval_fused` 分开写而不是复用它，是因为它俩的输出类型不同（`eval_fused` 要求
 * 输出与表达式同类型）。转换本来就是"改变类型"的操作，这里必须有自己的一份。
 */
template <typename Dst, typename Expr>
__global__ void cast_kernel(Expr expr, Dst* __restrict__ out, int rows, int cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols)
        return;
    const int i = idx / cols;
    const int j = idx - i * cols;
    out[idx] = narrow<Dst>(widen(static_cast<typename Expr::ele_type>(expr(i, j))));
}

} // namespace detail

/**
 * 把一个设备可求值的表达式**收窄**成 `Dst` 精度的设备矩阵。
 *
 * 这是把参数搬上设备、或者把激活降精度存下来的原语：`cast<bf16_t>(w.leaf())`
 * 得到的矩阵可以随后直接喂给 `gemm`（见 `jas_cuda_gemm.hpp` 的降精度特化）。
 *
 * 刻意**不做隐式转换**：`gemm(a_float, b_bf16)` 这种混用会在这里被类型系统挡住，
 * 而不是在 cuBLAS 那层变成一个含糊的 status 码。要混精度就显式写
 * `gemm(cast<bf16_t>(a), b_bf16, c)` —— 转换发生在哪一步、舍入发生了几次，
 * 都留在代码里看得见。
 */
template <typename Dst, typename Expr>
void cast_into(const Expr& x, dev_matrix_t<Dst>& out, int threads_per_block = 256)
{
    detail::assert_device_ready<Expr>();

    const int rows = x.row_num();
    const int cols = x.col_num();
    out.allocate(rows, cols);
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (total == 0)
        return;

    const int blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
    detail::cast_kernel<Dst, Expr><<<blocks, threads_per_block>>>(x, out.buffer().data(), rows, cols);
    JAS_CUDA_CHECK(cudaGetLastError());
}

template <typename Dst, typename Expr>
dev_matrix_t<Dst> cast(const Expr& x, int threads_per_block = 256)
{
    dev_matrix_t<Dst> out;
    cast_into<Dst>(x, out, threads_per_block);
    return out;
}

/**
 * 把任意精度的设备矩阵取回主机并**提升到 `double`**。
 *
 * 为什么不是 `download()` 直接给 `mat_t<T>`：降精度的 `mat_t<__nv_bfloat16>` 在主机上
 * 既不好比也不好打印，而"取回来跟参考值对拍"才是它唯一的用途。提升到 `double` 之后
 * 值完全不变（`bf16`/`fp16` 的每个值都能被 `double` 精确表示），所以这一步**不引入误差** ——
 * 这是它敢当参考值用的前提。
 */
template <typename T>
mat_t<double> to_host_double(const dev_matrix_t<T>& m)
{
    // 走原始缓冲区而不是 `download()`：后者返回 `mat_t<T>`，而 `mat_t` 只认算术类型
    // （见 jas_cuda_matrix.hpp 的 has_host_matrix_v）。提升到 double 之后每个值都是精确的。
    const std::size_t n = m.buffer().size();
    std::vector<T> raw(n);
    if (n > 0)
        m.buffer().download(raw.data(), n);

    mat_t<double> out(m.row_num(), m.col_num());
    const int cols = m.col_num();
    for (std::size_t idx = 0; idx < n; ++idx)
        out(static_cast<int>(idx) / cols, static_cast<int>(idx) % cols) =
            static_cast<double>(widen(raw[idx]));
    return out;
}

/** `to_host_double` 的逆向：把主机 `double` 矩阵量化成 `Dst` 再上传。 */
template <typename Dst>
dev_matrix_t<Dst> from_host_double(const mat_t<double>& m)
{
    dev_matrix_t<Dst> out(m.row_num(), m.col_num());
    const int cols = m.col_num();
    std::vector<Dst> raw(static_cast<std::size_t>(m.row_num()) * cols);
    for (int i = 0; i < m.row_num(); ++i)
        for (int j = 0; j < cols; ++j)
            raw[static_cast<std::size_t>(i) * cols + j] = narrow<Dst>(m(i, j));
    if (!raw.empty())
        out.buffer().upload(raw.data(), raw.size());
    return out;
}

// ---------------------------------------------------------------------------
// 本机能力
// ---------------------------------------------------------------------------

/**
 * 本机的降精度 GEMM 是否**有硬件支撑**（张量核）。
 *
 * `fp16` 张量核是 `sm_70` 起（Volta），`bf16` 是 `sm_80` 起（Ampere）。之前
 * cuBLAS 仍会给出正确结果，但走的是软件回退 —— 于是在 P4 这类卡上，
 * 降精度**省显存不省时间**。这个判断把两种情形分开，免得拿测试机上的结论去推目标机。
 */
inline bool reduced_precision_is_native()
{
    return device_info().compute_capability() >= 80;
}

} // namespace cuda
} // namespace jasmine

#endif
