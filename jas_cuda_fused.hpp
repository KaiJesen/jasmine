#ifndef __JAS_CUDA_FUSED_HPP__
#define __JAS_CUDA_FUSED_HPP__

/**
 * 融合逐元素求值：把**整棵表达式树**塞进一个 kernel，一次 launch 算完整条链。
 *
 * 这是表达式模板在 GPU 上最划算的用法。`(a + b) * c` 在 CPU 上要物化一个临时矩阵
 * （写一遍 18 MB、再读一遍），在 GPU 上则连中间结果都不存在 ——
 * 每个线程只读它需要的叶子元素，算完整棵树的标量运算，直接写出结果。
 * 省掉的不只是 kernel launch 开销，还有中间结果的显存往返。
 *
 * 能这么做的根本原因是 `work()` 天生就是纯标量函数（`return i + j;`），
 * 与内存布局完全解耦。设备叶子把「怎么读内存」也做成了设备可调用，
 * 剩下的就是把树按值递给 kernel。
 *
 * 注意边界：**只有逐元素算子能这样融合**。归约（`sum`/`hsum`）和 `dot` 不行 ——
 * 它们需要跨线程协作或两趟扫描，得走专用 kernel / cuBLAS，见 jas_cuda_gemm.hpp。
 * `mat_dot_t` 和 `mat_softmax_t` 刻意没有 `device_evaluable`，含它们的表达式
 * 会在这里被 `static_assert` 挡住。
 */

#include <cstddef>
#include <type_traits>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_mat_t.hpp"

namespace jasmine {
namespace cuda {
namespace detail {

/**
 * 每个线程算一个输出元素。
 *
 * 用一维网格 + 一次除法定位 (i, j)，而不是二维网格：二维网格的 gridDim.y 上限是 65535，
 * 大矩阵会撞墙。每线程一次整数除法相对于要走的显存流量可以忽略。
 */
template <typename Expr>
__global__ void fused_elementwise_kernel(Expr expr,
                                         typename Expr::ele_type* __restrict__ out,
                                         int rows, int cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total)
        return;

    const int i = idx / cols;
    const int j = idx - i * cols;
    out[idx] = static_cast<typename Expr::ele_type>(expr(i, j));
}

/** 编译期把「能不能上设备」和「能不能当 kernel 参数」三个条件一次问清楚。 */
template <typename Expr>
constexpr void assert_device_ready()
{
    static_assert(is_device_evaluable_v<Expr>,
                  "这个表达式含有无法在设备上求值的操作数（例如 mat_t / mat_view_t）。"
                  "设备表达式必须由 dev_mat_t 叶子构成；dot / softmax 也不能逐元素融合，"
                  "它们需要专用 kernel 或 cuBLAS。");
    static_assert(is_self_contained_v<Expr>,
                  "表达式树里有引用成员，不能作为 kernel 参数。kernel 参数是【按值】搬到设备上的，"
                  "引用成员搬过去的是主机地址，设备端一解引用就是 cudaErrorIllegalAddress。"
                  "凡 device_evaluable 的操作数都应当被按值拥有 —— 若这里失败，"
                  "检查 storage_of 的 owned_by_value 是否覆盖了该操作数类型。");
    static_assert(std::is_trivially_copyable_v<Expr>,
                  "kernel 参数必须平凡可拷贝。设备叶子与节点都是 POD，本应如此。");
}

} // namespace detail

/**
 * 在 GPU 上求值一个表达式，结果写进 `out`（会自动按需分配）。
 *
 * @param threads_per_block 每块线程数。默认 256；P4 这类 Pascal 卡 128~256 都比较稳。
 */
template <typename Expr>
void eval_fused(const Expr& expr, dev_buf_t<typename Expr::ele_type>& out,
                int threads_per_block = 256)
{
    detail::assert_device_ready<Expr>();

    const int rows = expr.row_num();
    const int cols = expr.col_num();
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

    out.allocate(total);
    if (total == 0)
        return;

    const int blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
    detail::fused_elementwise_kernel<<<blocks, threads_per_block>>>(expr, out.data(), rows, cols);
    JAS_CUDA_CHECK(cudaGetLastError());
}

/**
 * 求值并拷回主机。
 *
 * 同样是融合求值，只是顺手把结果搬回一块 `mat_t`，方便和 CPU 结果逐元素对拍。
 */
template <typename Expr>
mat_t<typename Expr::ele_type> eval_fused_to_host(const Expr& expr, int threads_per_block = 256)
{
    using ele_type = typename Expr::ele_type;

    dev_buf_t<ele_type> dev_out;
    eval_fused(expr, dev_out, threads_per_block);
    sync();

    mat_t<ele_type> host_out(expr.row_num(), expr.col_num());
    if (!dev_out.empty())
        dev_out.download(host_out.data(), dev_out.size());
    return host_out;
}

} // namespace cuda
} // namespace jasmine

#endif
