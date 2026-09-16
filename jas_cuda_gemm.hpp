#ifndef __JAS_CUDA_GEMM_HPP__
#define __JAS_CUDA_GEMM_HPP__

/**
 * cuBLAS 封装的矩阵乘。
 *
 * ## 为什么 GEMM 单独一条路，而不是塞进表达式模板
 *
 * 表达式模板擅长的是**逐元素**融合：每个输出元素只依赖同位置的输入，一个线程就能算完。
 * 矩阵乘不是这样 —— 每个输出元素都要跨一整行/一列求和，必须多线程协作 + 分块复用数据。
 * 把这种算子硬塞进"每线程算一个元素"的 kernel，等于让每个线程自己走一遍内积，
 * 访存完全无法复用，性能会塌掉。
 *
 * 所以正确的做法是**在 dot 处切分**：逐元素链交给融合 kernel，dot 交给 cuBLAS。
 * 这正是 PyTorch/OneFlow 这类框架张量表达式的调度方式，也是 `mat_dot_t`
 * 刻意不标 `device_evaluable` 的原因。
 *
 * ## 行优先 ↔ 列优先
 *
 * cuBLAS 只认列优先，而本项目一律行优先。做法是不用复制数据，靠"转置同一块内存"来对齐：
 *
 *   行优先存储的 W（ld = m_cols）在 cuBLAS 眼里就是一个列优先的 Wᵀ。
 *
 * 于是要算行优先的 `C(M×N) = opA(A) · opB(B)`，转到列优先就是算 `Cᵀ = Bᵀ·Aᵀ`，
 * 也就是说 **A 和 B 要交换位置、M 和 N 也要交换**：
 *
 *   cublasXgemm(handle, opB, opA, N, M, K, ..., B, ldb, A, lda, ..., C, ldc)
 *
 * 各组合都已验证：不转置时 op = `CUBLAS_OP_N`（直接用那个转置视图），
 * 转置时 op = `CUBLAS_OP_T`（再转回来）。前导维一律用**未经转置解释的存储步长**
 * `leading_dim()` —— 转置只改索引解释，内存布局没动。
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"

namespace jasmine {
namespace cuda {

class cublas_error : public std::runtime_error
{
public:
    cublas_error(cublasStatus_t status, const char* expr, const char* file, int line)
        : std::runtime_error(build_message(status, expr, file, line)), m_status(status)
    {
    }

    cublasStatus_t status() const noexcept { return m_status; }

private:
    static std::string build_message(cublasStatus_t status, const char* expr, const char* file,
                                     int line)
    {
        return std::string("cuBLAS 调用失败 (status=") + std::to_string(static_cast<int>(status))
               + ")\n  调用: " + expr + "\n  位置: " + file + ":" + std::to_string(line);
    }

    cublasStatus_t m_status;
};

} // namespace cuda
} // namespace jasmine

#define JAS_CUBLAS_CHECK(expr)                                                          \
    do {                                                                                \
        ::cublasStatus_t _jas_st = (expr);                                              \
        if (_jas_st != CUBLAS_STATUS_SUCCESS)                                           \
            throw ::jasmine::cuda::cublas_error(_jas_st, #expr, __FILE__, __LINE__);     \
    } while (0)

namespace jasmine {
namespace cuda {

/** 进程内共享一个 cuBLAS 句柄。句柄创建有开销，没必要每次 GEMM 都建。 */
inline cublasHandle_t cublas_handle()
{
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        JAS_CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}

/**
 * 行优先矩阵乘：`C = alpha * opA(A) * opB(B) + beta * C`。
 *
 * 转置由叶子的转置视图表达（`A.t()`），不需要额外参数：
 *   - `gemm(A, B, C)`           → C = A·B
 *   - `gemm(A.t(), B, C)`       → C = Aᵀ·B
 *   - `gemm(Q, K.t(), S)`       → S = Q·Kᵀ，正是注意力打分那一步
 *
 * 输出 `C` 必须已经指向一块够大的设备内存（用 dev_matrix_t 分配）。
 * 叶子本身只是薄壳，是否写它由 beta/使用方式决定。
 */
template <typename T>
void gemm(const dev_mat_t<T>& A, const dev_mat_t<T>& B, const dev_mat_t<T>& C,
          T alpha = T(1), T beta = T(0))
{
    // opA(A) 是 M×K，opB(B) 是 K×N（row_num/col_num 已经把转置考虑进去了）
    const int M = A.row_num();
    const int K = A.col_num();
    const int N = B.col_num();

    if (B.row_num() != K)
        throw std::invalid_argument(
            "gemm: 内维不匹配 —— opA(A) 的列数 " + std::to_string(K)
            + " != opB(B) 的行数 " + std::to_string(B.row_num()));
    if (C.row_num() != M || C.col_num() != N)
        throw std::invalid_argument(
            "gemm: 输出形状不对，应为 " + std::to_string(M) + "×" + std::to_string(N)
            + "，实际 " + std::to_string(C.row_num()) + "×" + std::to_string(C.col_num()));
    if (!A.valid() || !B.valid() || !C.valid())
        throw std::invalid_argument("gemm: 有操作数还是空叶子");

    // 前导维必须用未转置解释的存储步长
    const int lda = A.leading_dim();
    const int ldb = B.leading_dim();
    const int ldc = C.leading_dim();

    // 不转置 → 那个"列优先视图"正好就是要的左操作数的转置；转置 → 再转回来
    const cublasOperation_t opA = A.transposed() ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opB = B.transposed() ? CUBLAS_OP_T : CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>)
    {
        // 注意实参顺序：列优先下算的是 Cᵀ = Bᵀ·Aᵀ，所以 B 在前、A 在后，N 在前、M 在后
        JAS_CUBLAS_CHECK(cublasSgemm(cublas_handle(), opB, opA, N, M, K, &alpha,
                                     B.m_data, ldb, A.m_data, lda, &beta, C.m_data, ldc));
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        JAS_CUBLAS_CHECK(cublasDgemm(cublas_handle(), opB, opA, N, M, K, &alpha,
                                     B.m_data, ldb, A.m_data, lda, &beta, C.m_data, ldc));
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "gemm 只支持 float / double");
    }
}

/** 分配好输出、算完、拷回主机。方便和 CPU 结果对拍。 */
template <typename T>
mat_t<T> gemm_to_host(const dev_mat_t<T>& A, const dev_mat_t<T>& B, T alpha = T(1))
{
    dev_matrix_t<T> out(A.row_num(), B.col_num());
    gemm(A, B, out.leaf(), alpha, T(0));
    sync();
    return out.download();
}

} // namespace cuda
} // namespace jasmine

#endif
