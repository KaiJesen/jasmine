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
 *
 * 注意 `leading_dim()` **可以大于** `col_num()`（见 jas_cuda_leaf.hpp 的 `view()`）：
 * KV cache 那种「按 cap 分配、只暴露前 len 列」的子视图就是这样传给 cuBLAS 的，
 * 仍然是零拷贝。cuBLAS 本身完全支持 ld > cols，这里不能拿 col_num() 当 ld 用。
 *
 * 这样也不会越界读到 cap - len 那段"空闲"列：转置视图下每个矩阵槽被解释成
 * `(k × n)`，其中**被 ld 乘的那个下标只遍历逻辑列数**，而逻辑列数 ≤ ld，
 * 所以最大偏移落在缓冲区实长之内。即「按 cap 分配、只暴露前 len 列」多出来的
 * 那段空间永远不会被 GEMM 碰到 —— 这正是子视图能零拷贝的直接原因。
 *
 * ## 精度：默认不用 TF32
 *
 * Ampere 及以后单精度 GEMM 可以走 TF32 张量核（尾数只剩 10 位），而开不开取决于
 * math mode 与环境变量 —— 于是「同一份代码在不同机器上数值不同」。
 * 本项目整套测试的基准就是「与主机参考逐元素对齐」，所以默认钉死 `gemm_math::precise`
 * （`CUBLAS_PEDANTIC_MATH`），要 TF32 得显式 `set_gemm_math(gemm_math::tf32)`。
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
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

/**
 * GEMM 的数学模式。
 *
 * 这不是可有可无的开关。Ampere 及以后的卡上，单精度 GEMM 可以选择走 **TF32 张量核** ——
 * 指数位仍是 8 位，但尾数只剩 **10 位**，相对误差约 1e-3。
 * 而 CUDA 的行为受「math mode + NVIDIA_TF32_OVERRIDE 环境变量」共同影响，
 * 也就是说**同一份代码在不同机器上会给出不同数值**。
 *
 * 对本项目这是个陷阱：我们整套测试的价值就在于「设备结果与主机参考逐元素对齐」，
 * 如果 GEMM 精度随机器漂移，那条基准线就没意义了（float 对拍用的是 1e-5 量级的相对容差，
 * TF32 的 1e-3 直接把它冲掉）。
 *
 * 所以默认取 `precise`，把这层不确定性从默认路径上拿掉；
 * 想要速度的显式调 `set_gemm_math(gemm_math::tf32)`，那时也知道自己换掉了什么。
 * （double 的 Dgemm 不受 TF32 影响，两种模式下都是真双精度。）
 */
enum class gemm_math
{
    precise, // CUBLAS_PEDANTIC_MATH：强制真 FP32/FP64，跨机器可复现
    tf32,    // CUBLAS_TF32_TENSOR_OP_MATH：Ampere+ 单精度走 TF32 张量核，快但尾数少 13 位
};

namespace detail {

inline gemm_math& gemm_math_mode_ref()
{
    static gemm_math mode = gemm_math::precise;
    return mode;
}

/** 进程内共享一个 cuBLAS 句柄。句柄创建有开销，没必要每次 GEMM 都建。 */
inline cublasHandle_t& cublas_handle_ref()
{
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        JAS_CUBLAS_CHECK(cublasCreate(&h));
        // 建句柄时就钉死精度，别等某次 GEMM 才让结果悄悄变掉
        JAS_CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
        return h;
    }();
    return handle;
}

} // namespace detail

inline cublasHandle_t cublas_handle() { return detail::cublas_handle_ref(); }

/** 当前 GEMM 数学模式。 */
inline gemm_math gemm_math_mode() { return detail::gemm_math_mode_ref(); }

/** 本机是否支持 TF32 张量核（Ampere 及以后，即 sm_80+）。 */
inline bool tf32_supported()
{
    return device_info().compute_capability() >= 80;
}

/**
 * 切换 GEMM 数学模式。要求设备真实存在（会查询算力）。
 *
 * 在 P4（sm_61）这类没有 TF32 的卡上请求 `tf32` 会抛异常，而不是**静默降级成 FP32** ——
 * 静默降级会让你在测试机上「验证过」的加速在目标机上完全没生效，却毫无提示。
 */
inline void set_gemm_math(gemm_math mode)
{
    if (mode == gemm_math::tf32 && !tf32_supported())
        throw std::runtime_error(
            "set_gemm_math(tf32): 本设备算力 "
            + std::to_string(device_info().compute_capability())
            + " 不支持 TF32（需要 sm_80 及以上）");
    const cublasMath_t m = (mode == gemm_math::tf32) ? CUBLAS_TF32_TENSOR_OP_MATH
                                                     : CUBLAS_PEDANTIC_MATH;
    JAS_CUBLAS_CHECK(cublasSetMathMode(cublas_handle(), m));
    detail::gemm_math_mode_ref() = mode;
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

namespace detail {

template <typename T>
struct is_dev_matrix_t : std::false_type {};
template <typename T>
struct is_dev_matrix_t<dev_matrix_t<T>> : std::true_type {};

/**
 * 把任意操作数规约成一个 GEMM 能吃的薄壳叶子。
 *
 *   - 本来就是叶子 → 原样返回（拷薄壳，零成本）
 *   - 是拥有者     → 取只读薄壳
 *   - 是表达式     → **先融合物化**进 scratch，再用它的叶子
 *
 * 最后一种正是「任意设备可求值的表达式都能当 GEMM 操作数」的实现：
 * `matmul(x + residual, w)` 里那个加法会被融成一个临时矩阵，而不是逐元素硬塞进 GEMM。
 *
 * scratch 的生命周期由调用方（matmul_impl）的函数作用域保证，必须活得比 GEMM 调用长。
 */
template <typename T, typename X>
dev_mat_t<T> as_gemm_operand(const X& x, dev_matrix_t<T>& scratch)
{
    if constexpr (std::is_same_v<std::remove_cvref_t<X>, dev_mat_t<T>>)
    {
        return x;
    }
    else if constexpr (is_dev_matrix_t<std::remove_cvref_t<X>>::value)
    {
        return x.const_leaf();
    }
    else
    {
        scratch.allocate(x.row_num(), x.col_num());
        eval_fused(x, scratch.buffer());
        return scratch.leaf();
    }
}

template <typename T, typename A, typename B>
dev_matrix_t<T> matmul_impl(const A& a, const B& b, T alpha)
{
    // scratch 必须在这里声明：它的生命周期要覆盖后面的 gemm 调用
    dev_matrix_t<T> a_scratch, b_scratch;
    const dev_mat_t<T> a_leaf = as_gemm_operand<T>(a, a_scratch);
    const dev_mat_t<T> b_leaf = as_gemm_operand<T>(b, b_scratch);

    dev_matrix_t<T> out(a_leaf.row_num(), b_leaf.col_num());
    gemm(a_leaf, b_leaf, out.leaf(), alpha, T(0));
    return out;
}

} // namespace detail

/**
 * 设备端矩阵乘，返回**拥有显存**的结果。
 *
 * 这是主机端 `a.dot(b)` 的设备对应物，但**立即求值**而非构造惰性节点 ——
 * 原因见 jas_cuda_leaf.hpp 里 `dev_mat_t::dot` 的说明（GEMM 无法融合）。
 *
 * 转置靠叶子的转置视图表达，与 `gemm` 一致：
 *   - `matmul(A, B)`         → A·B
 *   - `matmul(A.t(), B)`     → Aᵀ·B
 *   - `matmul(Q, K.t())`     → Q·Kᵀ（注意力打分）
 *
 * 返回值可能是空的（形状为 0）或持有资源，均为可移动类型，按值返回无额外拷贝。
 */
template <typename T>
dev_matrix_t<T> matmul(const dev_mat_t<T>& a, const dev_mat_t<T>& b, T alpha = T(1))
{
    return detail::matmul_impl<T>(a, b, alpha);
}

/**
 * 操作数可以是任意**设备可求值**的表达式或拥有者，不必先手动取叶子：
 *
 *   matmul(x + residual, w);                       // 加法先融合成临时矩阵
 *   matmul(norm_result, w);                        // dev_matrix_t 直接传
 *
 * 约束排除「两边都是叶子」的情形，否则会与上面那个重载二义。
 */
template <typename A, typename B>
requires (!is_dev_leaf_v<A> || !is_dev_leaf_v<B>)
auto matmul(const A& a, const B& b)
{
    using T = std::common_type_t<typename std::remove_cvref_t<A>::ele_type,
                                 typename std::remove_cvref_t<B>::ele_type>;
    return detail::matmul_impl<T>(a, b, T(1));
}

// ---------------------------------------------------------------------------
// dev_matrix_t 的 .dot()：薄壳，转发到 matmul
// ---------------------------------------------------------------------------

template <typename T>
dev_matrix_t<T> dev_matrix_t<T>::dot(const dev_mat_t<T>& other) const
{
    return matmul(this->const_leaf(), other);
}

template <typename T>
dev_matrix_t<T> dev_matrix_t<T>::dot(const dev_matrix_t<T>& other) const
{
    return matmul(this->const_leaf(), other.const_leaf());
}

} // namespace cuda

// ---------------------------------------------------------------------------
// dev_mat_t 的 .dot()
//
// 注意命名空间：dev_mat_t 在 jasmine，dev_matrix_t 在 jasmine::cuda，
// 所以这两个定义必须放在 jasmine 作用域里（类外定义要在所属命名空间内）。
// ---------------------------------------------------------------------------

template <typename T>
cuda::dev_matrix_t<T> dev_mat_t<T>::dot(const dev_mat_t<T>& other) const
{
    return cuda::matmul(*this, other);
}

template <typename T>
cuda::dev_matrix_t<T> dev_mat_t<T>::dot(const cuda::dev_matrix_t<T>& other) const
{
    return cuda::matmul(*this, other.const_leaf());
}

} // namespace jasmine

#endif
