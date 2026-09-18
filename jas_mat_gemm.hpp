#ifndef _JAS_MAT_GEMM_HPP_
#define _JAS_MAT_GEMM_HPP_

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "jas_mat_t.hpp"

#ifdef JASMINE_USE_OPENMP
#include <omp.h>
#endif

#ifdef JASMINE_USE_BLAS
// System packages often ship cblas in libblas without installing cblas.h.
extern "C" {
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                 const float alpha, const float* A, const int lda, const float* B, const int ldb,
                 const float beta, float* C, const int ldc);
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                 const double alpha, const double* A, const int lda, const double* B, const int ldb,
                 const double beta, double* C, const int ldc);
}
#endif

namespace jasmine {
namespace detail {

// Prefer optimized path once flops exceed a small-matrix floor.
inline constexpr long long gemm_fast_threshold = 16LL * 16 * 16;
// OpenMP only when work is large enough that thread overhead pays off.
inline constexpr long long gemm_omp_threshold = 64LL * 64 * 64;
// Multi-head attend: rough work units ~ heads * seq * d_head
inline constexpr long long mha_head_omp_threshold = 4LL * 32 * 32;

inline bool gemm_should_parallel(int M, int N, int K)
{
#ifdef JASMINE_USE_OPENMP
    return static_cast<long long>(M) * N * K >= gemm_omp_threshold;
#else
    (void)M; (void)N; (void)K;
    return false;
#endif
}

inline bool mha_heads_should_parallel(int num_heads, int seq_len, int d_head)
{
#ifdef JASMINE_USE_OPENMP
    if (num_heads < 2)
        return false;
    return static_cast<long long>(num_heads) * seq_len * d_head >= mha_head_omp_threshold;
#else
    (void)num_heads; (void)seq_len; (void)d_head;
    return false;
#endif
}

/**
 * Blocked fallback kernel. TA/TB mean the operand is "the transpose of the stored matrix":
 *   TA == false: A is stored as (M x K) row-major (row stride lda)
 *   TA == true : A is logically (M x K) but stored as (K x M) row-major (row stride lda)
 * Transposing only changes addressing, not blocking, so it is a template parameter, not a branch.
 */
template <bool TA, bool TB, typename T>
void gemm_blocked_rowmajor_impl(int M, int N, int K,
                                const T* A, int lda,
                                const T* B, int ldb,
                                T* C, int ldc)
{
    constexpr int BS = 64;
    const bool parallel = gemm_should_parallel(M, N, K);

#ifdef JASMINE_USE_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[static_cast<std::size_t>(i) * ldc + j] = T{};

#ifdef JASMINE_USE_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (int i0 = 0; i0 < M; i0 += BS)
    {
        const int i1 = std::min(i0 + BS, M);
        for (int k0 = 0; k0 < K; k0 += BS)
        {
            const int k1 = std::min(k0 + BS, K);
            for (int j0 = 0; j0 < N; j0 += BS)
            {
                const int j1 = std::min(j0 + BS, N);
                for (int i = i0; i < i1; ++i)
                {
                    for (int k = k0; k < k1; ++k)
                    {
                        const T aik = TA
                            ? A[static_cast<std::size_t>(k) * lda + i]
                            : A[static_cast<std::size_t>(i) * lda + k];
                        for (int j = j0; j < j1; ++j)
                        {
                            const T bkj = TB
                                ? B[static_cast<std::size_t>(j) * ldb + k]
                                : B[static_cast<std::size_t>(k) * ldb + j];
                            C[static_cast<std::size_t>(i) * ldc + j] += aik * bkj;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void gemm_blocked_rowmajor(int M, int N, int K,
                           const T* A, int lda,
                           const T* B, int ldb,
                           T* C, int ldc,
                           bool trans_a, bool trans_b)
{
    if (!trans_a && !trans_b)
        gemm_blocked_rowmajor_impl<false, false, T>(M, N, K, A, lda, B, ldb, C, ldc);
    else if (!trans_a && trans_b)
        gemm_blocked_rowmajor_impl<false, true, T>(M, N, K, A, lda, B, ldb, C, ldc);
    else if (trans_a && !trans_b)
        gemm_blocked_rowmajor_impl<true, false, T>(M, N, K, A, lda, B, ldb, C, ldc);
    else
        gemm_blocked_rowmajor_impl<true, true, T>(M, N, K, A, lda, B, ldb, C, ldc);
}

template <typename T>
void gemm_blas_or_blocked(int M, int N, int K,
                          const T* A, int lda,
                          const T* B, int ldb,
                          T* C, int ldc,
                          bool trans_a, bool trans_b)
{
#ifdef JASMINE_USE_BLAS
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        const enum CBLAS_TRANSPOSE ta = trans_a ? CblasTrans : CblasNoTrans;
        const enum CBLAS_TRANSPOSE tb = trans_b ? CblasTrans : CblasNoTrans;
        // Single-threaded system BLAS: split C by row panels across OpenMP threads.
        // A can only be split by rows when not transposed (a transposed A has row stride 1).
        if (gemm_should_parallel(M, N, K) && !trans_a)
        {
#ifdef JASMINE_USE_OPENMP
#pragma omp parallel
            {
                const int nthreads = omp_get_num_threads();
                const int tid = omp_get_thread_num();
                const int i0 = tid * M / nthreads;
                const int i1 = (tid + 1) * M / nthreads;
                const int rows = i1 - i0;
                if (rows > 0)
                {
                    if constexpr (std::is_same_v<T, float>)
                    {
                        cblas_sgemm(CblasRowMajor, ta, tb,
                                    rows, N, K, 1.0f,
                                    A + static_cast<std::size_t>(i0) * lda, lda,
                                    B, ldb, 0.0f,
                                    C + static_cast<std::size_t>(i0) * ldc, ldc);
                    }
                    else
                    {
                        cblas_dgemm(CblasRowMajor, ta, tb,
                                    rows, N, K, 1.0,
                                    A + static_cast<std::size_t>(i0) * lda, lda,
                                    B, ldb, 0.0,
                                    C + static_cast<std::size_t>(i0) * ldc, ldc);
                    }
                }
            }
            return;
#endif
        }
        if constexpr (std::is_same_v<T, float>)
        {
            cblas_sgemm(CblasRowMajor, ta, tb,
                        M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
            return;
        }
        if constexpr (std::is_same_v<T, double>)
        {
            cblas_dgemm(CblasRowMajor, ta, tb,
                        M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
            return;
        }
    }
#endif
    gemm_blocked_rowmajor(M, N, K, A, lda, B, ldb, C, ldc, trans_a, trans_b);
}

template <typename T>
void gemm_rowmajor(int M, int N, int K,
                   const T* A, int lda,
                   const T* B, int ldb,
                   T* C, int ldc,
                   bool trans_a = false, bool trans_b = false)
{
    gemm_blas_or_blocked(M, N, K, A, lda, B, ldb, C, ldc, trans_a, trans_b);
}

/**
 * GEMM operand: ask for a "pointer + leading dimension + transposed" descriptor (gemm_view())
 * and, if it gets one, hand it to BLAS / the blocked kernel without copying. Otherwise (column-major,
 * nested transpose, different element type, ...) it materialises a row-major temporary -- the
 *
 * previous only path, kept as the fallback. mat_t / mat_view_t / mat_reshape_view_t all describe
 * themselves, so a.t().dot(b), a.dot(b.t()) and x.reshape_view(r,c).dot(w) copy nothing.
 */
template <typename Mat, typename T>
struct gemm_operand
{
    mat_t<T> owned;
    const T* ptr = nullptr;
    int ld = 0;
    bool transposed = false;

    /**
     * allow_transposed: whether a zero-copy descriptor for a transposed operand is accepted.
     *
     * Only the **right operand** gets this permission, and that is measured, not guessed (dW with
     * M=64,N=288,K=1024 and dcol with M=288,N=1024,K=64, reference BLAS on this machine):
     *
     *   | slot      | view (zero copy) | materialised GEMM | note |
     *   |------|-------------|------------|------|
     *   | B (right) | 2.15 ms | 2.08 ms | a wash: the stored (N x K) matrix is row-major, K stays sequential |
     *   | A (left)  | 8.04 ms | 1.63 ms | **5x slower**: TransA reads A by column (inner-product form) |
     *
     * A transposed left operand is therefore still materialised (unchanged; backward GEMMs such as
     * `W^T * delta` are unaffected), while a transposed right operand (`delta * col^T`) is zero copy.
     */
    explicit gemm_operand(Mat const& m, bool allow_transposed)
    {
        if constexpr (requires(const Mat& x) { x.gemm_view(); })
        {
            using desc_type = std::remove_cvref_t<decltype(m.gemm_view())>;
            if constexpr (std::is_same_v<typename desc_type::ele_type, T>)
            {
                const desc_type d = m.gemm_view();
                if (d.valid && (!d.transposed || allow_transposed))
                {
                    ptr = d.ptr;
                    ld = d.ld;
                    transposed = d.transposed;
                    return;
                }
            }
        }

        owned = mat_t<T>(m.row_num(), m.col_num(), true);
        for (int i = 0; i < m.row_num(); ++i)
            for (int j = 0; j < m.col_num(); ++j)
                owned(i, j) = static_cast<T>(m(i, j));
        ptr = owned.data();
        ld = owned.col_num();
        transposed = false;
    }
};

template <typename L, typename R, typename T>
bool try_fast_gemm(L const& left, R const& right, mat_t<T>& C)
{
    const int M = C.row_num();
    const int N = C.col_num();
    const int K = left.col_num();
    if (M <= 0 || N <= 0 || K <= 0)
        return false;
    if (static_cast<long long>(M) * N * K < gemm_fast_threshold)
        return false;
    if (!C.row_first() || C.data() == nullptr)
        return false;

    gemm_operand<L, T> A(left, /*allow_transposed=*/false);
    gemm_operand<R, T> B(right, /*allow_transposed=*/true);
    gemm_rowmajor(M, N, K, A.ptr, A.ld, B.ptr, B.ld, C.data(), C.col_num(),
                  A.transposed, B.transposed);
    return true;
}

} // namespace detail
} // namespace jasmine

#endif
