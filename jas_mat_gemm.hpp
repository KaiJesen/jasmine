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

template <typename T>
void gemm_blocked_rowmajor(int M, int N, int K,
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
                        const T aik = A[static_cast<std::size_t>(i) * lda + k];
                        for (int j = j0; j < j1; ++j)
                            C[static_cast<std::size_t>(i) * ldc + j] +=
                                aik * B[static_cast<std::size_t>(k) * ldb + j];
                    }
                }
            }
        }
    }
}

template <typename T>
void gemm_blas_or_blocked(int M, int N, int K,
                          const T* A, int lda,
                          const T* B, int ldb,
                          T* C, int ldc)
{
#ifdef JASMINE_USE_BLAS
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        // Single-threaded system BLAS: split C by row panels across OpenMP threads.
        if (gemm_should_parallel(M, N, K))
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
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    rows, N, K, 1.0f,
                                    A + static_cast<std::size_t>(i0) * lda, lda,
                                    B, ldb, 0.0f,
                                    C + static_cast<std::size_t>(i0) * ldc, ldc);
                    }
                    else
                    {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
            return;
        }
        if constexpr (std::is_same_v<T, double>)
        {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
            return;
        }
    }
#endif
    gemm_blocked_rowmajor(M, N, K, A, lda, B, ldb, C, ldc);
}

template <typename T>
void gemm_rowmajor(int M, int N, int K,
                   const T* A, int lda,
                   const T* B, int ldb,
                   T* C, int ldc)
{
    gemm_blas_or_blocked(M, N, K, A, lda, B, ldb, C, ldc);
}

template <typename Mat, typename T>
struct gemm_operand
{
    mat_t<T> owned;
    const T* ptr = nullptr;
    int ld = 0;

    explicit gemm_operand(Mat const& m)
    {
        if constexpr (std::is_same_v<std::decay_t<Mat>, mat_t<T>>)
        {
            if (m.row_first() && m.data() != nullptr)
            {
                ptr = m.data();
                ld = m.col_num();
                return;
            }
        }

        owned = mat_t<T>(m.row_num(), m.col_num(), true);
        for (int i = 0; i < m.row_num(); ++i)
            for (int j = 0; j < m.col_num(); ++j)
                owned(i, j) = static_cast<T>(m(i, j));
        ptr = owned.data();
        ld = owned.col_num();
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

    gemm_operand<L, T> A(left);
    gemm_operand<R, T> B(right);
    gemm_rowmajor(M, N, K, A.ptr, A.ld, B.ptr, B.ld, C.data(), C.col_num());
    return true;
}

} // namespace detail
} // namespace jasmine

#endif
