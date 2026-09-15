# BM_MatDot Release comparison

Environment: Release build, same machine; before = naive `mat_dot_t::clone`, after = BLAS `cblas_sgemm` via `jas_mat_gemm.hpp`.

| n | before (ns, median CPU) | after (ns) | speedup | before items/s | after items/s |
|---|-------------------------|------------|---------|----------------|---------------|
| 32 | 81201 | 8801 | 9.23x | 403.5 M/s | 3.723 G/s |
| 64 | 634628 | 81470 | 7.79x | 413.1 M/s | 3.218 G/s |
| 128 | 5018305 | 589090 | 8.52x | 417.9 M/s | 3.560 G/s |
| 256 | 39929696 | 4380895 | 9.11x | 420.2 M/s | 3.830 G/s |
| 512 | 318733335 | 33858917 | 9.41x | 421.1 M/s | 3.964 G/s |

Raw logs: `matmul_baseline_release.txt`, `matmul_after_gemm_blas.txt`.
