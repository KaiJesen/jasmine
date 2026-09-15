# OpenMP benchmark comparison (Release)

Baseline: `openmp_baseline.txt` (`OMP_NUM_THREADS=1`, no pragma effect on small/old sizes).
After: `openmp_after.txt` (`OMP_NUM_THREADS=8`).
Large MHA serial control: `openmp_mha_large_serial.txt` vs `openmp_mha_large_parallel.txt`.

| Benchmark | before (ns, median CPU) | after (ns) | speedup |
|---|--:|--:|--:|
| BM_MatDot/256 | 4,387,078 | 625,536 | **7.0x** |
| BM_MatDot/512 | 33,922,415 | 5,132,399 | **6.6x** |
| BM_MhaForward/2/32/8 | 114,401 | 114,701 | 1.00x（阈值下不并行） |
| BM_MhaForward/4/64/16 | 784,008 | 783,715 | 1.00x |
| BM_MhaForward/4/64/32 | 1,743,809 | 1,748,137 | 1.00x |
| BM_MhaForward/8/128/64 | 13,372,800 | 5,558,551 | **2.41x** |
| BM_MhaForward/8/256/64 | 45,892,679 | 21,486,988 | **2.14x** |

Notes: matmul OpenMP splits single-threaded system BLAS by row panels; MHA parallelizes QKV sections + multi-head attend above thresholds.
