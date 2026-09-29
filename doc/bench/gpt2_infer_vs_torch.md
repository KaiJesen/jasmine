# jasmine vs PyTorch: GPT-2 inference

Model: `distilgpt2` (6 layers, d_model 768, d_ff 3072, vocab 50257), float32, batch 1.
Workload: prefill 128 tokens, then decode 32 tokens one at a time against a warm KV cache.
Both sides load the same weights: jasmine from the exported `distilgpt2_weights.bin`, PyTorch from
the HF checkpoint the file was exported from. Median of 3 runs after 1 warmup run.

Reproduce with:

```bash
# as-shipped jasmine, plus a build linked against a faster BLAS for comparison
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJASMINE_USE_OPENMP=ON -DJASMINE_USE_BLAS=ON
cmake --build build -j --target bench_gpt2_infer

cmake -S . -B build-openblas -DCMAKE_BUILD_TYPE=Release -DJASMINE_USE_OPENMP=ON \
      -DJASMINE_USE_BLAS=ON -DJASMINE_BLAS_LIBRARY=/path/to/libopenblas.so \
      -DJASMINE_BUILD_TESTS=OFF -DJASMINE_BUILD_EXAMPLES=OFF
cmake --build build-openblas -j --target bench_gpt2_infer

python tools/compare_gpt2_torch.py \
    --jasmine-bin build-openblas/benches/bench_gpt2_infer \
    --compare-bin build/benches/bench_gpt2_infer \
    --weights build/distilgpt2_weights.bin --threads 4 --prefill 128 --decode 32
```

## Environment

| | |
|---|---|
| CPU | 20 logical cores, max 5.1 GHz (frequency scaling enabled, so absolute numbers are noisy; ratios within one run are reliable) |
| Compiler | g++ 13.3.0, CMake 3.28.3 |
| jasmine default BLAS | netlib reference BLAS 3.12.0 (`libblas3`) — **single-threaded, unoptimized** |
| jasmine comparison BLAS | OpenBLAS 0.3.15 (`DYNAMIC_ARCH NO_AFFINITY Prescott SINGLE_THREADED`) |
| PyTorch | 2.9.0+cu128, CPU, MKL |
| transformers | 5.17.0 |

## Four threads

| metric | jasmine (netlib) | jasmine (OpenBLAS) | PyTorch (MKL) |
|---|---|---|---|
| full forward, 128 tokens | 986.04 ms | 278.02 ms | **44.38 ms** |
| prefill through cache, 128 tokens | 4413.65 ms | 2085.40 ms | **44.37 ms** |
| decode, ms per token | 34.66 ms | 16.00 ms | **7.36 ms** |
| prefill + 32 decode steps | 5522.87 ms | 2597.54 ms | **279.83 ms** |

Slowdown vs PyTorch:

| metric | jasmine (netlib) | jasmine (OpenBLAS) |
|---|---|---|
| full forward | 22.2x | 6.27x |
| prefill through cache | 99.5x | 47.0x |
| decode per token | 4.71x | 2.18x |

## Thread scaling

jasmine (OpenBLAS) vs PyTorch, ms:

| metric | threads | jasmine | PyTorch | ratio |
|---|---|---|---|---|
| full forward | 1 | 646.94 | 162.92 | 3.97x |
| full forward | 4 | 278.02 | 44.38 | 6.27x |
| full forward | 8 | 257.24 | 62.88 | 4.09x |
| decode per token | 1 | 43.76 | 11.30 | 3.87x |
| decode per token | 4 | 16.00 | 7.36 | 2.18x |
| decode per token | 8 | 13.22 | 8.00 | 1.65x |
| prefill through cache | 1 | 5664.16 | 162.62 | 34.83x |
| prefill through cache | 4 | 2085.40 | 44.37 | 47.00x |
| prefill through cache | 8 | 1663.59 | 62.26 | 26.72x |

jasmine's forward scales 2.33x from 1 to 4 threads and then flattens (1.08x from 4 to 8). PyTorch
peaks at 4 threads and **regresses at 8** (44.38 ms to 62.88 ms). Both point at the same thing:
past 4 threads this machine is contended, so the 1- and 4-thread rows are the trustworthy ones.
Fixing the thread count also keeps the comparison meaningful; jasmine splits GEMMs across OpenMP
row panels in `jas_mat_gemm.hpp` while PyTorch parallelizes inside BLAS, so an unpinned run would
be measuring thread policy rather than code.

## Correctness control

The exported file and the HF checkpoint are only the same model if the final logits agree. Sum of
the last position's logits:

| threads | jasmine | PyTorch | relative spread |
|---|---|---|---|
| 1 | -2518023.4670 | -2518023.0000 | 1.85e-07 |
| 4 | -2518023.4670 | -2518023.0000 | 6.66e-07 |
| 8 | -2518023.4670 | -2518022.7500 | 2.85e-07 |

That is fp32 accumulation noise, so both sides ran the same weights and the timings describe the
same computation. `tools/compare_gpt2_torch.py` fails the run if this check does not hold.

## What the numbers say

**1. The BLAS choice dominates the as-shipped gap.** jasmine links whatever `CBLAS` CMake finds, and
on a stock Debian/Ubuntu that is the reference netlib BLAS. Swapping in OpenBLAS alone: full forward
986.04 → 278.02 ms (3.5x), decode 34.66 → 16.00 ms (2.2x). Most of the "22x slower than PyTorch"
headline is a packaging accident, not a transformer implementation problem. Either ship an
optimized BLAS or document the dependency; a benchmark against MKL with netlib underneath is not
an implementation comparison.

**2. With a comparable BLAS, the remaining gaps are 6.3x (batched forward) and 2.2x (single-token
decode).** The decode figure is the most apples-to-apples number here, and 2.2x is a much better
place to be than the raw comparison suggests.

**3. Prefill is not the same computation, and this is the biggest available win.** jasmine's
`prefill()` loops `forward_one` once per prompt token, while PyTorch's `use_cache=True` prefill is a
single batched forward. jasmine's own `forward()` on the same 128 tokens takes 278.02 ms against
its 2085.40 ms prefill: **7.5x is self-inflicted by stepping the prompt token by token**. This is a
missing code path, not a tuning problem — filling the KV cache from one batched forward would
remove it. PyTorch confirms the two should cost the same: 44.37 ms prefill vs 44.38 ms full forward.

**4. Decode is memory-bandwidth bound, and jasmine reaches about half the bandwidth.** One token
must read all 82M parameters (328 MB in fp32). jasmine's 16.00 ms implies ~20 GB/s, PyTorch's
7.36 ms ~44 GB/s. For a single token the GEMMs are effectively matrix-vector products, and jasmine
routes them through `cblas_sgemm` while splitting the output rows across OpenMP threads; both the
GEMM packing and the row-panel split are poor fits for a skin-deep matrix. A `gemv` path for the
decode case is the obvious thing to try next.

**5. Thread scaling saturates early and PyTorch also regresses at 8 threads**, so neither
implementation is a scaling win to celebrate on this machine. Any follow-up should fix threads at 4.

## Raw logs

`gpt2_vs_torch_1threads.txt`, `gpt2_vs_torch_4threads.txt`, `gpt2_vs_torch_8threads.txt`.
