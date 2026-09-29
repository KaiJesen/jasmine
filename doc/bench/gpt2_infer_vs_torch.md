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
| full forward, 128 tokens | 990.93 ms | 272.66 ms | **44.51 ms** |
| prefill through cache, 128 tokens | 988.35 ms | 282.08 ms | **44.46 ms** |
| decode, ms per token | 34.77 ms | 16.83 ms | **7.34 ms** |
| prefill + 32 decode steps | 2101.08 ms | 820.64 ms | **279.21 ms** |

Slowdown vs PyTorch:

| metric | jasmine (netlib) | jasmine (OpenBLAS) |
|---|---|---|
| full forward | 22.3x | 6.13x |
| prefill through cache | 22.2x | 6.34x |
| decode per token | 4.74x | 2.29x |

## Thread scaling

jasmine (OpenBLAS) vs PyTorch, ms:

| metric | threads | jasmine | PyTorch | ratio |
|---|---|---|---|---|
| full forward | 1 | 642.77 | 163.10 | 3.94x |
| full forward | 4 | 272.66 | 44.51 | 6.13x |
| full forward | 8 | 259.04 | 61.93 | 4.18x |
| prefill through cache | 1 | 646.25 | 162.82 | 3.97x |
| prefill through cache | 4 | 282.08 | 44.46 | 6.34x |
| prefill through cache | 8 | 257.01 | 62.20 | 4.13x |
| decode per token | 1 | 44.85 | 11.15 | 4.02x |
| decode per token | 4 | 16.83 | 7.34 | 2.29x |
| decode per token | 8 | 12.75 | 8.25 | 1.55x |

jasmine's forward scales 2.36x from 1 to 4 threads and then flattens (1.05x from 4 to 8). PyTorch
peaks at 4 threads and **regresses at 8** (44.51 ms to 61.93 ms). Both point at the same thing:
past 4 threads this machine is contended, so the 1- and 4-thread rows are the trustworthy ones.
Pinning the thread count also keeps the comparison meaningful; jasmine splits GEMMs across OpenMP
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

Note the jasmine checksum is **identical before and after** the batched-prefill change described
below (-2518023.4670), which is the evidence that batching changed only how many GEMM launches
happen, not the arithmetic.

## The batched prefill fix

The first version of this comparison showed prefill at 47x PyTorch, but that was not a like-for-like
number: `gpt2_model_t::prefill` looped `forward_one` once per prompt token, while PyTorch's
`use_cache=True` prefill is a single batched forward. jasmine's own `forward()` on the same 128
tokens took 278.02 ms against its 2085.40 ms prefill, so 7.5x was self-inflicted.

`mat_mha_t::forward_one` already accepted multi-column input and applied the causal mask by absolute
position, so the fix was to stop stepping one token at a time and run the whole prompt through it in
one pass (`jas_gpt2_t.hpp::prefill`). `forward_one` at the model level was also generalized to T
columns, which is what lets the interactive path append a whole user turn to an existing cache
without clearing it.

| metric (4 threads, OpenBLAS) | before | after | change |
|---|---|---|---|
| prefill through cache, 128 tokens | 2085.40 ms | 282.08 ms | **7.39x faster** |
| prefill + 32 decode steps | 2597.54 ms | 820.64 ms | 3.17x faster |
| slowdown vs PyTorch, prefill | 47.00x | 6.34x | — |
| full forward | 278.02 ms | 272.66 ms | unchanged (noise) |
| decode per token | 16.00 ms | 16.83 ms | unchanged (noise) |

prefill now costs about the same as `forward` on both sides (282.08 vs 272.66 for jasmine, 44.46 vs
44.51 for PyTorch), which is what it should be: prefill is a full forward plus writing K/V into the
cache.

## What the numbers say

**1. The BLAS choice dominates the as-shipped gap.** jasmine links whatever `CBLAS` CMake finds, and
on a stock Debian/Ubuntu that is the reference netlib BLAS. Swapping in OpenBLAS alone: full forward
990.93 → 272.66 ms (3.6x), decode 34.77 → 16.83 ms (2.1x). Most of the "22x slower than PyTorch"
headline is a packaging accident, not a transformer implementation problem. Either ship an
optimized BLAS or document the dependency; a benchmark against MKL with netlib underneath is not
an implementation comparison.

**2. With a comparable BLAS the remaining gaps are 6.1x (batched forward) and 2.3x (single-token
decode).** The decode figure is the most apples-to-apples number here, and 2.3x is a much better
place to be than the raw comparison suggests. Both multipliers describe real headroom in
`jas_mat_gemm.hpp` / `jas_mha_t.hpp`, not measurement artifacts.

**3. Decode is memory-bandwidth bound, and jasmine reaches about half the bandwidth.** One token
must read all 82M parameters (328 MB in fp32). jasmine's 16.83 ms implies ~19 GB/s, PyTorch's
7.34 ms ~45 GB/s. For a single token the GEMMs are effectively matrix-vector products, and jasmine
routes them through `cblas_sgemm` while splitting the output rows across OpenMP threads; both the
GEMM packing and the row-panel split are poor fits for a skin-deep matrix. A `gemv` path for the
decode case is the obvious next thing to try.

**4. Thread scaling saturates early and PyTorch also regresses at 8 threads**, so neither
implementation is a scaling win to celebrate on this machine. Any follow-up should fix threads at 4.

## Raw logs

`gpt2_vs_torch_1threads.txt`, `gpt2_vs_torch_4threads.txt`, `gpt2_vs_torch_8threads.txt`.
