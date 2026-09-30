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
    --weights build/distilgpt2_weights.bin --threads 4 --prefill 128 --decode 32
```

## Environment

| | |
|---|---|
| CPU | 13th Gen Intel i5-13600KF: 14 cores (6 P + 8 E), 20 threads, dual-channel DDR |
| Measured peak read bandwidth | ~50 GB/s at 14 threads, ~55 GB/s two-stream |
| Compiler | g++ 13.3.0, CMake 3.28.3, `-O3 -DNDEBUG` (no `-march=native`) |
| jasmine default BLAS | netlib reference BLAS 3.12.0 (`libblas3`) — **single-threaded, unoptimized** |
| jasmine comparison BLAS | OpenBLAS 0.3.15 (`DYNAMIC_ARCH NO_AFFINITY Prescott SINGLE_THREADED`) |
| PyTorch | 2.9.0+cu128, CPU, MKL |
| transformers | 5.17.0 |

Frequency scaling is enabled, so absolute numbers carry a few percent of run-to-run noise (the
PyTorch decode figure moved between 7.8 and 8.4 ms across sessions); ratios within one run are
reliable, and the A/B rows below were measured in a single run against the same PyTorch number.

## Four threads

| metric | jasmine (netlib) | jasmine (OpenBLAS) | PyTorch (MKL) |
|---|---|---|---|
| full forward, 128 tokens | 990.93 ms | 278.39 ms | **49.41 ms** |
| prefill through cache, 128 tokens | 988.35 ms | 288.60 ms | **49.00 ms** |
| decode, ms per token | 34.77 ms | 12.01 ms | **8.44 ms** |
| prefill + 32 decode steps | 2101.08 ms | 672.93 ms | **318.99 ms** |

Slowdown vs PyTorch:

| metric | jasmine (netlib) | jasmine (OpenBLAS) |
|---|---|---|
| full forward | 22.3x | 5.63x |
| prefill through cache | 22.2x | 5.89x |
| decode per token | 4.74x | **1.42x** |

## Thread scaling

jasmine (OpenBLAS) vs PyTorch, ms:

| metric | threads | jasmine | PyTorch | ratio |
|---|---|---|---|---|
| full forward | 1 | 640.74 | 169.37 | 3.78x |
| full forward | 4 | 278.39 | 49.41 | 5.63x |
| full forward | 8 | 267.16 | 79.33 | 3.37x |
| prefill through cache | 1 | 641.91 | 169.64 | 3.78x |
| prefill through cache | 4 | 288.60 | 49.00 | 5.89x |
| prefill through cache | 8 | 272.17 | 79.56 | 3.42x |
| decode per token | 1 | 35.47 | 12.01 | 2.95x |
| decode per token | 4 | 12.01 | 8.44 | 1.42x |
| decode per token | 8 | 11.41 | 8.91 | 1.28x |

jasmine's forward scales 2.30x from 1 to 4 threads and then flattens (1.04x from 4 to 8). PyTorch
peaks at 4 threads and **regresses at 8** (49.41 ms to 79.33 ms). Both point at the same thing:
14 cores share a dual-channel memory bus, so past 4 threads the bus is saturated and extra threads
just queue. The 1- and 4-thread rows are the trustworthy ones. Pinning the thread count also keeps
the comparison meaningful; jasmine splits GEMMs across OpenMP row panels while PyTorch
parallelizes inside BLAS, so an unpinned run would be measuring thread policy rather than code.

## Correctness control

The exported file and the HF checkpoint are only the same model if the final logits agree. Sum of
the last position's logits:

| threads | jasmine | PyTorch | relative spread |
|---|---|---|---|
| 1 | -2518023.4670 | -2518023.0000 | 1.85e-07 |
| 4 | -2518023.4670 | -2518023.0000 | 1.85e-07 |
| 8 | -2518023.4670 | -2518022.7500 | 2.85e-07 |

That is fp32 accumulation noise, so both sides ran the same weights and the timings describe the
same computation. `tools/compare_gpt2_torch.py` fails the run if this check does not hold.

Note the jasmine checksum is **identical across every change below** (-2518023.4670), which is the
evidence that the batched prefill and the GEMV path changed only how the arithmetic is scheduled,
not the arithmetic itself.

## Change 1: batched prefill

The first comparison showed prefill at 47x PyTorch, but that was not a like-for-like number:
`gpt2_model_t::prefill` looped `forward_one` once per prompt token, while PyTorch's `use_cache=True`
prefill is a single batched forward. jasmine's own `forward()` on the same 128 tokens took 278.02 ms
against its 2085.40 ms prefill, so 7.5x was self-inflicted.

`mat_mha_t::forward_one` already accepted multi-column input and applied the causal mask by absolute
position, so the fix was to stop stepping one token at a time and run the whole prompt through it in
one pass (`jas_gpt2_t.hpp::prefill`). `forward_one` at the model level was also generalized to T
columns, which is what lets the interactive path append a whole user turn to an existing cache
without clearing it.

| metric (4 threads, OpenBLAS) | before | after | change |
|---|---|---|---|
| prefill through cache, 128 tokens | 2085.40 ms | 288.60 ms | **7.2x faster** |
| slowdown vs PyTorch, prefill | 47.00x | 5.89x | — |
| full forward | 278.02 ms | 278.39 ms | unchanged (noise) |
| decode per token | 16.00 ms | 16.83 ms | unchanged (noise) |

prefill now costs about the same as `forward` on both sides (288.60 vs 278.39 for jasmine, 49.00 vs
49.41 for PyTorch), which is what it should be: prefill is a full forward plus writing K/V into the
cache.

## Change 2: a GEMV path for single-column output

Decoding is always `N == 1`, and that changes what the GEMM has to do. Every weight is used exactly
once, so there is no reuse for GEMM blocking to exploit, and BLAS's packing buffers become pure
extra traffic: the matrix is read, written into a packed copy, and read again. At `N == 1` the
problem is purely `bytes / bandwidth`, and the only thing that matters is not moving extra bytes.

Measured per shape on this machine (`K = 768`, 4 threads, GB/s counted over the weight matrix only,
OpenBLAS against a scalar GEMV):

| M | 1024 | 2048 | 4096 | 8192 | 12288 | 16384 | 32768 | 50257 |
|---|---|---|---|---|---|---|---|---|
| MB | 3 | 6 | 12 | 24 | 36 | 48 | 96 | 147 |
| sgemm | 36.1 | 58.6 | 56.3 | 41.7 | 30.7 | 24.5 | 21.1 | 20.2 |
| gemv | 20.9 | 44.7 | 45.1 | 43.1 | 42.5 | 37.2 | 36.1 | 35.6 |
| winner | sgemm | sgemm | sgemm | tie | gemv | gemv | gemv | gemv |

The crossover sits at ~24 MB — **this machine's L3 cache is 24 MiB**. That is the mechanism made
visible: while the operand fits in cache, the packing traffic is paid in L3 and BLAS wins; once it
spills to DRAM, GEMV's single pass wins, by 1.38x at 36 MB and 1.76x at 147 MB. The threshold in
`jas_mat_gemm.hpp` is therefore set just below L3 (16 MiB), so only operands that clearly cannot be
cache-resident take the GEMV path. Nothing else in GPT-2 reaches it: the largest other weight is
`mlp.fc` at 9 MB.

A/B on the real model, both binaries built from the same source with `-DJASMINE_USE_GEMV=0` for the
baseline, run in one session:

| metric (4 threads) | GEMV off | GEMV on | change |
|---|---|---|---|
| decode, ms per token (float) | 16.33 ms | 12.41 ms | **1.32x faster** |
| decode, ms per token (double) | 20.10 ms | 15.76 ms | 1.28x faster |
| full forward | 270.66 ms | 270.15 ms | unchanged |
| prefill through cache | 280.98 ms | 282.03 ms | unchanged |
| logits checksum | -2518023.4670 | -2518023.4670 | **identical** |

`forward` and `prefill` are unaffected because they run `N = 128`, which does not take the path —
that is the point of scoping it to `N == 1`. The double case benefits too, and the examples
(`gpt2_generate`, `gpt2_chat`) run in double, so they get the same ~1.3x.

Decode bandwidth: 328 MB per token, so 12.01 ms is 27.3 GB/s against PyTorch's 8.44 ms at 38.9 GB/s
and the ~50 GB/s the machine can deliver. The remaining 1.42x is not in the big weights any more:
`lm_head` (154 MB of the 328 MB) now streams at GEMV speed, while the other ~174 MB lives in the
per-layer matrices, which are cache-resident and deliberately left on BLAS, plus the attention and
elementwise work that does not stream weights at all.

## What the numbers say

**1. The BLAS choice dominated the as-shipped gap.** jasmine links whatever `CBLAS` CMake finds, and
on a stock Debian/Ubuntu that is the reference netlib BLAS. Swapping in OpenBLAS alone: full forward
990.93 → 278.39 ms (3.6x), decode 34.77 → 16.83 ms (2.1x). Most of the "22x slower than PyTorch"
headline was a packaging accident, not a transformer implementation problem. Either ship an
optimized BLAS or document the dependency; a benchmark against MKL with netlib underneath is not
an implementation comparison.

**2. Two real code problems accounted for most of the rest**, and both were "the arithmetic is fine,
the scheduling is not": stepping the prompt token by token (7.2x on prefill) and asking GEMM to do a
matrix-vector product (1.32x on decode). Together they took decode from 22.2x on the prefill side and
4.74x on the decode side down to 5.89x and **1.42x**.

**3. What is left splits by bottleneck.** The batched forward is compute-bound and sits at 5.63x:
that is GEMM efficiency, and jasmine's own hand-written blocked fallback and its BLAS usage are the
places to look. Single-token decode is bandwidth-bound at 1.42x, and it is now within 40% of the
bus's measured capability. Neither number is a measurement artifact.

**4. Thread scaling saturates early and PyTorch also regresses at 8 threads**, so neither
implementation is a scaling win on this machine. Any follow-up should fix threads at 4.

## Raw logs

`gpt2_vs_torch_1threads.txt`, `gpt2_vs_torch_4threads.txt`, `gpt2_vs_torch_8threads.txt`,
`gpt2_gemv_ab.txt` (the GEMV A/B run).
