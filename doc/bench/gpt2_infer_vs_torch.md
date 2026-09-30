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
| jasmine comparison BLAS | OpenBLAS 0.3.15, `OPENBLAS_CORETYPE=haswell` forced (see below) |
| PyTorch | 2.9.0+cu128, CPU, MKL |
| transformers | 5.17.0 |

**The BLAS core type is not a detail.** OpenBLAS chooses its kernels at runtime from the CPU it
detects. This build contains `sgemm_kernel_HASWELL` and `sgemm_kernel_SKYLAKEX` (AVX2/FMA) but on a
13600KF it reports `openblas_get_corename() == "Prescott"` — a 2004 SSE3-era part — and runs the
corresponding kernels. OpenBLAS 0.3.15 predates Raptor Lake, so the detection misses. Pinning
`OPENBLAS_CORETYPE=HASWELL` is worth **1.70x on the full forward** with no code change:

| 4 threads | core type auto-detected ("Prescott") | pinned "HASWELL" |
|---|---|---|
| full forward, 128 tokens | 189.54 ms | **111.46 ms** |
| prefill through cache | 186.47 ms | **112.78 ms** |
| decode, ms per token | 10.73 ms | **9.78 ms** |

`tools/compare_gpt2_torch.py` pins it by default (`--blas-coretype`, pass `""` to opt out) and prints
which value it used. Every number below has it pinned. On a machine whose OpenBLAS does recognise its
CPU this changes nothing; on this one it is the single largest lever in the whole document.

The netlib BLAS is unaffected by this (it has no kernels to choose from), which makes the
"netlib vs MKL" row below a fair picture of a stock Debian/Ubuntu build.

Frequency scaling is enabled, so absolute numbers carry a few percent of run-to-run noise (the
PyTorch decode figure moved between 7.6 and 8.4 ms across sessions); ratios within one run are
reliable.

## Four threads

| metric | jasmine (netlib) | jasmine (OpenBLAS) | PyTorch (MKL) |
|---|---|---|---|
| full forward, 128 tokens | 990.93 ms | 112.35 ms | **48.24 ms** |
| prefill through cache, 128 tokens | 988.35 ms | 111.92 ms | **45.17 ms** |
| decode, ms per token | 34.77 ms | 10.12 ms | **7.70 ms** |
| prefill + 32 decode steps | 2101.08 ms | 435.83 ms | **291.49 ms** |

Slowdown vs PyTorch:

| metric | jasmine (netlib) | jasmine (OpenBLAS) |
|---|---|---|
| full forward | 20.5x | **2.33x** |
| prefill through cache | 21.9x | **2.48x** |
| decode per token | 4.51x | **1.31x** |

## Thread scaling

jasmine (OpenBLAS, core type pinned) vs PyTorch, ms:

| metric | threads | jasmine | PyTorch | ratio |
|---|---|---|---|---|
| full forward | 1 | 234.68 | 165.33 | **1.42x** |
| full forward | 4 | 112.35 | 48.24 | 2.33x |
| full forward | 8 | 119.61 | 68.14 | 1.76x |
| prefill through cache | 1 | 232.90 | 163.44 | 1.43x |
| prefill through cache | 4 | 111.92 | 45.17 | 2.48x |
| prefill through cache | 8 | 129.73 | 65.70 | 1.97x |
| decode per token | 1 | 27.64 | 11.59 | 2.39x |
| decode per token | 4 | 10.12 | 7.70 | 1.31x |
| decode per token | 8 | 9.05 | 8.58 | **1.05x** |

Single-thread, where neither side's threading policy can confuse the comparison, jasmine is within
**1.42x** on the forward. At 4 and 8 threads the ratios are worse because PyTorch scales better
inside MKL than jasmine does across its OpenMP row panels.

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

> The "vs PyTorch" figures inside the change sections were measured when they were made, before the
> BLAS core type was pinned (see Environment). They are valid as before/after evidence for the change
> itself; the headline table at the top has the current numbers.

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
baseline, 5 repeats and 2 warmup runs each:

| metric (4 threads) | GEMV off | GEMV on | change |
|---|---|---|---|
| decode, ms per token (float) | 16.81 ms | 11.85 ms | **1.42x faster** |
| decode, ms per token (double) | 20.10 ms | 15.76 ms | 1.28x faster |
| full forward | 272.88 ms | 277.22 ms | unchanged (2% is noise) |
| prefill through cache | 276.73 ms | 275.57 ms | unchanged |
| logits checksum | -2518023.4670 | -2518023.4670 | **identical** |

`forward` and `prefill` are unaffected because they run `N = 128`, which does not take the path —
that is the point of scoping it to `N == 1`. An earlier run of this A/B showed prefill 13% slower
with GEMV on, which would have contradicted the design; repeating it with the two binaries
alternated showed that was a thermal outlier (the row now reads 1.00x), so the numbers above are
from the repeated run. Decode was consistent across all five alternations (1.35–1.42x), and the
double case matters because `gpt2_generate` and `gpt2_chat` run in double.

Decode bandwidth: 328 MB per token, so 12.01 ms is 27.3 GB/s against PyTorch's 8.44 ms at 38.9 GB/s
and the ~50 GB/s the machine can deliver. The remaining 1.42x is not in the big weights any more:
`lm_head` (154 MB of the 328 MB) now streams at GEMV speed, while the other ~174 MB lives in the
per-layer matrices, which are cache-resident and deliberately left on BLAS, plus the attention and
elementwise work that does not stream weights at all.

## Change 3: the matrix index accessor

The forward profile pointed at something much broader than decode. Splitting `forward` into its
phases on distilgpt2 at T=128:

| phase | ms | share |
|---|---|---|
| forward (total) | 266.75 | 100% |
| embed | 1.13 | 0.4% |
| 6 blocks, each | ~30.8 | 69% |
| head (ln_f + lm_head) | 77.14 | 28.9% |

Raw BLAS GEMMs of the same shapes sum to 115.74 ms, so **151 ms of the 267 ms was not GEMM at all**.
Drilling into one block:

| sub-layer | ms | note |
|---|---|---|
| ln_1 | 2.82 | a 0.4 MB matrix; a fused hand-written version does it in 0.60 |
| attn | 7.68 | |
| ln_2 | 2.82 | |
| mlp_fc | 4.14 | 3.20 of it is the GEMM |
| mlp_proj | 3.50 | 3.20 of it is the GEMM |
| residual + gelu | 7.54 | |

Isolating the cause on a plain `a - b` pass over 3 MB:

| accessor | ms | GB/s |
|---|---|---|
| raw pointer loop | 0.129 | **48.9** |
| raw pointer loop, 4 threads | 0.067 | 93.7 |
| **`mat_t::operator()` loop** | **4.700** | **1.3** |
| `(x - y).clone()` | 3.771 | 1.7 |
| copy ctor | 0.158 | 39.9 |

`operator()` was doing two integer modulo operations per element:

```cpp
int i = r % row_num();
int j = c % col_num();
```

An integer division is ~20-40 cycles on x86 and does not pipeline, and the accessor is called once
per element per operand, so a three-operand expression paid six divisions per element. Measured in
isolation the column index is the expensive one (it is the inner loop): protecting both indices with
a range check gives 6.8 → 22.0 GB/s.

The wraparound is deliberate — it is how the expression layer broadcasts `[R,1]` and `[1,C]` operands
and how periodic indexing works — so it stays; only the common in-range case takes the new branch.
The change is semantics-preserving for every input, including negative indices (the unsigned compare
fails for those, so they still reach the modulo).

| metric | before | after | change |
|---|---|---|---|
| full forward, 128 tokens | 278.39 ms | 210.83 ms | **1.32x faster** |
| prefill through cache | 288.60 ms | 216.92 ms | 1.33x faster |
| decode per token | 12.01 ms | 11.42 ms | 1.05x faster |
| block | 28.49 ms | 20.51 ms | 1.39x faster |
| ln_1 / ln_2, each | 2.82 ms | 0.96 ms | 2.95x faster |
| attention core, per layer | 4.71 ms | 2.04 ms | 2.31x faster |
| slowdown vs PyTorch, forward | 5.63x | **4.72x** | — |

It helps batched forward more than decode, because `operator()` is on every elementwise layer and
those dominate the batch-128 path, while decode is dominated by the weight stream. All 280 tests
pass, the three golden alignment tests against real weights pass, and `tools/verify_gpt2.py` still
matches HuggingFace token for token.

## Change 4: a fold-free accessor for validated loops

`mat_t::operator()` folds out-of-range indices with `%`. That is deliberate and stays — it is how the
expression layer broadcasts a `[R,1]` or `[1,C]` operand, and how periodic indexing works. But it
also means no loop that uses it can be vectorized, because folding is a data-dependent branch plus a
division. Measured on a plain `(a - b)` pass over 3 MB:

| accessor | GB/s | compiler verdict |
|---|---|---|
| raw pointer loop | 48.9 | `loop vectorized using 16 byte vectors` |
| `operator()`, folding range check | 22.0 | `not vectorized: control flow in loop` |
| `operator()`, unconditional `%` | 6.8 | `couldn't vectorize` |

So a range check alone is not enough — the branch itself blocks vectorization, and a loop that
promises folding can never drop it. `mat_t::unchecked(r, c)` was added for loops that have already
validated their shapes: identical addressing without the fold, asserting the precondition in debug
builds and compiling away under `NDEBUG`. The broadcast guarantee is untouched; the promise simply
moves into the hot paths that can actually keep it.

Applied to the elementwise kernel of the expression layer (output store, plus hoisting the loop
bounds, which for an expression node meant re-walking the subtree every iteration) and to the
`sum` / `vsum` / `pow` helpers:

| metric | before | after | change |
|---|---|---|---|
| `(x - y).clone()`, 3 MB | 3.771 ms | 0.971 ms | **3.9x faster** |
| full forward, 128 tokens | 210.83 ms | 191.41 ms | 1.10x faster |
| prefill through cache | 216.92 ms | 187.88 ms | 1.15x faster |
| ln_1 / ln_2, each | 0.958 ms | 0.757 ms | 1.27x faster |
| slowdown vs PyTorch, forward | 4.72x | **4.27x** | — |
| slowdown vs PyTorch, prefill | 4.86x | **4.21x** | — |

Two things to be honest about. First, the safe subset of this change (output store and hoisted
bounds) only bought 1.3% on the forward, because the *operands* still fold — making those fold-free
requires per-operand shape checks, since a broadcast operand is genuinely smaller than the result.
Second, `pow(x, 2.0)` now computes `x * x`; that is a numerical change, not just a scheduling one.
It is what LayerNorm's variance step wants, and the golden alignment tests plus the HuggingFace
cross-check both still pass, but it is not bit-identical to the library call in principle.

All 280 tests pass, the three golden alignment tests against real weights pass, and
`tools/verify_gpt2.py` still matches HuggingFace token for token.

## Change 5: fused softmax and a fold-free gelu

Two more explicit elementwise loops were still going through the folding accessor:

- `gelu_net_t::forward` (once per element of every FFN hidden state) now uses hoisted bounds and
  `unchecked`.
- `hsoftmax` built four intermediate matrices (`hmax`, `input - max`, `exp`, `hsum`, then the
  division). Attention calls it once per head per layer, so on a `[T,T]` score matrix the temporaries
  dominated, not the transcendentals. It is now one fused pass per row.

| metric (4 threads) | before | after |
|---|---|---|
| attention core, per layer | 1.826 ms | **1.467 ms** |
| full forward | 187.14 ms | 186.82 ms |

The forward barely moved, which is worth stating plainly: the remaining ~49% of the forward that is
not GEMM is spread thin across many small operations, and shaving individual ones no longer shows up
above the noise. That is a signal to stop tuning and look at the structure instead — which is what
led to the BLAS finding in the Environment section above.

## What the numbers say

**1. The BLAS is three separate problems wearing one name**, and together they were the whole
headline. In order of size:

| lever | effect on the 4-thread forward |
|---|---|
| core type: `OPENBLAS_CORETYPE=HASWELL` (was silently running SSE3 kernels) | 189.54 → 111.46 ms (**1.70x**) |
| BLAS choice: netlib reference vs OpenBLAS | 990.93 → 189.54 ms (**5.2x**) |

Neither is a line of jasmine code. A benchmark against MKL with the reference netlib BLAS underneath
measures packaging, not implementation, and a benchmark against MKL with a mis-detected OpenBLAS
measures CPU detection. Both had to be fixed before the remaining gap meant anything.

**2. Three real code problems accounted for most of the rest**, all of the same shape — "the
arithmetic is fine, the scheduling is not": stepping the prompt token by token (7.2x on prefill),
asking GEMM to do a matrix-vector product (1.42x on decode), and an index accessor doing two integer
divisions per element on every elementwise layer (1.32x on the forward, 3.9x on the raw elementwise
kernel).

**3. Starting point to now, 4 threads, like-for-like:**

| metric | at the start | now | change |
|---|---|---|---|
| full forward | 990.93 ms (20.5x vs PyTorch) | 112.35 ms (**2.33x**) | 8.8x |
| prefill through cache | 988.35 ms (21.9x) | 111.92 ms (**2.48x**) | 8.8x |
| decode per token | 34.77 ms (4.51x) | 10.12 ms (**1.31x**) | 3.4x |

At one thread, where no threading policy can confuse the comparison, the forward is within **1.42x**.

**4. What is left is not one thing.** With a correctly-tuned BLAS the projection GEMMs are roughly
55-60 ms of the 112 ms forward, and MKL does the same shapes in 40.66 ms, so GEMM efficiency is the
larger remaining piece and it is bounded by the BLAS we are allowed to link. The other ~50 ms is
attention, normalization and elementwise work spread across many small operations; no single one of
them is large enough to move the total, which is why further micro-tuning here has stopped paying.

**5. We have not beaten PyTorch on the forward, and on this machine we probably cannot.** Closing
2.33x would need a BLAS at MKL's level plus removing essentially all of the 50 ms of non-GEMM work.
Decode at 8 threads is at parity (1.05x), and single-token decode is bandwidth-bound and within 1.31x
at four threads. Those are the honest numbers.

## Raw logs

`gpt2_vs_torch_1threads.txt`, `gpt2_vs_torch_4threads.txt`, `gpt2_vs_torch_8threads.txt`,
`gpt2_gemv_ab.txt` (the GEMV A/B run). The three `gpt2_vs_torch_*.txt` logs were recorded with
`OPENBLAS_CORETYPE` pinned; earlier versions of them (auto-detected core type) are in the git history.
