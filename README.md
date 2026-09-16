# jasmine

A **C++20, header-only** matrix expression-template library, together with the Transformer stack
built on top of it. The same matrix and expression code runs on both **CPU (OpenMP + BLAS)** and
**CUDA (fused kernels + cuBLAS)**, and comes with inference alignment for GPT-2 and LLaMA-family
models (TinyLlama / Mistral / Qwen2 …), training implementations, and interactive chat demos.

```cpp
#include "jas_transformer_t.hpp"

mat_t<float> a(64, 64, /*...*/), b(64, 64, /*...*/), c(64, 64, /*...*/);

mat_t<float> y = a * b + c;                  // one expression tree, no intermediate temporaries
mat_t<float> z = (a + b) * c / 2.0f - exp(a);
mat_t<float> s = q.t().dot(k);               // matrix product (CPU: BLAS / GPU: cuBLAS)
```

> `*` is an **element-wise** product (like `+` and `exp`, it is a fusable element-wise operator).
> **The matrix product is `.dot()`** — it has to be split out to BLAS / cuBLAS, and cannot be fused.

---

## 1. Features

| Area | What's in it |
| --- | --- |
| Matrices & expressions | `mat_t` / `mat_view_t`; lazy expression trees, compile-time fusion, scalar operands, transposed views, zero-copy subviews |
| Operators | Arithmetic, comparisons, `exp` / `log` / `sqrt` / `sigmoid`, `dot`, row/column reductions, row-wise softmax, LayerNorm / RMSNorm |
| Layers | Linear, LayerNorm / RMSNorm, SiLU / GELU / ReLU, SwiGLU gating, residual, Embedding, cross-entropy / MSE |
| Models | decoder-only base; **GPT-2** (absolute positions + `gelu_new` + tied lm_head); **LLaMA family** (RoPE + RMSNorm + SwiGLU + GQA/MQA); enc-dec stack |
| Training | Backprop checked against numerical gradients; SGD / Adam / NAdam; gradient accumulation (`cache_updator_t`) |
| Inference | KV-cache prefill + per-token decoding; greedy / top-k / top-p sampling; streaming output |
| Weights | Single-file `JASMINE_WEIGHTS_V1` format; direct export from HuggingFace (`tools/`); layer-by-layer / logits golden-value alignment |
| CUDA | Element-wise chains fused into a single kernel launch, cuBLAS GEMM, reductions, device-side KV cache / RoPE / MHA / a whole LLaMA, backwards and optimizers, **fused attention**, **bf16 / fp16 mixed precision** |

**Status**: the host side passes `ctest` 178/178; the CUDA backend has 174 cases (compute-heavy cases
are skipped by default). The CUDA backend is still being iterated on; target-machine vs test-machine
differences are covered in [`CUDA.md`](CUDA.md), section 11.

---

## 2. Getting started

### 2.1 Dependencies

| Dependency | Required | Notes |
| --- | --- | --- |
| C++20 compiler | ✅ | Needs concepts / `requires` (GCC 13+, Clang 16+) |
| CMake ≥ 3.16 | ✅ | |
| OpenMP | optional | Disable with `-DJASMINE_USE_OPENMP=OFF` |
| BLAS (OpenBLAS / BLAS) | optional | Large matmuls go through `cblas_*gemm`; falls back to a blocked GEMM when absent |
| CUDA Toolkit 12.4 | optional | Only to build the CUDA backend (`-DJASMINE_USE_CUDA=ON`) |
| Python 3 + `transformers` / `torch` | optional | Only for exporting weights and running the tokenizer services; **the C++ side does not depend on Python** |

googletest and Google Benchmark are fetched automatically by CMake `FetchContent`
(the first configure step needs network access).

### 2.2 Build and test

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure     # 178 cases
```

Common configuration options:

| Option | Default | Notes |
| --- | --- | --- |
| `JASMINE_BUILD_TESTS` | `ON` | Build the googletest unit tests |
| `JASMINE_BUILD_BENCH` | `ON` | Build the Google Benchmark targets |
| `JASMINE_BUILD_EXAMPLES` | `ON` | Build the programs under `examples/` |
| `JASMINE_USE_OPENMP` | `ON` | Enable OpenMP |
| `JASMINE_USE_BLAS` | `ON` | Route large matmuls through BLAS |
| `JASMINE_USE_CUDA` | `OFF` | Build the CUDA backend and its tests |
| `JASMINE_CUDA_ARCHITECTURES` | `61` | CUDA target architecture; set it for your target machine (e.g. `80` / `86` for Ampere) |

### 2.3 CUDA backend

```bash
cmake -S . -B build-cuda -DJASMINE_USE_CUDA=ON
cmake --build build-cuda -j
./build-cuda/tests/cuda_tests                        # fast, no heat
JASMINE_CUDA_STRESS=1 ./build-cuda/tests/cuda_tests  # includes compute-heavy cases (4096² fused, 256³ GEMM)
```

Usage overview (full details in [`CUDA.md`](CUDA.md)):

```cpp
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"

auto y = cuda::eval_fused_to_host((a.leaf() + b.leaf()) * c.leaf());  // whole chain, one launch
auto s = cuda::gemm_to_host(q.leaf(), k.leaf().t());                  // S = Q·Kᵀ, via cuBLAS
auto p = cuda::softmax_rows(s.leaf() / scale + mask.leaf());          // mask folded into the expression
```

> ⚠️ **Test-machine thermals**: the development machine for this repository has a fanless Tesla P4,
> and sustained full load will overheat it. When running GPU cases, watch the temperature and the
> duration, and prefer `cuda_tests` (which excludes the stress cases). The P4 is *not* the target machine.

### 2.4 Running the demos

First export HuggingFace weights into jasmine's single-file format using `tools/`:

```bash
# GPT-2 family (distilgpt2 is ~313 MiB, gpt2 ~476 MiB)
python3 tools/export_gpt2.py --model distilgpt2 --out build/gpt2_weights.bin

# LLaMA family (TinyLlama-1.1B-Chat is ~4.2 GiB)
python3 tools/export_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                              --out build/tinyllama_weights.bin
```

Then just start chatting (the tokenizer is provided by a long-lived Python subprocess):

```bash
./build/examples/gpt2_chat  build/gpt2_weights.bin
./build/examples/llama_chat build/tinyllama_weights.bin
```

The main options for `llama_chat`:

```
--model NAME        HF model name for the tokenizer (default: read from the weight manifest)
--system TEXT       system prompt
--max-new N         max tokens to generate per turn (default: 256)
--temperature T     sampling temperature (default: 0.7)   --top-k K   top-k cutoff (default: 50)
--top-p P           nucleus threshold (default: 0.95)     --greedy    argmax decoding
--max-context N     context limit (default: the model's n_pos)
--no-stream         disable streaming output
```

In-session commands: `/exit` to quit, `/reset` to clear the context, `/context` to show the context
length (`gpt2_chat` additionally has `/params` and `/set KEY VALUE`).

Other executables:

| Target | Purpose |
| --- | --- |
| `train_transformer` / `train_transformer_ce` | Small-scale Transformer training examples (MSE / cross-entropy) |
| `train_tf_base` / `train_tf_kernel` | Demonstrate training with the base layers and with the kernel layers, respectively |
| `gpt2_generate` | Batch GPT-2 generation (the counterpart that `tools/verify_gpt2.py` cross-checks against) |
| `cuda_fused_demo` | Device-side fused evaluation / attention demo (requires `JASMINE_USE_CUDA=ON`) |

---

## 3. Design notes

### 3.1 Expression templates and the value-category contract

`a * b + c` is not evaluated immediately; it builds an expression tree that is only materialized when
assigned to a `mat_t`, evaluating element by element — no intermediate result is ever materialized.
That requires the operands to stay valid for the lifetime of the tree, so storage is chosen by
**value category**:

- lvalue operands are held by `const T&` (the caller guarantees the lifetime);
- rvalue operands are **taken by value** (otherwise they would dangle);
- types that must be owned by value even when passed as lvalues (such as device leaves) opt in via
  the `operand_owned_by_value` customization point.

The details, the dangling-reference bug that used to be here, and the deliberately retained
boundaries are in [`TESTING.md`](TESTING.md), section 9.

### 3.2 Networks and models

`jas_net_t.hpp` provides the generic shell of "weights + updater + per-layer cache", while
`jas_transformer_kernel_t.hpp` / `jas_transformer_t.hpp` assemble the encoder-decoder and
decoder-only stacks, and `jas_gpt2_t.hpp` / `jas_llama_t.hpp` each lay out the layer order of their
target model. Every difference from jasmine's original stack is documented with its *why* in the file
header (e.g. GPT-2's pre-norm + absolute positions + `gelu_new` + tied lm_head, and LLaMA's
RMSNorm + RoPE + SwiGLU + GQA).

RoPE rotation tables are shared per `(d_head, pairing convention)` through a process-wide registry
and filled concurrently by multiple threads — a second shared mutable structure alongside the
"don't materialize the probability matrix" story. Its invariant and the two rounds of concurrency
fixes are in [`TESTING.md`](TESTING.md), section 10.

### 3.3 CUDA backend

The reason expression templates work on the device at all is that evaluating a whole tree is a **pure
scalar function** at the device level, while a leaf degenerates into a trivially-copyable thin shell
("device pointer + dimensions + leading dimension", `dev_mat_t`). So a whole element-wise chain fuses
into **one kernel launch**. The matrix product, however, has to be split out at `dot` and handed to
cuBLAS — doing a GEMM inside a fused kernel is a pure loss. That boundary, plus the compile-time
probes for what can and cannot go on the device, are in [`CUDA.md`](CUDA.md), sections 3 and 4.

---

## 4. Repository layout

```
jasmine/
├── jas_*.hpp              the library itself (header-only)
│   ├── jas_mat_t.hpp          matrices and views
│   ├── jas_mat_express_t.hpp  expression templates and operators
│   ├── jas_mat_gemm.hpp       GEMM (BLAS / blocked fallback)
│   ├── jas_net_t.hpp          layer shell (weights + updater + cache)
│   ├── jas_mha_t.hpp          multi-head attention (RoPE, causal mask, KV cache)
│   ├── jas_RoPE_t.hpp         rotary positional embeddings and the shared registry
│   ├── jas_kv_cache_t.hpp     KV cache
│   ├── jas_transformer_*.hpp  the Transformer stacks
│   ├── jas_gpt2_t.hpp         GPT-2 forward
│   ├── jas_llama_t.hpp        LLaMA-family forward
│   ├── jas_weight_io.hpp      weight file I/O
│   └── jas_cuda_*.hpp         the CUDA backend (only built when CUDA is enabled)
├── examples/              runnable examples and the interactive demos
├── tests/                 unit tests (host unit_tests / device cuda_tests)
├── benches/               Google Benchmark benchmarks (matmul, MHA)
├── tools/                 Python-side weight export, tokenizer services and cross-check scripts
├── doc/                   checklists and benchmark data (including before/after comparisons)
├── TESTING.md             testing and alignment methodology
└── CUDA.md                CUDA backend design document
```

`main.cpp` / `makefile` / `run.sh` are early standalone entry points kept only for compatibility;
use the CMake targets instead.

---

## 5. Testing and reproduction

```bash
# full host suite
cmake --build build -j && ctest --test-dir build --output-on-failure

# just one group
./build/tests/unit_tests --gtest_filter='RoPE.*:LlamaStructure.*'
```

The scripts under `tools/` cross-check end to end against HuggingFace (export + comparison in one command):

```bash
python3 tools/verify_gpt2.py --model distilgpt2 --text "The capital of France is" --logits
```

A few methodological choices are deliberate; see [`TESTING.md`](TESTING.md) for the full story:

- **Golden-value alignment**: both the per-layer hidden states and the final logits must match —
  comparing logits alone lets normalization layers mask deviations.
- **Two independent judges**: backprop is checked against both the host analytic solution and finite
  differences; fused attention is checked against both the non-fused implementation and the host formula.
- **Falsification-style cases**: not just "is the result right", but "did the optimization actually
  kick in" — e.g. a launch counter asserting the single-pass softmax path was taken, and asserting
  `m_weights` is empty to prove the probability matrix really was not materialized.
- **Tolerances are computed, not fudged**: mixed-precision tolerances come from the error model
  rather than from a number that happens to make the test pass.

---

## 6. License

This project is released under the **GNU General Public License v3.0**, see [`LICENSE`](LICENSE).
