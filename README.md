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
| Matrices & expressions | `mat_t` / `mat_view_t` / `mat_reshape_view_t`; lazy expression trees, compile-time fusion, scalar operands, transposed views, zero-copy subviews and shape views, zero-copy BLAS operands |
| Operators | Arithmetic, comparisons, `exp` / `log` / `sqrt` / `sigmoid`, `dot`, row/column reductions, row-wise softmax, LayerNorm / RMSNorm |
| Layers | Linear, **RBM / DBN (contrastive divergence + static stacking)**, Conv2d (im2col + GEMM, full backward)**, **MaxPool2d / AvgPool2d (full backward)**, **Flatten**, **MeanPool (token sequences)**, **Dropout**, LayerNorm / RMSNorm, SiLU / GELU / ReLU, SwiGLU gating, residual, **Transformer encoder**, Embedding, cross-entropy / MSE |
| Models | decoder-only base; **GPT-2** (absolute positions + `gelu_new` + tied lm_head); **LLaMA family** (RoPE + RMSNorm + SwiGLU + GQA/MQA); enc-dec stack |
| Training | Backprop checked against numerical gradients; SGD / Adam / NAdam / **AdamW (decoupled weight decay)**; gradient accumulation (`cache_updator_t`); **cosine annealing with warm restarts** (`cosine_annealing_decay`) |
| Inference | KV-cache prefill + per-token decoding; greedy / top-k / top-p sampling; streaming output |
| Weights | Single-file `JASMINE_WEIGHTS_V1` format; **training-result serialization** (per-layer params + meta in one file, `save`/`load` round-trip); direct export from HuggingFace (`tools/`); layer-by-layer / logits golden-value alignment |
| CUDA | Element-wise chains fused into a single kernel launch, cuBLAS GEMM, reductions, device-side KV cache / RoPE / MHA / a whole LLaMA, backwards and optimizers, **fused attention**, **bf16 / fp16 mixed precision** |

**Status**: the host side passes `ctest` 258/258 (one case self-skips, see TESTING.md 15); the CUDA backend has 174 cases (compute-heavy cases
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
JASMINE_CUDA_STRESS=0 ./build-cuda/tests/cuda_tests  # force fast mode
JASMINE_CUDA_STRESS=1 ./build-cuda/tests/cuda_tests  # force compute-heavy cases
# sm_80+ targets run the stress cases by default; the fanless P4 does not.
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
│   ├── jas_mat_view_t.hpp    sub-views, transposes and zero-copy reshape views
│   ├── jas_mat_gemm.hpp       GEMM (BLAS / blocked fallback)
│   ├── jas_net_t.hpp          layer shell (weights + updater + cache)
│   ├── jas_rbm_t.hpp          RBM + DBN (greedy CD pretraining, static stacking)
│   ├── jas_conv_t.hpp         2-D convolution (im2col + GEMM, forward/backward)
│   ├── jas_pool_t.hpp         2-D max / average pooling (forward/backward)
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

### 4.1 MNIST + DBN (RBM static stacking)

`examples/mnist_dbn.cpp` stacks 2 RBMs + a classification head + CE into a DBN with
`dbn_net_t<2, upr_tpl>` (`jas_rbm_t.hpp`): layer-wise greedy CD-k pretraining, then supervised
fine-tuning, then serialization. Measured (2000 training images, 3 pretraining epochs, 3 fine-tuning
epochs, 1000 test images):

```text
[pretrain] reconstruction error 0.380 -> 0.247   (greedy CD-1)
[finetune] epoch 3 loss=0.112 test_acc=0.68
```

Note: the fine-tuning gradients are **summed over the batch**, not averaged, so `--ft-lr` has to be
roughly one batch smaller than the pretraining learning rate (at the default 1e-3 the test accuracy is
only 0.09; 2e-4 gives 68%). The trap is documented in `TESTING.md`, section 15.

### 4.2 MNIST convolution toy (static layer stacking)

`examples/mnist_conv.cpp` builds every architecture with the static layer stacking API
(`complex_net_builder_t` -> `complex_net_t`, so `forward` / `backward` / `step` / `init_weight` all go
through the chain) and compares three topologies on the same data, hyper-parameters and seed:

```text
cnn2 = conv->relu->pool->conv->relu->pool->flatten->(fc->relu->fc)->ce   "standard CNN"
cnn1 = conv->relu->pool->flatten->(fc->relu->fc)->ce                     (conv + MLP encoder)
trf  = conv->relu->pool->conv->relu->pool->(patch embedding)->Transformer encoder->mean pool->fc->ce
       the conv stem matches cnn2 (two downsamplings to 7x7); the patch embedding projects every
       spatial position into one token (16 channels -> d_model); the encoder is the library's
       encoder_t (bidirectional self-attention + LayerNorm + FFN + residual, assembled from
       complex_net, with RoPE supplying the token order); mean_pool averages the 49 tokens.

# compare both topologies for 3 epochs (arch=both is the default)
./build/examples/mnist_conv --data-dir build/mnist --epochs 3 --train-limit 6000 \
    --batch 16 --lr 2e-3 --arch both --save build/mnist/cmp

# 6000 training images x 3 epochs: the two CNN variants
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       3    0.098669      0.9695      0.9724   120.618
cnn1          202330       3    0.119815    0.962667       0.964   113.324
test_acc difference (cnn1 - cnn2) = -0.0084

# 3000 x 6 epochs, fixed lr 1e-3, no dropout
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       6    0.104656    0.969333      0.9535    73.53
trf            21386       6     0.27073    0.921667       0.899   118.73
test_acc difference (trf - cnn2) = -0.0545

# same scale, but with cosine annealing + warm restarts (--scheduler cosine) and dropout 0.2
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       6    0.132647    0.959333      0.9705    66.86
trf            21386       6     0.33591    0.899333      0.9165   110.03
test_acc difference (trf - cnn2) = -0.0540

# equal capacity (trf scaled to 3 layers / d=64 / CLS token; 0.4% parameter gap), 1500 x 4 epochs
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       4    0.254249    0.917333       0.926    25.88
trf           105642       4     1.12019    0.571333      0.6365   121.94
test_acc difference (trf - cnn2) = -0.2895

# equal capacity with an auto-solved CNN width
--arch match            # computes trf's parameter count, then shrinks cnn2's hidden width to match
```

Reading the numbers:

- **Within the CNN family**: cnn2 (two downsamplings) wins by 0.84 points with half the parameters
  (105k vs 202k) -- the second convolution turns spatial detail into fewer, more abstract features,
  which beats feeding a wide feature map into a large fully-connected encoder.
- **CNN vs Transformer**: after cutting the token count from 196 to 49 (a second pool; self-attention
  cost drops 16x) and giving the Transformer the same two-conv stem, it improves from 55.9% to **89.9%**,
  closing the gap to 5.5 points while using only 1/5 of the CNN's parameters (21.4k vs 105.2k) -- per
  parameter it is the cheaper model. The price is being 1.6x slower per epoch.
- **Training tools**: the loop drives the library's `cosine_annealing_decay` (cosine annealing with warm
  restarts, stepped per mini-batch; half of the total steps forms the first cycle, so the run contains
  exactly one restart) plus `dropout_net_t` (inverted dropout, explicitly disabled for evaluation).
  Together they lift cnn2 95.35% -> **97.05%** and trf 89.9% -> **91.65%** (~1.7 points each); the logs
  show the `cycle 0 -> 1` restart and the accuracy jump after it.
- **Equal capacity (0.4% parameter gap)**: scaling trf to 3 layers, d=64 and a CLS token (105,642 params)
  makes it *worse* at this budget -- 63.7% vs 92.6% over 1500 images x 4 epochs. That is not a capacity
  problem but a data problem: its train accuracy is only 0.57 and its loss is still 1.12, i.e. it has not
  even fitted the training set. Compared with the 21k-parameter transformer above (91.65% at 3000 x 6),
  adding capacity at this data scale only makes training harder; a fair test needs >=20-30k samples and
  10+ epochs (30-60 minutes on this machine).
- Other knobs: `--weight-decay` (AdamW decoupled decay), `--dropout`, `--scheduler cosine|fixed`,
  `--hidden`, `--arch match` (equal capacity) and `--save/--load` (including the CLS vector and metadata).

The same example also demonstrates serializing training results: per-sample forward/backward with
mini-batch gradient accumulation via `cache_updator_t`, after which the weights plus metadata are written
into a single `JASMINE_WEIGHTS_V1` file and read back with `--load` to verify the round-trip:

```bash
# real MNIST (uncompressed IDX files under build/mnist/)
mkdir -p build/mnist && cd build/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sSLO "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" && gunzip -f "$f.gz"
done
cd ../..

./build/examples/mnist_conv --data-dir build/mnist --epochs 3 --train-limit 6000 \
    --batch 16 --lr 2e-3 --save build/mnist/model.jas
# -> [epoch 3] train_loss=0.0987 train_acc=0.9695 test_acc=0.9615
# -> [check] OK: save/load round-trip consistent
./build/examples/mnist_conv --data-dir build/mnist --load build/mnist/model.jas --epochs 0

# no data / no network: the built-in synthetic digit patterns keep the whole pipeline runnable
./build/examples/mnist_conv --synthetic --epochs 2 --train-limit 500 --test-limit 200
```

Measured on this machine (reference BLAS, single-threaded): 6000 of the 60000 training images sampled
at random, 3 epochs in ~85-115 s, reaching **96.2% (seed 1234) / 97.5% (seed 7)** on the **full
10000-image test set**. Training and test come from the official, disjoint IDX splits (an md5 comparison
of the image contents finds a single duplicated image, a known MNIST quirk). Predictions after save/load
match those before saving exactly. The read/write API is the pre-existing `weight_writer_t` /
`weight_file_t` in `jas_weight_io.hpp`; this change adds the per-layer naming glue
(`add_layer_params` / `read_layer_params`) and scalar metadata (`add_scalar`).

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
