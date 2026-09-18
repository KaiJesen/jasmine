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
| Layers | Linear, **Conv2d (im2col + GEMM, full backward)**, **MaxPool2d / AvgPool2d (full backward)**, **Flatten**, **MeanPool（token 序列）**, LayerNorm / RMSNorm, SiLU / GELU / ReLU, SwiGLU gating, residual, **Transformer encoder**, Embedding, cross-entropy / MSE |
| Models | decoder-only base; **GPT-2** (absolute positions + `gelu_new` + tied lm_head); **LLaMA family** (RoPE + RMSNorm + SwiGLU + GQA/MQA); enc-dec stack |
| Training | Backprop checked against numerical gradients; SGD / Adam / NAdam; gradient accumulation (`cache_updator_t`) |
| Inference | KV-cache prefill + per-token decoding; greedy / top-k / top-p sampling; streaming output |
| Weights | Single-file `JASMINE_WEIGHTS_V1` format; **training-result serialization** (per-layer params + meta in one file, `save`/`load` round-trip); direct export from HuggingFace (`tools/`); layer-by-layer / logits golden-value alignment |
| CUDA | Element-wise chains fused into a single kernel launch, cuBLAS GEMM, reductions, device-side KV cache / RoPE / MHA / a whole LLaMA, backwards and optimizers, **fused attention**, **bf16 / fp16 mixed precision** |

**Status**: the host side passes `ctest` 233/233; the CUDA backend has 174 cases (compute-heavy cases
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

### 4.1 MNIST 小玩具（卷积 + 编码器 + 序列化）

`examples/mnist_conv.cpp` **用 `complex_net_builder_t` 把层静态堆叠成 `complex_net_t`**
（`forward` / `backward` / `step` / `init_weight` 全部走链），并支持两种结构在同一份数据、
超参、随机种子下对比：

```text
cnn2 = conv→relu→pool→conv→relu→pool→flatten→(fc→relu→fc)→ce      "标准 CNN"
cnn1 = conv→relu→pool→flatten→(fc→relu→fc)→ce                    （卷积 + MLP encoder）
trf  = conv→relu→pool→(patch embedding)→Transformer encoder→mean pool→fc→ce
       其中 Transformer encoder 用的是库里的 encoder_t（双向自注意力 + LayerNorm + FFN + 残差，
       由 complex_net 堆成），patch embedding 把每个空间位置投影成一个 token，
       mean_pool 把 T 个 token 平均成分类头要的向量。

# 两条结构各训 3 epoch 后对比（默认 arch=both）
./build/examples/mnist_conv --data-dir build/mnist --epochs 3 --train-limit 6000 \
    --batch 16 --lr 2e-3 --arch both --save build/mnist/cmp

# 6000 张 × 3 epoch：CNN 系两变体
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       3    0.098669      0.9695      0.9724   120.618
cnn1          202330       3    0.119815    0.962667       0.964   113.324
test_acc 差值（cnn1 - cnn2）= -0.0084

# 2000 张 × 2 epoch：CNN vs Transformer encoder（--arch both 的默认组合）
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       2    0.218465       0.936      0.9255    16.65
trf            17914       2     1.31306       0.514       0.559   104.41
test_acc 差值（trf - cnn2）= -0.3665
```

读法：

- **CNN 系内部**：两次下采样的 `cnn2` 用一半参数（10.5 万 vs 20.2 万）反而高 0.84 个点——
  第二个卷积层带来的层级特征比把宽特征直接灌进全连接更划算；
- **CNN vs Transformer**：这个规模下（2000 张 × 2 epoch）Transformer 落后 37 个点，而且慢 6 倍
  （196 个 token 的自注意力）。它的 loss 仍在稳定下降（2.18 → 1.31，train_acc 0.19 → 0.51），
  属于**还没训够**：Transformer 没有卷积那样的局部性/平移等变先验，需要更多数据、更多 epoch、
  更小的学习率或更强正则才能追上。想跑出有意义的曲线就加大 `--epochs` / `--train-limit`
  （例如 `--arch trf --epochs 20 --train-limit 20000 --lr 5e-4`）。

按样本前向/反向、用 `cache_updator_t` 做 mini-batch 梯度累加，训练完把权重与元信息写进一个
`JASMINE_WEIGHTS_V1` 文件，再 `--load` 回来验证往返一致：

```bash
# 真实 MNIST（IDX 文件未压缩；放到 build/mnist/）
mkdir -p build/mnist && cd build/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sSLO "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" && gunzip -f "$f.gz"
done
cd ../..

./build/examples/mnist_conv --data-dir build/mnist --epochs 3 --train-limit 6000 \
    --batch 16 --lr 2e-3 --save build/mnist/model.jas
# -> [epoch 3] train_loss=0.0987 train_acc=0.9695 test_acc=0.9615
# -> [check] OK: 保存/载入往返一致
./build/examples/mnist_conv --data-dir build/mnist --load build/mnist/model.jas --epochs 0

# 没有数据/没有网络时：内置合成数字图案，整条链路照样跑通
./build/examples/mnist_conv --synthetic --epochs 2 --train-limit 500 --test-limit 200
```

实测（本机、参考 BLAS、单线程）：训练集 60000 张里随机抽 6000 张、3 个 epoch 约 85~115 秒，
在**完整 10000 张测试集**上达到 **96.2%（seed 1234）/ 97.5%（seed 7）**；训练与测试用的是
官方 IDX 划分的两个文件（按图像内容比对，两集仅 1 张重复，属 MNIST 自身的已知瑕疵）。
保存/载入后的预测与保存前完全一致。序列化的读写接口是 `jas_weight_io.hpp` 里已有的
`weight_writer_t` / `weight_file_t`，本轮补上了按层命名的黏合层（`add_layer_params` /
`read_layer_params`）与标量元信息（`add_scalar`）。

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
