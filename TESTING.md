# Testing & Benchmarking

Library headers contain **only** library code. Checks and demos live under:

| Dir | Tool | Purpose |
|-----|------|---------|
| `tests/` | GoogleTest | Correctness / regression |
| `benches/` | Google Benchmark | Performance |
| `examples/` | plain binary | Interactive demos (`cin`) |

## Build

```bash
cmake -S . -B build -DJASMINE_USE_OPENMP=ON
cmake --build build -j
```

## Run

```bash
ctest --test-dir build --output-on-failure
./build/benches/bench_jasmine --benchmark_filter=BM_MhaForward
./build/examples/train_transformer      # MSE：连续向量 + 特征维 SOS/EOS
./build/examples/train_transformer_ce   # CE：离散 token + embedding + CE
./build/examples/train_tf_base
./build/examples/train_tf_kernel
```

Demo harnesses（非 GoogleTest）：

- `examples/transformer_mse_demo.hpp` — `mse_transformer_demo_t`
- `examples/transformer_ce_demo.hpp` — `ce_transformer_demo_t`

## GPT-2 推理对齐（加载开源权重）

加载 HuggingFace 的 GPT-2 / distilgpt2 权重，在 jasmine 里做因果前向 + KV-cache 生成，
并与 HF 的逐层 hidden 和 logits 对齐。**推理 only**（无训练/微调）。

### 1. 导出权重

依赖 `transformers` / `torch`（词表与权重都在 Python 侧处理，jasmine 只吃 id）：

```bash
pip install transformers torch

# distilgpt2（6 层，313 MiB）——建议先跑通这个
python tools/export_gpt2.py --model distilgpt2 --out build/distilgpt2_weights.bin \
    --golden-prompt "Hello, my dog is cute"

# gpt2（12 层，476 MiB）——同一套 C++ 代码，不用改
python tools/export_gpt2.py --model gpt2 --out build/gpt2_weights.bin \
    --golden-prompt "Hello, my dog is cute"
```

脚本会：转置 Conv1D、拆 fused `c_attn` 为 q/k/v、绑定 `lm_head` 与 `wte`、写 `cfg.*` 结构参数，
并在给定 `--golden-prompt` 时额外 dump HF 的 `golden.hidden.{i}` 与 `golden.logits` 供单测比对。
同时产出 `<out>.json` manifest（模型名、结构、tensor 列表）。

### 2. 逐层 / logits 对齐

```bash
./build/tests/unit_tests --gtest_filter='Gelu*:Gpt2WeightIo*:Gpt2Structure*'
./build/tests/unit_tests --gtest_filter='Gpt2AlignmentTest.*'   # 需要权重文件

# 指定权重文件（否则按 ctest 工作目录下的默认名字自动搜索）
JASMINE_GPT2_WEIGHTS=build/gpt2_weights.bin \
    ./build/tests/unit_tests --gtest_filter='Gpt2AlignmentTest.*'
```

未找到权重文件时 `Gpt2AlignmentTest` 会 `GTEST_SKIP()`，因此 CI 不依赖大文件。

对齐断言的量：

| 测试 | 比对对象 | 容差 |
|------|----------|------|
| `HiddenStatesMatchLayerByLayer` | 逐层 hidden（`forward_stages` vs HF `hidden_states`） | `1e-3` |
| `GoldenLogitsMatch` | 末层 logits + 每位置 argmax | `1e-3` / argmax 完全一致 |
| `KVCacheDecodeMatchesGolden` | KV-cache 逐步解码的每步 logits | `1e-3` |

实测两个模型均通过：逐层 `max_abs_diff ≤ 6e-4`（float32 精度量级），logits `≤ 1e-3`。

`HiddenStatesMatchLayerByLayer` 是**定位问题的主要手段**：单看 logits 不够，因为末尾 `ln_f`
会掩盖仿射级别的偏差；逐层比对能直接指认是哪一层开始不一致。

### 3. 文本 → id → 生成 → 文本
```bash
# 文本 -> prompt ids（每行一个 id）
python tools/gpt2_tokenize.py encode --model distilgpt2 \
    --text "Hello, my dog is cute" --out build/prompt_ids.txt

# KV-cache 生成（greedy / top-k 采样）
./build/examples/gpt2_generate build/distilgpt2_weights.bin build/prompt_ids.txt \
    --max-new 20 --out-ids build/out_ids.txt

# id -> 文本
python tools/gpt2_tokenize.py decode --model distilgpt2 --ids-file build/out_ids.txt
```

`gpt2_generate` 选项：`--max-new N`、`--eos ID`、`--greedy|--sample`、`--temperature T`、
`--top-k K`、`--seed S`、`--out-ids FILE`、`--quiet`。

实测：distilgpt2 与 gpt2 的 greedy 生成结果（26 token）与 HuggingFace `generate(do_sample=False)`
**逐 token 完全一致**。

### 4. 与 HuggingFace 端到端对撞（一条命令）
上面第 2、3 步是分开验证；`tools/verify_gpt2.py` 把「同一份权重、同一个 prompt、两边分别跑、
逐 token 比对」串成一条命令，是判断「能不能用」最快的方式：

```bash
# 默认 distilgpt2；权重缺失时自动导出
python tools/verify_gpt2.py --text "The capital of France is"

# 12 层模型
python tools/verify_gpt2.py --model gpt2 --text "The capital of France is"

# 再比一次整段序列的 logits（两侧喂同一串 id，与 RNG 无关）
python tools/verify_gpt2.py --text "The capital of France is" --max-new 20 --logits

# 只验数值、不生成
python tools/verify_gpt2.py --text "Hello" --max-new 0 --logits

# 采样解码（此时逐 token 无法比对，靠 --logits 兜底）
python tools/verify_gpt2.py --sample --temperature 0.8 --text "Once upon a time" --logits
```

输出示例：

```
=== jasmine vs HuggingFace: distilgpt2 ===
[1/5] weights: build/distilgpt2_weights.bin
[2/5] prompt: 'The capital of France is' -> 5 tokens
[3/5] jasmine: 20 new tokens (total 25)
[4/5] HF     : 20 new tokens (total 25)
[5/5] compare
  token ids : MATCH (25/25)
  logits    : max_abs_diff = 9.155e-05 over 50257x25 (PASS, tol 0.001)
  argmax    : MATCH all positions
RESULT: PASS
```

退出码 `0` = 一致、`1` = 不一致或出错，可直接用于 CI。

`--logits` 的实现方式值得说明：它把 **jasmine 生成的整段序列**喂给 HF 做一次 forward 再比对，
而不是各自 forward 自己的序列。这样两侧输入完全相同，因此**采样模式下也有意义**
（采样时两侧 RNG 不同、token 序列不同，逐 token 比对本就不可能一致）。

### 5. 交互式对话 demo（`gpt2_chat`）

在终端里和模型来回聊。模型权重**只加载一次**，KV cache 跨轮复用，所以每轮只算新 token。

```bash
# 直接开聊（权重路径之后可跟任意选项）
./build/examples/gpt2_chat build/distilgpt2_weights.bin

# 12 层模型
./build/examples/gpt2_chat build/gpt2_weights.bin

# 用对话模板（User:/Assistant:）并在换行处停下
./build/examples/gpt2_chat build/distilgpt2_weights.bin --template chat

# 更多控制
./build/examples/gpt2_chat build/distilgpt2_weights.bin \
    --max-new 60 --temperature 0.8 --top-k 40
./build/examples/gpt2_chat build/distilgpt2_weights.bin --greedy --no-color
```

> ⚠️ **GPT-2 是 base 语言模型，不是指令微调模型。** 它不会回答问题，只会**续写**你给的文字。
> `--template chat` 只是用 `User: ... / Assistant:` 这种文本格式引导它进入对话式续写，
> 效果远不如真正的 chat 模型（如 TinyLlama-Chat）。这个 demo 的价值在于**验证 jasmine 的
> 推理链路端到端可用**，而不是得到一个好用的助手。

REPL 内命令：

| 命令 | 作用 |
|------|------|
| `/help` | 命令列表 |
| `/exit` | 退出（Ctrl-D 也可以） |
| `/reset` | 清空对话与 KV cache |
| `/context` | 显示上下文长度和最近内容的解码结果 |
| `/params` | 显示当前采样参数 |
| `/set KEY VALUE` | 运行中改参数：`temperature` / `top_k` / `max_new` / `greedy` / `template` |

常用选项：

| 选项 | 默认 | 说明 |
|------|------|------|
| `--max-new N` | 40 | 每轮最多生成多少个 token |
| `--temperature T` | 0.9 | 采样温度（`--greedy` 时忽略） |
| `--top-k K` | 40 | top-k 截断 |
| `--greedy` | 关 | argmax 解码 |
| `--seed S` | 1234 | 随机种子 |
| `--template raw\|chat` | raw | `raw`=直接续写；`chat`=包一层 User/Assistant 并遇换行停止 |
| `--max-context N` | 模型 `n_pos` | 上下文上限；超出后丢最早内容并**重建 cache**（绝对位置编码，位置会整体前移） |
| `--no-stream` | 关 | 关掉逐 token 流式输出 |
| `--model NAME` | 从 manifest 读 | tokenizer 用哪个 HF 模型 |

实现要点：tokenizer 由常驻子进程 `tools/gpt2_tokenizer_server.py` 提供（`transformers`
冷启动约 2 秒，每轮起进程会慢到不可用）；文本走 base64 传输以避开所有转义问题。
流式输出不能逐 token `decode`（多字节字符会被切成半个、HF 会替换成 U+FFFD），
而是每步重解全量并按新增部分输出，详见 `examples/gpt2_chat.cpp` 顶部注释。

实测响应速度：权重加载约 2.5 秒（一次性），之后每轮"用户输入 + 生成 40 token"约 1 秒。

### 相关文件

| 文件 | 作用 |
|------|------|
| `jas_gelu_t.hpp` | `gelu_net_t`（`gelu_new` tanh 近似，含解析梯度） |
| `jas_gpt2_t.hpp` | `gpt2_block_t` / `gpt2_model_t`：pre-norm + 绝对位置 + `ln_f` + tied `lm_head` |
| `jas_weight_io.hpp` | 权重文件读（`weight_file_t`）/ 写（`weight_writer_t`）+ `load_gpt2` 名字映射 |
| `tools/export_gpt2.py` | HF → jasmine 权重导出（含黄金值） |
| `tools/verify_gpt2.py` | 与 HF 的端到端对撞（生成逐 token + logits 数值） |
| `tools/jasmine_weights.py` | 权重文件格式的 Python 读写（导出与校验共用） |
| `tools/gpt2_tokenizer_server.py` | 常驻 tokenizer 服务（供 `gpt2_chat` 调用） |
| `tools/gpt2_tokenize.py` | 文本 ↔ id（命令行） |
| `tests/test_gpt2_weights.cpp` | GELU / 加载器读写 / 结构 / 黄金对齐单测 |
| `examples/gpt2_generate.{hpp,cpp}` | KV-cache 生成 demo（`--dump-logits` 供 verify 脚本比对） |
| `examples/gpt2_chat.cpp` | 交互式对话 REPL |

对齐机制与踩坑记录见 [`doc/SMALL_MODEL_REPRO_CHECKLIST.md`](doc/SMALL_MODEL_REPRO_CHECKLIST.md)。

## LLaMA / TinyLlama-Chat 推理对齐（加载开源权重）

加载 HuggingFace 的 LLaMA 系权重（以 `TinyLlama-1.1B-Chat-v1.0` 验证），在 jasmine 里做
因果前向 + KV-cache 生成，并与 HF 的逐层 hidden / logits 对齐。**推理 only**。

与 GPT-2 的差异（同一套 `mat_mha_t` / 基础设施，只换模型类与导出脚本）：

| | GPT-2 | TinyLlama-1.1B-Chat |
|---|---|---|
| 位置编码 | 绝对 `wpe`（`set_use_rope(false)`） | RoPE，**`half_split` 配对** |
| Norm | Post-norm LayerNorm + `ln_f` | Pre-norm **RMSNorm** |
| FFN | `gelu_new` 两层 | **SwiGLU** 三投影 |
| 注意力 | MHA | **GQA**（32 Q / 4 KV） |
| 线性层 bias | 有（`lm_head` 显式置 0） | **全部无 bias** |
| `lm_head` | 与 `wte` 绑定 | **不绑定**（untied） |
| 输入构造 | 裸文本续写 | **`apply_chat_template`**（ SentencePiece，`enc(a)+enc(b) ≠ enc(a+b)`） |

### 1. 导出权重

```bash
# 约 4.2 GiB；同时 dump 两条 golden：裸文本续写 + chat 模板序列
python tools/export_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --out build/tinyllama_weights.bin \
    --golden-prompt "The capital of France is" \
    --golden-chat "What is the capital of France?"
```

脚本会：校验模型可支持性（`rope_theta` 必须 10000、无 bias、无 RoPE scaling、激活必须是 silu）、
按 HF 的 `[out, in]` 直接存（LLaMA 的 `nn.Linear` 无 Conv1D 转置）、`k_proj/v_proj` 不复制，
并用 forward hook 抓取 23 个阶段（embed + 22 层 block，**pre-`ln_f`**）作为逐层黄金值。

### 2. 逐层 / logits / KV-cache / chat 模板对齐

```bash
./build/tests/unit_tests --gtest_filter='Llama*'

# 指定权重文件（否则按 ctest 工作目录下的默认名字自动搜索）
JASMINE_LLAMA_WEIGHTS=build/tinyllama_weights.bin \
    ./build/tests/unit_tests --gtest_filter='LlamaAlignmentTest.*'
```

未找到权重文件时 `LlamaAlignmentTest` 会 `GTEST_SKIP()`，CI 不依赖大文件。

| 测试 | 比对对象 | 容差 |
|------|----------|------|
| `RawHiddenStatesMatchLayerByLayer` | 23 个阶段逐层 hidden | `1e-3` |
| `RawGoldenLogitsMatch` | 末层 logits + 每位置 argmax | `1e-3` / argmax 完全一致 |
| `KVCacheDecodeMatchesGolden` | KV-cache 逐步解码的每步 logits | `1e-3` |
| `PrefillMatchesGoldenLastPosition` | `prefill()` 末位 logits（demo 实走路径） | `1e-3` |
| `ChatTemplateLogitsMatch` | `apply_chat_template` 序列的 logits | `1e-3` |

实测：全部 `max_abs_diff ≤ 3.1e-5`，即 float32 参考值自身的舍入噪声量级（jasmine 侧是 double）；
top-5 逐位相同。容差取 `1e-3` 是为了留足跨平台/BLAS 归约顺序余量，同时足以拦住结构性错误
（下面那个 RoPE 坑会造成 O(1) 偏差）。

### 3. 交互式对话 demo（`llama_chat`）

```bash
./build/examples/llama_chat build/tinyllama_weights.bin
```

- 输入按 **`apply_chat_template`** 渲染（由 `tools/llama_tokenizer_server.py` 的 `M` 操作提供）
- 会话命令：`/help` `/context` `/reset` `/exit`
- 常用参数：`--max-new` `--temperature` `--top-k` `--top-p` `--greedy` `--seed` `--system` `--max-context`
- KV cache 跨轮复用：把当前 messages 渲染成规范 token 序列，与已有 cache 比**公共前缀**，
  相同则只喂新增 token，不同则整体重建；超 `--max-context` 时丢弃最早轮次
- tokenizer 服务脚本定位有兜底（按 `:exe` 目录逐级向上找 `tools/`），从 `build/` 里跑也行

实测（greedy）：`What is the capital of France?` → `The capital of France is Paris.`；
多轮上下文正确累积（66 → 93 → 141 token），第二轮能引用第一轮内容。

**性能（20 核，`libblas.so.3` + OpenMP，`--no-stream --greedy`）**：
启动到可输入约 **85 秒**（含读取 4.2 GiB 权重并转成 double），之后约 **2.5 秒/token**。
（用两次不同 `--max-new` 的墙钟时间做差得到每 token 成本：`(203.4 − 104.9) / 40 ≈ 2.46 s`。）

这个速度慢得显眼，但成因明确、且与「没开 BLAS / 没开多线程」无关——
**batch=1 的解码是内存带宽瓶颈，不是算力瓶颈**：`mat_t<double>` 让 1.1B 参数以 double 常驻
（约 8.8 GB），每生成一个 token 都要把这 8.8 GB 完整读一遍，
8.8 GB ÷ 2.5 s ≈ **3.5 GB/s**，正好是这台机器的实际内存带宽。此时 GEMV 没有可并行的计算量，
多线程和 BLAS 都帮不上；想快只有换 float 权重（约 2 倍）或量化（一个数量级）。
换句话说，这个数字衡量的是内存带宽，不是实现质量。

### ⚠️ 这个坑最贵：RoPE 特征配对约定

θ_i = m / 10000^(2i/d) 在两种约定下**完全一致**，区别只在于第 i 个角作用在哪两个特征上：

- `interleaved`：`(2i, 2i+1)` —— 原论文 / GPT-NeoX，jasmine 原生实现与默认值
- `half_split`：`(i, i + d/2)` —— GPT-J / **HuggingFace LLaMA**（`rotate_half`）

接 TinyLlama 时最初沿用了原生约定，结果**除位置 0 以外所有位置的 logits 都差 O(1)**
（逐层 hidden `max_abs_diff` 达 7.17）。之所以难查，是因为**位置 0 完全正确**：
那里旋转矩阵退化为单位阵，任何配对约定都对得上 —— 于是很容易反过来怀疑
attention / softmax / 权重加载。

定位手法（可复用）：把 HF 的 `apply_rotary_pos_emb` monkeypatch 成本仓库的交错配对，
logits 立刻与自己的输出**逐位吻合**，从而证明差异**只**在配对约定。

```python
def apply_rotary_interleaved(q, k, cos, sin, unsqueeze_dim=1):
    D = q.shape[-1]; half = D // 2
    c = cos[..., :half].unsqueeze(unsqueeze_dim)   # HF 的 cos/sin 是全维（两半重复）
    s = sin[..., :half].unsqueeze(unsqueeze_dim)
    def rot(x):
        xp = x.unflatten(-1, (-1, 2))
        x0, x1 = xp[..., 0], xp[..., 1]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)
    return rot(q), rot(k)
```

处理：新增 `rope_pair_layout` 枚举，默认 `interleaved`（既有模型与 119 个测例行为不变），
LLaMA 在 `jas_llama_t.hpp` 里显式设 `half_split`；RoPE 注册表 key 从 `d` 改为 `(d, layout)`，
避免同维度不同约定互相复用。

**回归测试**（这组别删，它们就是钉这个坑的）：

```bash
./build/tests/unit_tests --gtest_filter='RoPE.*'
```

| 测试 | 钉住什么 |
|------|----------|
| `RoPE.HalfSplitPairsOffsetFeatures` | `(i, i+d/2)` 配对的独立公式逐元素比对 |
| `RoPE.LayoutsAgreeAtPositionZeroOnly` | **位置 0 两种约定必须相同、位置 ≥1 必须不同**（即上面那个陷阱本身） |
| `RoPE.BackwardIsAdjointForBothLayouts` | `<forward(x), δ> == <x, backward(δ)>`，两种约定都成立 |
| `RoPE.RegistrySeparatesPairLayouts` | 注册表按 `(d, layout)` 分开，不复用 |
| `RoPE.MhaPropagatesPairLayoutToHeads` | `set_rope_pair_layout` 真的换掉了每个头绑定的 RoPE |

### 相关文件（LLaMA）

| 文件 | 作用 |
|------|------|
| `jas_llama_t.hpp` | `llama_model_t`：pre-norm RMSNorm + RoPE(half_split) + GQA + SwiGLU，无 bias、untied head |
| `jas_weight_io.hpp` | `load_llama` 名字映射 + `llama_config_t` / `read_llama_config` |
| `jas_RoPE_t.hpp` | `rope_pair_layout`（`interleaved` / `half_split`）+ 按 `(d, layout)` 共享的注册表 |
| `tools/export_llama.py` | HF → jasmine 权重导出（含可支持性校验与黄金值） |
| `tools/llama_tokenizer_server.py` | 常驻 tokenizer 服务（`E`/`D`/`M`/`V`，`M` = apply_chat_template） |
| `tests/test_llama_weights.cpp` | 结构 / 加载 / 黄金对齐单测 |
| `tests/test_rope.cpp` | RoPE 基础 + **配对约定回归** |
| `examples/llama_chat.cpp` | 交互式对话 REPL（LLaMA 系） |
| `examples/chat_common.hpp` | GPT-2 与 LLaMA 共用的 REPL 基础设施（颜色 / base64 / 流式 / tokenizer 客户端） |

## 6. 门控 FFN（SwiGLU）：容器与激活

对齐 LLaMA 系模型需要 SwiGLU。拆成两块：**激活** `silu_net_t` + **门控容器**
`gated_net_t<gate, up>`，容器只负责拓扑（双分叉 + 逐元素乘汇合），层的类型由分支决定：

```text
完整 FFN =  gated( gate: Linear → SiLU ,  up: Linear )  →  down_proj
                 └────── gated_net_t ──────┘              └ 普通 weight_net_t
```

`down_proj` **不在**容器里，和 `residual_net_t` 只负责 `+skip` 是同一个设计取向。

```bash
# 单测：容器正向 / 反向数值梯度 / reinit / 接入 complex_net / 残差包裹
./build/tests/unit_tests --gtest_filter='Gated.*'

# 与 PyTorch 端到端对撞：y = down(silu(gate(x)) * up(x))
python3 tools/verify_swiglu.py
# -> SwiGLU FFN vs PyTorch: d_model=6 d_ff=16 T=4  max_abs_diff=8.327e-17  PASS
```

用法要点：

```cpp
// 换个激活就是 GEGLU / ReGLU（silu_net_t / gelu_net_t / relu_net_t）
using upr_tpl = cache_updator_t<double, nadam_t>;
gated_ffn_branches_t<double, upr_tpl, silu_net_t> gated;
gated.reinit(std::vector<int>{d_model, d_ff});   // gate/up 形状相同，一份就够

// 分支是 Linear→SiLU，访问它要再下一层：gate 分支的 Linear 是 get<0, 0>()
auto& gate_lin = gated.gate_branch().template get<0>();
auto& up_lin   = gated.up_branch();              // up 分支本身就是 Linear
```

反向要点：两条分支读的是**同一个 `x`**，所以 `∂L/∂x` 是两条路径梯度**之和**；
而逐元素乘的梯度就是「乘对方」（`∂L/∂gate = delta⊙up`、`∂L/∂up = delta⊙gate`），
因此 `forward` 必须缓存两分支的输出（容器只多存这两份矩阵）。

### 相关文件（门控 FFN）

| 文件 | 作用 |
|------|------|
| `jas_silu_t.hpp` | `silu_net_t`：`x⊙σ(x)`，含解析梯度、饱和区不产 NaN |
| `jas_net_t.hpp` | `gated_net_t<gate, up>` 容器 + `gated_ffn_branches_t` 别名 |
| `tests/test_silu.cpp` | `SiLU.*`：定义 / PyTorch 参考值 / 数值梯度 / 非单调性 / 饱和 |
| `tests/test_gated.cpp` | `Gated.*`：逐元素乘（非矩阵乘）/ 数值梯度 / reinit / 链路与残差集成 |
| `tools/verify_swiglu.py` | 用 PyTorch 生成权重与期望输出，端到端对撞完整 SwiGLU FFN |

## 7. GQA / MQA：`n_kv_heads` 参数化

GQA 不是新算法，而是把「K/V 头数」从 Q 头数上解耦，因此**没有新文件、没有新注意力核**——
注意力核 `mat_head_gen_t` 本来就只认切好的 Q/K/V，一行没改。

```text
n_kv_heads == n_heads    → 经典 MHA（默认，与旧实现逐位一致）
1 < n_kv_heads < n_heads → GQA
n_kv_heads == 1          → MQA
```

```cpp
// 构造函数 / set_param 的最后一个参数，默认 0 表示"等于 num_heads"
mat_mha_t<mat_t<float>, nadam_t> gqa(n_q_heads, d_model, /*mask=*/true, seq_len, /*n_kv_heads=*/2);

gqa.num_heads();      // Q 头数
gqa.num_kv_heads();   // K/V 头数
gqa.group_size();     // 每个 KV 头被几个 Q 头共享 = num_heads / num_kv_heads
gqa.d_head();         // 每头宽度 = d_model / num_heads
gqa.d_kv();           // K/V 投影输出宽度 = num_kv_heads * d_head

gqa.k_proj().weight();  // [d_kv, d_model] —— 注意不是 [d_model, d_model]
gqa.q_proj().weight();  // [d_model, d_model] 不变
```

与 MHA 的差异只有三处：

1. **K/V 投影变窄**：输出宽度 `n_kv_heads*d_head`（`q_proj`/`out_proj` 仍是 `d_model`）；
2. **KV cache 只存 `n_kv_heads` 份**（省显存之处）；
3. **反向时共享同一 KV 头的多个 Q 头梯度累加**（不是覆盖）。

```bash
./build/tests/unit_tests --gtest_filter='Gqa.*'
```

### 两个必须知道的坑

- **KV cache 必须"每个 KV 头只 append 一次"。** 若图省事让共享 KV 头的多个 Q 头各自走
  `forward_one_at`，同一份 K/V 会被写入 `group_size` 次 —— 形状不报错、单步看似可用，
  但 cache 长度与内容全错，到多轮对话才炸。正确做法是拆两步：`append_kv_head`（写 + RoPE）
  再 `attend_cached`（只读不写）。回归测试 `Gqa.KvCacheGetsOneAppendPerKvHead` 专门钉这一点。
- **`backward` 不能接在 `forward_one` 之后。** 训练前向把所有头内缓存都填了，而推理前向
  `attend_cached` 不填 `m_v`、`m_softmax.m_output` 也只有单步形状。顺序应是
  `forward`（整段）→ `backward`，需要推理时先 `clear_kv_cache()` 再 `forward_one`。

### 相关文件（GQA）

| 文件 | 作用 |
|------|------|
| `jas_mha_t.hpp` | `mat_mha_t` 的 `n_kv_heads` 参数化（投影/cache/分组/反向累加/cross-attn） |
| `tests/test_gqa.cpp` | `Gqa.*`：朴素参考对撞 / 等价 K-V 复制版 MHA / 分组累加 / cache 回归 / 数值梯度 / MQA |

---

## 8. RMSNorm：`rms_norm_net_t`

放在 `jas_net_t.hpp` 里紧邻 `layer_norm_net_t`，**便于对照阅读**——因为两者的差别小到
可以逐行对比，而"抄错"恰恰是这里最容易犯的错。

```text
LayerNorm(x) = gamma ⊙ (x - mean(x)) / sqrt(var(x) + eps) + beta
RMSNorm(x)   = gamma ⊙  x            / sqrt(mean(x²) + eps)
```

差异只有三处：**不减均值**、**没有 beta**、eps 加在**均方值**上而不是方差上。

```cpp
rms_norm_net_t<mat_t<double>, nadam_t> norm;
norm.set_param(d_model, /*eps=*/1e-6);   // 不传 eps 则用 kDefaultEps(1e-5)

norm.gama();       // [d_model, 1] —— 只有缩放，没有 beta
norm.eps();        // 读当前 eps
norm.set_eps(1e-5);// 单独改 eps
```

和 `layer_norm_net_t` 一样，初始化接口是 `set_param` 而**不是** `reinit`
（否则 `is_reinitable_net` 判定会变，`complex_net_t::reinit` 的槽位会对不上）。
`val_type` 是 public，因为 RMSNorm 常位于 pre-norm 链首，`complex_net_t` 要从它推断整链类型。

```bash
./build/tests/unit_tests --gtest_filter='RmsNorm.*'
```

### 为什么"删掉减均值"还能 work

归一化真正起作用的是 **缩放不变性**（`RMSNorm(c·x) == RMSNorm(x)`，`c>0`），
它让梯度不依赖激活的绝对幅度，从而抑制爆炸/消失。重新中心化是最不重要的一环：
紧跟其后的仿射层本来就会重新引入偏置。代价是丢掉平移不变性（`RMSNorm(x+c) ≠ RMSNorm(x)`）。
收益：省掉一次行归约，快约 5~15%，且少一组 `beta` 参数。

几何上两者都落在半径 `sqrt(d)` 的球面上——**LayerNorm 先减均值改变方向，RMSNorm 不改变方向**。
（严格说，`eps>0` 时半径是 `sqrt(d)·sqrt(ms/(ms+eps))`，略小于 `sqrt(d)`。）

### 两个必须知道的坑

- **反向必须删掉 `mean(g)` 那一项。** 直接从 `layer_norm_net_t::backward` 复制会多留一项
  （LayerNorm 减均值才需要的 `sum(g)/d`）。这个错误**不会报错、训练也能跑**，只是方向系统性偏，
  解析梯度与数值梯度差约 1.0。测试 `RmsNorm.BackwardOmitsCenteringTermLestItBeWrong`
  从两侧钉住：解析梯度既**不等于**那个错误版本，也要与数值梯度一致——避免"两处同错"互相掩盖。
- **别拿 HF `LlamaRMSNorm` 的输出当 float64 基准。** 该类内部先 `.to(torch.float32)` 再算，
  即使你传入 float64，输出也与精确解差约 `1e-7`。jasmine 是 double，基准应取
  **float64 定义式**。已核实：测试里的 `kY` 与定义式差 `0.0`，与官方类之差恰为该精度转换。

### 相关文件（RMSNorm）

| 文件 | 作用 |
|------|------|
| `jas_net_t.hpp` | `rms_norm_net_t`（紧邻 `layer_norm_net_t`）+ `rms_norm_net_t::kDefaultEps` |
| `tests/test_rms_norm.cpp` | `RmsNorm.*`：定义式逐元素 / PyTorch float64 基准 / 缩放不变 / 平移不敏感 / 与 LayerNorm 对照 / 数值梯度 / 均值项删除断言 / pre-norm 接入 |

---

## 9. 表达式模板的操作数生命周期（值类别契约）

表达式模板 (`mat_add_t` / `mat_mul_t` / `mat_dot_t` …) 是**惰性**的：节点里存的是操作数，
真正的求值发生在 `clone()` / `operator mat_t<>`。因此「节点怎么持有操作数」直接决定了
表达式树能不能被安全地存下来、拷贝、或者传给别的执行后端。

### 曾经的 bug：一律按引用持有 → 悬垂

早期实现里，非标量操作数一律按 `T const&` 存（`storage_selector`）。这对具名变量没问题，
但对临时量必然悬垂 —— 而**嵌套表达式本身也是临时量**：

```cpp
auto tree = (a + b) * c;   // (a+b) 所在的临时 mat_add_t 在本语句结束即销毁
mat_t<double> r = tree;    // tree 持有的引用已失效
```

ASan 会直接抓到：

```
ERROR: AddressSanitizer: stack-use-after-scope
    #1 mat_express_2_param_stable_t<mat_add_t<...>, mat_t<double>, mat_mul_t>::col_num() const
    #2 ...::clone() const
```

同一个成因还有一种更隐蔽的形态。成员函数里的 `*this` **永远是左值**，哪怕对象本身是临时量：

```cpp
auto tree = a.t().dot(b);      // a.t() 是临时视图，却被当成左值借了引用
auto tree = (a + b).dot(c);    // 同上：临时表达式节点作为 .dot() 的接收者
auto tree = make_mat().dot(b); // 同上：临时矩阵作为接收者
```

### 规则

按**值类别**分派（见 `storage_type` / `storage_of`）：

| 操作数 | 存储方式 | 语义 |
|--------|----------|------|
| 标量 | `scalar_leaf_t<T>` 按值 | 1×1 的 POD（见下方"设备扩展"一节），与矩阵共用 `row_num`/`col_num`/`operator()` |
| 非标量**左值** | `T const&` | **借引用，零拷贝**。调用方保证它比表达式活得久 |
| 非标量**右值** | `T` 按值 | **拥有**。右值就是临时量，必须拥有 |
| **设备叶子** | `T` 按值 | 即使传的是左值也按值拥有（见 `operand_owned_by_value`） |

配套的三条实现约定：

1. 二元/一元运算符用**转发引用**接收操作数，并以 `lval_type&&` / `val_type&&` 作模板实参，
   这样值类别信息才传得到 `storage_type`。
2. 各类构造函数形参取**存储类型**（而非 `lval_type`）。这既让右值走移动、左值零拷贝，
   又保证构造函数首参类型不可能是节点自身 —— 于是不会顶掉隐式拷贝/移动构造，
   表达式树仍然可拷贝（这正是它能被安全传递的前提）。
3. `.dot()` 成员用 **ref-qualifier** 区分接收者：`const&` 借引用，`const&&` 按值拥有
   （`mat_t` / `mat_view_t` / 两个表达式基类 / `mat_dot_t` 都是如此）。
   右操作数同样转发。

`is_caculable` 判标量前必须 `remove_cvref`：左值标量（`double s; m / s;`）推导出的是
`double&`，直接 `is_arithmetic_v<double&>` 是 false，会让整个重载悄悄消失。

### 仍然存在的边界（有意保留）

- **左值操作数**：借引用是表达式模板避免临时量的根本，改成按值会让每次构造表达式都深拷贝
  整块矩阵。所以「左值必须比表达式活得久」是调用方契约。副作用是表达式保持惰性 ——
  构造之后、物化之前修改原矩阵，结果会跟着变（`ExpressionLifetime.LvalueOperandsAreBorrowedSoMutationIsVisible`）。
- **视图所引用的矩阵**：视图本身被按值拥有，但它只是「引用 + 偏移」，其指向的矩阵仍由调用方
  持有。`f().view(...) + c` 这种「对临时矩阵取视图」的写法依然不安全（与 Eigen 的约定一致）。

### 单测

`tests/test_expression_lifetime.cpp`（18 例）把上面每条都锁死了：编译期 `static_assert`
左值借引用 / 右值按值拥有，运行期覆盖嵌套树、拷贝、按值传参、移入容器、临时矩阵/视图接收者、
`.dot()` 链、以及惰性可见性。**这些用例在 ASan 下必须零告警**：

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -O1 -g" \
  -DJASMINE_BUILD_BENCH=OFF -DJASMINE_BUILD_EXAMPLES=OFF
cmake --build build-asan -j
ASAN_OPTIONS=detect_leaks=0 ./build-asan/tests/unit_tests --gtest_filter='-LlamaAlignmentTest.*'
```

（`detect_leaks=0` 是因为在 `ptrace` 环境下 LeakSanitizer 会直接 fatal；
`LlamaAlignmentTest` 需要 4.2 GiB 权重且耗时很长，按需单独跑。）

### 设备扩展（CUDA 分支新增）

同一套存储策略被 CUDA 后端原样复用，只加了三样东西：

1. **标量存储从 `mat_t` 换成 `scalar_leaf_t`**。原来是「包成 1×1 的 `mat_t`」，但 `mat_t`
   用 `new[]` 拿内存、`m_data` 指向**主机地址** —— 表达式一旦上设备，设备端解引用它必崩。
   换成只含一个值的 POD 之后主机设备语义一致，也省掉了每次的分配与 `%` 取模。
2. **「按值拥有」的判据从「设备叶子」放宽到「设备可求值」**。最初只给 `dev_mat_t` 开了
   `operand_owned_by_value`，结果漏掉一个隐蔽的坑：接口形如 `softmax_rows(Expr const& x)` 时，
   传进来的具名表达式节点是**左值**，会被按引用借进派生出的子树；树能编译、能拷贝、
   `is_trivially_copyable` 也为真，但 kernel 参数是按值搬到设备上的，搬过去的是**主机栈地址**，
   设备端一解引用就是 `cudaErrorIllegalAddress`。现在判据是

   ```cpp
   operand_owned_by_value<raw_type> || is_device_evaluable_v<raw_type>
   ```

   因为 `device_evaluable` 沿类型树递归传播，等价于**凡是设备树就全程自持**。
   主机树不受影响（`mat_t` 不是设备可求值），零拷贝的卖点仍然成立。
3. **`is_self_contained_v` 编译期哨兵**：拿「拷贝赋值是否可用」当「树里没有引用成员」的代理指标
   （含引用成员的类，隐式拷贝赋值会被删除），把上面那个运行期段错误扭成构建失败。
   注意 `is_trivially_copyable` **拦不住**这类错误 —— 它只保证没有非平凡的特殊成员函数，
   与引用指向哪里完全无关。

`JAS_HD`（CUDA 下展开为 `__host__ __device__`，否则为空）只加在 `row_num` / `col_num` /
`operator()` / `work()` 上；`std::exp` / `std::max` / `std::sqrt` 在设备端不可用这类差异都在
`jas_cuda_compat.hpp` 里收口（`device_exp` / `device_max` / `device_sqrt`）。
归约 kernel 用到的 `__syncthreads` / `warpSize` 是设备独有，故另有 `JAS_DEV`（仅 `__device__`）。
`device_evaluable` 沿类型树递归传播，启动 kernel 前 `static_assert` 拦下主机表达式。

设备端的归约入口刻意**不叫**主机端的 `hsum` / `vsum` / `hsoftmax`，而叫
`row_sum` / `col_sum` / `softmax_rows`：设备叶子与主机表达式同住 `jasmine` 命名空间，
同名会让 ADL 把主机重载静默拉进候选集，产生极难定位的错误。

`dot` 在两侧的语义刻意不同，且由类型系统承载：主机端 `.dot()` 返回惰性 `mat_dot_t` 节点
（可继续并进表达式树），设备端 `.dot()` **立即求值**并返回 `cuda::dev_matrix_t`。
理由见 `CUDA.md` 第 6 节 —— `mat_dot_t::operator()` 是「每个输出元素自己走一遍 K 循环」，
融进逐元素 kernel 会把访存复用全丢掉，所以它显式标注 `device_evaluable = false`。
`is_mat_dot` 这个判别 trait 就是为了在编译期把两种返回类型分别断言出来。

`tests/test_cuda_fused.cu` 覆盖逐元素/GEMM 契约，`tests/test_cuda_reduce.cu` 覆盖归约、
softmax、归一化层，`tests/test_cuda_dot.cu` 覆盖 `dot` 分派与注意力端到端，
`tests/test_cuda_kv_cache.cu` 覆盖设备 KV cache（容器级与主机 `kv_cache_t` 对拍、
decode/prefill 注意力、GQA），`tests/test_cuda_rope.cu` 覆盖设备端 RoPE
（与主机 `RoPE_net_t` 对拍、两种配对约定、逐头旋转、RoPE+KV cache 端到端），
`tests/test_cuda_backward.cu` 覆盖反向传播与设备端层库，`tests/test_cuda_mha.cu` 覆盖设备端
MHA 与 Embedding，`tests/test_cuda_llama.cu` 覆盖整模型的参数搬运 / 增量解码 / 反向 / 收敛，
`tests/test_cuda_attention.cu` 覆盖融合注意力（与非融合实现 / 主机算式三方对拍、分块无关性、
"不物化"的状态断言与收益量化），`tests/test_cuda_precision.cu` 覆盖混精度（误差模型的校准与
逐源验证、`cast` 与主机 `narrow` 逐位一致、`matmul` 拒绝隐式混精度）。
细节见 `CUDA.md`。

### softmax 的两条路径，以及「优化没生效」这类失败

`softmax_rows` 现在按「整行能否放得进共享内存」自动选路：放得下就把整行留在**共享内存**里，
`max` / `exp`+求和 / 归一化全在片上做（全局读 1 遍 + 写 1 遍、每元素 `exp` 1 次）；
放不下就回退到原来的三趟实现（读 3 遍 + 写 1 遍、`exp` 2 次）。两条路径的数值结构刻意一致，
所以能互相回退。细节见 `CUDA.md` 5.6。

这里有个测试上的陷阱值得单独记：**快路径一上线，所有小矩阵用例都会走它，回退路径就再也没人测了**。
所以除了 `softmax_shared_launch_count()` 计数器（用来断言「该走快路径时确实走了」——
优化的典型失败方式是**压根没生效**，结果照样正确、用例照样全绿），还留了
`softmax_max_cols_override()` 把阈值压到 0，让**同一份输入两边各跑一遍再对拍**。
另外两条边界单独有用例：恰好用满共享内存预算的行仍走单趟（顺带验证
`cudaFuncSetAttribute` 的 opt-in 生效），超预算的行自动回退而不是启动失败。

### RoPE：以主机实现为参照，而不是重写一遍公式

设备端 RoPE 的对拍基准是主机端 `RoPE_net_t` 本身。若另写一份公式做参考，
测出来的只是「两处实现都照着我写的公式抄对了」，而抄错的公式两边一致、照样全绿。
用主机实现当参照，测的才是**口径一致**。

三个口径是分开钉的：

| 口径 | 为什么容易错 | 怎么钉 |
|------|--------------|--------|
| 列 `j` 用绝对位置 `start_pos + j` | decode 每步只喂一列，位置必须来自 cache 长度而非列下标 | `start_pos != 0` 的专项用例 |
| 两种配对约定的区别 | 位置 0 处两者都是恒等变换，单 token 无法区分 | 多列输入 + 显式 `rope_pair_layout` |
| 表与 kernel 的行下标映射一致 | `interleaved` 是 `(2i, 2i+1)`、`half_split` 是 `(i, i+d/2)` | 逐元素对拍 + **保持每对模长**（不依赖参考实现的性质检查） |

`start_pos = 0` 时**只有第 0 列**是恒等（第 `j` 列的绝对位置就是 `j`），这点最容易被想当然，
也有单独用例盯着。最后一条端到端用例把 RoPE → KV cache → attention 串起来，
以主机侧同一条链（`RoPE_net_t` + `kv_cache_t` + 手写注意力公式）为参照。

### 反向传播：两套互不相干的裁判，以及「只对拍主机」为什么不够

每个梯度都跑两条独立验证（细节见 `CUDA.md` 9.4）：

| 裁判 | 能抓什么 | 抓不到什么 |
|------|----------|------------|
| 与主机解析解逐元素对拍 | 实现与主机不一致（符号、转置、广播方向、除数是行数还是列数） | **两边犯同一个错**，或主机实现本身写错 |
| 有限差分 `(L(x+ε) − L(x−ε)) / 2ε` | 公式本身写错（不依赖任何一份解析反向的正确性） | 数值放大后的偏差（噪声随 `1/ε` 增长） |

有限差分用**设备前向**算 `L`，让两边算术一致，容差才能卡到 1e-5 而不是被 GEMM 求和顺序淹没。

**顺序有硬约束**：必须先做有限差分、再做解析反向。因为设备端层的 `backward` 会**就地更新参数**
（与主机一致），而有限差分必须在一整套固定参数上完成。同理，算切线的用例都要
`set_lr(0)`。

有一条纪律在这里第一次收到回报：`dev_mse_loss_t::loss` 最初漏了平方（`mean(y−target)` 
而不是 `mean((y−target)²)`）。**它有下降趋势、形状也对、逐层梯度全部正常** ——
只有端到端那条对拍把差异暴露成 `0.44 vs 2.07`。所以「训练能跑、损失在降」不能当验收标准。

参数梯度没有直接返回值（是在 `backward` 里就地更新的），对拍时用
「初始参数 − 更新后参数，再除以学习率」反推：sgd 下这是精确值，顺带把**参数更新语义**一起验了。

### 训练的容差：第 0 步卡 1e-12，之后必须放宽

端到端用例（`LayerNorm → Linear → SiLU → Linear → MSE` 训练 6 步，与主机同构栈逐步对拍）
的容差刻意分两段：

- **第 0 步 1e-12**：两边走的是**同一批参数**，只有 GEMM 求和顺序不同。这一步卡紧才有意义 —— 
  它是「梯度确实算对了」的证明。
- **之后 1e-7**：训练是个反馈过程，第 0 步的 ulp 级差异会被逐步放大。再卡紧就不是在测正确性，
  而是在测混沌。

同理，除了损失值，**参数也要跟着比**：某一条路径的更新语义不同（比如学习率用错、更新顺序颠倒）
时，损失可能仍然在降，只有参数轨迹会分道扬镳。另有一条与主机无关的自检（60 步后损失确实下降）
用来确认「一致的方向是对的」，而不是两条实现一起朝着错误方向走。

### 差分要覆盖**每一个**参数矩阵，不能只钉头、中、尾

整模型反向（`tests/test_cuda_llama.cu`）没有主机 backward 可以对照 —— `llama_model_t` 是纯推理
实现 —— 所以有限差分是**唯一**的裁判。最初只在三处取参数（`wte`、某一层的 `W_Q`、`lm_head`），
看起来「头、中、尾各钉一个」很合理，其实漏得不轻：**整栈反向里任何一处漏算、符号反了、
或者 GQA 下少累加一个共享 KV 头，都只影响它自己那一段**，三处之外的错误可以安然通过。
现在差分覆盖每一个参数矩阵（每层 9 个 + `wte` + `ln_f` + `lm_head`），总共约两千次前向、
两秒以内 —— 这个价格换来的确定性远比省下的时间值钱。

由此还得到一条分诊经验，写在这里免得下次误判：

> **损失炸了先怀疑步长，不要先怀疑梯度。**
> 朴素 SGD 只保证在 `lr < 2/λ_max` 内收敛，越界就是单调发散。实测同一套参数：`lr=0.05` 时
> 60 步从 0.69 降到 3e-5；`lr=0.2` 时 60 步涨到 1e76。**这两件事长得一模一样**，
> 分开它们靠的正是上面那条差分用例（它用一次更新 + 小学习率反推梯度，与步长无关）。
> 顺带一提，收敛用例打印的是**整条损失轨迹**而不是首尾两点：「降得慢」和「先降后炸」
> 只看首尾是分不出来的。

### 零拷贝子视图：`dev_mat_t` 的逻辑列数和前导维是分开的

`dev_mat_t` 原来只有 `m_rows` / `m_cols`，而 `m_cols` 同时充当「逻辑列数」和「存储步长」。
于是「缓冲区按 `cap` 分配、只对外暴露前 `len` 列」这种视图**表达不出来** —— 每个 decode step
都得把已用部分拷成紧凑缓冲，正好把 KV cache 的意义抵消掉（每步 O(len) 拷贝 × 步数 = O(len²)）。

现在 `m_cols` 是逻辑列数、`m_ld` 是存储步长，`view(rows, cols)` 只改前者。
`cuBLAS` 本来就支持 `ld > cols`，`gemm` 读的也是 `leading_dim()`，所以零拷贝视图能直接进 GEMM。

**为什么不会读到 `cap - len` 那段"空闲"列**：转置视图下每个矩阵槽被解释成 `(k × n)`，
其中**被 ld 乘的那个下标只遍历逻辑列数**（≤ ld），最大偏移因此落在缓冲区实长之内。
用例里用 `ld=48, cols=16` 的视图直接做 GEMM 并与主机端逐元素对拍。

顺带一条容易踩的：`keys()` 返回的薄壳**在扩容后即悬空**（扩容会重新分配缓冲区）。
decode 前 `reserve()` 一次就不再分配，也是推荐用法。

### GQA：用 API 形状消灭「重复 append」

主机端 `jas_mha_t.hpp` 记录过一次事故：GQA 下「每个 Q 头各自 append」会把同一份 K/V
写进同一个 KV 头两次，形状不报错、单步看似可用，到多轮对话才炸。

设备端不靠注释，而是让这种写法**根本不可达**：多 KV 头容器 `dev_kv_caches_t` 的唯一写入入口是
`append_all(k_full, v_full)`（一次写给**所有** KV 头），没有按单个 KV 头 append 的接口；
按 KV 头的访问只有 `const` 只读形式；`kv_head_of(q_head)` 的映射由容器持有，调用方不必自己算分组。
于是「各 KV 头长度恒等」是结构性质，而不是需要靠用例维护的不变量。

### 精度：默认不用 TF32（跨机器可复现的前提）

Ampere 及以后的卡上，单精度 GEMM 可以走 **TF32 张量核**（尾数只剩 10 位，相对误差约 1e-3），
而开不开取决于 math mode 与 `NVIDIA_TF32_OVERRIDE` 环境变量 —— **同一份代码在不同机器上数值不同**。

本项目的整套对拍基准是「与主机参考逐元素对齐」，float 用的是 1e-5 量级的相对容差，
TF32 的 1e-3 一冲就没了；而且它在测试机（Pascal P4）上**看不出来**，到目标机才发现。

所以 `cublasCreate` 之后立刻钉死 `CUBLAS_PEDANTIC_MATH`（真 FP32/FP64），
要 TF32 得显式 `cuda::set_gemm_math(cuda::gemm_math::tf32)`；在不支持 TF32 的卡上请求它会
**抛异常**而不是静默降级（静默降级会让你在测试机上"验证过"一个在目标机上没生效的加速）。
`Dgemm` 不受 TF32 影响，两种模式下都是真双精度。

### 融合注意力：怎么证明「不物化」不只是一句口号

`jas_cuda_attention.hpp` 的卖点是「概率矩阵不进显存」。但这件事最容易退化成一个自述：
代码看着是分块流的、用例全绿，而中间矩阵其实还在某处 `allocate`。所以验收分三层：

| 层次 | 怎么验 |
|------|--------|
| 数值 | 融合前向与**非融合实现**和**主机算式**三方对拍；反向与有限差分对拍 |
| 结构 | 直接断言 `m_weights` 为空、`m_lse` 非空 —— "省掉了"必须能在**状态**上看见，而不是只能从耗时去推测 |
| 收益 | `weights_bytes()` 把省下的量算出来，比值必须**恰好**等于 `len` |

第三层不是走形式。`weights_bytes` 的第一版把 `d_head` 也乘了进去（成了每头一份
`q_len × len × d_head` 的矩阵），比值就成了 `len·d_head/(d_head+1)` —— 一个 2048 上下文、
32 头 / `d_head=64` 的层会被报成 65536 MiB（真值是 1024 MiB）。它**看起来只差百分之几**，
光看"省了两千倍"根本发现不了。把比值卡成恰好 `len` 之后，这类夸大口径立刻现形。

分块（`bc`）的测法与 softmax 的两条路径同构：`flash_attention_block_cols_override()` 把分块
硬压到 1、压到 `len`、再留自动值，三档必须给出**同一个结果**。这是分块逻辑唯一的独立裁判 ——
只跑自动值的话，分块写错会因为"恰好整块装下"而完全看不出来。

有一条与分块绑死的 bug 值得记：前向里旧累加和的缩放写成 `s_l[warp] * corr`（每个 lane 都算一遍），
被 warp 归约求和后等于把 `s_l[warp]` 加了 32 次。`bc >= len` 时 `s_l` 恒为 0，
所以**单块用例全绿**，只有分块才错，而且错得像是"精度问题"。

### 混精度：容差必须是算出来的，不是哄出来的

降精度最容易滑向"把容差放宽到 1e-2、然后什么都不验"。这组用例拒绝那条路，改成
**证明误差只来自它该来的地方**。误差只有四个来源，每个单独验：

| 来源 | 上界 | 怎么验 |
|------|------|--------|
| 操作数量化 | `2·u_op` | 与"不舍入"的参考比，差异必须**明显大于**累加项（否则说明降精度压根没生效） |
| 乘积 | **0** | 与"先把操作数舍入、再用 double 算"的参考比，差异只该落在累加项之内 |
| 累加 | `K·u_fp32` | 同一条断言的反面：**若 cuBLAS 用 16 位累加，这条会以量级之差失败** |
| 结果舍入 | `u_out` | 输出降精度那一档，与"同一个 gemm 但输出留 fp32"比 —— 差的正好是那一次舍入 |

最后一行有个反例值得留档：这一条最初拿"把数学参考也量化一次"当基准。那份参考把结果舍入也做了
一遍，**两次舍入互相抵消**，于是测出来的只有累加误差（实测 0），"输出类型到底生效没有"
根本没被验证。换基准之后实测 2.7e-3 —— 正好是 `bf16` 的 `u` 量级。

模型本身也要先校准：`u = 2^-(m+1)` 里 `m` 写错一个比特就是差一倍，所以第一条用例先拿
"实测最大舍入误差"去比模型上界（实测占上界 99.6%，说明模型没写错）。另有一条比两种格式的
实测误差，比值应当落在 8 附近（尾数 7 位对 10 位，差 3 位）—— 这条的样本要够多，
否则比的是抽样噪声而不是模型（单种子下曾量到 17.6，扩到四个种子 × 64 元素之后回到 9.2）。

`cast` 单独钉一条：设备端 `cast<bf16_t>` 必须与主机端 `narrow` **逐位一致**，
不然"设备上算的"和"主机上算的"是两份不同的量化，所有对拍都会失去意义。

`matmul` 则从另一头设防：**隐式混精度在编译期就不通过**（`matmul(a<float>, b<bf16>)` 报错），
要混必须显式 `cast`。理由是隐式提升会在表达式树里四处发生，而"哪一步降了精度"必须留在源码里。

### 相关文件

| 文件 | 作用 |
|------|------|
| `jas_mat_express_t.hpp` | `storage_of` / `storage_type` 存储策略；`scalar_leaf_t`；`is_self_contained_v`；`is_mat_dot`；各表达式节点与运算符；`mat_dot_t` |
| `jas_mat_concepts.hpp` | `is_caculable`（判标量前 `remove_cvref`） |
| `jas_mat_t.hpp` / `jas_mat_view_t.hpp` | `.dot()` 的 ref-qualified 声明；访问器的 `JAS_HD` 标注 |
| `jas_cuda_compat.hpp` | `JAS_HD` / `JAS_DEV`、设备安全数学、`device_evaluable` 探测 |
| `jas_cuda_leaf.hpp` | 设备叶子 `dev_mat_t`（薄壳、转置、**独立前导维 + `view()` 零拷贝子视图**、**`row_slice()` 行子块**） |
| `jas_cuda_gemm.hpp` | cuBLAS GEMM、`matmul` 三入口、`.dot()` 的定义、**精度 math mode 控制** |
| `jas_cuda_reduce.hpp` | 广播叶子、`dev_colvec_t`/`dev_rowvec_t`、归约 kernel、`softmax_rows`（**单趟共享内存 + 三趟回退**） / `layer_norm` / `rms_norm` |
| `jas_cuda_attention.hpp` | 融合注意力：online softmax + 共享内存 K/V 分块，前向/反向不物化概率矩阵（`choose_block_cols`、`attention_cache_t` / `attention_grad_t`、分块覆写与启动计数两个测试钩子） |
| `jas_cuda_precision.hpp` | 混精度：`bf16_t` / `fp16_t`、误差模型（`precision_traits` / 量化 / 累加 / 总上界）、`narrow` / `widen` / `quantize`、`cast`、`to_host_double` / `from_host_double` |
| `jas_cuda_kv_cache.hpp` | `dev_kv_cache_t` / `dev_kv_caches_t`（GQA）、`attend_cached` |
| `jas_cuda_rope.hpp` | `dev_rope_t`：cos/sin 表上设备、两种配对约定、`start_pos` 偏移、逐头行子块、原地旋转、**反向** |
| `jas_cuda_updator.hpp` | 设备端优化器 `dev_sgd_t` / `dev_adam_t` / `dev_nadam_t` / `dev_cache_updator_t`（原地 kernel） |
| `jas_cuda_net.hpp` | 设备端层库：linear / layer_norm / rms_norm / silu / gated / residual / mse，forward + backward |
| `tests/test_cuda_kv_cache.cu` | 设备 KV cache：容器对拍、decode/prefill、GQA、扩容与边界、精度模式 |
| `tests/test_cuda_rope.cu` | 设备 RoPE：与主机 `RoPE_net_t` 对拍、配对约定、`start_pos`、逐头、`row_slice` 地址运算、RoPE+KV cache 端到端 |
| `tests/test_cuda_backward.cu` | 反向传播与层库：优化器对拍、各层反向的「主机 + 有限差分」双裁判、RoPE 转置回旋、整栈训练逐步对拍 |
| `tests/test_cuda_mha.cu` | 设备端 MHA / Embedding：单头与多头（MHA/GQA/MQA）对拍、参数梯度、独立有限差分、KV cache 解码路径、重复 id 的原子累加、越界 id 报错 |
| `tests/test_cuda_llama.cu` | 整模型：参数搬运后逐层对拍主机、增量解码对拍、全参数有限差分、训练收敛自检 |
| `tests/test_cuda_attention.cu` | 融合注意力：与非融合/主机三方对拍、因果性、分块无关性、反向有限差分、引擎记录语义、`m_weights` 必须为空 + 收益量化 |
| `tests/test_cuda_precision.cu` | 混精度：`u` 校准、误差逐源验证（量化/乘积精确/累加/结果舍入）、`cast` 逐位一致、`matmul` 拒绝隐式混精度、`alpha`/`beta`、显存减半 |
| `tests/test_expression_lifetime.cpp` | `ExpressionLifetime.*`：值类别契约 + 生命周期回归 |
| `tests/test_cuda_fused.cu` | `CudaEnvironment.*` / `CudaDeviceTest.*`：设备契约 + 融合/GEMM 对拍 |
| `tests/test_cuda_reduce.cu` | `CudaReduceContract.*` / `CudaReduceTest.*`：归约、softmax、归一化、注意力端到端 |
| `tests/test_cuda_dot.cu` | `CudaDotContract.*` / `CudaDotTest.*`：`dot` 分派、转置组合、链式与表达式操作数 |

---

## 10. RoPE 缓存的并发填充（L 形补齐 + 发布/订阅）

RoPE 的旋转块 `[[cosθ, -sinθ], [sinθ, cosθ]]` 只跟 `(特征对 i, 位置 m, d)` 有关，
所以整张表按需算、算过就存：`mat_RoPE_t` 持有 `mat_cache_t`，行是特征二维对、列是位置展开的 2×2 块。

麻烦在于**它不归某一个头所有**：`rope_registry_t<val_type>` 是按 `(d, layout)` 索引的**进程级单例**，
而 `mat_mha_t` 的每个头都在 `#pragma omp parallel for` 里跑，于是「哪些格子已经算过」这件事
既要不重复计算（否则每加一个位置就重算整张表），又必须在并发下成立。

### 不变量

> **缓存自称已填充的矩形 `[0, m_filled_rows) × [0, m_filled_cols)` 里，不允许存在从未被写过的格子。**

这不是洁癖：`mat_t` 的存储来自 `new val_type[...]` 且分配后 `memset` 清零，没写过的格子读出来是 0.0，
不崩、不报错，只是那块 2×2 旋转矩阵变成全 0 —— **对应特征对被静默清零**，误差随层数放大。

### 曾经的 bug（两个成因，同一个不变量）

**1. 用「两个轴的 max」记账，而不是真写过的范围。** 旧 `init` 只在请求到的矩形里逐格跳过「已填」
格子，最后把两个轴的上限各自取 max。于是这个序列会留下洞：

```cpp
rope.init(0, d, 0, 2);   // 高而窄：写了 [0,d) × [0,2)
rope.init(0, 2, 0, 8);   // 矮而宽：只写了 [0,2) × [2,8)
// 记账成了 (d, 8)，但 [2,d) × [2,8) 从没被遍历过 —— 却被记成已填
```

单线程时请求顺序是「只长列」（每次都只有列方向越界），不会踩到；多线程交错时，
先来的可能是「宽而矮」再是「高」，洞就出现了 —— 这正是
`LlamaAlignmentTest.RawHiddenStatesMatchLayerByLayer` **时好时坏**、且失败时逐层漂移的原因。

**2. 补齐过程没有同步。** 判「够不够」读的两个 `int`（旧 `m_enable_rows/cols`）被多线程同时读写；
更糟的是扩容会**替换底层存储**，而别人的 `mat_view_t` 正指着旧缓冲区。

### 修法

| 问题 | 现在 |
|------|------|
| 补齐范围不精确 | `fill_from_origin()` 只补**精确的 L 形带**：新来的行 × 全部列 + 老行 × 新来的列。只增不减，所以「补到哪」永远是一块完整矩形，不需要逐格判断 |
| 并发补齐 | 整个「扩容 + 补齐」在 `m_fill_mutex` 内串行化；进锁后会**重读一次上限取 max**，两个线程同时走慢路径也不丢更新 |
| 上限的可见性 | 上限改成 `std::atomic<int> m_filled_rows/cols`，写完再 `release`，读用 `acquire`。命中时走**完全无锁**的只读快路径（两次 acquire 读 + 一次 `raw_view`），所以读不是瓶颈 |
| 扩容换存储 | 旧存储**不析构**，进 `m_buffers` 退休（读者手里可能还攥着它的视图）。增长按 1.5 倍几何扩（见下），所以退休总量收敛：在职 + 退休 ≤ 3 倍最终容量，不随扩容次数线性累积 |
| 误用面 | 删掉 `mat_cache_t` 上「读一次顺手扩容+改记账」的惰性 `operator()` / `range`，只留 `grow_to`（容量）+ `cell`（写）+ `raw_view`（共享读，不改任何状态） |

副作用与边界：**dynamic 模式现在也可以多线程共享**（原先头文件注释写的是「不安全」）；
代价是扩容过的实例会多留一份旧存储。`static_fixed`（先 `reserve` 一次填满、运行期不写）不受影响，
它的语义也依旧更严：越界直接抛 `std::out_of_range`，绝不在运行期扩容。

### 追加：增长路径的对象同一性（第二轮，内存口径修好之后才暴露）

第一轮修完 TSan 是 **0 处告警**——因为当时构造函数预分配了 `1024×1024`（8 MiB），
小规模用例**根本走不到扩容**，增长路径等于没被覆盖。把预分配去掉之后，
`ConcurrentFillMatchesSerialFill` 变成**随机段错误**，TSan 给出的报点很干脆：

```
T2 写: mat_cache_t::ensure_capacity        ← fill_from_origin ← range
T1 读: mat_t<double>::col_num()            ← mat_view_t::operator()   ← 早已取出的视图
```

根因不在锁，而在**对象同一性**：`mat_view_t` 里存的是 `mat_t*`，**每次取元素都重新解引用那个对象**
（读它的 data 指针与维度）。旧实现把缓冲存成 `mat_cache_t` 的一个 `mat_t` 成员，扩容时原地 `= std::move(新缓冲)`——
于是任何一个已经交出去的视图，在并发扩容下都会读到一个**正在被改写的对象**。这不是数值偏差，是 UB。

修法是把它变成不变式：**对象一经发布就不再改动**。缓冲全部由 `m_buffers`（`vector<unique_ptr<mat_t>>`）持有，
当前的那个由 `std::atomic<mat_t*> m_cur` 指过去；扩容 = 拷进**新对象** + `release` 换指针，
读方一次 `acquire` 载入指针后，拿到的维度与 data 就都稳定了（写方只写「已发布范围之外」的格子，
与读者读的格子不重叠）。这也让「退休」从「防 use-after-free」升级成「保证读者看到的是一个完整的旧对象」。

### 内存口径：`EXPAND_SIZE` 这个下限是反的

同一轮顺手修掉的第二个问题：`mat_cache_t` 原本一出生就 `mat_cache(EXPAND_SIZE, EXPAND_SIZE)`，
而 `ensure_capacity` 又用 `m_cache.row_num() + EXPAND_SIZE` 保底。RoPE 表的**行数是 `d_head`**
（几个到上百个），给它加一个 1024 的下限，等于每次都为小维度白分配 1024 行。TinyLlama（`d_head=64`）
量出来的差别：

| 场景（`d_head=64`，2048 位置） | 改前 | 改后 |
|------|------|------|
| dynamic 惰性填充 | +32 MiB | +9 MiB（驻留 2 MiB + 退休 6 MiB） |
| static 预留 | +2 MiB | +2 MiB（精确，退休 0） |

32 MiB 正是那个「1024 行下限 × 4096 列 × 8 字节」。现在改成空对象起步 + 纯 1.5 倍几何增长
（够用优先，其次 1.5 倍，避免逐位置增长时 O(n) 次重分配）。

### 把预留真正接到推理路径上

上面这些都属于「修好那条路径」；但**推理路径压根没走预留**：`mat_mha_t::set_param` 里
`reserve_kv_cache(seq_len)` 与 `bind_rope()` 并排，而 `bind_rope()` 的 `max_seq_len` 默认 0，
`rope_registry_t` 又默认 `dynamic` —— 两个条件都不成立，`reserve()` 一次都不会调。
`set_rope_cache_mode` / `reserve_rope` / `bind_rope(max_seq_len)` 这三个 API 全仓只在测试里出现过。

现在补上 `mat_mha_t::reserve_rope_cache(max_seq)`（经 `llama_model_t::reserve_rope_cache` 下沉到各层），
`llama_chat` 在 `reserve_kv_cache(max_context)` 旁边一行调用。几个刻意的选择：

- **不改全局默认模式**，只对拿到的那一个共享条目 `set_cache_mode`——注册中心按 `(d_head, layout)` 共享，
  改全局默认会波及同进程里其它维度的用法。
- **预留长度单调取 max**，且已就位就早返回——否则上层逐层调下来时，`reserve()` 会重置记账并整表重填 `n_layers` 遍。
- 没有 RoPE 的模型（GPT-2 那类绝对位置）调它是**无操作**，不是抛异常。
- 越界抛 `std::out_of_range`：这是把「上下文上限」从注释变成可检查的不变量。

### 单测

`tests/test_rope_cache_race.cpp`（4 例，`RoPeCacheThreading.*`）：

| 用例 | 钉住什么 |
|------|----------|
| `FilledRectangleHasNoUnwrittenCell` | 上面那个 init 序列留下的洞；与线程无关，**旧实现确定性失败** |
| `ConcurrentExtentsMatchAnalyticValues` | 8 线程乱序请求不同矩形，每个块必须等于解析值 |
| `ConcurrentFillMatchesSerialFill` | 整块矩形「并发 vs 串行」对拍（全量块，不只是抽样） |
| `ReservedCacheIsSafeToReadConcurrently` | `static_fixed` 只读共享的对照组 |

TSan 是这条不变量的关键证据。kernel 6.17 上 `-fsanitize=thread` 的二进制会撞
「unexpected memory mapping」（ASLR / shadow 映射），要关掉随机化；`setarch -R` 需要
`personality(2)`，在沙箱里会被拦，**要在普通 shell 里跑**：

```bash
GT=build-tsan/_deps/googletest-src/googletest/include
g++ -std=c++20 -fsanitize=thread -g -O1 -fno-omit-frame-pointer \
  -I. -I"$GT" -Itests tests/test_rope_cache_race.cpp \
  build-tsan/lib/libgtest.a build-tsan/lib/libgtest_main.a -lpthread -o /tmp/tsan_rope
setarch "$(uname -m)" -R /tmp/tsan_rope --gtest_filter='RoPeCacheThreading.*'
```

修前 / 修后的实测（同一份测试、同一套编译参数）：

```
修前：exit 66，25 处 WARNING: ThreadSanitizer: data race，FilledRectangleHasNoUnwrittenCell 失败
      报点集中在 jas_RoPE_t.hpp 的 range()/init() 与 mat_cache_t::range()，还有直接落在存储上的读写竞争
第一轮修后：exit 0，0 处告警，4 例全过
      —— 但这时预分配还在，增长路径没被跑到，所以这个「0」是有水分的
第二轮（去掉预分配 + 原子换指针）：exit 0，0 处告警，4 例全过，
      且此时 ConcurrentFillMatchesSerialFill 真的在走扩容；修前它是随机段错误（60 次重复 0 失败）
```

复现命令（段错误那条也可以用 ASan，但它是时序相关、不一定每次撞上，TSan 是更稳的裁判）：

```bash
# 抓时序：重复跑，任何一次非 0 都是回归
for i in $(seq 1 60); do ./build/tests/unit_tests --gtest_filter='RoPeCacheThreading.*' || echo "FAILED at $i"; done
```

### 相关文件

| 文件 | 作用 |
|------|------|
| `jas_RoPE_t.hpp` | `mat_cache_t`（`m_buffers` + `m_cur` 原子指针 / `cell` / `raw_view` / 退休存储）+ `mat_RoPE_t`（`fill_from_origin` / `range` / `reserve`） |
| `tests/test_rope_cache_race.cpp` | `RoPeCacheThreading.*`：填充不变量 + 并发回归 + TSan 复现入口 |
| `tests/test_rope.cpp` | `RoPE.ReserveRopeCache*`：预留的长度/模式/越界/单调性/无 RoPE 时无操作 |
| `tests/test_llama_weights.cpp` | `LlamaStructure.ReservedRopeCacheKeepsOutputIdenticalAndFailsFast`：预留后的数值必须与惰性填充**逐位一致** |
| `jas_mha_t.hpp` | 各头并发调用 `m_rope->forward_at()`；`reserve_rope_cache()` 在这里切换模式并预留 |
| `examples/llama_chat.cpp` | 推理路径唯一的调用点（`reserve_kv_cache` 旁边） |

## 11. 二维卷积（`conv2d_net_t`）：im2col 与「两条 GEMM」的反向

仓库里原先没有任何卷积层的影子，这一节记录新增的 `jas_conv_t.hpp` 为什么长这样，
以及两个必须知道的坑。

### 为什么是 im2col，而不是写一个卷积核

jasmine 的数据载体只有 2 维的 `mat_t`，没有 NCHW/NHWC 张量，也没有 `im2col` 之外的第三个选择。
把卷积的感受野摊平成列之后，整层退化成一件仓库里早就优化过的事：

```text
输入 x   : [C_in,  H * W]            通道先行，空间维展平在列上
卷积核 W : [C_out, C_in * Kh * Kw]   每个输出通道一行，行内按 (c, kh, kw) 展平
偏置 b   : [C_out, 1]                广播到所有输出位置
输出 y   : [C_out, H_out * W_out]
```

这**就是 `weight_net_t` 的形状约定**（权重 `[out, in]`、输入 `[in, T]`），只是把 "in" 换成了
`C_in*Kh*Kw`、把 "T" 换成了 `H_out*W_out`。于是：

```text
forward : y   = W · col + b                      一次 GEMM（大尺寸自动走 BLAS）
backward: dW  = delta · colᵀ                     第二条 GEMM
          dcol= Wᵀ · delta                       第三条 GEMM
          dx  = col2im(dcol)                     散射累加
          db  = Σ_{输出位置} delta
```

卷积的复杂度全部收敛在 `im2col` / `col2im` 这对循环里，矩阵乘一行没重写。反向的三份梯度
（输入 / 权重 / 偏置）都在 `backward` 里算完，权重与偏置直接经 updator 原地更新，与
`weight_net_t::backward` 的语义一致。

一维卷积（序列卷积）不做单独分支：取 `H = 1, Kh = 1, pad_h = 0`，把长度 L 的序列当作
`[C_in, 1*L]` 喂进来即可，形状检查与反向逻辑都不需要分叉。这一串参数有便捷写法，
它返回的仍是同一个类的实例，**不是第二套实现**：

```cpp
using conv_t = conv2d_net_t<mat_t<double>, sgd_t>;
auto conv = conv_t::one_d(c_in, c_out, /*len=*/L, /*k=*/3, /*stride=*/2, /*pad=*/1);
auto y = conv.forward(x);        // x: [C_in, L]  →  y: [C_out, L_out]
```

### 形状从哪来：`set_param`，而不是 `reinit`

`[C_in, H*W]` 反推不出 `H` 与 `W`（两者可互换），核大小更没法从输入推断，所以本层的形状由
`set_param(...)` 显式配置，构造函数只是它的一层转发。

**刻意不叫 `reinit`**：`is_reinitable_net` 靠 `requires { net.reinit(std::vector<int>()); }`
判定（见 `jas_mat_concepts.hpp`），给了 `reinit` 就会改变所有含本层的复杂网络的 reinit 语义
（容器会要求多一对 `{in, out}` 槽位）。这与 `layer_norm_net_t` 用 `set_param` 的理由完全相同，
所以本层在 `complex_net_t::reinit` 里是**被跳过**的一层：

```cpp
using conv_relu_t = complex_net_builder_t<double>
    ::push_back_updatable<conv2d_net_t, sgd_t>
    ::push_back_staticnet<relu_net_t>
    ::type;

conv_relu_t net;                                  // 默认构造：此时权重还是空的
net.get<0>().set_param(2, 3, 3, 3, 2, 2);         // 显式配置 + 分配权重/偏置
net.get<0>().init_weight<he_gaussian_t>();        // 再填充
```

### 坑一：`mat_t::reshape(1, 1)` 在空矩阵上会除零

`C_out == 1` 时偏置是 1x1，而 `mat_t` 把 1x1 一律当作**标量矩阵**。`reshape(1,1)` 一旦走进
标量分支，会先执行 `(*this)(0, 0)` 去取旧值——而 `mat_t::operator()` 内部是
`r % row_num()`，对默认构造的 0x0 空矩阵就是 `0 % 0`。这不是「抛异常」，是直接 SIGFPE。

所以本层所有「分配新形状」的位置一律写成**整体赋值一份新矩阵**：

```cpp
m_weight = mat_t<val_type>(c_out, c_in * kh * kw);   // 而不是 reshape(...)
m_bias   = mat_t<val_type>(c_out, 1);
m_col    = mat_t<val_type>(c_in * kh * kw, h_out * w_out);
```

`col2im` 里的 `dx` 同理（`BackwardAccumulatesOverlappingPatches` 与 `DegenerateShapesStillWork`
两例就是钉这个：后者是 1 通道 + 1x1 输入 + 1x1 核，权重/偏置/im2col 全是标量）。

### 坑二：除零发生在校验之前

输出尺寸公式里有 `... / stride`。若先算尺寸再校验，`stride == 0` 会先在整数除法上崩掉，
根本轮不到 `throw`。所以 `set_param` 的顺序被固定为：

1. `validate_basic(...)` —— 通道/空间/核/stride/dilation/padding 的**不依赖尺寸**的检查；
2. 再算 `H_out / W_out`；
3. 再检查 `H_out/W_out >= 1`（核比 padded 输入还大的情形）；
4. 最后才写入成员，抛出时对象保持原状。

### 坑三：`W.dot(col) + b` 会把整次 GEMM 踢出快速路径（已修）

写这段时用探针量出来的一件事，比 im2col 的拷贝本身重要得多：

```cpp
return m_weight.dot(m_col) + m_bias;      // 慢
```

`mat_add_t` 继承的是 `mat_express_2_param_stable_t::clone()`，它是**逐元素**求值的：
对每个 `(i,j)` 调一次 `mat_dot_t::operator()`，而后者是「自己扫一遍 K」的朴素循环。
于是 `dot` 外包一层 `+` 之后，`try_fast_gemm` / BLAS **根本不会被调用**。实测
（M=64, N=1024, K=288, 37.8 MFLOP, 20 次平均）：

| 写法 | 耗时 |
|------|------|
| `W.dot(col) + b`（表达式） | ~100 ms |
| `W.dot(col)`（裸 dot，走 BLAS） | 7.4 ms |
| `mat_t y = W.dot(col); y += b;` | 7.2 ms |
| `conv2d_net_t::forward`（im2col + 上式的两步写法） | 8.8 ms |

拆成两句后快了 **约 12 倍**：`+=` 是纯逐元素 O(M·N) 的广播加，不碰 GEMM。
所以本层前向固定写成「先裸 dot、再 `+=` 偏置」，并在 `forward` 里留了注释。

**同一个坑还在 `jas_net_t.hpp:52` 的 `weight_net_t::forward`**——那是全仓库所有线性层
（QKV/输出投影、FFN、lm_head……）的必经之路，也就是说它们现在都在走朴素三重循环。
改法与这里完全相同（拆成两句），但它会改变全局数值路径（BLAS 的求和顺序与朴素循环不同，
量级 1e-15），所以**没有在这次卷积改动里顺手改**：那属于独立的性能改动，
要连同它对 GPT-2 / LLaMA 对齐测试的影响一起评估。对齐测试的容差是 1e-9（float64），
理论上足以吸收这个量级，但这是一个需要单独决策的库级变更。

### 顺手验证过的：im2col 能不能做成「不拷贝的懒视图」

顺便回答一个自然的设计问题——既然 `dot` 认的只是 `row_num()/col_num()/operator()(i,j)`
（见 `jas_mat_concepts.hpp` 的 `is_matrix`），那能不能写一个懒 im2col 视图，把索引计算放在
`operator()` 里，从而完全不拷贝？（探针里真写了一个 `im2col_view_t`：`static_assert(is_matrix<...>)`
通过，数值与物化版逐位相同，`max|Δ| = 0`。）

结论是**在这个代码库里不行**，原因是拷贝并没有被消除，只是被搬进了 GEMM：

- `mat_dot_t::clone()` 会走 `try_fast_gemm` → `gemm_operand`，后者只对「行优先的 `mat_t`」直接取
  `data()` 指针；**任何其它类型（视图、表达式、懒视图）都会被物化成一个临时行优先矩阵**再交给 BLAS。
  所以懒视图只是把 im2col 的拷贝挪到了 `gemm_operand::owned` 里。
- 在朴素路径（尺寸小于 BLAS 阈值、或外面又套了表达式）里更糟：`operator()(k, j)` 的索引算术
  会被**重复 M 次**（每个输出通道读同一列元素一次），而物化版本只算一遍。

实测（C_in=32, C_out=64, H=W=32, 20 次平均，`lazy` 指上面那个懒视图）：

| 核 | im2col 拷贝 | 物化: 裸 dot | 懒视图: 裸 dot | col / input |
|----|------------|--------------|----------------|-------------|
| 1x1 | 0.23 ms | 0.72 ms | 1.09 ms | 1x |
| 3x3 | 1.75 ms | 7.5 ms | 9.5 ms | 9x |
| 5x5 | 4.4 ms | 19.7 ms | 28.7 ms | 25x |
| 7x7 | 8.5 ms | 46.9 ms | 59.6 ms | 49x |
| 11x11（28x28 输入） | 15.5 ms | 56.7 ms | 82.1 ms | 121x |

也就是说：拷贝是 O(K·N)，GEMM 是 O(M·K·N)，拷贝被摊薄了 1/M；而且它把「带除法取模的
跨步收集」变成了连续、可被 BLAS 按块复用的数据。**物化 im2col 的拷贝在这个设计里是划算的。**

懒视图真正能省的是**内存**（col 是输入的 Kh·Kw 倍，11x11 时 121 倍），以及在自己写
implicit GEMM / 直接卷积时才谈得上省时间——那要求 GEMM 能直接吃「收集式操作数」，
而不是先物化再乘。设备端同理：cuBLAS 只认指针 + leading dimension，懒视图在那里无从发挥，
要省只能在 kernel 里做 implicit GEMM。所以这次维持「物化 col 并缓存给反向复用」的方案。

### 单测

`tests/test_conv.cpp`（14 例，`Conv2d.*`）：

| 用例 | 钉住什么 |
|------|----------|
| `ForwardMatchesReference` | 朴素六重循环参考实现逐点对拍；覆盖 stride / padding / dilation / 非方输入 / 非方核，含会走 BLAS 的大尺寸 |
| `OneByOneIsChannelMix` | 1x1 核必须退化成 `W·x`（与 `weight_net_t` 同形） |
| `Conv1dIsOneRowSpecialCase` | `H=1, Kh=1` 的一维卷积与参考实现一致 |
| `BackwardMatchesNumericalGradient` | 输入 / 权重 / 偏置三份梯度都做中心差分对拍（SGD lr=1 直接读出梯度） |
| `BackwardAccumulatesOverlappingPatches` | 重叠窗口必须**累加**而非覆盖：全 1 权重下的解析梯度恰好是「四角 1 / 边 2 / 中心 4」 |
| `DegenerateShapesStillWork` | 1 通道 1x1 卷积的标量路径（上面那个 `% 0` 坑）不崩且梯度正确 |
| `BackwardOnTwelveByTwelveHitsGemmFastPath` | BLAS/分块 GEMM 路径与朴素参考反向逐点一致 |
| `RejectsBadShapes` / `UnconfiguredLayerFailsFast` / `SetParamRejectsInvalidConfig` | 输入形状、delta 形状、未配置、非法配置的快速失败 |
| `InitWeightFillsConfiguredShapes` | `init_weight<he_gaussian_t>()` 分配形状正确且非零 |
| `IsUpdatableButNotReinitDetectedByConcept` | `is_updatable_net` 为真、`is_reinitable_net` 为假（即不占用 reinit 槽位） |
| `ChainInsideComplexNet` | 作为 `complex_net_t` 的一层跑 forward / backward / step |
| `NetTypeReportsShapes` | `net_type()` 打印输入/输出/核/步长等结构信息 |

### 相关文件（卷积）

| 文件 | 作用 |
|------|------|
| `jas_conv_t.hpp` | `conv2d_net_t`：`set_param` + im2col 前向 + 三条 GEMM 反向 + `col2im` |
| `tests/test_conv.cpp` | `Conv2d.*`：前向参考对拍 / 数值梯度 / 退化形状 / 概念与链路集成 |
| `jas_net_t.hpp` | 形状约定与反向语义参照物（`weight_net_t`） |
| `jas_mat_gemm.hpp` | 前向与反向实际走的 GEMM（BLAS / 分块回退） |

补一条**外部裁判**：除了单测里的朴素参考实现，卷积层还与 PyTorch 做过随机对撞——77 组随机配置
（`C_in/C_out`、非方输入、1..3 的核、1..3 的步长、0..2 的 padding、1..2 的 dilation）下，
前向 `y` 与三份梯度 `∂L/∂x / ∂L/∂W / ∂L/∂b` 与 `F.conv2d` 的最大偏差 ≤ `6.7e-15`（float64 舍入量级）。
工具在 `/tmp` 下临时搭建（`jas_conv_t.hpp` + `F.conv2d` 读同一批数据），不入库。

## 12. 二维池化（`pool2d_net_t`）：无参静态层与 argmax 路由

卷积之后紧接着的是池化。这一节记录 `jas_pool_t.hpp` 的取舍，以及它和 PyTorch 对齐的四条语义。

### 一个类，两种模式

```cpp
enum class pool_mode { max, average };
```

最大池化与平均池化的**窗口遍历完全相同**，差异只在「窗口内怎么算」和「梯度怎么回」上，
所以做成一个类 + 运行期枚举，而不是两个高度重复的类（对照 GQA：那是把 `n_kv_heads` 参数化，
而不是新写一个注意力核）。运行期参数还让「默认构造 + `set_param`」的构建器模式照旧可用。

**池化没有可训练参数**，而且逐通道独立，所以本层是**静态层**：

- 与 `relu_net_t` / `gelu_net_t` 一样只吃 `input_type`，不持有 updator；
- 因此 `is_updatable_net` 为假，链上的 `set_updator` / `set_lr` 会自动跳过它；
- 形状来自 `set_param` 而不是 `reinit`，所以 `complex_net_t::reinit` 也不占槽位
  （`is_reinitable_net` 同样为假，单测里用 `static_assert` 钉住）；
- `init_weight<init_type>()` / `step()` 是空实现——但**必须存在**，因为
  `complex_net_t::init_weight` / `step` 会对所有成员无条件调用它们。

### 与 PyTorch 对齐的四条语义

| 语义 | 取值 | 为什么 |
|------|------|--------|
| 并列最大值 | **梯度只给第一个**（行优先） | PyTorch CPU 如此：`x=[[1,1],[1,1]]`、`max_pool2d(2)` 的 grad 是 `[1,0,0,0]`，不是 `[0.25,...]`。实现上就是 `if (v > best)` 的严格大于 |
| 平均池化除数 | 默认 `count_include_pad = true`，恒为 `Kh*Kw` | `nn.AvgPool2d` 的默认行为：padding 的 0 也进分母。置 false 才除以窗口内有效元素个数 |
| padding 上限 | `2*pad <= kernel` | PyTorch 的 "pad should be at most half of effective kernel size"。它同时保证每个窗口至少覆盖一个真实元素，于是最大池化不会遇到「整窗 -inf」、平均池化也不会除以 0 |
| stride 缺省 | `0` 作哨兵，表示「同核大小」 | PyTorch `MaxPool2d(2)` 就是 `stride=2`。0 在别处不是合法步长，拿它当哨兵不会歧义 |

输出尺寸与卷积同式（floor 模式）：`H_out = (H + 2*pad - Kh) / stride + 1`。
floor 语义下最后一个不满窗会被丢掉，例如长度 7、核 2、步长 2 出 3 个输出（末元素被丢弃）——
这一点也在单测里钉了。

为什么先不做 dilation / ceil_mode：平均池化在 PyTorch 里根本没有 dilation；`ceil_mode` 会引入
「最后一个窗口算不算满」的规则并与 `count_include_pad` 交织（PyTorch 在 ceil 模式下还要额外修正
分母），先把最常用的 floor + 整数核做扎实。

### 反向：一个 argmax 表，或一个位置函数

```text
max    : forward 记下每个输出元素的 argmax（展平输入下标）→ backward 把 delta 原路送回那个位置
average: 除数是位置函数（Kh*Kw 或窗口内有效元素个数）→ backward 把 delta/除数 散射累加到窗口内每个有效位置
```

- argmax 存的是 `std::vector<int>` 而不是浮点矩阵：大图上 `H*W` 超过 2^24 之后 `float` 存不下
  整数下标，而这里本来也不需要浮点。
- 重叠窗口（stride < kernel）在两种模式下都必须**累加**：最大池化的同一个像素可能同时是多个窗口
  的最大值，平均池化更是每个输出都往窗口内每个像素各贡献一份。单测用全 1 输入 + 全 1 delta 把
  梯度钉成「覆盖次数」与「覆盖次数/4」的解析值。
- 平均池化的除数反向现算而不缓存：它是 `(oh, ow)` 的纯函数，forward/backward 算出来必然相同。

### 踩到的一个编译坑：文档注释里不能写 `/*name=*/`

`jas_pool_t.hpp` 里给 `one_d` 写的 `/** ... */` 文档注释中原本有一行

```text
 *     auto pool = pool_t::one_d(pool_mode::max, L, /*kernel=*/2);
```

行内的 `/*kernel=*/` 里的 `*/` **会提前终止外层块注释**，后面的文字变成代码，报一串
"expected unqualified-id" 与 "one_d is not a member"。正确写法是把行内说明去掉或改用 `//`。
代码里（非注释内）的 `/*h=*/1` 这种参数标注是没问题的，只有嵌在块注释里才出事。

### 单测与外部对撞

`tests/test_pool.cpp`（15 例，`Pool2d.*`）：

| 用例 | 钉住什么 |
|------|----------|
| `MaxForwardMatchesReference` / `AverageForwardMatchesReference` | 朴素参考实现逐点对拍；覆盖重叠/不重叠、padding、非方核、非方输入、多通道、1x1 池化 |
| `MatchesPyTorchMaxGoldens` / `MatchesPyTorchAverageGoldens` | 硬编码 torch 2.9 的 `F.max_pool2d` / `F.avg_pool2d` 输出与梯度（含 `count_include_pad` 两种、非方核、多通道） |
| `MaxTieRoutesWholeGradientToFirstIndex` | 并列最大值只给第一个（全并列与部分并列两种） |
| `AverageOverlappingWindowsAccumulate` | 重叠窗口必须累加：全 1 输入下梯度 = 覆盖次数/4 |
| `BackwardMatchesNumericalGradient` | 最大池化（互异值避开不可导点）与平均池化（含/不含 padding）都做中心差分对拍 |
| `StrideDefaultsToKernelSize` | `stride=0` 哨兵解析为核大小 |
| `OneDIsOneRowSpecialCase` | `one_d` 一维池化与参考实现一致，且 H/Kh/H_out 确实被置为 1 |
| `RejectsBadShapes` / `UnconfiguredLayerFailsFast` / `SetParamRejectsInvalidConfig` | 输入/delta 形状、未配置、非法配置（核 0、负步长、负 pad、pad 超半核、核比 padded 输入还大）快速失败 |
| `IsStaticLayerNotUpdatableNorReinit` | `is_updatable_net` / `is_reinitable_net` 均为假 |
| `ChainInsideComplexNet` | conv → relu → maxpool 整条链 forward / backward / step |
| `NetTypeReportsConfig` | `net_type()` 打印模式、形状、核、步长、padding（仅 average 打印 `count_include_pad`） |

另外与 PyTorch 做了随机对撞：108 组随机配置（两种模式、2..9 的输入、1..4 的核、0..3 的步长、
`pad <= kernel/2` 的全部组合、1..3 通道、随机 delta）下，前向 `y` 与反向 `∂L/∂x` 与
`F.max_pool2d` / `F.avg_pool2d` 的**最大偏差为 0**（逐位一致）。

### 相关文件（池化）

| 文件 | 作用 |
|------|------|
| `jas_pool_t.hpp` | `pool_mode` + `pool2d_net_t`：窗口遍历、argmax 路由、平均池化除数、`one_d` 便捷入口 |
| `tests/test_pool.cpp` | `Pool2d.*`：前向参考 / PyTorch golden / 数值梯度 / 并列与重叠语义 / 概念与链路集成 |
| `jas_net_t.hpp` | 静态层（`relu_net_t`）的接口参照物：`init_weight` / `step` 空实现 |
| `jas_conv_t.hpp` | 形状约定（`[C, H*W]`）与 `one_d` 风格的同源实现 |

## 13. 形状视图（`mat_reshape_view_t`）与「视图操作数零拷贝进 GEMM」

前面第 11 节留下的两个缺口——「reshape 只能重建矩阵、不能改形状」和「懒 im2col 视图最终仍被
`gemm_operand` 物化」——在这一节一起补掉。

### `mat_reshape_view_t`：只改参数、不碰数据

```cpp
mat_t<double> x(1, 12);            // 一行 12 个数
auto patches = x.reshape_view(4, 3);   // (4 x 3)，零拷贝；patches(i,j) 就是 x(0, i*3+j)
auto col     = patches.t();            // (3 x 4) 转置视图，仍然零拷贝
```

语义要点（都有单测钉住）：

- `rows*cols` 必须等于元素总数，否则构造抛 `std::invalid_argument`；
- **别名**而不是副本：通过视图写入会直接改到底层（反之亦然），单测两个方向都验；
- 索引按**与底层相同的存储顺序**解释：底层行优先时 `view(i,j)` 是展平后的第 `i*cols+j` 个元素，
  底层列优先时是第 `j*rows+i` 个（等价于 numpy 的 `order='C'/'F'`），所以列优先存储也不会读错；
- `t()` 返回带转置标志的同类视图，依然零拷贝。

注意它和 `mat_view_t` 的分工：`mat_view_t` 是**取子区域**（子块/行/列/转置），
`mat_reshape_view_t` 是**换形状**（元素一一对应）。两者都不拷贝。

顺带说明：`mat_t::reshape(rows, cols)` 的行为没有改。它在元素总数相同时**什么都不做**
（既不重建也不更新维度，`jas_mat_t.hpp` 里只有 `rows*cols != row_num()*col_num()` 才重建），
在总数不同时会**重新分配并清零**。要"改形状但不拷贝"请用 `reshape_view()`。

### 让 GEMM 接受「指针 + 前导维 + 是否转置」

原来的 `gemm_operand` 只认行优先的 `mat_t`：任何其它类型（子视图、转置视图、懒视图）都被
物化成一块临时行优先矩阵再交给 BLAS。现在改成先向操作数索要描述符：

```cpp
detail::gemm_buffer<T> { const T* ptr; int ld; bool transposed; bool valid; };
```

`mat_t` / `mat_view_t` / `mat_reshape_view_t` 都能给出它（`mat_t::gemm_view()`、
`mat_view_t::gemm_view()`、`mat_reshape_view_t::gemm_view()`），给不出（列优先存储、嵌套转置、
元素类型不同）就退回物化——也就是原来的唯一路径。BLAS 侧用 `CblasTrans` 表达，
自带阻塞回退也加了 `TA/TB` 模板分支（`gemm_blocked_rowmajor_impl<TA,TB>`），两条路径都与朴素乘法
逐位一致（`tests/test_reshape_view.cpp` 里有断言描述符被真正给出的用例，避免"测试通过但其实在物化"）。

### 一个必须记住的坑：转置左操作数会慢 5 倍

量出来的，不是猜的。同样一批数据、同一套编译标志，只差操作数是视图还是物化矩阵
（`dW = delta(64x1024) · col(288x1024)^T`、`dcol = W(64x288)^T · delta(64x1024)`，本机参考 BLAS）：

| 位置 | 视图（零拷贝 + CblasTrans） | 物化后再 GEMM | 结论 |
|------|---------------------------|--------------|------|
| **右操作数** B | 1.53 ms | 1.66 ms | 零拷贝更快（还省一份拷贝） |
| **左操作数** A | 8.04 ms | 1.63 ms | **慢 5 倍**，必须物化 |

原因是参考 BLAS 的 `TransA` 走内积配方（按列读 A，stride = lda），而 `TransB` 的存储矩阵是
`(N x K)` 行优先、K 方向仍顺序访问。所以 `gemm_operand` 只对**右操作数**开 `allow_transposed`
（`jas_mat_gemm.hpp` 里写了这张表）；左操作数的转置视图仍然物化，`W^T·delta` 这类反向 GEMM
行为与改动前一致。换成 OpenBLAS/MKL 这条结论可能变，改策略前请重新量。

### 收益（本轮实测）

| 场景 | 改动前 | 改动后 | 说明 |
|------|--------|--------|------|
| patchify / 非重叠一维卷积：`W · x.reshape_view(n,t)^T` | 物化 col + GEMM 3.85 ms | 2.11 ms | **1.8x**，且不再需要 `t*n` 的临时矩阵 |
| 同上，t=9 的窗口 | 3.02 ms | 2.10 ms | **1.4x** |
| 卷积反向里的 `delta·col^T` | 1.66 ms | 1.53 ms | 1.09x（省掉 288x1024 的转置拷贝） |
| `conv2d_net_t::backward` 整层（-O2，与改动前同标志） | 19.35 ms | 16.5~17.1 ms | ~1.15x |

另外顺带修掉一个隐患：`mat_dot_t::clone()` 的快速路径现在会在**操作数层面**判断能否零拷贝，
所以「视图参与 dot」不再无声地退化成"每次调用都物化一次"。

### 单测

`tests/test_reshape_view.cpp`（9 例，`ReshapeView.*`）：

| 用例 | 钉住什么 |
|------|----------|
| `ReinterpretsShapeWithoutCopying` | (2x6)→(3x4) 的元素对应关系（按展平下标逐点验） |
| `WritesAliasBothWays` | 通过视图写 ⟹ 底层可见；写底层 ⟹ 视图可见 |
| `TransposeIsAlsoAView` | `t()` 的数值与写穿 |
| `SizeMismatchThrows` / `ColumnMajorBaseKeepsStorageOrder` | 元素总数校验；列优先底层按列优先展平 |
| `GemmDescriptorIsExposedForRowMajorStorage` | 描述符的 ptr/ld/transposed 正确（含子视图偏移、列优先给不出） |
| `TransposedOperandsMatchNaive` | 转置左/右/双侧操作数、子视图转置与朴素乘法逐点一致 |
| `ReshapeOperandMatchesNaive` | reshape 视图参与 dot；1xL 切成 (n,t) 再转置 = patchify 的窗口矩阵 |
| `ViewOperandIsNotModified` | 零拷贝只读：算完源矩阵原样不动 |

### 13.1 补记：把「按行做一维分解」这条路走通并量清楚

有人在设计讨论里提出过另一种卷积循环序：**按输入行扫描**，把一行用 reshape 视图变成 `(m, tc)` 的窗口
矩阵，再和卷积核的一行做点乘、累加到对应输出行。这在非重叠理想情形（`stride == kernel`、
`ir = n*tr`、`ic = m*tc`）下**确实可以零拷贝**，本轮把它实现出来对撞了：

```text
A  全量 im2col + 一次大 GEMM（现状）
B1 逐输入行 + reshape 窗口视图 + 逐核行 dot（即提问者原样，GEMV）
B2 逐输出行：只物化当前行需要的 im2col tile，立刻做一次 GEMM
```

| 形状（非重叠，无 padding） | A | B1 | B2 |
|---|---|---|---|
| 1 通道, 3x3, 输出 32x32 | **0.050 ms** | 0.065 ms | 0.066 ms |
| 1 通道, 3x3, 输出 128x128 | **0.823 ms** | 1.045 ms | 1.038 ms |
| 8 通道, 3x3, 输出 32x32 | **0.651 ms** | 4.227 ms | 1.829 ms |
| 16 通道, 3x3, 输出 64x64 | **4.661 ms** | 66.6 ms | 26.1 ms |

三者数值互相一致（maxdiff ~1e-16），但**A 全程最快**，通道数一上来差距就拉开（B1 慢 14 倍）。

原因不是内存访问，而是**算术强度**：

- B1 每个 `(输入行, 核行)` 做一次 `(m x tc)·(tc x 1)`，这是 **GEMV**：左操作数没有任何复用，
  每个元素只参与 `tc` 次乘加 ≈ 0.5 flop/byte，纯带宽受限；而且 `C_in x C_out` 一多就变成
  成千上万次微小 BLAS 调用（16 通道时每次输入行 256 次调用）。
- B2 把窗口物化成 tile 后仍然是**小 GEMM**（M=16、K=144、N=64），BLAS 调用开销和 tile 填充
  摊不薄，只有 A（M=C_out、N=全部输出位置、K=C_in·K² 的一次大 GEMM）能把寄存器复用吃满。
- 结论：**模板要一次用完所有输出通道/位置**（保持 GEMM 大），而不是按行拆成 GEMV/小 GEMM。
  零拷贝视图的价值在于"喂给那个大 GEMM 时不用复制"，而不是"把 GEMM 拆小"。

### 13.2 顺带补上：`mat_view_t::reshape_view` 与它的两个前提

「按输入行取窗口」需要先拿到一个**行视图**再 reshape，所以补了 `mat_view_t::reshape_view()`，
它带来两条必须显式处理的前提（都已写进代码注释与单测）：

1. **只有紧凑的视图才能线性展平**。`mat_view_t::densely_packed()` 的判据：行优先底层下
   "只有一行"（行内连续）或"横跨整行"（行间无空洞）；列优先对称；转置视图一律不算。
   不满足就抛 `std::invalid_argument` —— 跨步子视图展平出来的顺序不是内存顺序，会读错元素。
2. **不能从临时量取视图**。`x.view(...).reshape_view(...)` 里那个临时 `mat_view_t` 在整表达式
   结束时析构，而 reshape 视图持有它的引用 → 悬垂（写这个探针时真的踩到了 SIGFPE）。
   因此 `reshape_view` 只在左值上可用，右值重载被 `= delete`：

   ```bash
   echo 'auto v = mat_t<double>(2,6).reshape_view(3,4);' | \
     g++ -std=c++20 -fsyntax-only -I. -x c++ -
   # error: use of deleted function ... mat_t<double>::reshape_view(int, int) &&
   ```

   这与 TESTING.md 第 9 节的值类别契约是同一套思路：悬垂引用要在编译期挡住，而不是留给运行期。

## 14. 训练结果序列化与 MNIST 小玩具

### 14.1 序列化：格式本来就有，缺的是「按层命名」的黏合层

`jas_weight_io.hpp` 早就有对称的一对：

- `weight_file_t`：读取 `JASMINE_WEIGHTS_V1`（文本索引 + float32 数据区），`read_into` / `read_scalar`；
- `weight_writer_t`：写出同一个格式，`add(name, mat)` / `write(path)`。

之前它们只被用来**加载** GPT-2 / LLaMA 权重（以及 dump logits 给 Python 对拍），没有用来保存训练结果。
本轮补上三件小接口，让"保存训练结果"变成两行：

```cpp
weight_writer_t w;
add_layer_params(w, "conv1", net.conv1);      // -> conv1.weight / conv1.bias
add_layer_params(w, "fc1",   net.fc1);        // conv2d_net_t / weight_net_t 都适用
w.add_scalar("meta.epochs", 3);               // 训练元信息 = 1x1 tensor，同文件保存
w.write("model.jas");
```

```cpp
weight_file_t wf;  wf.load("model.jas");
read_layer_params(wf, "conv1", net.conv1);    // 形状不符会抛错，不会静默写坏权重
int epochs = static_cast<int>(wf.read_scalar<float>("meta.epochs"));
```

`add_layer_params` / `read_layer_params` 只依赖 `weight()` / `bias()` 的形状约定，所以
`conv2d_net_t`、`weight_net_t`、`output_proj_net_t` 通用。选择沿用既有格式（而不是新造 JSON /
safetensors）的理由：Python 侧的 `tools/jasmine_weights.py` 已经能解析它，训练结果可以直接和
导出脚本、golden 对拍脚本共用一套读取代码。

`tests/test_model_serialization.cpp`（`ModelSerialization.*`，2 例）钉住：

| 用例 | 钉住什么 |
|------|----------|
| `RoundTripPreservesForwardOutput` | 权重 + 元信息往返后前向输出一致（float32 存储 → 1e-6 容差）；文件头是 `JASMINE_WEIGHTS_V1` / `f32` / tensor 计数；缺 tensor 抛错 |
| `ShapeMismatchIsRejected` | 张量形状不匹配（如输出通道数变了）时读取抛 `std::runtime_error`，不静默截断 |

### 14.2 MNIST 小玩具：`examples/mnist_conv.cpp`

用本库的层直接拼一个小 CNN（**没有为它新加任何算子**）：

```text
输入 [1, 28*28]
  conv1 1→8  5x5 pad2  → [8,784]   → ReLU → maxpool 2x2 → [8,196]
  conv2 8→16 5x5 pad2  → [16,196]  → ReLU → maxpool 2x2 → [16,49]
  flatten（reshape_view，零拷贝）   → [784,1]
  fc1 784→128 → ReLU
  fc2 128→10  →  ce_loss_t（softmax + 交叉熵）
```

- mini-batch 用的是库里现成的 `cache_updator_t<val_type, adam_t>`：逐样本 `backward` 累加梯度，
  每 `batch` 个样本 `step()` 一次（这就是"梯度累加 = mini-batch"的既有机制，不是新写的）；
- flatten 用的是本轮新增的 `reshape_view`（零拷贝），反向时把 `[784,1]` 的梯度再 reshape 回 `[16,49]`；
- 数据优先读 `--data-dir` 下的未压缩 MNIST IDX 文件；找不到就退化成内置的合成数字图案，
  所以无网络/无数据时这个 demo 仍然能跑通整条链路（也方便当冒烟测试）。

实测（本机、参考 BLAS、单线程、`-O3 -march=native`，评估都在**完整 10000 张测试集**上）：

```text
$ ./build/examples/mnist_conv --data-dir build/mnist --epochs 3 --train-limit 6000 --batch 16 --lr 2e-3 --seed 1234
[epoch 1] train_loss=0.3472 train_acc=0.8917 test_acc=0.9350
[epoch 2] train_loss=0.1444 train_acc=0.9537 test_acc=0.9535
[epoch 3] train_loss=0.0987 train_acc=0.9695 test_acc=0.9615
[save] build/mnist/model.jas  (12 tensors: 4 层参数 + 4 元信息)
[check] OK: 保存/载入往返一致

$ ./build/examples/mnist_conv ... --seed 7          # 换种子重跑
[epoch 1] test_acc=0.9418    [epoch 2] test_acc=0.9684    [epoch 3] test_acc=0.9747
```

**为什么 6000 张 × 3 epoch 就有 96~97%**：MNIST 本身很容易，而这个网络只有约 10.5 万参数
（conv1 208 + conv2 3216 + fc1 100480 + fc2 1290），20 个 epoch 都远没到过拟合；3 个 epoch
已经足以把明显可分的手写数字学会。要更高的数字就把 `--train-limit` 调到 60000（约 15 分钟）。

**训练/测试是真分开的**：

- 数据来自官方 IDX 两个文件：`train-*-idx3/idx1-ubyte`（60000 张）与 `t10k-*`（10000 张），
  训练循环只访问 `train.images[order[k]]`，评估只访问 `test.images`，两条路径不交叉；
- 按**图像内容**（md5）比对两集：交集只有 1 张，是 MNIST 数据集自身的已知瑕疵，不足以解释精度；
- `--train-limit 6000` 不是文件前 6000 张，而是先用 `std::shuffle` 打乱全部 60000 的索引再取前 6000，
  所以子集是有代表性的（标签分布 ≈ 每类 600 张）；默认评估用全部 10000 张测试图。

### 14.3 顺带修掉的三处「表达式包住 dot」

本轮把上一节之外剩下的同类问题一起修了（都是同一个机制：`mat_add_t` / `mat_div_t` 的 `clone()`
逐元素求值，会绕过 GEMM 快速路径）：

| 位置 | 改法 | 实测 |
|------|------|------|
| `weight_net_t::forward`（全仓库线性层：QKV/FFN/lm_head） | `dot` 后 `+=` 偏置 | d_model=512,T=64：**41.1ms → 1.34ms（30.7×）** |
| `jas_mha_t.hpp` 三处注意力打分 `q.t().dot(k) / scale` | 先 `dot` 再单独缩放 | T=128,d=64：**3.91ms → 0.25ms（15.9×）** |
| `jas_mha_t.hpp` 两处反向 `delta_q/delta_k` 的 `/ scale` | 同上 | — |

副作用是**整条测试套件从 185s 掉到 75s**：GPT-2 / LLaMA 的逐层对齐测试原本大部分时间都花在这些
朴素 GEMM 上。数值上只差 BLAS 求和顺序（1e-15 量级），对齐测试的 1e-9 容差覆盖得住。

另外非重叠一维卷积（patchify）现在会走 `reshape_view` 零拷贝路径：判据是
`C_in == 1 && Kh == 1 && stride_h == 1 && pad == 0 && dilation == 1 && stride_w == Kw`
（单行时窗口是输入的前缀，多行时要求 `W == W_out*Kw` 整除对齐）。探针实测 1.06~1.22×，
`Conv2d.col_view_enabled()` 可以在测试里断言这条路径确实被启用。

### 相关文件（序列化 / MNIST）

| 文件 | 作用 |
|------|------|
| `jas_weight_io.hpp` | `weight_file_t` / `weight_writer_t` + `add_layer_params` / `read_layer_params` / `add_scalar` |
| `examples/mnist_conv.cpp` | MNIST IDX 读取（含合成数据退化）、CNN 训练/评估、保存/载入自检 |
| `tests/test_model_serialization.cpp` | `ModelSerialization.*`：往返一致、文件头、缺失/形状不符抛错 |
| `tests/test_conv.cpp` | `Conv2d.ZeroCopyColViewForPatchifyGeometry` / `OneDZeroCopyMatchesReference`：零拷贝路径的启用与数值 |

### 14.4 用静态层堆叠重写 MNIST 玩具，并对比两种结构

最初那版 MNIST 例子是把每层写成结构体成员、手工串 forward/backward（能跑，但没用到库的
「静态层堆叠」）。现在改成 `complex_net_builder_t` → `complex_net_t`：

```cpp
template <bool two_conv>
using mnist_net_t = std::conditional_t<two_conv,
    complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, upr_tpl>   // 0 conv
        ::push_back_staticnet<relu_net_t>              // 1
        ::push_back_staticnet<pool2d_net_t>            // 2
        ::push_back_updatable<conv2d_net_t, upr_tpl>   // 3 conv
        ::push_back_staticnet<relu_net_t>              // 4
        ::push_back_staticnet<pool2d_net_t>            // 5
        ::push_back_staticnet<flatten_net_t>           // 6
        ::push_back_updatable<weight_net_t, upr_tpl>   // 7 encoder
        ::push_back_staticnet<relu_net_t>              // 8
        ::push_back_updatable<weight_net_t, upr_tpl>   // 9 输出
        ::push_back_staticnet<ce_loss_t>               // 10 loss
        ::type,
    /* 单卷积版：同上但去掉 3~5 */ >;
```

训练代码对两种结构**完全共用**：

```cpp
const dmat logits = net.forward(x);     // 链式前向；末端 ce_loss_t 透传并缓存 logits
epoch_loss += net.back().loss(label);   // 末端损失层
net.backward(label);                    // 链式反向：net_backward 逆序回传
net.step();                             // 各层 updator（cache_updator_t）落地
```

链式反向的顺序我先单独验证过（`net_backward` 是折叠表达式，容易看错结合方向）：
`net.backward(delta)` 与「手工按逆序逐层 backward」逐位一致。

**为这条链新加的层：`flatten_net_t`**（`jas_net_t.hpp`）。语义就是 reshape，但它必须是**层**
才能参与堆叠：forward 缓存输入形状并把 `[C, W]` 按行优先展平成 `[C*W, 1]`，backward 再把
`[C*W, 1]` 的梯度还原成 `[C, W]`。与 relu/pool 一样是无参数静态层（`init_weight`/`step` 空实现、
`is_reinitable_net` 为假，不占 `reinit` 的容器槽位）。`tests/test_flatten.cpp` 6 例钉住展平顺序、
反向还原、懒初始化、概念判定，以及 conv→flatten→fc 链的数值与梯度形状。

**对比实验**（真实 MNIST，同一份数据/超参/种子，全量 10000 张测试集）：

```text
$ ./build/examples/mnist_conv --data-dir build/mnist --arch both --epochs 3 \
      --train-limit 6000 --batch 16 --lr 2e-3 --seed 1234

arch          params  epochs  train_loss   train_acc    test_acc   seconds
conv2         105194       3    0.098669      0.9695      0.9724   120.618
conv1         202330       3    0.119815    0.962667       0.964   113.324
test_acc 差值（conv1 - conv2）= -0.0084
```

读法：

- `conv2` = conv→relu→pool→conv→relu→pool→flatten→encoder→ce（两次下采样，特征 16×7×7=784）；
- `conv1` = conv→relu→pool→flatten→encoder→ce（一次下采样，特征 8×14×14=1568）；
- **conv2 用一半参数（10.5 万 vs 20.2 万）反而高 0.84 个点**：第二次卷积 + 下采样把「空间特征」
  换成了「更抽象、更少」的特征，比把宽特征图直接灌进全连接更划算；
- 两者训练时间接近（121s vs 113s）：conv1 的前向/反向更便宜，但它的第一层全连接大 4 倍
  （1568×128 = 20 万参数），正好抵掉；
- 两种结构的保存/载入自检都通过（`--save build/mnist/cmp` 会写成
  `cmp.conv2.jas` / `cmp.conv1.jas`，`--load` 同样按结构名找文件）。

顺带说明为什么这条链能直接用 `complex_net_t`：conv/pool/flatten 都是**无 `reinit`** 的层
（形状由 `set_param` 给），所以 `complex_net_t::reinit(container)` 只会作用到链里的两个
`weight_net_t`，容器给 `{特征数, 128, 10}` 就够；`set_updator` / `init_weight` / `step` 会
自动跳过静态层。

### 14.5 第三种结构：卷积 stem + **Transformer encoder**（`--arch trf`）

上面 14.4 里我把"encoder"理解成了 MLP encoder —— 那是错的：这里的 encoder 指的是 **Transformer
encoder**。修正后的结构（`--arch trf`）：

```text
conv 1→8  5x5 pad2 → ReLU → maxpool 2x2          → [8, 196]
conv 8→16 5x5 pad2 → ReLU → maxpool 2x2          → [16, 49]      ← 卷积 stem 与 cnn2 相同
patch embedding: fc 16→d_model（逐位置投影，一个空间位置 = 一个 token）→ [d_model, 49]
encoder_t：bidirectional Transformer encoder（MHA+LayerNorm+FFN+残差 × n_layers，RoPE 给顺序）
mean_pool：49 个 token 取平均 → [d_model, 1]
fc d_model→10 → ce
```

v1 版本只做了一次池化（196 个 token、单卷积 stem），第二轮做了三处修改，效果显著：

1. **token 数 196 → 49**（第二次 2x2 池化）：自注意力是 `O(T²·d)`，这一步把注意力开销降了 16 倍，
   单样本训练时间 30.7ms → 7.2ms（1000 样本 1 epoch：30.7s → 7.2s）；
2. **卷积 stem 补成两次卷积**，与 cnn2 完全相同的下采样路径（特征 16×7×7），
   这样比的是"后端用 CNN 还是 Transformer"，而不是"前端特征谁强"；
3. 训练更久（2000×2 → 3000×6 epoch）且学习率降到 1e-3（Transformer 对步长更敏感）。

用的都是库里现成的积木，**没有为 Transformer 变体新写任何算子**：

- `encoder_t`（`jas_transformer_kernel_t.hpp`）本身就是用 `residual_net_t` + `mat_mha_t` +
  `layer_norm_net_t` + `base_ffn_t` 堆出来的；接进链里是
  `push_back_impl<encoder_t<double, upr_tpl>>`，形状用
  `set_param(层数, 头数, d_model, d_ff, seq_len)`（seq_len = token 数）给；
- token 顺序信息用 `mat_mha_t::bind_rope(max_seq_len)` 挂 RoPE（从注册表按 d_head 取共享条目）；
- 分类头前新加 **`mean_pool_net_t`**（`jas_net_t.hpp`）：把 `[d_model, T]` 平均成 `[d_model, 1]`，
  backward 把梯度按 `1/T` 摊回每个 token。与 flatten/pool 一样是无参静态层
  （`tests/test_mean_pool.cpp` 5 例：逐行平均、梯度均摊、形状校验、概念判定、mean pool→linear 链）。

实测（真实 MNIST，同一份数据/超参/种子，2000 张训练 × 2 epoch，2000 张测试图）：

```text
# v1：196 token、单卷积 stem、2000 张 x 2 epoch
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       2    0.218465       0.936      0.9255    16.65
trf            17914       2     1.31306       0.514       0.559   104.41     ← 落后 36.7 个点

# v2：49 token、双卷积 stem、3000 张 x 6 epoch、lr 1e-3
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       6    0.104656    0.969333      0.9535   73.53
trf            21386       6     0.27073    0.921667       0.899  118.73     ← 落后 5.5 个点
```

读法与下一步：

- 三处修改让 Transformer 从 55.9% → **89.9%**（+34 个点），每 epoch 只慢 1.6 倍；
- 更关键的是**参数效率**：trf 用 21.4k 参数拿到 89.9%，cnn2 用 105.2k 拿到 95.35%——
  按参数算 Transformer 反而更省（它的 loss/acc 曲线还在爬，没到平台）；
- **等容量对比已经做进工具里**：`--arch match` 会先算出 trf 的参数量，再自动反解 cnn2 的隐层宽度
  （实测把 cnn2 裁到 21,719 参数 vs trf 21,386，相差 1.6%），两者跑同样的训练即可比"同容量谁更强"；
  也可以用 `--hidden N` 手动指定 CNN 宽度；
- 还想继续追平：dropout / 权重衰减、CLS token 取代 mean pool、更多 epoch（Transformer 缺卷积的
  局部性/平移等变先验，需要更多数据与步数）。

### 14.6 接上库里的学习率调度（余弦退火 + 热重启）与 dropout

**调度**：训练循环改用库自带的 `cosine_annealing_decay`（`jas_mat_utility.hpp`），按 **mini-batch 步**
推进（`--scheduler cosine`，默认）：

```cpp
const int steps_per_epoch = (n_train + batch - 1) / batch;
const int total_steps = steps_per_epoch * epochs;
cosine_annealing_decay sched(/*epoch_max=*/total_steps,
                             /*init_decay_steps=*/total_steps / 2,   // 中途恰好一次热重启
                             /*max_lr=*/lr, /*min_lr=*/lr * 0.05,
                             /*warmup_rate=*/0.1, /*T_multiplier=*/2.0);
...
sched.step();                       // 先推进再取，跳过预热里 lr=0 的第 0 步
net.set_lr(sched.get_lr());         // complex_net_t::set_lr 会遍历链上所有可更新层
net.step();
```

日志会打印当前 lr 与 cycle，可以直接看到 `cycle 0 → 1` 的热重启：cnn2 在重启前 95.55%，
重启后第 3 个 epoch 到 **97.05%**；trf 从 82.3% 一路走到 **91.65%**。

**dropout**：新增 `dropout_net_t`（`jas_net_t.hpp`，inverted dropout）：

- forward：每个元素以 `keep = 1-p` 保留，保留的乘 `1/keep`（输出期望不变，推理时恒等即可）；
- backward：`delta ⊙ mask`，forward 被丢掉的位置梯度为 0；
- 训练/推理开关是**显式**的 `set_enabled(bool)`，评估路径里关掉、训练里打开。
  之所以不做成"infer 时自动跳过"：那要求链上每层都有 `forward_one`，而库里的 `encoder_t` 没有，
  会限制它出现在哪些链里；
- 与 flatten/pool/mean_pool 一样是无参静态层（不占 `reinit` 槽位）。
- `tests/test_dropout.cpp`（7 例）：关闭/p=0 恒等、保留元素按 `1/(1-p)` 缩放、丢弃比例与均值
  （inverted dropout 的无偏性）、backward 用同一份 mask、非法 p 与形状报错、概念判定、net_type。

**两者一起的效果**（真实 MNIST，同一规模 3000 张 × 6 epoch，2000 张测试图）：

| 配置 | cnn2 | trf |
|------|------|-----|
| 固定 lr 1e-3，无 dropout | 0.9535 | 0.8990 |
| **cosine（含热重启）+ dropout 0.2** | **0.9705** | **0.9165** |

各涨约 1.7 个点，差距仍是 5.4 个点，但 trf 只用了 21.4k 参数（cnn2 是 105.2k）。

### 14.7 AdamW、CLS token，以及「同规模」对比

**AdamW**（`jas_updator_t.hpp`，照 `nadam_t` / `adam_t` 的结构写）：矩估计只吃真实梯度，权重衰减
**直接作用在参数上**：

```text
Adam : g ← ∇L + wd·θ，再进一/二阶矩      （衰减被矩估计归一化，退化成 L2 正则）
AdamW: θ ← θ - lr·( m̂/(√v̂+ε) + wd·θ )   （真正的"权重衰减"）
```

`tests/test_adamw.cpp` 4 例钉住：`wd = 0` 时与 `adam_t` **逐位一致**；梯度为 0 时衰减仍生效
（θ ← θ·(1-lr·wd) 的解析值）；衰减与矩估计解耦（单步解析值对拍，并与 L2 版本对比不同）；
`set` / `set_weight_decay` 接口。example 里加了 `--weight-decay`（默认 0.01）。

**CLS token**（`jas_net_t.hpp` 新增两层，ViT 风格）：

| 层 | forward | backward |
|----|---------|----------|
| `cls_token_net_t` | `[d, T] → [d, T+1]`，第 0 列是可学习向量 | 第 0 列作为该向量的梯度交给 updator，其余列原样回传 |
| `take_token_net_t` | `[d, T] → [d, 1]`，取第 `index` 列（默认 0） | 只有第 `index` 列拿到梯度，其余为 0 |

`cls_token_net_t` 是**可更新但不可 reinit**（用 `set_param(d_model)`，不占 `complex_net_t::reinit`
槽位）；`take_token_net_t` 是无参静态层。`tests/test_cls_token.cpp`（8 例）覆盖拼接/后移、
梯度分配与回传、越界报错、概念判定（CLS 可更新、take_token 静态）、net_type。

**同规模对比**：把 trf 放大到与 cnn2 同一量级（3 层、d_model=64、ff=128、CLS token）
→ **105,642 vs 105,194 参数（差 0.4%）**。数据 1500 张 × 4 epoch：

```text
arch          params  epochs  train_loss   train_acc    test_acc   seconds
cnn2          105194       4    0.254249    0.917333       0.926   25.88
trf           105642       4     1.12019    0.571333      0.6365  121.94
test_acc 差值（trf - cnn2）= -0.2895
```

**结论：不是容量不够，是数据不够。** trf 的 train_acc 只有 0.57、loss 仍高达 1.12——连训练集都
没拟合上（而上一轮 21k 参数的小 trf 在 3000 张 × 6 epoch 上能到 91.65%）。同一数据量下给
Transformer 加容量只会更难训；要给它公平机会需要 ≥20~30k 样本、10+ epoch（本机约 30~60 分钟），
以及 `--arch match` 提供的等容量对照。

## 15. RBM 与 DBN（`jas_rbm_t.hpp`）：两种训练语义 + 静态堆叠

### 15.1 RBM 的双重身份

`rbm_net_t` 同时扮演两个角色，这是理解它的关键：

| 角色 | 接口 | 用在哪 |
|------|------|--------|
| **层** | `forward` = σ(Wv+c)（隐层**概率**）、`backward` = 「线性+sigmoid」求导；只更新 W 与 c | DBN 的监督微调（整链反向） |
| **RBM 自己** | `contrastive_divergence(v0, k)` = CD-k；更新 W / b / c | 逐层贪心无监督预训练 |

形状：`W: [n_hidden, n_visible]`、`b: [n_visible, 1]`、`c: [n_hidden, 1]`；一次一个样本一列。

两处容易搞错、已经写进代码注释与测试的地方：

1. **CD 的负增量方向**：W 是 `[n_hidden, n_visible]`，而 CD 规则 ΔW = η(v0h0ᵀ - vkhkᵀ) 是
   `[n_visible, n_hidden]`；给 updator（做 `p ← p - lr·g`）的必须是 `hk·vkᵀ - h0·v0ᵀ`。
   我第一版写反了，而 `updator.update()` 只按 grad 的形状工作、会**静默把参数 resize 掉**，
   于是加了一道 `check_grad_shape` 护栏（梯度与参数形状不符直接抛错）。
2. **层语义下可见偏置 b 不更新**：`h = σ(Wv+c)` 根本不经过 b，所以 ∂L/∂b = 0；b 只属于生成方向
   P(v|h)，由 CD 负责。这也是 DBN 判别式微调的标准做法（微调阶段 RBM 就是一个 sigmoid 层）。

### 15.2 DBN：用静态堆叠把 RBM 串起来

```cpp
template <int n_rbm, template <typename> class updator_tpl>
using dbn_net_t = /* n_rbm 个 rbm_net_t → weight_net（分类头）→ ce_loss_t */;
dbn_net_t<2, upr_tpl> dbn;
dbn.reinit({784, 256, 128, 10});                    // 每个 RBM / 分类头各消费一对数
dbn_pretrain<2>(dbn, data, /*cd_k=*/1, /*epochs=*/1);  // 逐层贪心（第 i 层吃第 i-1 层的隐层概率）
dbn.forward(x); dbn.backward(label); dbn.step();      // 监督微调 = 整链反向
```

`jas_rbm_t.hpp` 里为此加了 `push_back_n_updatable`（把同一个「可更新层」重复压进 builder N 次），
所以 DBN 的每个 RBM 都是 `complex_net_t` 的正式成员，`forward/backward/step/init_weight/reinit/
set_updator/set_lr` 全部走链 —— 与库里其它网络完全一致的构建方式（以后 demo 与测试都按这个风格写）。

### 15.3 单测与 demo

`tests/test_rbm.cpp`（7 例）：

| 用例 | 钉住什么 |
|------|----------|
| `HiddenProbabilityIsSigmoidOfLinearTerm` | P(h|v)=σ(Wv+c) 与 P(v|h) 的对称形式（手算值） |
| `LayerBackwardMatchesNumericalGradient` | 层语义下 W/c/v 的梯度与数值梯度一致，b 的梯度为 0（∂L/∂b=0） |
| `ContrastiveDivergenceFollowsTheCdRule` | 确定性 CD-1 的参数增量 = ΔW/Δb/Δc 的解析值（逐项对拍） |
| `ReconstructionImprovesWithTraining` | 200 轮 CD-1 后重建误差显著下降（< 0.15） |
| `ConceptsAndReinit` | `is_updatable_net` / `is_reinitable_net`（带权重 → 参与容器协议） |
| `GreedyPretrainReducesReconstruction` | DBN 贪心预训练降低重建误差、堆叠前向形状、参数形状不被 updator 改掉 |
| `FullChainBackward` | **已知问题（已缩小到具体一步）**：整链反向（CE→分类头→RBM→RBM）单独编译通过，链接进完整 `unit_tests` 会抛 `mat_dot_t: inner dimensions do not match`。这里 `GTEST_SKIP` 并记录，不假装通过 |

`examples/mnist_dbn.cpp`：MNIST（784→256→128→10）贪心 CD-1 预训练 + 监督微调 + 序列化。
实测（2000 张训练、3 个预训练 epoch、3 个微调 epoch、1000 张测试）：

```text
[pretrain] 重建误差 0.380 -> 0.247
[finetune] epoch 3 loss=0.112 test_acc=0.68
```

**踩到的坑（值得记）**：微调的梯度是「按 batch 求和」而不是求平均，所以 batch=16 时
`--ft-lr` 实际被放大 16 倍——默认 1e-3 时 test_acc 只有 0.09（等于瞎猜，RBM 学到的特征被一步打飞），
降到 2e-4 后正常到 0.68。用 `cache_updator_t` 做 mini-batch 时都要注意这一点。


### 15.4 那个整链反向问题的排查记录（已缩小到一步，根因待查）

排查手段与结论：

1. **ASan 不复现**：`build-asan`（`-fsanitize=address -O1 -g`）里整链反向不抛错，改成断言失败（预训练不稳定），
   没有任何 use-after-free / stack-use-after-scope 报告 → 不是典型的访存越界。
2. **不是编译参数差异**：单独编译 `test_rbm.cpp` 时用与 CMake 完全相同的参数
   （`-O3 -DNDEBUG -fopenmp -DJASMINE_USE_BLAS -DJASMINE_USE_OPENMP`）能通过；`flags.make` 也确认
   大二进制同样是这套宏。
3. **不是两两组合**：`test_rbm.cpp` 分别与 `test_dropout.cpp` / `test_cls_token.cpp` / `test_adamw.cpp`
   一起链接都通过，需要更大的 TU 组合才复现。
4. **打点定位（在 Release 大二进制里逐层手工回放反向）**：
   ```
   [D] start      rbm0 W=(5,8) c=(5,1) | rbm1 W=(4,5) c=(4,1) | head W=(3,4)   ← 形状都对
   [D] ce.backward -> (3,3)                                                     ← CE 正常
   [L] head.forward  this=0x…e08 in=(4,3) -> m_input=(4,3)                       ← 前向缓存写了
   [L] head.backward this=0x…e08 delta=(3,3) m_input=(0,0) m_weight=(3,4)        ← 同一对象，缓存空了
   ```
   即：**分类头 `weight_net_t` 的前向缓存 `m_input` 在 forward 时是 (4,3)，到 backward 时变成 (0,0)**
   （`this` 相同 → 不是对象副本的问题），于是 `delta.dot(m_input.t())` 的维度检查失败。
   问题不在 RBM，而在「前向缓存被清空」这一步。

顺着这个结论做的两件事：

- **加了明确的护栏**（永久保留）：`weight_net_t::backward` 现在先检查 `m_input` 是否有效、
  delta 形状是否匹配，直接抛
  `weight_net_t::backward: forward cache is empty (forward must be called before backward, ...)`，
  而不是留下难懂的 `mat_dot_t: inner dimensions do not match`。
- **待查方向**（供后续接手）：
  (a) 怀疑与 `detail::store_for_backward` 的移动分支有关——链上前一层的返回值是临时量，
  `dst = std::move(src)` 会把源清空；若某处**同时**引用/移动了同一个临时矩阵，
  就可能出现"写进去又被清空"。RBM 的 `forward` 返回成员 `m_hidden` 的拷贝，值得再核对一遍
  `complex_net_t::forward` 的折叠展开与各层返回值的生命周期；
  (b) 在**完整二进制 + `-O2 -fsanitize=address`**（保留优化，让布局接近 Release）复跑，
  ASan 的 `-fsanitize=address` 在 -O2 下仍能报 use-after-scope；
  (c) 二分更大的 TU 组合（例如 `test_rbm.cpp` + 所有新增测试 + `test_net.cpp`），
  找出触发条件最小的集合。


### 15.5 第二轮排查：从「缓存被清空」收窄到「mat_t 头被写坏」

接着 15.4 继续，用「加探针 + 在 test 侧分点观察」的方式把范围压到一句话能说清：

1. **不是对象副本**：在 test 侧打印 `&dbn` / `&dbn.get<2>()`，与库内 `weight_net_t::forward` /
   `backward` 打印的 `this` 对比 —— **三处地址完全一致**（同一对象、同一成员）。
2. **不是 forward 被调用多次**：给 `weight_net_t::forward` 加计数器，整个用例里
   `head.forward` 只被调用 **1 次**（`#1 in=(4,3)`），且 store 之后 `m_input=(4,3) valid=1`。
3. **清空发生在 `dbn.backward(...)` 内部**：在 test 侧分点观察（`dbg_input_valid()` 临时探针）——
   ```
   [W] after chain forward      head cache valid=1 (4,3)
   [W] after ExpectShape(logits) head cache valid=1 (4,3)
   [W] after ExpectShape(head)   head cache valid=1 (4,3)
   [W] before ce.backward        head cache valid=1 (4,3)
   [A] ce.backward -> (3,3)
   [W] after  ce.backward        head cache valid=1 (4,3)   ← 手工单独调 CE 反向不会清它
   ... 随后 dbn.backward(labels)
   [L] head.backward this=<同一地址> delta=(3,3) m_input=(0,0) valid=0   ← 进链反向之后才空
   ```
   即：手工调用 `ce_loss_t::backward` **不会**动分类头的缓存，但同一条链上的 `dbn.backward` 会在
   调 `head.backward` 之前把它变成 `(0,0)`（连 `m_data` 都没了 → `valid()==false`）。
4. **`m_input` 的地址没变**（同一个 `mat_t` 对象），变化的只是它的**内容（维度 + 数据指针）** ——
   也就是说 `mat_t` 的**头部被写了**，而不是"换了个对象"。

由此得到的结论：**这属于内存破坏类问题（把 `mat_t` 的维度/指针字段写坏了），不是逻辑分支错误。**
ASan（`-O1`）没有报错并不矛盾：`-fsanitize=address` **不检测对象内部/相邻栈变量之间的越界写**，
而 `m_input` 正是嵌在 tuple 里的栈对象。

**下一步（按性价比排序）**：
1. **在 `weight_net_t` 里给 `m_input` 旁边加一个 canary 成员**（`std::uint64_t m_canary = 0xDEADBEEF`），
   在 forward/backward 各检查一次 —— 如果 canary 被打掉，就直接证明是相邻越界写，并能用二分法定位是
   哪一层/哪个更新的写入越界；
2. 用 **`-O2 -fsanitize=address`** 重编**完整** `unit_tests`（保留优化，布局接近 Release），
   并把 `Dbn.FullChainBackward` 临时取消 skip 触发一次；
3. 换 `valgrind --tool=memcheck`（能抓栈上相邻写）跑那一个用例 —— 本机没装也能用 `-static` 版试；
4. 若都无果：把 DBN 的监督微调从"整链 backward"改成"逐层显式 backward"（demo 里就是这么驱动也正常），
   绕过这条路径并把问题记录在案。

**目前状态**：护栏（`weight_net_t::backward` 检查前向缓存 + delta 形状）已保留，报错信息明确；
`Dbn.FullChainBackward` 仍以 `GTEST_SKIP` + 精确症状记录；`examples/mnist_dbn.cpp` 在独立二进制里
端到端正常（预训练 + 微调到 68%）。


### 15.6 第三轮排查：更正结论 —— 是「优化级别相关的 UB」，不是陈旧构建目录

这一轮先得出了一个**错误**结论（"`build/` 陈旧、全新目录就正常"），随后被自己的对照实验推翻，记录如下
以免后人重走：

**推翻过程**：新建 `build-check` 后 RBM/DBN 全过（含整链反向），我一度以为是陈旧 `.o`。但对比
`flags.make` 发现：

```
build/        CXX_FLAGS = -O3 -DNDEBUG -std=gnu++20 … -fopenmp      ← 失败
build-check/  CXX_FLAGS =            -std=gnu++20 … -fopenmp       ← 通过（未指定构建类型 = 无优化）
```

即"全新目录"同时把**优化关掉了**，两个变量被混在一起 → 结论无效。另外 `build-check` 还因为缺
`-DJASMINE_USE_BLAS` 走了分块回退路径，数值容差类用例成片失败，说明这个对照本身也不干净。

**真正稳定的现象**：

| 条件 | 结果 |
|------|------|
| 单独编译 `test_rbm.cpp`，`-O3 -DNDEBUG` | 通过 |
| 整包 `unit_tests`，无优化 | 通过 |
| 整包 `unit_tests`，`-O3 -DNDEBUG`（Release，本仓库默认） | **失败** |
| 整包 + `-O1 -fsanitize=address` | 不复现（ASan 无报告） |
| 整包 + `-O3 -DNDEBUG -fno-strict-aliasing` | **仍失败**（排除严格别名） |
| 两两 TU 组合（rbm+dropout / +cls / +adamw） | 通过 |

**已确认的失败细节**（Release 大二进制，用临时 canary + 探针，均已移除）：

- 分类头**同一个对象、同一个 `this`**，`forward` 只被调用 1 次，写入后 `m_input=(4,3) valid=1`；
- 在 test 侧分点观察，直到 `dbn.backward(...)` 之前缓存都有效；**手工**单独调 `ce_loss_t::backward`
  也不会动它；
- 但进入 `dbn.backward(...)` 之后、`head.backward` 执行时，`m_input` 已是 `(0,0) valid=0`；
- 紧贴 `m_input` 的 canary 被写成 `0x00007fff....0001`（像半个指针），提示**有 32 字节的 `mat_t`
  头被写到了 `m_input` 附近**；
- 更怪的是两次打印里 `this` 相同、`&m_input` 却差了 `0x10`（同一对象不该如此）——这一点指向
  **inline 模板函数在不同 TU 里拿到了不一致的类布局/代码**（ODR 或 codegen 层面的问题），
  但我们已确认所有 TU 用的是同一套宏（`-DJASMINE_USE_BLAS -DJASMINE_USE_OPENMP`），所以还没锁定。

**下一步（按性价比）**：

1. **二分内联**：在 Release 全量二进制上加 `-fno-inline` / 只对 `jas_net_t.hpp` 相关的 TU 加，
   看是否与内联决策相关（若是，问题多半在 CRTP 表达式模板的 `reinterpret_cast<derived_type*>(this)`）；
2. **在新目录里同时保留 `-O3` 与 canary**（这次别再顺手关优化），并用 canary 二分"在哪一步被打掉"：
   在 `complex_net_t::backward` 的折叠里逐层插检查；
3. 逐一排除可疑写入者：把 CE 的 `backward` 换成"手工算 softmax+CE 梯度"的等价实现，看是否还复现
   （怀疑点集中在 CE 反向里那几个临时 `mat_t`）；
4. 兜底方案（可立即采用）：DBN 的监督微调改成**逐层显式 backward**（`examples/mnist_dbn.cpp` 那种
   驱动方式在 Release 下也正常），绕开 `complex_net_t::backward` 这条路径，并保留问题记录。

### 15.7 结案：根因是「匿名命名空间里同名别名模板」导致的 ODR/COMDAT 冲突

**问题解决。** 15.4–15.6 三轮排查的最后一步是这样定案的：

**决定性证据**：在 `weight_net_t::forward` 与 `::backward` 里各打印一次 `sizeof(*this)`：

```text
[SZ] forward:  this=0x…f38 &m_input=0x…0d8 sizeof(*this)=448 type=weight_net_t<mat_t<double>, _GLOBAL__N_1::upr_tpl>
[SZ] backward: this=0x…f38 &m_input=0x…0c8 sizeof(*this)=432 type=weight_net_t<mat_t<double>, _GLOBAL__N_1::upr_tpl>
```

**同一个名字、同一个 `this`，却有两套布局**（差 16 字节 = 两个 updator × 8 字节，正是 `adamw_t` 的
`m_weight_decay`）。这不是越界写，而是 **ODR 违规**：

- 多个测试/示例文件都在**匿名命名空间**里写了同名的别名模板
  `template <typename T> using upr_tpl = cache_updator_t<T, ???>;`，
  **有的用 `adamw_t`、有的用 `sgd_t`/`nadam_t`**（5 个文件、2 种不同目标）；
- GCC 对匿名命名空间的 mangle 一律是 `_GLOBAL__N_1`，**两个 TU 里的 `upr_tpl` 因此 mangle 成同一个符号**，
  于是 `weight_net_t<mat_t<double>, upr_tpl>` 在链接期被视为**同一个实例**；
- 这个类模板的成员函数是内联（vague linkage/COMDAT），链接器只保留**其中一份**，
  而对象却是由各自的头文件版本构造出来的 → **一份代码 + 另一份布局** → 成员偏移错位
  （`m_input` 偏了 0x10）→ 前向缓存读到别处、canary 被覆盖、`delta.dot(m_input.t())` 报维度不符。

**验证与修复**：把 `test_rbm.cpp` 里的别名改成唯一名字后，两处 `sizeof` 立刻一致（都是 448）、
`&m_input` 同一地址、整链反向通过。随后把 tests/ 与 examples/ 里所有同类匿名命名空间别名按用途
重命名（`adamw_upr_tpl` / `test_flatten_upr_tpl` / …），并清掉全部临时探针。

**规则（写给以后的自己）**：**不要在不同 TU 的匿名命名空间里用同一个名字命名别名模板**，
尤其是它们的目标类型不同的时候——mangle 会撞车而产生 ODR 违规。按「目标 updator」命名
（`adamw_upr_tpl`、`sgd_upr_tpl`），或者把它放进具名 struct/namespace。

**这一类 bug 的特征**（值得记住）：
- 单独编译某个 TU 正常，链接进大二进制才炸；
- 与优化级别、链接顺序相关，换目录/换构建类型表现不同（本次就是这样反复误导排查方向）；
- **ASan 抓不到**（不是越界写，而是"两份合法代码被错误合并"）；
- `-fno-strict-aliasing` 无效；
- 症状可以是"数据看起来被写坏了"（canary 被覆盖），容易被误判成越界。

**收尾时保留的改动**：`weight_net_t::backward` 的护栏（前向缓存有效 + delta 形状检查，
给出明确错误信息）——正是它把这条 ODR bug 从"难懂的 dot 维度报错"变成可读信息的开端。

**结果**：`ctest` **258/258 通过，0 失败，无 skip**；`Dbn.FullChainBackward` 不再是 skip 用例。
