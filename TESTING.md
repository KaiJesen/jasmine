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
| 扩容换存储 | 旧存储**不析构**，进 `m_retired` 退休（读者手里可能还攥着它的视图）。扩容按「至少 1.5 倍」（`EXPAND_SIZE` 保底）增长，所以退休总量收敛：在职 + 退休 ≤ 4 倍最终容量，不随扩容次数线性累积 |
| 误用面 | 删掉 `mat_cache_t` 上「读一次顺手扩容+改记账」的惰性 `operator()` / `range`，只留 `grow_to`（容量）+ `cell`（写）+ `raw_view`（共享读，不改任何状态） |

副作用与边界：**dynamic 模式现在也可以多线程共享**（原先头文件注释写的是「不安全」）；
代价是扩容过的实例会多留一份旧存储。`static_fixed`（先 `reserve` 一次填满、运行期不写）不受影响，
它的语义也依旧更严：越界直接抛 `std::out_of_range`，绝不在运行期扩容。

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
修后：exit 0，0 处告警，4 例全过
```

### 相关文件

| 文件 | 作用 |
|------|------|
| `jas_RoPE_t.hpp` | `mat_cache_t`（容量 / `cell` / `raw_view` / 退休存储）+ `mat_RoPE_t`（`fill_from_origin` / `range` / `reserve`） |
| `tests/test_rope_cache_race.cpp` | `RoPeCacheThreading.*`：填充不变量 + 并发回归 + TSan 复现入口 |
| `jas_mha_t.hpp` | 各头并发调用 `m_rope->forward_at()`，缓存不变量在这里被真正用上 |
