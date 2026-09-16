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
softmax、归一化层，`tests/test_cuda_dot.cu` 覆盖 `dot` 分派与注意力端到端
（编译期断言 + 与主机逐元素对拍）。细节见 `CUDA.md`。

### 相关文件

| 文件 | 作用 |
|------|------|
| `jas_mat_express_t.hpp` | `storage_of` / `storage_type` 存储策略；`scalar_leaf_t`；`is_self_contained_v`；`is_mat_dot`；各表达式节点与运算符；`mat_dot_t` |
| `jas_mat_concepts.hpp` | `is_caculable`（判标量前 `remove_cvref`） |
| `jas_mat_t.hpp` / `jas_mat_view_t.hpp` | `.dot()` 的 ref-qualified 声明；访问器的 `JAS_HD` 标注 |
| `jas_cuda_compat.hpp` | `JAS_HD` / `JAS_DEV`、设备安全数学、`device_evaluable` 探测 |
| `jas_cuda_gemm.hpp` | cuBLAS GEMM、`matmul` 三入口、`.dot()` 的定义 |
| `jas_cuda_reduce.hpp` | 广播叶子、`dev_colvec_t`/`dev_rowvec_t`、归约 kernel、`softmax_rows` / `layer_norm` / `rms_norm` |
| `tests/test_expression_lifetime.cpp` | `ExpressionLifetime.*`：值类别契约 + 生命周期回归 |
| `tests/test_cuda_fused.cu` | `CudaEnvironment.*` / `CudaDeviceTest.*`：设备契约 + 融合/GEMM 对拍 |
| `tests/test_cuda_reduce.cu` | `CudaReduceContract.*` / `CudaReduceTest.*`：归约、softmax、归一化、注意力端到端 |
| `tests/test_cuda_dot.cu` | `CudaDotContract.*` / `CudaDotTest.*`：`dot` 分派、转置组合、链式与表达式操作数 |
