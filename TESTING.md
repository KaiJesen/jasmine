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
