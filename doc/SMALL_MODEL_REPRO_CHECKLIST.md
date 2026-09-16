# 复现小模型 Checklist（Decoder-only LM）

目标：用 jasmine 复现 / 对齐一类 **小规模开源因果语言模型**（如 TinyLlama、SmolLM、GPT-2 small 量级），而不是完整 DeepSeek。  
前提：任务是 **next-token 语言建模**，拓扑为 **decoder-only**（无 encoder / cross-attn）。

状态约定：`[ ]` 未做 · `[~]` 进行中 · `[x]` 完成 · `[—]` 非本阶段必须

与优化清单分工：算子/性能见 [`OPTIMIZATION_CHECKLIST.md`](OPTIMIZATION_CHECKLIST.md)；本文件管 **「能训出/加载一个可用的小 LM」**。

---

## 0 — 范围与成功标准

- [x] **选定参考模型**
  - **GPT-2 系**（`distilgpt2` 6 层 → `gpt2` 12 层两条都验证过）
    - 词表 50257、`d_model=768`、`n_heads=12`、`d_ff=3072`、`n_pos=1024`
    - 绝对位置（`wpe`）、Post-norm LN + 末尾 `ln_f`、`gelu_new`、无 GQA
  - **LLaMA 系**（`TinyLlama-1.1B-Chat-v1.0`）—— 与 GPT-2 的差异正好覆盖另一套配方
    - 词表 32000（SentencePiece）、`d_model=2048`、`n_heads=32`、`n_kv_heads=4`（GQA）、
      `d_ff=5632`、`n_layers=22`、`n_pos=2048`
    - **RoPE**（无 `wpe`）、**Pre-norm RMSNorm**、**SwiGLU** FFN、**全连接层无 bias**、`lm_head` 不与 embedding 绑定
- [x] **成功标准（至少一条）**
  - **从零训**：固定小语料上 loss 稳定下降，生成可读（可过拟合验证）—— 见 §5，未做
  - **加载权重**：与参考实现同输入 id 时，logits / 隐层在约定容差内（黄金对齐）—— **已完成**
  - **不要求**：完整复现商业大模型词表与万亿 token 预训练

---

## 1 — 拓扑（相对 enc–dec 玩具）

- [x] **`decoder_only_t` / `decoder_only_base_t`**
  - 因果 MHA + FFN，无 MHCA；`forward` / `forward_one` + KV cache
  - 文件：`jas_transformer_kernel_t.hpp`、`jas_transformer_t.hpp`
  - 验收：`TransformerKernel.DecoderOnly*`（loss 下降、因果首位置稳定、`forward_one`≡全量）
- [ ] **Decoder-only CE demo**
  - 单路：`ids → embedding → decoder_only → lm_head → ce_loss`
  - 标签右移；可选 `position_mask` 屏蔽 prompt
  - 建议：`examples/decoder_only_ce_demo.hpp` + 对应 binary（对标现有 `transformer_ce_demo`）
- [x] **（对齐常用 LLM）Pre-norm**
  - 原有 Post-norm 组件保持不变；GPT-2 的 pre-norm 用独立类型别名拼出
  - `gpt2_attn_branch_t` / `gpt2_ffn_branch_t` = `LayerNorm → SubLayer`，外套 `residual_net_t`
  - 栈顶另有 `ln_f`（`gpt2_model_t::head`）；实现见 `jas_gpt2_t.hpp`
  - 验收：`Gpt2Structure.LayerNormAffineLoadableBeforeForward`、
    `Gpt2AlignmentTest.HiddenStatesMatchLayerByLayer`
- [x] **GELU FFN**（`gelu_new`，tanh 近似）
  - `jas_gelu_t.hpp` 的 `gelu_net_t`，接入 pre-norm FFN 分支
  - 注意：必须用 tanh 近似版，精确 erf 版与 GPT-2 对不上
  - 验收：`Gelu.*`（含数值梯度自检）
- [x] **RMSNorm**
  - `jas_net_t.hpp` 的 `rms_norm_net_t`，紧邻 `layer_norm_net_t` 便于对照
  - 与 LayerNorm 的实质差别只有三处：**不减均值**、**没有 beta**、eps 加在均方值上
  - eps 可通过 `set_param(d_model, eps)` 配置（对齐开源权重时要读模型 config：LLaMA 系 1e-5 / 1e-6 都有）
  - 反向 = LayerNorm 的反向**删掉 `mean(g)` 那一项**，σ 换成 rms
  - 验收：`RmsNorm.*`（13 例，含数值梯度 + PyTorch float64 基准 + "均值项必须删对"的反向断言）
- [x] **SwiGLU FFN**
  - [x] 门控容器 `gated_net_t<gate, up>`：两分支共享输入、**逐元素乘**汇合
  - [x] `silu_net_t`（`x⊙σ(x)`，含解析梯度）
  - [x] `gated_ffn_branches_t<val_type, updator, act>`：换激活即得 GEGLU / ReGLU
  - [x] 接入 LLaMA 系模型；完整 FFN = `gated(gate, up) → down_proj`（`jas_llama_t.hpp`）
  - 验收：`Gated.*`（11 例，含数值梯度 + PyTorch 端到端对撞）、`tools/verify_swiglu.py`、
    `LlamaAlignmentTest.RawHiddenStatesMatchLayerByLayer`（接进模型后逐层对黄金值）
- [x] **GQA / MQA**
  - 实现方式：**参数化 `mat_mha_t`**，新增尾置参数 `n_kv_heads`（默认 0 ⇒ 等于 `num_heads`）
    - `n_kv_heads == n_heads` → 经典 MHA（与旧实现逐位一致，现有 119 例全绿）
    - `1 < n_kv_heads < n_heads` → GQA；`n_kv_heads == 1` → MQA（白送）
  - 与 MHA 的差异只有三处：K/V 投影输出宽度变成 `n_kv_heads*d_head`、KV cache 只存 `n_kv_heads` 份、反向时共享 KV 头的多个 Q 头梯度**累加**
  - 注意力核 `mat_head_gen_t` **零改动** —— 它本来就只认切好的 Q/K/V
  - `forward/forward_one/backward/forward_at`（cross-attn）全部支持
  - 验收：`Gqa.*`（11 例）；关键两例已做**变异测试**验证有效
    - `Gqa.KvCacheGetsOneAppendPerKvHead`：naive 复用 `forward_one_at` 会让 cache 长度变成 `T*group_size`
    - `Gqa.BackwardSumsKvGradientsAcrossGroup`：错用 `assign` 会只剩下组内最后一个 Q 头的贡献
  - 权重加载：`num_kv_heads()/group_size()/d_head()/d_kv()` 供切分 fused QKV；LLaMA 的 `k_proj/v_proj` 直接是 `n_kv_heads*d_head` 行，无需复制

---

## 2 — Tokenizer 与词表

Tokenizer **不在** embedding 内：文本 → **离散 id**；embedding 只做 **id → 向量**。

- [ ] **确定分词方案**
  - 自训玩具：字符级 / 小 BPE 即可
  - 对齐开源权重：**必须用该模型自带 tokenizer**（`tokenizer.json` / SPM 等），不可混用
- [ ] **词表与特殊符号**
  - `vocab_size == embedding` 列数 / `lm_head` 输出维
  - 明确 `bos` / `eos` / `pad` / `unk`（及 chat 模板若有）
- [ ] **编解码闭环**
  - `encode(text) → ids → model → next_id → decode` 往返可测
- [—] **库内 BPE 实现**
  - 可继续用 HuggingFace / SentencePiece 在库外；jasmine 只吃 `1×T` id

**BPE 词表从哪来（概念）**：在大规模语料上做预分词 + 高频字节/字符对合并，得到固定大小词表；推理按同一规则切分查 id。开源模型（如 DeepSeek BBPE≈128K）随仓库发布词表文件，合并规则与预训练语料配方通常不完整公开。

---

## 3 — 嵌入、头与损失

- [x] **`embedding_net_t` + `output_proj` + `ce_loss_t`**
  - 含 `ignore_index` / `position_mask`
  - 验收：`tests/test_embedding_ce.cpp`
- [x] **权重绑定（可选）**
  - `lm_head.W` 与 `embedding.E` 转置共享（GPT-2 采用）
  - `gpt2_model_t::tie_word_embeddings()`：`lm_head.W = wte.W^T`，且 **bias 显式置 0**
  - 验收：`Gpt2Structure.TiedWordEmbeddings`
- [ ] **Next-token 数据布局**
  - 输入 `t[0..T-2]`，标签 `t[1..T-1]`；pad / prompt 不计入 CE
- [ ] **（可选）对话模板**
  - 用户/助手标记拼进同一条因果序列；loss 常只算助手段

---

## 4 — 位置编码与注意力契约

- [x] **RoPE 在 Q/K 上**（`rope_registry` / `mat_head_gen_t`）
- [x] **因果 mask + 黄金测试**
- [x] **RoPE 开关**（绝对位置模型用）
  - `mat_mha_t::set_use_rope(false)`：`bind_rope()` 变为对所有头 `set_rope(nullptr)`
  - GPT-2 组装时统一关闭；默认仍为 `true`，不影响既有 RoPE 模型
  - 验收：`Gpt2Structure.RopeCanBeDisabled`
- [x] **与参考模型位置约定一致**
  - RoPE 侧：base `θ`、是否 NTK/YaRN、最大长度需与参考一致
  - **RoPE 侧：特征配对约定**（`rope_pair_layout`，见下方「踩过的坑」里最贵的那个）
    - `interleaved`：第 i 个角作用在 `(2i, 2i+1)`（原论文 / GPT-NeoX；jasmine 原生默认）
    - `half_split`：第 i 个角作用在 `(i, i+d/2)`（GPT-J / **HF LLaMA**，`rotate_half`）
    - 两种约定在 **位置 0 输出完全相同**（旋转退化为恒等），只有多 token 序列才暴露差异
    - LLaMA 系一律 `half_split`：`mat_mha_t::set_rope_pair_layout()`，`jas_llama_t.hpp` 中已设
    - 验收：`RoPE.HalfSplitPairsOffsetFeatures`、`RoPE.LayoutsAgreeAtPositionZeroOnly`、
      `RoPE.BackwardIsAdjointForBothLayouts`、`RoPE.RegistrySeparatesPairLayouts`
  - **绝对位置侧（GPT-2）**：`wte(ids) + wpe(pos)`，pos 为 0..T-1；`forward_one` 用显式 `pos` 取 `wpe`
  - 验收：`Gpt2Structure.AbsolutePositionChangesOutput` + `Gpt2AlignmentTest.GoldenLogitsMatch`

---

## 5 — 训练配方（从零复现）

- [ ] **语料管线**
  - 明文 → tokenizer → 定长/packing 的 id 流；记录 `tokens/sec`
- [ ] **优化器与调度**
  - 现状有 Nadam/Adam、`cosine_annealing_decay`；对齐参考再定 β、wd、warmup
- [ ] **过拟合单 batch**
  - 几条短句反复训到近零 CE → 证明数据流与反传通
- [ ] **小规模真实训**
  - 固定 seed；记 `train_loss` / 抽样生成；避免只报「能跑」
- [—] **Dropout / 权重衰减**
  - 见优化清单 P3 可选项

---

## 6 — 权重加载（对齐开源 checkpoint）

从零训可不做；**数值复现已有权重**则必须：

- [x] **参数名与 shape 对照表**
  - HF 名 → jasmine 名的完整映射见 `jas_weight_io.hpp` 的 `load_gpt2` / `load_llama` 注释
  - GPT-2：`transformer.wte`→`wte.weight`（转置）、`h.{i}.attn.c_attn`→`attn.{q,k,v}.{weight,bias}`（拆 fused）
  - GPT-2：`h.{i}.attn.c_proj`→`attn.out`、`h.{i}.mlp.c_fc/c_proj`→`mlp.fc/proj`、`ln_1/ln_2/ln_f`
  - LLaMA：`model.embed_tokens`→`wte`、`model.layers.{i}.self_attn.{q,o}_proj`→`attn.{q,out}`、
    `k_proj/v_proj`→`attn.{k,v}`（GQA 下本就是 `n_kv_heads*d_head` 行，**不复制**）、
    `mlp.{gate,up,down}_proj`→`mlp_{gate,up,down}`、`input_layernorm/post_attention_layernorm`→`ln_1/ln_2`、
    `model.norm`→`ln_f`、`lm_head`→`lm_head`
- [x] **布局约定**
  - GPT-2 Conv1D `[in, out]` → 转置为 `[out, in]`（jasmine `weight_net_t` 是 `y = Wx`）
  - LayerNorm / RMSNorm 1-D `[d]` → 列向量 `[d, 1]`；`wte [V, d]` → `[d, V]`
  - QKV fused 拆分：转置后按行切 `[0:d]=q, [d:2d]=k, [2d:3d]=v`
  - **LLaMA 无 bias**：`nn.Linear` 默认无 bias，导出不产 bias tensor；
    `llama_model_t::init_weight` 里 `zero_all_biases()` 兜底（否则 `reinit` 会重新随机出 bias）
  - **LLaMA `lm_head` 不绑定**：TinyLlama 是 untied；三处偏差（gate/up/down、q/k/v/o）都要按 HF 的 `[out, in]` 直接存
- [x] **加载器**
  - `jas_weight_io.hpp`：单文件「文本索引 + 二进制 float32」，`read_into` 校验 shape 后写入
  - 导出脚本 `tools/export_gpt2.py` / `tools/export_llama.py` 负责所有布局换算（C++ 侧不做转置）
  - 大模型（TinyLlama 4.2 GiB）加载后须 `release()` 释放原始 blob，否则 float64 模型 + blob 双份常驻
  - 验收：`Gpt2WeightIo.*`、`LlamaLoad.*`（含 shape 不符 / 缺 tensor / magic 错误的拒绝路径）
- [x] **前向黄金对齐**
  - 导出时同时 dump HF 的 `golden.hidden.{i}`（逐层）与 `golden.logits`
  - 验收：`Gpt2AlignmentTest.HiddenStatesMatchLayerByLayer`（逐层定位）、
    `GoldenLogitsMatch`（含 argmax 序列一致）、`KVCacheDecodeMatchesGolden`
  - 验收（LLaMA）：`LlamaAlignmentTest.{RawHiddenStatesMatchLayerByLayer, RawGoldenLogitsMatch,
    KVCacheDecodeMatchesGolden, PrefillMatchesGoldenLastPosition, ChatTemplateLogitsMatch}`
  - 实测 distilgpt2：逐层 max_abs_diff ≤ 6e-4，greedy 生成 26 token 与 HF 完全一致
  - 实测 TinyLlama-1.1B（double 对 float32 参考）：23 个阶段（embed + 22 层）逐层与 logits
    全位置 `max_abs_diff ≤ 3e-5`，即 float32 参考值本身的舍入噪声量级；top-5 逐位相同
- [x] **（可选）反传不对齐也可**
  - 推理复现只需 forward；训练复现再对梯度（`gelu_net_t::backward` 已补，GPT-2 模型类暂无）

说明：仅有 `decoder_only_t` **不能**直接加载任意开源权重；缺 tokenizer、缺加载与结构差异（Pre-norm / SwiGLU / GQA）时，只能「同思路小模型」，不是「同一 checkpoint」。

---

## 7 — 推理

- [x] **KV cache + `forward_one`**（decoder-only / MHA）
- [x] **生成循环**
  - `gpt2_generate()`（`examples/gpt2_generate.hpp`）：prefill → 逐步 greedy / top-k 采样 → `eos` 停止
  - 位置用绝对下标索引 `wpe`；`reserve_kv_cache(n_pos)` 固定容量避免中途扩容
  - CLI：`examples/gpt2_generate.cpp`（`--greedy/--sample --temperature --top-k --out-ids`）
- [x] **与无 cache 全量 forward 一致性**
  - `Gpt2Structure.PrefillMatchesForward`、`ForwardOneMatchesFullForward`
  - 真实权重：`Gpt2AlignmentTest.KVCacheDecodeMatchesGolden`（每步都对黄金 logits）
- [x] **cli 闭环**
  - `tools/gpt2_tokenize.py encode/decode` 负责文本 ↔ id，jasmine 只吃 `1×T` id
- [x] **交互式对话 REPL**（`examples/gpt2_chat.cpp`）
  - 模型只加载一次、KV cache 跨轮复用；常驻 tokenizer 子进程
  - `/reset /context /params /set` 运行中可调参；支持 raw / chat 两种模板
  - 不变量由 `Gpt2Structure.MultiTurnCacheMatchesFullForward` 守住
    （跨轮复用 cache 的 logits ≡ 整段一次 forward）
- [x] **交互式对话 REPL（LLaMA 系）**（`examples/llama_chat.cpp`）
  - 与 GPT-2 共用 `examples/chat_common.hpp`：ANSI 颜色、base64、流式增量、tokenizer 客户端
  - 输入构造走 **`apply_chat_template`**（由 `tools/llama_tokenizer_server.py` 的 `M` 操作提供）：
    SentencePiece 下 `enc(a)+enc(b) != enc(a+b)`，模板必须整段交给 tokenizer 渲染
  - 上下文管理：`render_prompt`（当前 messages → 规范 token 序列）→ 与已有 KV cache 比前缀 →
    相同则只喂新增 token，不同则整体重建；超 `--max-context` 时丢弃最早轮次
  - 采样支持 temperature / top-k / **top-p（nucleus）**；`/context` 显示 token 占用
  - 脚本定位有兜底（`:exe` 目录逐级向上找 `tools/`），从 `build/` 里跑也能找到服务脚本
  - 不变量由 `LlamaStructure.MultiTurnCacheMatchesFullForward`、
    `LlamaAlignmentTest.ChatTemplateLogitsMatch` 守住

---

## 8 — 文档与验收命令

- [x] **本 checklist 随进度勾选**，重大结构旁注 commit
- [x] **`TESTING.md` 增加 decoder-only demo / 复现入口**（见「GPT-2 推理对齐」小节）
- [x] **一条命令的对撞验证**：`python tools/verify_gpt2.py --text "..." --logits`
  （退出码 0/1，可直接进 CI；权重缺失时自动调用导出脚本）
- [x] **快速自检（现状）**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJASMINE_USE_OPENMP=ON -DJASMINE_USE_BLAS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
# decoder-only 内核
./build/tests/unit_tests --gtest_filter='TransformerKernel.DecoderOnly*'
# GPT-2 对齐（无权重文件时自动 skip）
./build/tests/unit_tests --gtest_filter='Gelu*:Gpt2*'
# 现有 enc–dec CE（对照，非 decoder-only）
./build/examples/train_transformer_ce
```

### GPT-2 推理对齐（完整闭环）

```bash
# 1. 导出权重 + HF 黄金值（首次需下载，约 313 MiB for distilgpt2）
python tools/export_gpt2.py --model distilgpt2 --out build/distilgpt2_weights.bin \
    --golden-prompt "Hello, my dog is cute"

# 2. 与 HF 端到端对撞（一条命令：生成逐 token + logits 数值；权重已在则跳过导出）
python tools/verify_gpt2.py --text "The capital of France is" --logits

# 3. 逐层 / logits 对齐单测
./build/tests/unit_tests --gtest_filter='Gpt2AlignmentTest.*'

# 4. 交互式对话（在终端里聊）
./build/examples/gpt2_chat build/distilgpt2_weights.bin

# 5. 文本 → id → 生成 → 文本（脚本化）
python tools/gpt2_tokenize.py encode --model distilgpt2 \
    --text "Hello, my dog is cute" --out build/prompt_ids.txt
./build/examples/gpt2_generate build/distilgpt2_weights.bin build/prompt_ids.txt \
    --max-new 20 --out-ids build/out_ids.txt
python tools/gpt2_tokenize.py decode --model distilgpt2 --ids-file build/out_ids.txt
```

实测（distilgpt2 / gpt2，两者都过）：逐层 hidden `max_abs_diff ≤ 6e-4`，logits `≤ 1.2e-4`，
argmax 全位置一致，greedy 生成逐 token 与 HF 完全相同。
换 `--model gpt2`（12 层）走同一套代码即可，无需改 C++。

`tools/verify_gpt2.py` 的 `--logits` 把 **jasmine 生成的序列**喂给 HF 做 forward 再比对，
两侧输入因此完全相同——所以采样模式下这条检查依然有效（采样时两侧 RNG 不同，
逐 token 比对本就不可能一致，此时脚本会明确标注 N/A 而不误报）。

### LLaMA / TinyLlama-Chat 推理对齐（完整闭环）

```bash
# 1. 导出权重 + HF 黄金值（首次需下载，约 4.2 GiB；两条 golden：裸文本续写 + chat 模板）
python tools/export_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --out build/tinyllama_weights.bin \
    --golden-prompt "The capital of France is" \
    --golden-chat "What is the capital of France?"

# 2. 逐层 / logits / KV-cache / chat 模板四条对齐单测（权重缺失时自动 skip）
./build/tests/unit_tests --gtest_filter='Llama*'

# 3. RoPE 配对约定等基础设施回归（这组是当初踩坑的地方，别删）
./build/tests/unit_tests --gtest_filter='RoPE.*'

# 4. 交互式对话（在终端里聊；tokenizer 服务脚本会自动定位）
./build/examples/llama_chat build/tinyllama_weights.bin

# 5. 会话里可用：/help /context /reset /exit
```

实测（TinyLlama-1.1B-Chat，double 对 float32 HF 参考）：23 个阶段逐层 + logits 全位置
`max_abs_diff ≤ 3.1e-5`（= 参考值 float32 舍入噪声量级），top-5 逐位相同；
chat 模板序列（22 token）同样对齐；多轮对话上下文正确累积（66 → 93 → 141 token）且第二轮能引用第一轮内容。

跨模型对照（同一套代码，只换 `jas_*_t.hpp` 与导出脚本）：

| | GPT-2 / distilgpt2 | TinyLlama-1.1B-Chat |
|---|---|---|
| 位置编码 | 绝对 `wpe`（`set_use_rope(false)`） | RoPE `half_split` |
| Norm | Post-norm LN + `ln_f` | Pre-norm RMSNorm |
| FFN | `gelu_new` 两层 | SwiGLU 三投影 |
| 注意力 | MHA | GQA（32 Q / 4 KV） |
| bias | 有（`lm_head` 置 0） | 全部无 bias |
| `lm_head` | 与 `wte` 绑定 | 不绑定 |
| 黄金对齐 | 逐层 ≤ 6e-4 | 逐层 ≤ 3.1e-5 |

---

## 建议实施顺序

```text
过拟合单 batch（embedding + decoder_only + CE）
    → decoder-only demo / generate
        → 选定参考小模型的 Norm/激活/RoPE 对齐
            → （可选）加载开源权重 + logits 黄金对齐
                → 小语料真训 / 报告
```

已完成到「加载开源权重 + logits 黄金对齐」（GPT-2 / distilgpt2 与 LLaMA / TinyLlama-1.1B-Chat，见 §8）；
LLaMA 系的四块积木 —— **RMSNorm / RoPE / GQA / SwiGLU** —— 已拼进 `jas_llama_t.hpp` 并逐层对齐，
对话 demo（`examples/llama_chat.cpp`）可玩。
剩余：从零训练配方（§5）。

---

## 已具备（勿重复造轮子）

- [x] 因果 MHA（含 GQA / MQA 参数化）、RoPE(Q/K，可开关 + **两种配对约定**)、LayerNorm / **RMSNorm**、FFN(ReLU)
- [x] `decoder_only_t`（无 cross-attn）
- [x] Embedding / CE / pad·position mask
- [x] 推理 KV cache（`forward_one`）
- [x] enc–dec 玩具 demo（MSE / CE）——可作对照，不是 LM 复现终点
- [x] Pre-norm 分支 + `gelu_net_t` + 绝对位置 + 权重绑定（`jas_gpt2_t.hpp`）
- [x] 门控容器 `gated_net_t` + `silu_net_t` + `gated_ffn_branches_t`（SwiGLU / GEGLU / ReGLU 骨架）
- [x] GQA / MQA（`mat_mha_t` 的 `n_kv_heads` 参数化，注意力核零改动）
- [x] RMSNorm（`rms_norm_net_t`，eps 可配）
- [x] **LLaMA 系模型 `jas_llama_t.hpp`**：Pre-norm RMSNorm + `half_split` RoPE + GQA + SwiGLU，无 bias、`lm_head` 不绑定
- [x] 权重加载器（`jas_weight_io.hpp`）：`load_gpt2` / `load_llama` + 两个导出脚本 + 黄金对齐单测
- [x] KV-cache 生成 demo（`examples/gpt2_generate.*`）
- [x] 共用 REPL 基础设施（`examples/chat_common.hpp`）+ 两个可玩对话 demo（GPT-2 / TinyLlama-Chat）

---

## 踩过的坑（GPT-2 对齐实录）

| 坑 | 现象 | 处理 |
|----|------|------|
| `gelu_new` ≠ erf 版 GELU | logits 有系统性偏差 | 按 tanh 近似实现 `gelu_net_t` |
| `layer_norm_net_t` 懒初始化 γ/β | 加载器拿到未分配矩阵 | 新增 `set_param(d_model)` 显式分配（**不要**叫 `reinit`，会改变 `is_reinitable_net` 判定） |
| `complex_net_t` 从链首成员取 `val_type` | LN 放在链首时编译失败（`val_type` 为 private） | `layer_norm_net_t` / `gelu_net_t` 的 `val_type` 改为 public |
| RoPE 被 `bind_rope()` 无条件绑定 | 绝对位置模型 logits 对不上 | `set_use_rope(false)`，由 `bind_rope()` 尊重该开关 |
| Conv1D 布局与 fused `c_attn` | 直接灌权重形状/数值都不对 | 导出时转置并在 Python 侧切 q/k/v |
| `lm_head` 有 bias | logits 整体偏移 | 显式置 0，且与 `wte` 绑定 |
| **`transformers>=5` 的 `hidden_states[-1]` 是 `ln_f` 之后的值** | 最后一层「黄金值」看似量级相近却对不上（`max_abs_diff` 达 483）；LayerNorm 会掩盖仿射差异，极易误判为己方有 bug | 导出脚本手动跑最后一个 block 取真正的 pre-`ln_f` 输出，并自检 `ln_f(last) == hidden_states[-1]` |
| 逐 token `decode` 做流式输出 | 多字节字符被切成半个 token 时 HF 会替换成 U+FFFD，原始字节丢失，终端显示 `I��m`，且与全量 decode 不一致 | 每步重解**全量** token 列表，只输出相对上次的新增部分，并掐掉末尾 U+FFFD（详见 `gpt2_chat.cpp` 注释） |
| 每轮起一个 Python 进程做 tokenize | `transformers` 冷启动约 2 秒，每轮两次调用 → 对话卡到不可用 | 常驻 tokenizer 子进程（`gpt2_tokenizer_server.py`），行协议 + base64 传输 |
| raw 续写模式不补分隔符 | 上一轮结尾与这一轮开头粘成 `stillWhat`，模型困惑，下一个 token 直接预测 `endoftext`（表现为「0 个新 token」） | 轮与轮之间补一个换行 |
| 门控 FFN 的反向：两分支**共享**同一输入 | 若把两条分支的 `backward` 结果取平均/只留一条，梯度就错了 | `gated_net_t::backward` 返回两条路径梯度**之和**；`∂L/∂gate = delta⊙up`、`∂L/∂up = delta⊙gate`（逐元素乘的梯度就是乘对方） |
| 门控 FFN 用矩阵乘合并 | 维度能凑上（方阵时）但数值全错、且 `silu(x)*x` 与矩阵积在语义上完全不同 | 合并算子必须是**逐元素乘**（`operator*` 即 Hadamard），单测 `Gated.ForwardIsElementwiseProductNotMatmul` 专门钉住这一点 |
| 数值梯度自检时容器 `backward` 会**原地**改权重与偏置 | 若数值基准取的是 `backward` 之后的参数状态，梯度比对的绝对误差约 `1e-3`，且集中在权重幅值最大的最后一行，看着像「容器反向有 bug」 | 数值梯度必须基于 `backward` **之前**的参数快照（`weight_net_t` 对 bias 也做 update，别只冻结权重） |
| GQA 复用已有 `forward_one_at` 逐 Q 头写 cache | 共享同一 KV 头的 `group_size` 个 Q 头各写一次 → 同一份 K/V 被 append `group_size` 次。**形状不报错、单步看似可用**，但 cache 长度与内容全错，多轮对话才炸 | KV cache 的写入必须"每个 KV 头一次"：拆成 `append_kv_head`（写 + RoPE）+ `attend_cached`（只读不写），后者本就是现成原语 |
| GQA 反向沿用 MHA 的 `assign` 写回 KV 梯度 | 组内多个 Q 头的梯度互相覆盖，只剩最后一个 → 训练能跑但学不到 K/V | `mat_view_t` 没有 `+=`，用 `add_rows` 在共享 KV 头上**累加**；判据见 `Gqa.BackwardSumsKvGradientsAcrossGroup` |
| `backward` 依赖 `forward` 留下的头内缓存 | 若在 `forward_one`（推理路径）之后调用 `backward`，`attend_cached` 不填 `m_v`、`m_softmax.m_output` 也只有单步形状 → `inner dimensions do not match` | 训练前向与推理前向不可混用后接反向：先 `forward`（整段）再 `backward`；`clear_kv_cache()` 之后才走 `forward_one` |
| 从 LayerNorm 抄反向实现给 RMSNorm | 会**多留一个 `mean(g)` 项**（LayerNorm 减均值带来的），解析梯度与数值梯度差约 1.0 —— 训练能跑但方向系统性偏 | RMSNorm 反向 = LayerNorm 反向去掉 `sum(g)/d` 那一项；`RmsNorm.BackwardOmitsCenteringTermLestItBeWrong` 双向钉住 |
| 用 HF 的 `LlamaRMSNorm` 输出当 float64 基准 | 该类内部 `.to(torch.float32)`，即使传 float64，输出也与精确解差约 1e-7 | 基准取 float64 定义式（jasmine 是 double）；已核实差值恰为该精度转换 |
| **RoPE 特征配对约定与 HF LLaMA 不一致** | **最贵的一个坑**：jasmine 原生按 `(2i, 2i+1)` 配对，HF LLaMA 按 `(i, i+d/2)`。逐层 hidden 与 logits 全部对不上（`max_abs_diff` 达 7.17），但**位置 0 完全正确**（那里旋转是恒等变换），所以极易误判成「attention/softmax/权重加载有问题」去乱查 | 新增 `rope_pair_layout`（`interleaved` / `half_split`），默认保持原生行为；LLaMA 显式用 `half_split`。定位手法：把 HF 的 `apply_rotary_pos_emb` monkeypatch 成本仓库的交错配对，logits 立刻与自己的输出逐位吻合 → 证明差异**只**在配对约定 |
| 只比对单 token / 位置 0 的输出来验证位置编码 | 位置 0 是恒等变换，**任何**配对约定、任何 base θ 都对得上，等于没验 | 位置编码必须用 **≥2 个 token** 且带非零相对距离的序列验证；`RoPE.LayoutsAgreeAtPositionZeroOnly` 专门钉住这个陷阱 |
| RoPE 注册表只按维度 `d` 建 key | 同一 `d` 上不同配对约定会共用同一份 RoPE，`half_split` 被静默降级成 `interleaved` | 注册表 key 改为 `(d, layout)`；`RoPE.RegistrySeparatesPairLayouts` 守住 |
| LLaMA 的 `nn.Linear` 无 bias，但 `init_weight` 会 `reinit` 出随机 bias | `set_param` 里清零无效（随后 `init_weight` 又随机化），`LlamaStructure.AllBiasesAreZero` 失败 | 在 `llama_model_t::init_weight` 末尾统一 `zero_all_biases()`，与调用顺序无关 |
| `transformers>=5` 的 `apply_chat_template(..., return_tensors='pt')` 返回 `BatchEncoding` | 直接取 `.shape` 抛 `AttributeError`（`ids.shape[1]` 报错，看着像 tokenizer 坏了） | 统一用 `as_batch_ids()` 兜一层：优先 `.input_ids`，兼容张量/列表两种返回 |
| 4.2 GiB 权重文件与 float64 模型同时常驻 | 加载后整模型 double ≈ 8.8 GB，再加原始 blob 4.4 GB，小内存机器会 OOM | `weight_file_t::release()` 在 `read_into` 完成后释放 blob；大模型加载路径必须调用 |

---

## 常见误解（备忘）

| 说法 | 澄清 |
|------|------|
| Decoder-only = 迭代 encoder | 否；是因果自注意力栈 + 自回归生成 |
| Tokenizer = embedding | 否；前者文本↔id，后者 id↔向量 |
| 有因果 mask 就等于开源小模型 | 否；还差词表、头/Norm/FFN 配方、权重与数据 |
| 能加载 DeepSeek 词表就能复现 DeepSeek | 否；结构/规模/数据差几个数量级；本清单只盯「小模型」 |
| logits 对得上就万事大吉 | 否；LayerNorm 会掩盖仿射偏差，**逐层 hidden 对齐**才能定位问题 |
