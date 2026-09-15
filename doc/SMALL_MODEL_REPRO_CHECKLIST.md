# 复现小模型 Checklist（Decoder-only LM）

目标：用 jasmine 复现 / 对齐一类 **小规模开源因果语言模型**（如 TinyLlama、SmolLM、GPT-2 small 量级），而不是完整 DeepSeek。  
前提：任务是 **next-token 语言建模**，拓扑为 **decoder-only**（无 encoder / cross-attn）。

状态约定：`[ ]` 未做 · `[~]` 进行中 · `[x]` 完成 · `[—]` 非本阶段必须

与优化清单分工：算子/性能见 [`OPTIMIZATION_CHECKLIST.md`](OPTIMIZATION_CHECKLIST.md)；本文件管 **「能训出/加载一个可用的小 LM」**。

---

## 0 — 范围与成功标准

- [ ] **选定参考模型**
  - 例：GPT-2 small / TinyLlama-1.1B / 自研玩具配置（`d_model=256, n_layers=4, n_heads=4, V≈8k`）
  - 写清：词表来源、层数、`d_model`、`n_heads`、`d_ff`、RoPE/绝对位置、Norm 位置、激活、是否 GQA
- [ ] **成功标准（至少一条）**
  - **从零训**：固定小语料上 loss 稳定下降，生成可读（可过拟合验证）
  - **加载权重**：与参考实现同输入 id 时，logits / 隐层在约定容差内（黄金对齐）
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
- [ ] **（对齐常用 LLM）Pre-norm**
  - 现状：Post-norm（`SubLayer` → add → Norm）
  - 目标：`x + SubLayer(Norm(x))`；栈顶可选 Final Norm
  - 验收：与参考小配置数值或训练稳定性对比文档化
- [—] **GELU / SwiGLU FFN**
  - 现状：`Linear → ReLU → Linear`
  - 对齐 LLaMA 系再换 SwiGLU；对齐 GPT-2 再换 GELU
- [—] **GQA**
  - 现状：标准 MHA；小模型可先不做，KV cache 已够用

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
- [ ] **权重绑定（可选）**
  - `lm_head.W` 与 `embedding.E` 转置共享（GPT-2 常见）；不共享亦可，需与参考一致
- [ ] **Next-token 数据布局**
  - 输入 `t[0..T-2]`，标签 `t[1..T-1]`；pad / prompt 不计入 CE
- [ ] **（可选）对话模板**
  - 用户/助手标记拼进同一条因果序列；loss 常只算助手段

---

## 4 — 位置编码与注意力契约

- [x] **RoPE 在 Q/K 上**（`rope_registry` / `mat_head_gen_t`）
- [x] **因果 mask + 黄金测试**
- [ ] **与参考模型位置约定一致**
  - base `θ`、是否 NTK/YaRN、最大长度；绝对位置模型则不要强行 RoPE
  - 验收：固定 id、固定权重时位置 0..L 的 Q/K 旋转与参考误差在阈值内

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

- [ ] **参数名与 shape 对照表**
  - 参考 `state_dict` ↔ jasmine 层（`W_Q/K/V/O`、FFN、LN γ/β、embed、lm_head）
- [ ] **布局约定**
  - 行/列主序、`d_model×V` vs `V×d_model`、QKV 是否 fused
- [ ] **加载器**
  - 读 safetensors / numpy；校验每个 tensor shape 再写入
- [ ] **前向黄金对齐**
  - 同 `input_ids`：对比 logits（或最后一层 hidden）；先 1 层再全栈
- [ ] **（可选）反传不对齐也可**
  - 推理复现只需 forward；训练复现再对梯度

说明：仅有 `decoder_only_t` **不能**直接加载任意开源权重；缺 tokenizer、缺加载与结构差异（Pre-norm / SwiGLU / GQA）时，只能「同思路小模型」，不是「同一 checkpoint」。

---

## 7 — 推理

- [x] **KV cache + `forward_one`**（decoder-only / MHA）
- [ ] **生成循环**
  - `clear_kv_cache` → 预填 prompt → 逐步采样（greedy / temperature）→ `eos` 停止
- [ ] **与无 cache 全量 forward 一致性**
  - 已有单测模式；接到 demo 的 generate API 上再验一次

---

## 8 — 文档与验收命令

- [ ] **本 checklist 随进度勾选**，重大结构旁注 commit
- [ ] **`TESTING.md` 增加 decoder-only demo / 复现入口**（有 binary 后）
- [ ] **快速自检（现状）**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJASMINE_USE_OPENMP=ON -DJASMINE_USE_BLAS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
# decoder-only 内核
./build/tests/unit_tests --gtest_filter='TransformerKernel.DecoderOnly*'
# 现有 enc–dec CE（对照，非 decoder-only）
./build/examples/train_transformer_ce
```

---

## 建议实施顺序

```text
过拟合单 batch（embedding + decoder_only + CE）
    → decoder-only demo / generate
        → 选定参考小模型的 Norm/激活/RoPE 对齐
            → （可选）加载开源权重 + logits 黄金对齐
                → 小语料真训 / 报告
```

---

## 已具备（勿重复造轮子）

- [x] 因果 MHA、RoPE(Q/K)、LayerNorm、FFN(ReLU)
- [x] `decoder_only_t`（无 cross-attn）
- [x] Embedding / CE / pad·position mask
- [x] 推理 KV cache（`forward_one`）
- [x] enc–dec 玩具 demo（MSE / CE）——可作对照，不是 LM 复现终点

---

## 常见误解（备忘）

| 说法 | 澄清 |
|------|------|
| Decoder-only = 迭代 encoder | 否；是因果自注意力栈 + 自回归生成 |
| Tokenizer = embedding | 否；前者文本↔id，后者 id↔向量 |
| 有因果 mask 就等于开源小模型 | 否；还差词表、头/Norm/FFN 配方、权重与数据 |
| 能加载 DeepSeek 词表就能复现 DeepSeek | 否；结构/规模/数据差几个数量级；本清单只盯「小模型」 |
