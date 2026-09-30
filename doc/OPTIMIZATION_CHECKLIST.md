# Jasmine 优化 Checklist

按优先级排列。勾选前请补测试；小矩阵上盲目并行通常会更慢。

状态约定：`[ ]` 未做 · `[~]` 进行中 · `[x]` 完成

复现小规模 decoder-only LM（词表 / Pre-norm / 权重加载等）见 [`SMALL_MODEL_REPRO_CHECKLIST.md`](SMALL_MODEL_REPRO_CHECKLIST.md)。

---

## P0 — 正确性（先修）

- [x] **RoPE 2×2 共用同一 θ**
  - 文件：`jas_RoPE_t.hpp`
  - 问题：同一旋转块的 cos / −sin 使用了不同的角度索引，不是正交 2×2
  - 验收：单元测试对比已知角度的旋转结果（黄金值）

- [x] **RoPE 频率与位置公式对齐常见定义（或文档化为刻意变体）**
  - 目标：`θ_{i,m} = m / 10000^(2i/d)`（或等价形式），位置用 token 下标 `m` 而非 packed column `j`
  - 验收：与手算 / 参考实现在小 `d_model` 上误差 < 1e-10

- [x] **明确 RoPE 作用位置**
  - 现状：各层 MHA/MHCA 在 Q/K 投影后旋转（`mat_head_gen_t`）；`rope_registry_t` 按 `d_head` 共享
  - 决策：不再在 `transformer_base_t` 入口旋转整段输入
  - 验收：文档与实现一致；encoder 侧若可训练 embedding，反向能穿过 RoPE（经 Q/K 路径）

- [x] **RoPE / 因果注意力黄金测试**
  - 扩展：`tests/test_rope.cpp`（`BackwardIsTransposeRotation`）、`tests/test_causal_attention.cpp`（mask softmax 手算 + 未来 token 不泄漏）
  - 不只查「有限」，要查数值契约

---



## P1 — 模型结构保真

- [x] **MHA 改为经典「全维 QKV → 按头拆分」**
  - 文件：`jas_mha_t.hpp`
  - 现状：`W_Q/W_K/W_V` 在 `mat_mha_t` 全维投影，再 `vsplit` 到各头 attend；头内不再自带小投影
  - 目标：`W_Q,W_K,W_V ∈ R^{d_model×d_model}`（或等价），再 reshape/split heads，最后 concat + `W_O`
  - 验收：shape 测试 + 与单头全维注意力在 `num_heads=1` 时一致（`Mha.SingleHeadCanFitSimpleTarget`）

- [x] **（可选）交叉注意力同样走标准多头投影**
  - 与 self-attn 同一套 head 约定，避免两套语义

- [x] **修正过时注释**
  - 例：`jas_mha_t.hpp` 中 encoder 梯度「记得除以层数」——当前实现是累加后一次 `backward`，不应除
  - 已改为：K/V 全维梯度直接累加到 `encoder_delta`

---



## P2 — 推理与训练效率

- [x] **Decoder self-attn KV cache**
  - 场景：`predict` 逐步生成（`examples/transformer_mse_demo.hpp` / CE 见 `transformer_ce_demo.hpp`）
  - 实现：`jas_kv_cache_t.hpp` + `mat_mha_t::forward_one` / `decoder_t::forward_one`；cache 存 RoPE 后的 K/V
  - 验收：`tests/test_kv_cache.cpp` 逐步输出与无 cache 全量 forward 一致；`predict` 走 `complex_net_t::infer`（`skip_on_infer` 层被跳过）

- [x] **Encoder memory 复用**
  - `encoder_forward` 一次写入 `m_encoder_output`；decode 多步只读，不重跑 encoder

- [x] **Cross-attn K/V cache**
  - `set_encoder_output` → 各层 `prepare_cross_kv`：投影 memory 为 K/V（RoPE 后）写入 cache
  - `mat_mhca_t::forward_one` 只算 Q，经 `attend_cached` 读 cache
  - 验收：`DecoderKvCache.*` / `MhcaKvCache.PrepareThenDecodeMatchesFull`
- [x] **Teacher forcing 路径不强制走 cache API**
  - 整段一次前向保持简单；cache 仅 inference / AR 使用（`forward` 训练 vs `infer` 推理，结构同一 `net_type`）

- [x] **prompt 预填走批量路径（不是逐 token 循环）**
  - `gpt2_model_t::prefill` 现在是整段一次前向：`mat_mha_t::forward_one` 本就接受多列并按
    绝对位置做 causal mask，逐 token 循环白付了 T 次 GEMM 启动
  - `gpt2_model_t::forward_one` 同时支持多列（T>1），供交互式路径把一整轮用户输入
    一次追加进已有 cache（跨轮复用不能 clear）
  - 验收：`Gpt2Structure.PrefillCacheMatchesStepByStepFeeding`（prefill 后的 cache 继续解码 ≡ 整段
    forward）、`Gpt2Structure.MultiColumnForwardOneMatchesPerTokenLoop`；
    真实权重下 `Gpt2AlignmentTest` 三例仍全绿，`tools/verify_gpt2.py` 逐 token 一致
  - 实测（distilgpt2 / fp32 / 128 token prefill / 4 线程）：2085.40 ms → 282.08 ms（**7.39×**），
    且 prefill 与整段 `forward` 同价（282.08 vs 272.66）——见 `doc/bench/gpt2_infer_vs_torch.md`

- [x] **评估 OpenMP 落点（有收益再开）**
- [x] **评估 OpenMP 落点（有收益再开）**
  - `jas_mat_gemm.hpp`：大 GEMM（`MNK≥64³`）对单线程 BLAS 按行分片并行；blocked fallback 外层并行
  - `jas_mha_t.hpp`：大 shape 下 QKV 三投影 `sections` 并行；多 head attend `parallel for`（阈值：`heads·seq·d_head≥4·32·32`）
  - **小矩阵 / 短序列不并行**（`if` 阈值）
  - 验收：`doc/bench/openmp_*.txt`；小 shape 不回归，大 shape 有加速

- [x] **大矩阵走 BLAS / 优化 GEMM（可选）**
  - `jas_mat_gemm.hpp`：大矩阵 `mat_dot_t::clone` 走 cache-blocked GEMM；若找到 `libblas`/`openblas` 则 `cblas_sgemm/dgemm`（`JASMINE_USE_BLAS`）
  - 验收：`tests/test_gemm.cpp`；`doc/bench/matmul_*.txt` 中 `BM_MatDot` 在 `n≥256` 有数量级加速
  - 基线（Release，优化前）→ 优化后对比见 `doc/bench/`

- [x] **确认 BLAS 真的在用本机的最优内核（不是「链上了就算完」）**
  - 问题：OpenBLAS 在**运行时**按检测到的 CPU 选内核。本机那份 0.3.15（来自一个无关安装）
    在 13600KF 上 `openblas_get_corename()` 返回 **`Prescott`**（2004 年、仅 SSE3），
    尽管库里带着 `sgemm_kernel_HASWELL/SKYLAKEX`——0.3.15 早于 Raptor Lake，检测没命中
  - 三种解法实测（4 线程，整段 forward）：

    | 配置 | forward | 相对自动检测 |
    |---|---|---|
    | OpenBLAS 0.3.15，自动检测（Prescott） | 190.6 ms | — |
    | OpenBLAS 0.3.15 + `OPENBLAS_CORETYPE=HASWELL` | 111.4 ms | **1.71×** |
    | **OpenBLAS 0.3.26（Ubuntu 24.04 `libopenblas0-pthread`），自动检测（Haswell）** | **106.5 ms** | **1.79×** |

  - **结论：换一个新版 OpenBLAS 即可，无需任何配置**；0.3.26 自动识别正确，且内核比旧库强制
    Haswell 再快约 4%。环境变量只是旧版本下的权宜之计
  - **务必验证实际用的是哪一份**：两个库可以同名 `libopenblas.so.0`，而 `LD_LIBRARY_PATH`
    会把其中一个悄悄替换掉（本轮就踩过一次，差点得出错误结论）：
    `ldd ... | grep libopenblas` 看解析到哪个文件，`OPENBLAS_VERBOSE=2` 看选了哪个内核
  - `tools/compare_gpt2_torch.py` 默认钉住 `OPENBLAS_CORETYPE=haswell` 并打印所用值
  - **教训**：这条比同期任何代码优化都大。做性能对比前必须先确认 BLAS 的实际内核，
    否则测的是打包/CPU 检测，不是实现

- [x] **试用 Intel MKL（对比「是不是换个 BLAS 就能追平」）**
  - 拿法：`apt-get download` Ubuntu multiverse 的 `intel-mkl 2020.4`（`libmkl-rt/core/avx2/gnu-thread/intel-lp64`），
    解压后用 RPATH 链接，**不安装任何系统包**
  - 实测（4 线程，与 OpenBLAS 0.3.26 同轮交替）：

    | BLAS | forward | decode/token |
    |---|---|---|
    | OpenBLAS 0.3.26（Haswell，自动） | **106.5 ms** | 9.52 ms |
    | MKL 2020.4（AVX2） | 108.2 ms | **8.30 ms** |

  - **forward 上 MKL 无改善（2% 内，噪声）**；**decode 上 MKL 快 1.15×**，使 4 线程 decode 与
    PyTorch 打平（7.75 vs 7.85 ms）
  - **关键认知**：PyTorch 用的 MKL **不是这个**。`find / -name 'libmkl*.so*'` 为空——
    torch 的 MKL **静态链进了 `libtorch_cpu.so`**（23605 个符号已定义、未定义为 0），且来自
    2024 年的 pip wheel，而 Ubuntu 里能拿到的是 2020.4。torch 那 570 GFLOP/s 来自那份更新的
    静态 MKL，本机无法作为共享库链接。所以「forward 差 2.36×」不能简单归因于「换 MKL 就好」
  - **链接 MKL 的必要前提**：`libmkl_rt` 默认加载 Intel 自带的 OpenMP；而 jasmine 的并行循环也是
    OpenMP，同进程两个 OpenMP runtime 是已知的挂死来源。故用 `libmkl_gnu_thread` +
    `MKL_THREADING_LAYER=GNU`，并**故意不装** intel-thread；不加该变量会直接
    `Cannot load libmkl_intel_thread.so` 退出。`compare_gpt2_torch.py` 已自动处理
  - 日志：`doc/bench/gpt2_vs_torch_mkl_{1,4,8}threads.txt`

- [x] **单列输出（解码）走 GEMV，不走 GEMM**
  - 问题：解码恒为 `N=1`，每个权重只用一次，GEMM 的分块打包（把 A/B 复制成缓存友好的副本）
    收益为零，却把「必须搬 328 MB」变成搬了更多——时间没花在读权重，花在复制权重
  - 实测交叉点在 **24 MB，正好是本机 L3（24 MiB）**：矩阵能驻留 L3 时打包流量不出片、BLAS 赢；
    溢出到 DRAM 后 GEMV 单遍扫描赢，36 MB 时 1.38×、147 MB 时 1.76×
  - 实现：`jas_mat_gemm.hpp` 的 `gemv_rowmajor`，阈值 `gemv_byte_threshold = 16 MiB`（留在 L3 之下），
    只对 `N==1 && !trans_a` 生效；可用 `-DJASMINE_USE_GEMV=0` 关掉做 A/B
  - 注：换更好的 BLAS 会把交叉点从 ~24 MB 上推到 ~40 MB，使 16 MiB 偏保守。**未调整**，因为
    fp32 下最大的非 lm_head 矩阵只有 9.4 MB（无论怎么调都走 BLAS），而 double 下开 GEMV 仍明显更快
    （13.8–15.0 vs 19.3–19.6 ms/token）——double 下 lm_head 是 309 MB，占绝对主导
  - 验收：`MatGemm.GemvDispatchPolicy` / `GemvKernelMatchesNaive` / `GemvKernelHandlesBothVectorStrides`；
    真实权重 `Gpt2AlignmentTest` 三例全绿、`verify_gpt2.py` 逐 token 一致
  - 实测（distilgpt2 / fp32 / 4 线程）：decode 16.33 → 12.41 ms/token（**1.32×**），
    double 下 20.10 → 15.76 ms（1.28×）；`forward`/`prefill` 不变（N=128 不走该路径），
    logits 校验和逐位相同 → 见 `doc/bench/gpt2_gemv_ab.txt`

- [x] **削减不必要的** `.clone()`
  - `jas_mat_storage.hpp::store_for_backward` + 层间 `net_forward` 完美转发（mat rvalue 移动）
  - 推理 KV：attend 直接读 cache view，不再每步 clone 全量 K/V
  - 层 forward/backward 去掉与 `operator mat_t()` 重复的 `.clone()`；`fill_kv_cache` 视图直 append
  - 验收：`tests/` 全绿；`doc/bench/clone_reduction_*.txt` 对比

---



## P3 — 任务与损失（从玩具走向可用）

- [x] **离散 token + embedding + 输出投影**
  - `jas_embedding_t.hpp`：`embedding_net_t` 查表 `E ∈ R^{d_model×V}`，输入 `1×T` id → `d_model×T`
  - 输出投影：`output_proj_net_t` = `weight_net_t`（`d_model → V` logits）
  - 验收：`tests/test_embedding_ce.cpp`（gather / scatter / pipeline）

- [x] **Cross-entropy / NLL loss**
  - `ce_loss_t`：fused 稳定 softmax + NLL；`forward` 透传 logits；`skip_on_infer`
  - `backward(target_ids)` → `(softmax(z) - onehot) / N`；保留 `mse_loss_t`
  - 验收：`CeLoss.SoftmaxGradMatchesOneHot`；`TokenPipeline.*`

- [x] **掩码：padding / 未来位置与 loss 对齐**
  - `set_ignore_index(pad_id)`：pad 不计入 loss/梯度；EOS 为普通类别 id
  - `set_position_mask(1×T)`：可屏蔽未来位等；与 ignore_index 同时生效
  - 验收：`CeLoss.IgnoreIndexSkipsPad` / `CeLoss.PositionMaskSkipsFuture`

- [ ] **（可选）dropout / 权重衰减等训练配方**
  - 非必须；有真实数据过拟合时再加

---



## P4 — 工程卫生

- [x] **`mat_t::operator()` 的 `%` 折回（保留语义，但避免热路径取模）**
  - 设计原则：**底层默认放开，不在索引层做严格越界禁止**；形状合法性由上层逻辑保证
  - 用途：express 对位广播（标量 / 行向量 / 列向量）、以及大矩阵对小矩阵的周期/折回访问（例如 CNN 卷积核在输入上滑动时的下标映射）
  - 决策：保持 `%`；仅在确有需要的上层路径再显式约束，不把「禁止」下沉到 `mat_t`
  - **实测问题**：每元素两次整数取模（约 20–40 周期且难以流水），而访问器是「每元素每操作数一次」，
    三操作数表达式等于每元素 6 次除法 → 隔离测试仅 **1.3 GB/s**，裸指针同循环是 48.9 GB/s（**38×**）
  - 修法：**先做范围检查，越界（含负数）才走原来的 `%`** → 对所有输入可证明等价；
    隔离测试 6.8 → 22.0 GB/s（列下标那次取模是主因，它是内层循环）
  - 验收：全量 280 例 0 失败；`Gpt2AlignmentTest` 三例 + `tools/verify_gpt2.py` 逐 token 一致
  - 实测：forward 278.39 → **210.83 ms（1.32×）**，对 PyTorch 的倍数 5.63× → **4.72×**；
    单层 LayerNorm 2.82 → 0.96 ms（2.95×），attention core 4.71 → 2.04 ms（2.31×）
  - 见 `doc/bench/gpt2_infer_vs_torch.md` 的 “Change 3”

- [ ] **数值梯度检查（抽几层）**
  - weight / LayerNorm / MHA 对输入的有限差分 vs `backward`

- [ ] **文档同步**
  - `TESTING.md`：如何跑 correctness + bench
  - 本 checklist 完成后在条目旁标 `[x]` 与 PR/commit

- [ ] **Benchmark 基线入库**
  - 固定 shape（如 matmul 512、MHA `d=512,h=8,T=128`）记录一版数字，优化前后对比

---



## 建议实施顺序

```text
RoPE 正确性 + 黄金测试
    → MHA 标准多头
        → KV cache（decode）
            → 大矩阵 GEMM / 有选择的 OpenMP
                → 离散 CE 任务（若目标是 LM/seq2seq）
```



## 已完成（勿重复投入）

以下已相对稳妥，优化时默认保留行为：

- [x] `namespace jasmine`
- [x] FFN：`Linear → ReLU → Linear`，`d_ff = 4·d_model`
- [x] LayerNorm：按 token（列）在特征维标准化
- [x] 训练：每步重算 encoder；decoder TF = `SOS+label`；scheduled sampling；EOS 损失加权
- [x] `set_lr` 不重置优化器动量（避免每步 `set_updator`）
- [x] CMake + GoogleTest / Benchmark / examples 分离
- [x] `mat_t::operator()` 用 `%` 折回（底层默认放开；广播/卷积等通用机制，越界由上层约束）



## 快速自检命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJASMINE_USE_OPENMP=ON -DJASMINE_USE_BLAS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/benches/bench_jasmine --benchmark_filter=BM_
./build/examples/train_transformer
./build/examples/train_transformer_ce
```

