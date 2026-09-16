# Jasmine 的 CUDA 后端

这份文档讲清楚三件事：**表达式模板为什么能上 GPU**、**什么能上什么不能**、以及**在这台机器上跑测试的散热注意事项**。

---

## 1. 结论先行

表达式模板能上 CUDA，而且能拿到它最值钱的那个好处：**整条逐元素链融合成一次 kernel launch，中间结果一个都不物化**。

`(a + b) * c` 在 CPU 上要写一个临时矩阵再读回来；在 GPU 上，每个线程只读它需要的叶子元素，算完整棵树的标量运算，直接写出结果。

但必须**在 `dot` 处切分**：逐元素算子交给融合 kernel，矩阵乘交给 cuBLAS。理由见第 4 节。

| 能力 | 状态 |
| --- | --- |
| 逐元素融合（`+ - * /`、比较、`exp`、`sigmoid`） | ✅ 已实现 |
| 标量操作数 | ✅ 已实现 |
| 转置视图（逐元素路径） | ✅ 已实现 |
| GEMM 与转置组合（`A·B`、`Aᵀ·B`、`Q·Kᵀ`、双转置、`beta` 累加） | ✅ 已实现（cuBLAS） |
| `.dot()` 的设备分派（立即求值 + 链式 + 表达式操作数） | ✅ 已实现（第 6 节） |
| 归约（`row_sum` / `col_sum` / `sum_all` / …） | ✅ 已实现（第 5 节） |
| 逐行 softmax（含 `-inf` 因果掩码） | ✅ 已实现（`softmax_rows`，单趟共享内存路径 + 三趟回退，见 5.6） |
| LayerNorm / RMSNorm | ✅ 已实现 |
| softmax / LayerNorm / RMSNorm / RoPE 的**反向** | ✅ 已实现（第 9 节） |
| 设备端优化器 sgd / adam / nadam + 梯度累积 | ✅ 已实现（第 9 节） |
| 设备端层（linear / norm / silu / residual / gated / mse）的 forward + backward | ✅ 已实现（第 9 节） |
| 单头注意力前向端到端 | ✅ 已实现（GEMM + 掩码 + softmax + GEMM，有对拍用例） |
| 设备端 KV cache（零拷贝视图、GQA 多 KV 头、decode 与 prefill） | ✅ 已实现（第 7 节） |
| 设备端 RoPE（两种配对约定、逐头、`start_pos` 偏移） | ✅ 已实现（第 8 节） |
| 设备端 MHA 层 `dev_mha_t`（前向 + 反向，含 GQA 梯度归并、decode 路径） | ✅ 已实现（9.7） |
| 设备端 Embedding 层 `dev_embedding_t`（gather + 按 id 原子累加） | ✅ 已实现（9.8） |
| **整个 LLaMA 模型 `dev_llama_t`**（参数搬运 + 整段前向 / 增量解码 / 反向 / 训练） | ✅ 已实现（9.9） |
| GPT-2 模型上设备 | ❌ 尚未实现（结构同 9.9，缺的是「按 `gpt2_model_t` 的层序再拼一遍」） |

---

## 2. 快速上手

```cmake
cmake -S . -B build-cuda -DJASMINE_USE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda            # 只跑已启用的用例，不会烤卡
```

用法：

```cpp
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_kv_cache.hpp"
#include "jas_cuda_rope.hpp"

using namespace jasmine;

cuda::dev_matrix_t<double> a(rows, cols, ha);   // 分配 + 上传
cuda::dev_matrix_t<double> b(rows, cols, hb);
cuda::dev_matrix_t<double> c(rows, cols, hc);

// 逐元素：整条链一次 launch
auto y = cuda::eval_fused_to_host((a.leaf() + b.leaf()) * c.leaf());

// 矩阵乘：走 cuBLAS
auto s = cuda::gemm_to_host(q.leaf(), k.leaf().t());   // S = Q·Kᵀ

// .dot() 立即求值，写法与主机端几乎同形（第 6 节）
auto scores = q.leaf().t().dot(k.leaf());              // → dev_matrix_t
auto prod   = a.dot(b);                                // 链式：a.dot(b).dot(c)

// 操作数可以是表达式：非叶子的先融合物化
auto mixed = cuda::matmul(a.leaf() + b.leaf(), c.leaf());   // (a + b)·c

// 归约：整棵表达式树都能塞进去，exp 融进求和那一趟
auto row_totals = cuda::row_sum(exp(x.leaf() * 2.0));

// 逐行 softmax：掩码（0 / -inf）直接加进表达式
auto weights = cuda::softmax_rows(scores.leaf() / scale + mask.leaf());

// 归一化层：gamma 是 (d_model × 1) 的列向量，沿列广播
cuda::dev_colvec_t<double> gamma(d_model);
gamma.buffer().upload(host_gamma.data(), d_model);
auto normed = cuda::rms_norm(x.leaf(), gamma, 1e-5);

// GEMM 的结果用 .leaf() 回到表达式世界
auto out = v.leaf().dot(weights.leaf().t());

// KV cache：多轮 decode（第 7 节）。K 必须已做过 RoPE —— 与主机端同一契约。
cuda::dev_kv_caches_t<double> caches;
caches.configure(num_heads, num_kv_heads, d_head, max_seq);   // GQA 的 Q→KV 头映射在这里定
caches.append_all(k_full, v_full);          // 唯一写入入口：一次写进**所有** KV 头
auto attn = caches.attend_q_head(h, q.leaf());   // Q→KV 头映射由容器内部完成

// RoPE（第 8 节）：作用在 d_head 上，口径与主机端 RoPE_net_t 一致
cuda::dev_rope_t<double> rope(d_head, jasmine::rope_pair_layout::half_split);
rope.reserve(max_seq);
// 单头单个 token：位置 = 当前 cache 长度，这就是「列 j 用绝对位置 start_pos + j」的入口
auto k_rot = rope.forward_at(dk.leaf(), cache.length());
// 打包 QKV 时按头切开（零拷贝行子块），逐头旋转到目标矩阵的对应行段
for (int g = 0; g < num_kv_heads; ++g)
    rope.rotate_into(row_slice(qkv.const_leaf(), g * d_head, d_head),
                     row_slice(qkv_rot.leaf(), g * d_head, d_head), pos);
```

归约结果的两种用法：`.download()` 回主机对拍，或 `.leaf()` 取广播叶子继续参与表达式运算。

**训练也跑在同一套东西上**（第 9 节）——`jas_cuda_net.hpp` 里那些层的 `forward` / `backward`
与主机端 `jas_net_t.hpp` 同名同签名，`backward` 返回梯度、顺带就地更新参数：

```cpp
#include "jas_cuda_net.hpp"        // 层
#include "jas_cuda_updator.hpp"    // 优化器

using namespace jasmine;

cuda::dev_layer_norm_t<double, cuda::dev_sgd_t> ln;   ln.set_param(d);
cuda::dev_linear_t<double, cuda::dev_sgd_t>     l1;   l1.set_param(d, hidden);
cuda::dev_silu_t<double>                        act;
cuda::dev_linear_t<double, cuda::dev_sgd_t>     l2;   l2.set_param(hidden, d);
cuda::dev_mse_loss_t<double>                    loss;

l1.upload_weight(w1);  l1.upload_bias(b1);            // 参数从主机搬上来
l2.upload_weight(w2);  l2.upload_bias(b2);
l1.set_lr(0.02);  l2.set_lr(0.02);  ln.set_lr(0.02);

cuda::dev_matrix_t<double> x(d, t, host_x), target(d, t, host_target);

for (int step = 0; step < n_steps; ++step) {
    // 前向：每层缓存反向需要的东西（layernorm 存 hx/std、linear 存输入）
    auto pred = l2.forward(act.forward(l1.forward(ln.forward(x.const_leaf()))));
    loss.forward(pred);
    const double value = loss.loss(target);           // 会同步到主机

    // 反向：损失层收 target（不是梯度），其余层收上游梯度
    cuda::dev_matrix_t<double> delta = loss.backward(target);
    delta = l2.backward(delta);
    delta = act.backward(delta);
    delta = l1.backward(delta);
    ln.backward(delta);

    l1.step();  l2.step();  ln.step();                // 只有梯度累积器在这里做事
}
```

优化器可以整体换：`dev_cache_updator_t<double, cuda::dev_nadam_t>`（本项目主机端的默认配置）
就是「先用 `update()` 累积若干 micro-batch，再 `step(param_buffer)` 一次性更新」。

**整个 LLaMA 模型也在设备上了**（9.9）。权重仍旧由主机端那套加载器解析，再整块搬上去 ——
设备端不重复实现权重解析；搬完之后设备端可以前向、可以增量解码、可以训练：

```cpp
#include "jas_cuda_llama.hpp"

using namespace jasmine;

// 主机端：老流程加载权重（已有黄金值测试）
llama_model_t<mat_t<double>, sgd_t> host;
host.set_param(n_layers, n_heads, d_model, d_ff, vocab, n_pos, n_kv_heads);
host.load(...);                                // jas_weight_io.hpp 那一套
host.finalize_after_load();

// 设备端：同一套配置，然后只搬数值
cuda::dev_llama_t<double, cuda::dev_sgd_t> dev;
dev.set_param(n_layers, n_heads, d_model, d_ff, vocab, n_pos, n_kv_heads);
dev.upload_from(host);                         // 逐矩阵搬运，形状不符立刻报错
dev.set_lr(0.01);

const mat_t<double> ids = /* 1 × T 的 token id —— 主机矩阵可以直接传，见 9.5 */;
cuda::dev_matrix_t<double> logits = dev.forward(ids);      // vocab × T

// 增量解码：prefill 一次，之后每步一个 token（位置来自 cache 长度）
dev.prefill(ids);                                          // 清空 cache 并整段喂入
cuda::dev_matrix_t<double> next = dev.forward_one(next_id); // next_id 是 1 × 1 主机矩阵

// 训练：forward → 损失 → backward（参数就地更新）→ step
cuda::dev_mse_loss_t<double> loss;
cuda::dev_matrix_t<double> target_dev(vocab, T, host_target);
for (int step = 0; step < n_steps; ++step) {
    cuda::dev_matrix_t<double> out = dev.forward(ids);
    loss.forward(out.const_leaf());
    dev.backward(loss.backward(target_dev));
    dev.step();
}
```

---

## 3. 为什么表达式模板能上设备

不是硬凑的，是因为它本来就把**算子逻辑**和**内存布局**拆开了。三个条件凑齐即可：

### 3.1 `work()` 天生是纯标量函数

每个表达式节点的核心就一行：

```cpp
JAS_HD auto work(lval_base_type i, rval_base_type j) const { return i * j; }
```

它不知道 `i`、`j` 从哪来、什么类型、在内存里怎么排。所以**同一个 `work()` 既能在 CPU 上逐元素求值，也能在 kernel 里被每个线程调用** —— 这就是「一套代码两个后端」的本钱。

标注只加了一个宏：`JAS_HD` 在 CUDA 编译下展开成 `__host__ __device__`，否则展开为空。**纯 CPU 构建完全不受影响**。

### 3.2 叶子要能在设备上取元素

主机叶子 `mat_t` 用 `new[]` 拿内存、`m_data` 指向主机地址 —— 设备端解引用它就是段错误。所以引入设备叶子 `dev_mat_t`：

- 只有「设备指针 + 维度 + 转置标志」，是个**薄壳**，内存归 `dev_buf_t` 管；
- 平凡可拷贝，于是能**按值**当 kernel 参数递进去；
- `operator()` **不做取模**（`mat_t` 为了容忍越界索引有 `r % row_num()`）。设备上每个线程都重算模运算纯属浪费，还会挡住合并访存。

**逻辑列数与存储前导维是分开的两个字段**（`m_cols` 与 `m_ld`）。这不是可有可无的灵活性：KV cache 按 `cap` 分配缓冲、只对外暴露前 `len` 列，一旦两者是同一个字段，这种视图就只能靠**拷贝**成紧凑缓冲来实现 —— 而 KV cache 存在的全部意义恰恰是不做这件事。`view()`（左上角子块）与 `make_dev_leaf_strided()` 是它的两个入口，行方向的零拷贝切片另有 `row_slice()`（逐头 RoPE 要用，见 8.3），细节见第 7 节。

标量操作数原来包成 `mat_t`（1×1，同样持有主机指针），这在设备端同样会解引用主机地址。已改成 POD 的 `scalar_leaf_t`，主机设备语义一致，也没有任何分配。

### 3.3 表达式树必须能被按值拷贝

CUDA kernel 的参数就是按值传的，而且必须平凡可拷贝。所以设备叶子要**按值拥有**，哪怕调用方传的是具名左值 —— 设备栈上不可能有主机对象的地址，引用毫无意义。

这靠一个定制点完成：

```cpp
template <typename T>
inline constexpr bool operand_owned_by_value = false;

template <typename T>
inline constexpr bool operand_owned_by_value<dev_mat_t<T>> = true;   // 设备叶子：按值
```

于是上一轮修好的「值类别存储策略」原封不动地复用了：左值借引用（主机零拷贝），右值按值拥有，设备叶子无论左右值都按值拥有。

#### 这条规则曾经漏了「表达式节点」，后果很隐蔽

规则最初只覆盖**叶子**。但接口形如 `softmax_rows(Expr const& x)` 时，传进来的具名表达式节点是**左值**，于是它被按引用借进了派生出的子树：

```cpp
auto named = d.leaf() * 2.0;      // 左值表达式节点
auto w = cuda::softmax_rows(named);   // x - m.leaf() 里借了 named 的引用
```

这棵树**能编译、能拷贝**，`std::is_trivially_copyable` 也**为真** —— 含引用成员的类照样平凡可拷贝，所以当时那道 `static_assert` 根本拦不住。但 kernel 参数是按值搬到设备上的，搬过去的是 `named` 的**主机栈地址**，设备端一解引用就是 `cudaErrorIllegalAddress`。

**修法**：把「按值拥有」的判据从「是不是设备叶子」放宽到「是不是设备可求值」——

```cpp
static constexpr bool owned_by_value =
    operand_owned_by_value<raw_type> || is_device_evaluable_v<raw_type>;
```

因为 `device_evaluable` 沿类型树递归传播，这条规则等价于：**凡是设备树，全程自持**。主机树不受影响（`mat_t` 不是设备可求值），零拷贝的卖点仍然成立。

**为什么用 `device_evaluable` 而不是「树里有没有引用」**：后者需要在每个节点上做成员级检查，C++ 没有反射；而前者已经在类型上传播好了，判据与「要不要上设备」天然一致。

### 3.4 一条编译期防线

`device_evaluable` 沿类型树递归传播：叶子写 `true`，节点与操作数做与运算，`mat_t` 这类主机独占类型连这个成员都没有，自然是 `false`。启动 kernel 前 `static_assert` 挡一道，把「拿主机表达式直接上设备」这类错误在**编译期**问出来，而不是运行时段错误。

针对上一节那个坑，防线还补了一条**自持性**检查：

```cpp
template <typename T>
inline constexpr bool is_self_contained_v = std::is_copy_assignable_v<T>;
```

含引用成员的类，隐式拷贝赋值会被删除 —— 拿它当「树里没有引用成员」的哨兵。这是个代理指标，但对本项目的表达式节点足够精确（成员只有操作数存储），而且**编译期**就把上一节那个运行期段错误扭成了构建失败。

> 这两道 `static_assert` 是本次开发最有价值的一处收获：`is_trivially_copyable` 看着像能保证「可以安全传给 kernel」，其实只保证「没有非平凡的特殊成员函数」，和「引用指向哪里」完全无关。

---

## 4. 为什么 `dot` / `softmax` 不能融合

融合逐元素 kernel 的模型是「一个线程算一个输出元素」。这要求每个输出只依赖同位置的输入。

矩阵乘不是这样：每个输出元素要跨一整行/一列求和，必须多线程协作 + 分块复用数据。硬塞进「每线程算一个元素」的 kernel，等于让每个线程自己走一遍内积，访存完全无法复用，性能会塌掉。

所以正确的做法是**在 `dot` 处切分**：逐元素链 → 融合 kernel，`dot` → cuBLAS。这正是主流框架张量表达式的调度方式。

`softmax` 另有一层原因：它的 `max` / `sum` 统计量在**构造时于主机上算好**，而 `max()`/`sum()` 是主机独占的归约。对一个设备矩阵调 `softmax()`，会在主机上解引用设备指针。所以它既不能融合，也不该被构造出来。

这两类节点因此刻意**不标** `device_evaluable`（`mat_dot_t` 里显式写了 `= false`，免得日后有人「顺手」补上），含它们的表达式会在编译期被 `static_assert` 挡住。

> 注意区分：**`mat_softmax_t` 这个节点**不能用于设备（理由如上），但「在设备上做 softmax」是另一回事 —— 见第 5 节，走归约 kernel 而不是靠节点融合。`dot` 同理，见第 6 节。

---

## 5. 归约：softmax 与归一化层

逐元素算子能融进「一个线程管一个输出」的模型，归约不能 —— 它要把一整行/一整列汇总成一个数，必须跨线程协作。所以归约走**块内归约**（warp shuffle + 共享内存），与前面那套正交地组合。

组合点在**读**那一侧：归约 kernel 读的是**整棵表达式树**，于是 `row_sum(exp(x - m))` 里 `exp` 被融进了求和的同一趟，指数结果一个都不物化。对访存受限的 P4 来说，省下的正是最贵的那部分。

### 5.1 两个轴都要，因为项目里两处用法方向相反

这套代码库的约定是**特征维在行方向、序列/批次在列方向**，于是两个轴都有真实用途 —— 一开始我差点只做了逐行那一半：

| 用途 | 归约方向 | 结果形状 | 对应主机端 |
| --- | --- | --- | --- |
| 注意力 softmax | 逐行（沿 key 方向） | (seq × 1) | `hsoftmax` / `hmax` / `hsum` |
| LayerNorm / RMSNorm | 逐列（沿 d_model 方向） | (1 × T) | `layer_norm_net_t` / `rms_norm_net_t` 里的 `vmean` |

`attn_scores` 是 (seq × seq)，掩码规则是 `j > i` 置 `-inf`，softmax 沿 **j** 走；而归一化层的 `gamma` 形状是 `[d_model, 1]`、统计量用 `vmean` 沿**行**算。两边方向正好相反，所以 `row_*` 和 `col_*` 都得有。

### 5.2 广播叶子：把「忽略哪个下标」变成类型信息

归约结果是「一行」或「一列」，要参与 (rows × cols) 的逐元素运算必须能广播。主机端靠 `mat_t::operator()` 里的取模实现，设备端**刻意不用取模** —— 那会让每个线程重算一遍模运算，还会挡住合并访存分析。改用两种专用叶子：

- `dev_col_leaf_t`：(rows × 1)，`operator()(r, c)` **忽略 `c`** → 沿列广播
- `dev_row_leaf_t`：(1 × cols)，`operator()(r, c)` **忽略 `r`** → 沿行广播

这不只是性能考量。若拿一个 (rows × 1) 的普通 `dev_mat_t` 去参与 (rows × cols) 表达式，`operator()(r, c)` 会按 `data[r * 1 + c]` 寻址，`c > 0` 时直接**越界读**。用忽略下标、无越界可能的专用类型，这个错误在类型层面就写不出来了。

两种叶子都存 `const T*`：它们是纯只读广播句柄，没有任何人会从它们写回（GEMM 的输出叶子另走 `dev_mat_t`，那里才需要可写指针）。

### 5.3 命名与主机端刻意不同（ADL 陷阱）

设备端的归约入口叫 `row_sum` / `col_sum` / `row_max` / `col_max` / `col_mean` / `sum_all` / `max_all` / `softmax_rows`，**不叫**主机端那些 `hsum` / `vsum` / `hsoftmax`。

原因不是口味：设备叶子 `dev_mat_t` 与主机端表达式都住在命名空间 `jasmine`，于是任何非限定的 `vsum(expr)` 都会经 ADL 把 `jasmine::vsum` 拉进候选集。实测中主机端那个重载被**静默**选中（主机 `vsum` 返回 `mat_t`，随后报出「`mat_t` has no member `leaf`」这种风马牛不相及的错），而不是给出重载歧义提示 —— 这类错误极难定位。用互不相同的名字从根上消除隐患，顺带 `row_*` / `col_*` 也比 `h*` / `v*` 直白。

### 5.4 数值契约：求和顺序变了，末位 ulp 必然不同

主机端是顺序累加，设备端是 warp shuffle 的树形归约，**求和顺序不同**，所以结果不会逐位相同。测试与示例因此一律用**相对容差**对拍，而不是要求位相等。这是设计上的取舍（用一点末位差异换并行度），不是缺陷。

另外 `max` 的单位元取 `lowest()`（与主机端 `max()` 的初值一致），于是「整行全是 `-inf`」时得到的是 `lowest` 而非 `-inf` —— 与主机端逐字相同。因果掩码下每行至少有一个合法位置，这种情况实际不可达。

### 5.5 全归约分两阶段，为了确定性

`sum_all` / `max_all` 先用多个 block 各算一段连续区间的部分和，再用一个 block 收尾。没有用「单 block 网格步长」（只能吃到 1 个 SM，对 20 个 SM 的 P4 是浪费），也没有用原子加（浮点原子加的顺序不确定，结果会随运行变化）。分块固定、归约树固定 → 结果确定，可复现。

### 5.6 `softmax_rows`：单趟共享内存路径 + 三趟回退

`softmax_rows` 有两条实现，按「整行能否放得进共享内存」自动选择：

| | 全局访存 | `exp` 次数 | 行长上限 |
| --- | --- | --- | --- |
| **单趟**（默认路径） | 读 1 遍 + 写 1 遍 | 每元素 **1** 次 | 受共享内存限制 |
| **三趟**（回退） | 读 3 遍 + 写 1 遍 | 每元素 2 次 | 无 |

三趟是 `row_max` → `row_sum(exp(...))` → 归一化，每趟都融合但各自要读一遍 x。单趟则是**把整行留在共享内存里**：表达式树只求值一遍写进共享，`max` / `exp`+求和 / 归一化全在片上做，`exp` 结果直接写回共享从而省掉第二次 `exp`。

两条路径的**数值结构刻意一致**（先减行最大值再求指数和，且逐线程跨步累加后做同一种树形归约），所以结果不会因为实现不同而分叉 —— 这是它们能互相回退的前提。

超出共享内存上限时回退而不是硬启动：`(cols + 32) * sizeof(T)` 超过设备允许的动态共享内存就直接走三趟，P4 上这个分界是 6112 个 `double` / 12256 个 `float`。恰好用满预算的那一行仍走单趟，`cudaFuncSetAttribute` 的 opt-in 在启动器里按需调用。

> 这里**没有**用 online-softmax。它的卖点是把三趟读降到两趟，但单趟版已经是**一趟**读；只有在「整行放不进共享内存」时 online 才有意义，而那时三趟回退已经够用、且没有 online 的数值重标定风险。online-softmax 真正不可替代的场合是 **fused attention**（softmax 与 `V` 的乘不物化中间概率矩阵），那需要绕过 cuBLAS 自己写注意力 kernel，属于另一个量级的改动 —— 见第 14 节第 6 条。

**一个必须显式检查的点**：优化的典型失败方式不是算错，而是**压根没生效**（阈值算错、分支写反），此时结果照样正确、测例照样全绿。所以启动器维护了一个 `softmax_shared_launch_count()` 计数器，测例断言「该走快路径时确实走了」；同时提供了 `softmax_max_cols_override()` 把快路径阈值压到 0，让**同一份输入**能两边各跑一遍再对拍 —— 否则快路径一上线，回退路径就再也没人测了。

---

## 6. `dot` 的设备分派：立即求值

主机端的 `.dot()` 返回一个**惰性**的 `mat_dot_t` 节点。这个设计在 CPU 上是对的 —— 节点能被并进更大的表达式树，由 `work()` 逐元素求值，省掉中间的矩阵物化。

设备端不能照搬，而且原因不是「还没实现」，是**它不该被那样用**：`mat_dot_t::operator()` 的实现是「每个输出元素自己走一遍 K 循环」。真把它融进逐元素 kernel，每个线程都会重算一遍整行内积，访存复用全部丢掉 —— 正确但性能塌方。

既然 GEMM 必然要单独执行一次，那就不该假装它是个惰性节点。所以设备端的 `.dot()` **立即求值**：

```cpp
// 主机端：返回惰性节点，可继续并进表达式
auto s = m_q.t().dot(m_k);

// 设备端：立即落成 cuBLAS，返回拥有显存的结果
auto s = dq.leaf().t().dot(dk);              // → dev_matrix_t<double>
```

于是两边写法几乎同形，语义差异（惰性 vs 立即）由类型系统显式承载：主机端返回 `mat_dot_t`，设备端返回 `dev_matrix_t`。测试里有编译期断言盯住这一条。

### 6.1 三种入口

| 写法 | 用途 |
| --- | --- |
| `a.leaf().dot(b.leaf())` | 叶子之间，最贴近主机端写法 |
| `A.dot(B)` | `dev_matrix_t` 之间；结果是拥有者，可继续链式 `.dot()` |
| `cuda::matmul(a, b, alpha)` | **操作数可以是任意设备可求值的表达式**：非叶子的会先被融合物化成临时矩阵，再进 GEMM |

`matmul` 的第三种能力是这里唯一比主机端更灵活的地方，也补上了「分派点必须可组合」这一环：

```cpp
auto y = cuda::matmul(x.leaf() + residual.leaf(), w.leaf());   // (x + residual)·W
```

结果 `dev_matrix_t` 用 `.leaf()` 就能回到表达式世界：

```cpp
auto scores = q.leaf().t().dot(k.leaf());
auto prob   = cuda::softmax_rows(scores.leaf() / scale + mask.leaf());
auto out    = v.leaf().dot(prob.leaf().t());
```

### 6.2 转置仍然只翻标志位

`.t()` 在设备端和主机端一样只是翻一个 `m_transposed`、不碰内存，所以 GEMM 的四种转置组合（`A·B`、`Aᵀ·B`、`A·Bᵀ`、`Aᵀ·Bᵀ`）都自动成立，前导维始终是**未经转置解释**的存储步长。内维校验读的是叶子报出的（已含转置的）形状，因此「同一块内存，直接乘该抛、转置后合法」这种边界也有专门用例盯着。

### 6.3 一处刻意的 `const_cast`

`dev_mat_t::m_data` 是 `T*`（因为 GEMM 的输出要写它），所以从 `const dev_matrix_t` 里取不出 `dev_mat_t<T>`。为此 `dev_matrix_t` 提供了一个 `const_leaf()`，里面是一次显式 `const_cast`。

它只在**语义上只读**的地方用（`gemm` 的 A/B 形参本就是 `const dev_mat_t<T>&`，只读 `m_data`，从不写），拿到它只应传给 GEMM 这类只读接口；要写仍然必须用非 const 的 `leaf()`。这比让所有 `.dot()` 都要求非 const 接收者要好 —— 后者会把「读操作」的 const 正确性代价转嫁给每个调用方。

---

## 7. 设备端 KV cache

多轮 decode 的状态就是 K/V。主机端的 `kv_cache_t` 与 `mat_mha_t` 已经把这套语义定义清楚了，设备端要做的不是重新设计，而是**在保持同一套契约的前提下让它零拷贝地跑起来**。有三处必须踩准。

### 7.1 零拷贝视图是前提，不是优化

主机端的 `keys()` 是零拷贝的：

```cpp
return m_k.view(0, 0, m_d, m_len);   // 形状 (d × len)，底层缓冲其实是 (d × cap)
```

设备端一开始**做不到**这件事 —— `dev_mat_t` 原来只有 `m_rows`/`m_cols`，而 `m_cols` 同时充当了「逻辑列数」和「存储步长」。于是「缓冲区按 cap 分配、只暴露前 len 列」这种视图根本表达不出来，每个 decode step 都得把已用部分拷成紧凑缓冲。

那正好把 KV cache 的意义抵消掉了：每步 O(len) 的拷贝乘以步数就是 O(len²)，把 attention 的访存瓶颈换成拷贝瓶颈。

修法是把两者拆开（`m_cols` 与 `m_ld`），并给 `dev_mat_t` 加一个 `view(rows, cols)` —— 指针与步长都不动，只改逻辑形状。

**为什么这样喂给 cuBLAS 不会读到越界的"空闲"列**：转置视图下每个矩阵槽被解释成 `(k × n)`，其中**被 ld 乘的那个下标只遍历逻辑列数**（≤ ld），所以最大偏移落在缓冲区实长之内。也就是说 `cap - len` 那段空间永远不会被 GEMM 碰到。四种槽位的推导这里不展开，结论有专门用例盯着（`ld=48, cols=16` 直接进 GEMM 并与主机端逐元素对拍）。

### 7.2 GQA：把「重复 append」从注释变成写不出来的代码

主机端 `jas_mha_t.hpp` 里记着一次真实事故：

> GQA 的 KV cache 必须「每个 KV 头 append 一次」。共享同一个 KV 头的多个 Q 头如果各自 append，同一份 K/V 会被写入两次：形状不报错、单步看似可用，但 cache 长度与内容全错，到多轮对话才炸。

设备端不靠注释提醒，而是**让 API 形状排除这种写法**：

- 多 KV 头容器 `dev_kv_caches_t` 的**唯一写入入口**是 `append_all(k_full, v_full)` —— 一次给**所有** KV 头追加同一批 token（`k_full` 形状 `(num_kv_heads × d_head, n_new)`，按行区间切给各头）；
- 压根没有「按单个 KV 头 append」的接口，所以「同一个头被写两次」不可达；
- 按 KV 头的访问只有 `const` 只读形式，供 Q 头读；
- `kv_head_of(q_head)` 由容器内部持有 `group_size`，调用方不必自己算分组（算错分组也是同一类隐蔽故障）。

于是「各 KV 头长度必然一致」成为结构性质而非待维护的不变量，测试里只需断言它确实成立。

### 7.3 契约与主机端逐字一致：RoPE 不在这一层

`append` 假定 **K 已经做过 RoPE**。这不是偷懒，是沿用主机端的切分方式 —— `kv_cache_t::append` 的注释原文就是「调用方保证已做 RoPE（K）」。所以 RoPE 是**在进 cache 之前**做的独立一步（第 8 节），不混在这一层里；`dev_kv_cache_t` 只管「给定已旋转的 K/V，零拷贝地存下来并参与 attention」。

好处是两件事可以各自单独测：cache 的容器语义以主机 `kv_cache_t` 为参照，RoPE 的口径以主机 `RoPE_net_t` 为参照，两者合起来还有一条端到端的 decode 用例。

### 7.4 扩容改成翻倍

主机端 `grow_to` 是 `m_cap + 64`，累计到 cap 要拷贝 O(cap²) 个元素。设备上这个拷贝走显存带宽，代价更实在，所以改成**翻倍**，摊销 O(cap)。prefill 那种「一次来一大段」的情形由 `max(need, cap * 2)` 一并覆盖。

代价是一条必须记住的约束：**扩容会重新分配缓冲区，先前取出的 `keys()` / `values()` 叶子随即悬空**（和 `dev_matrix_t::leaf()` 一样，薄壳不是稳定的）。decode 前 `reserve()` 一次就不再分配，也是推荐用法。`static_fixed` 模式下越界直接抛异常，不静默扩容。

### 7.5 `attend_cached`

```
scores  = qᵀ · K / sqrt(d_head)
weights = softmax_rows(scores)
out     = V · weightsᵀ        → (d_head × q_len)
```

1/sqrt(d_head) 直接折进 GEMM 的 `alpha`，省掉一趟逐元素缩放。

不传掩码的版本对 decode 是**天然正确**的：`q_len == 1` 时不存在"未来"，所以单 token 解码不需要任何掩码。prefill（`q_len > 1`）才需要一个 `(q_len × len)`、合法位置填 0、屏蔽位置填 `-inf` 的掩码 —— 而它由**调用方**构造，因为语义取决于 q 首列的绝对位置（`q_pos`），那属于调用方的上下文。

---

## 8. 设备端 RoPE

### 8.1 数学定义与主机端逐字一致

\[\theta_i = \frac{m}{10000^{2i/d}}, \quad
\begin{pmatrix} y_{2i} \\ y_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}\]

`i` 是**特征对**下标（特征下标 `n` 对应 `i = n/2`），`m` 是绝对位置。两种配对约定：

- `interleaved`：第 `i` 对是 `(2i, 2i+1)`（原始 RoPE / GPT-NeoX，本仓库既有实现，默认）
- `half_split`：第 `i` 对是 `(i, i + d/2)`（GPT-J / HuggingFace `LlamaRotaryEmbedding`）

**两者的 `θ_i` 完全相同**，只差「第 `i` 对是哪两行」—— 所以 cos/sin 表是同一张（第 8.2 节）。

> 位置 `m = 0` 时旋转是恒等变换，两种约定输出相同，因此**单 token、位置 0 的比对区分不出二者**；多 token 序列上必须显式指定。导出 LLaMA 系权重（HF 实现）时必须选 `half_split`。同理「`start_pos = 0`」也只有第 0 列是恒等，第 `j` 列的绝对位置是 `j` —— 这条单独有测例钉住。

### 8.2 三处与主机端刻意的实现差异

**一、不走 2×2 小矩阵乘。** 主机端把每对特征做成一个 2×2 旋转矩阵、再 `dot` 到 `(2 × 1)` 视图上 —— 那是为了复用矩阵设施。设备端一个线程直接读写两个元素就够了，走 GEMM 反而荒谬。

**二、`half_split` 不搬数据。** 主机端是「行重排 → interleaved 旋转 → 搬回去」（因为文档里已论证两者只差一个特征行置换）。设备端把这个置换折成 kernel 里的**行下标映射** `r0 = i`、`r1 = i + half`，一趟走完，省掉两次全量搬运。

**三、表在主机上算好再上传。** cos/sin 表只依赖「位置 × 维度对」，与数据无关，`reserve()` 时算一次就长期复用。用 libm 的 `std::pow` / `std::cos` 比在设备端实现更省心，也更容易和主机端对齐。表用 **double** 计算再落成 `T`：`θ_i` 的精度直接决定旋转角的精度，用 double 算常量是白拿的准确度（`RoPE_net_t<double>` 算的就是 double）。

### 8.3 逐头 + 行子块：真实用法

RoPE 作用在 `d_head` 上，而真实的 QKV 是打包成 `(num_heads * d_head × seq)` 的。所以要把打包矩阵按头切开 —— `dev_mat_t` 原本只有列方向的 `view()`（左上角子块），缺行方向的切片，于是补了 `row_slice(leaf, row0, rows)`：**只挪指针、列数与存储步长都不动**。转置视图下也成立（那时逻辑行对应存储的列，偏移退化成 `row0` 个元素）。

```cpp
// 逐头旋转到目标矩阵的对应行段，全程零拷贝
for (int g = 0; g < num_heads; ++g)
    rope.rotate_into(row_slice(qkv.const_leaf(), g * d_head, d_head),
                     row_slice(qkv_rot.leaf(), g * d_head, d_head), pos);
```

这就是 demo 里多轮 decode 的写法，也是 `rotate_into(x, out, pos)` 为什么存在：**目的地是一个行子块**，而不是一个新建的矩阵。

### 8.4 融合与原地旋转

`rotate_into` / `forward_at` 的输入可以是**任意设备可求值的表达式**，于是旋转能直接融进上游的投影/加法链，不必先落一个临时矩阵。

`forward_inplace(m, pos)` 对别名安全：每个线程只读自己那一对元素、再写回同一对，没有任何别的线程会碰它们（这也是 kernel 不需要任何同步的原因）。decode 路径上省掉一次分配。

### 8.5 表容量

和 KV cache 一样分 `dynamic`（自动扩容、翻倍）与 `static_fixed`（越界抛异常）两种模式。同样的一条约束：**扩容会重新分配表**——但表是**只读常量**、本类也不对外暴露它的叶子，所以不像 KV cache 那样会让先前取出的叶子悬空。

---

## 9. 设备端训练栈：反向、层、模型

前向能算不等于能训练。第 9 节补上的是**梯度**：让同一套设备端算子也能沿反向走一遍；
有了梯度之后，再把算子攒成**层**（9.5）、把层攒成**模型**（9.7–9.9）。

做法上有一条贯穿始终的原则：**公式逐字照抄主机实现**，包括那些看起来可以"顺手优化"的地方。
理由是训练是个反馈过程 —— 梯度差一个 1e-12 的量级，几十步之后就能把两条轨迹推到完全不同的地方
（后面 9.4 那条端到端用例就是拿这件事当验收标准的）。

### 9.1 反向其实是"广播 + 转置配对"

把主机端各层的 `backward` 摊开看，用到的设备端算子**前向就已经全都有了**，没有一个新的数学原语：

| 主机反向里的操作 | 设备端对应物 | 备注 |
| --- | --- | --- |
| `hsum(·)`（每行求和） | `row_sum` | 偏置梯度、norm 的 gamma 梯度 |
| `vsum(·)` / `vmean(·)`（每列求和/均值） | `col_sum` / `col_mean` | norm 的输入梯度 |
| `delta.dot(x.t())` / `w.t().dot(delta)` | `matmul` + 转置叶子 | 转置配对由叶子表达 |
| `w ⊙ (dy − row_sum(w ⊙ dy))` | `softmax_backward` | 见 9.2 |
| `delta * (s + x·s·(1−s))` | 融合逐元素 + `sigmoid` 节点 | SiLU，见 9.3 |
| `Rᵀ·delta`（RoPE） | 同一个旋转 kernel 换方向 | 见 9.3 |

所以**归约的反向（`hsum` 的反向）不是一个新算子，而是它的"形状"**：`hsum` 把 `(rows × cols)`
压成 `(rows × 1)`，反向就是把 `(rows × 1)` 的梯度**广播**回 `(rows × cols)` ——
而那正是 `dev_colvec_t::leaf()` 这个广播叶子在做的事。同理 `vsum` 的反向对应 `dev_rowvec_t`。

> 这解释了为什么第 5 节先做归约、第 9 节才做反向：归约一旦有了，反向就只剩"组装"。

### 9.2 `softmax_backward`：为什么必须要前向留下的概率矩阵

```
dx = w ⊙ (dy − row_sum(w ⊙ dy))
```

`w` 是**前向输出的概率矩阵**，必须由调用方留着 —— 主机端也是这么做的
（`hsoftmax_net_t::m_output` 是 public 成员，`mat_head_gen_t::backward` 直接读它）。
那个行和项就是 softmax「整行归一化」这个约束在反向里的体现：每个元素都受整行影响，
所以梯度里要减掉整行的加权和。这一项漏掉的话，梯度**仍然是个合理的形状、也仍然能下降**，
只是不收敛到正确的地方 —— 属于 9.4 里有限差分专门要抓的那类错误。

顺带一个易错点已经在前向踩过、反向照旧：设备端函数的命名是 `row_sum` / `col_sum`
而不是 `hsum` / `vsum`，`softmax_backward` 同理不叫 `hsoftmax_backward`（ADL 的坑见 5.1）。

### 9.3 RoPE 与 SiLU 的反向

**RoPE** 的主机反向实现是 `rope_mat.t().dot(delta_view)` —— 也就是把 2×2 旋转矩阵换成转置。
旋转矩阵是正交的，所以 `R⁻¹ = Rᵀ`：**反向不是"除以什么"，而是同一个旋转、反方向转**。
设备端因此不需要第二个 kernel，只在 `rope_rotate_kernel` 上加一个 `Inverse` 模板参数：

```
正向： out0 = c·a − s·b     ;    反向： dx0 =  c·g + s·h
      out1 = s·a + c·b           dx1 = −s·g + c·h
```

主机端 `RoPE_net_t::backward` **没有** `start_pos` 参数（它把列下标直接当绝对位置，
即隐含 `start_pos == 0`）。设备端把它显式化：`backward(delta)` 与主机行为完全对齐，
另加 `backward_at(delta, start_pos)` 供其它起点使用。

**SiLU** 的反向需要 sigmoid 参与三项（`s + x·s·(1−s)`）。`sigmoid` 已经是设备可求值的
表达式节点（`mat_sigmoid_t` 内部就用 `detail::device_exp`），所以可以直接融进融合 kernel；
但同一个 `exp` 在一棵树里出现三次会被求值三次，所以先把 `s` 物化一遍再用。
主机端反向是**重算** sigmoid（不缓存），设备端同理 —— 物化的只是同一次反向内部的复用。

### 9.4 两套独立的裁判，而不是一套

只跟主机对拍是不够的：**主机实现自己也可能是错的**，而"两份实现犯了同一个错误"
恰恰最难发现。所以每个梯度都跑两条互不相干的验证：

1. **对主机解析解**：公式逐字对齐（能抓实现与主机不一致 —— 符号、转置、广播方向）。
2. **有限差分**：`(L(x+ε) − L(x−ε)) / 2ε` 与解析梯度比。这条**不依赖任何一份解析反向的正确性**，
   纯从"前向是个求值函数"这个定义出发，抓的是**公式本身写错**。

有限差分刻意用**设备前向**来算 `L`，让两边算术一致 —— 否则容差要放到 1e-5 量级，
反而把真实错误一起放过去了。

> **这条纪律当场就抓到了一个真 bug**：`dev_mse_loss_t::loss` 最初写成
> `mean(y − target)`（漏了平方），而主机是 `mean(pow(y − target, 2.0))`。
> 它有下降趋势、形状也对、逐层梯度全部正常 —— 只有端到端那条对拍把差异暴露成
> `0.44 vs 2.07`。**如果只写"损失应该下降"这种自检，这个 bug 会一直活着。**

参数梯度没有直接返回值（主机是在 `backward` 里就地更新参数的），
所以对拍时用「初始参数 − 更新后参数，再除以学习率」反推 —— sgd 下这是精确值，
而且顺带把**参数更新本身**也一起验证了。

### 9.5 设备端层库 `jas_cuda_net.hpp`

到 RoPE 为止设备端攒下的是**算子**；能在 GPU 上训练需要的是**层** ——
层才有「前向留下什么、反向怎么算、参数怎么更新」这套结构。`jas_cuda_net.hpp` 提供：

| 设备端类 | 主机对应物 | 前向留下的东西 |
| --- | --- | --- |
| `dev_linear_t` | `weight_net_t` | 输入 `x`（算 dW 与 dx 都要） |
| `dev_layer_norm_t` | `layer_norm_net_t` | `hx`（不含 gamma）、`std` |
| `dev_rms_norm_t` | `rms_norm_net_t` | `hx`、`rms` |
| `dev_silu_t` | `silu_net_t` | 输入 `x` |
| `dev_gated_t` | `gated_net_t` | 两支的输出（SwiGLU） |
| `dev_residual_t` | `residual_net_t` | skip 连接 |
| `dev_mse_loss_t` | `mse_loss_t` | 输入（预测值） |

**协议与主机同形**（`forward` / `backward` / `step` / `set_lr` / `set_updator` /
`init_weight<init_t>` / `net_type` / `reinit`），因为主机端那套是鸭子类型 + C++20 concept，
设备端保持同名同签名就能直接复用那些编译期装置（比如 `is_reinitable_net` 靠
`reinit(std::vector<int>)` 判定，所以这个签名必须原样保留）。

三条必须守住的约定：

1. **梯度是返回值，参数更新发生在 `backward` 里。** 主机就是这样的
   （`weight_net_t::backward` 里先 `updator.update(...)` 再 return），设备端照搬；
   `step()` 只为梯度累积器存在。
2. **顺序不能动。** 主机用**更新前**的权重算输入梯度（`mat_t ret = m_weight.t().dot(delta)`
   是立即物化的），然后才更新权重。设备端同序，否则同一个输入喂两边会得到不同的数。
3. **输入要显式物化。** 设备端没有 `store_for_backward` 那种"左值拷贝/右值移动"的自动分派 ——
   拥有者走设备间 memcpy，叶子与表达式走融合 kernel，这一层由 `detail::materialize_into` 吸收。

一个不对称值得单独记：**主机 `mat_t` 既能当拥有者、又能直接进表达式；`dev_matrix_t` 不行。**
它刻意没有 `operator()`、也不是 `JAS_HD`（否则就不再是"缓冲区 + 尺寸"这种简单所有者），
所以它进不了表达式，必须先取 `leaf()` / `const_leaf()`。设备端各层的 `forward` 因此
都用模板 + `materialize_into` 吸收这个不对称，让调用方既能传 `dev_matrix_t`、也能传叶子和表达式。

这里补一句最容易被忽略的**接口约定**：`set_lr` / `set_updator` / `step` 这三个方法，
**无参数的层也必须提供**（空实现）。因为 `dev_chain_t` / `dev_gated_t` 这类容器会对每一个
成员统一调用一遍，缺一个就会在容器实例化时报「没有成员」——主机端的 `silu_net_t` 同样有空实现，
这不是设备端新加的规矩。

顺着这个不对称还有一条实用约定：**主机 `mat_t` 可以直接当设备端的输入**。
token id 本来就产自主机上的 tokenizer，测试里的参照数据也一律先在主机上生成，
所以各层的 `forward` 统一走 `detail::materialize_input(input, cache)` 做分流 ——
是主机矩阵就 upload，是设备叶子/拥有者/表达式就 `eval_fused`。这比在几十个调用点
各写一次 upload 更不容易漏，代价只是一次 H2D 拷贝。

### 9.6 设备端优化器

`jas_cuda_updator.hpp` 给出 `dev_sgd_t` / `dev_adam_t` / `dev_nadam_t` / `dev_cache_updator_t`。

一个结构性差别：主机端更新器是**纯数值对象**（`update(grad, mat)` 直接对 `mat_t` 做算术），
设备端必然是**主机侧对象 + 显存缓冲**（adam 的 `m_m` / `m_v`），`update` 是一次 kernel 启动 ——
和 `dev_matrix_t` 的定位一样：主机侧是"拥有者 + 启动器"，设备侧只有数据。

这里**没有**用 `eval_fused`，而是自己开 kernel：`eval_fused` 把结果写进**新分配的**缓冲，
而参数更新是**原地**写回已有参数（权重缓冲的地址不能变，否则之前取出的叶子全部悬空 —— 
KV cache 那套教训）。顺带一个好处：一阶矩、二阶矩、偏差修正、参数写入能合成**一个** kernel、
一趟显存读写；主机端那套写法会物化 `m_hat` / `v_hat` 等一串临时矩阵。

公式逐字照抄主机，包括**偏差修正的更新时机**（先累乘 `beta^t` 再取 `1 − beta^t`）。
nadam 需要 `b1_t` 本身（`1 − b1_t·b1` 这一项无法由 `1 − b1_t` 反推），所以 `b1t` 也一并传进 kernel
而不是在设备端用 `pow` 重算。

### 9.7 设备端注意力层 `dev_mha_t`

`jas_cuda_mha.hpp` 分两层：单头 `dev_head_gen_t`（对应主机 `mat_head_gen_t`）与多头外壳
`dev_mha_t`（对应 `mat_mha_t`）。前向几乎全是**拼装**：Q/K/V 各一次 `dev_linear_t`，
按行切成头，每头内部走「RoPE → `Q·Kᵀ` → 掩码 → softmax → `·V`」，最后拼回去过 `W_O`。
已有的零件一个不少，所以这一层的工作量主要在**边界**上：

- **掩码不物化。** 主机端用 `mat_t` 乘一个叠加了 `-inf` 的掩码矩阵；设备端若照做，就要为
  每个头分配一个 `(q_len × len)` 的浮点矩阵。改成在算完打分后由一个 kernel 就地写 `-inf`，
  概率矩阵**必然**会物化（它是 softmax 的输出、反向要用），但掩码自身不会。
- **反向要按 GQA 归并。** 这是整个设备端 MHA 里唯一一处「不是逐字照抄」的地方：GQA 下
  多个 Q 头共享一个 KV 头，所以 `delta_k` / `delta_v` 必须**按行累加**到同一个 KV 头段上
  （主机 `add_rows` 那一步），而 `delta_q` 是每个 Q 头独占自己的行段、可以**覆盖**写。
  写反的后果很不对称：该覆盖的写成累加会得到毫无意义的梯度，该累加的写成覆盖则只剩
  最后一个 Q 头的贡献 —— 两者形状都对，只有差分能抓。
- **decode 路径的 RoPE 要逐 KV 头做。** 训练路径没有这个问题（旋转发生在单头核内部），
  但 `forward_one` 是先在完整 K 投影（`n_kv_heads × d_head` 行）上旋转再进 cache 的，
  而旋转表只有 `d_head` 维：整块旋转会把**相邻两个头的前半行配对**。单 KV 头时恰好等价，
  所以这个 bug 只在 GQA/MQA 上现形 —— 实测就是这么被 `MhaForwardOneMatchesHostWithKvCache`
  抓到的（`输入行数 8 与 d 4 不一致` 只是它连形状都没对齐的第一层症状）。

### 9.8 设备端 Embedding 层 `dev_embedding_t`

前向是 gather（每个 token id 取 `W` 的一列），反向是 scatter —— 一行代码的量级，
难点全在**反向怎么写**。主机端每步分配一个 `[d_model × vocab]` 的稠密梯度矩阵，
设备端照抄的话显存直接乘以词表大小（`d_model × vocab × 8` 字节，7B 级别就是 260 MB × 层数），
而每步真正被更新的只有 batch 里出现的那几个 id。

设备端因此改成**按 id 的原子累加**：一个稠密缓冲复用（不是每步新建），先 `zero()`，
再让每个出现过的 id 用 `atomicAdd` 把自己的梯度加回对应列。两个后果要记住：

- **重复 id 会被正确累加**（同一个 token 在一段里出现多次，梯度必须叠加），
  这正是 `atomicAdd` 存在的理由，也是测试里专门造重复 id 的原因。
- **`zero()` 不能省。** 稀疏写入留下的旧值会被当成梯度喂给优化器。缓冲复用是显存上的必要选择，
  代价就是这一步「清空」必须由本层自己负责，不能指望调用方。

id 的越界检查放在**主机侧、上传之前**：一个越界的 id 在 kernel 里就是一次非法访存
（`cudaErrorIllegalAddress`，且后续所有 CUDA 调用都会连带失败，极难定位），
而查一遍 id 只有 `1 × T` 个元素。

### 9.9 设备端整模型 `dev_llama_t`

`jas_cuda_llama.hpp` 把 9.7 / 9.8 与 9.5 的层按 `llama_model_t` 的层序拼起来：
`wte → block[0..N-1] → ln_f → lm_head`，其中每个 block 是
`x + attn(rms_1(x))` 后接 `h1 + down(silu(gate(rms_2(h1))) ⊙ up(rms_2(h1)))`。
两条路径都保留：`forward`（整段、因果掩码、不动 cache）与 `forward_one`（增量解码、KV cache、位置来自 cache 长度）。

三个设计决定值得单独记：

1. **权重加载留在主机。** 主机端已经有一整套经过黄金值验证的加载器（`jas_weight_io.hpp`、
   `finalize_after_load()`）和 `bind_rope()` 依赖的进程级单例 `rope_registry_t`；
   照搬到设备端只是多出一份要维护的解析逻辑。所以流程是
   `llama_model_t 加载 → dev_llama_t::upload_from(host) → 设备端前向/训练`，
   而「搬过去的数值对不对」是可以逐元素对拍的 —— 这也正是 `test_cuda_llama.cu` 的第一条用例。
2. **残差加法写成显式的 `eval_fused(a + skip, ...)`，不套 `dev_residual_t`。** 这一层要同时
   缓存两份 skip（注意力支与 FFN 支），套容器反而要在外面再包一层显式加法。
   `dev_residual_t` 本身仍然可用、也仍然有独立用例 —— 这是「哪个更好读」的选择，不是能力缺失。
3. **`upload_from` 是模板（`template <typename HostModel>`）。** 主机模型的类型是「更新器也是
   模板参数」的另一半，把签名钉成 `llama_model_t<mat_t<T>, updator_type>` 会逼调用方把两边的
   更新器凑成同一个类型 —— 而权重搬运跟主机端用什么更新器毫无关系。

`forward_one` 只返回**最后一个位置**的 logits（与主机同义），靠 `col_slice` 零拷贝取尾列，
不必先拷一份紧凑副本。

---

## 10. 行优先 ↔ 列优先

cuBLAS 只认列优先，而本项目一律行优先。做法是不复制数据，靠「转置同一块内存」对齐：

> 行优先存储的 W（前导维 `m_cols`）在 cuBLAS 眼里就是一个列优先的 Wᵀ。

于是要算行优先的 `C(M×N) = opA(A) · opB(B)`，转到列优先就是算 `Cᵀ = Bᵀ·Aᵀ`，也就是 **A 和 B 要交换位置、M 和 N 也要交换**：

```
cublasXgemm(handle, opB, opA, N, M, K, ..., B, ldb, A, lda, ..., C, ldc)
```

- 不转置 → `op = CUBLAS_OP_N`（那个列优先视图正好就是要的转置）
- 转置 → `op = CUBLAS_OP_T`（再转回来）
- 前导维一律用**未经转置解释的存储步长** `leading_dim()` —— 转置只改索引解释，内存布局没动。它**可以大于逻辑列数**（KV cache 的子视图就是这样，见 7.1），此时不能拿 `col_num()` 当 ld 用。

四种转置组合都有对拍测试，其中 `Q·Kᵀ`（注意力打分）单独一个用例，因为它是最容易搞反的一处。

---

## 11. 目标机与测试机

先分清两台机器，因为它们的约束完全不同：

| | 测试机（本机） | 目标机 |
| --- | --- | --- |
| GPU | Tesla P4，**无风扇**，sm_61 | Ampere 及以后（有 Tensor Core、支持 TF32/bf16/fp16） |
| 显存 | 8 GB | 远不止，量化/分页 KV cache 的紧迫性低 |
| 主要约束 | **散热**、CUDA 版本 | **精度可复现性**（见 11.3） |

所以下面 10.1 / 10.2 / 10.5 是**只在测试机上成立**的约束，10.3 是**只在目标机上才会出现**的坑。

### 10.1 必须用 CUDA 12.4，不能用 13.x（测试机）

**CUDA 13 起移除了 Pascal 支持**，而本机 `/usr/local/cuda` 默认指向 13.2。用它编译 `sm_61` 会直接失败。

CMake 已经处理：`JASMINE_USE_CUDA=ON` 时会主动去找一个还支持目标架构的 toolkit（按 `12.6 → 12.5 → 12.4 → 12 → /usr/local/cuda` 顺序），并显式设置 `CMAKE_CUDA_COMPILER`。需要覆盖时：

```bash
cmake -S . -B build-cuda -DJASMINE_USE_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
      -DJASMINE_CUDA_ARCHITECTURES=61
```

### 10.2 散热（测试机，重要）

本机只有一张 **Tesla P4，没有风扇**，靠机箱风道散热，TDP 75W。长时间满负载会持续升温。

针对这一点，CUDA 测试做了三层保护：

1. **默认用例都很小**：逐元素最大 512×512，GEMM 最大 256³。单次 kernel 亚毫秒级，整个 `cuda_tests` 跑完约 2 秒，结温纹丝不动（实测 44℃）。
2. **温度守门**：每个用例前后测温，超过 80℃ 直接 `skip` 而不是硬跑。温度也会打印出来，方便观察趋势。
3. **算力型用例默认关闭**：4096×4096 融合、256³ GEMM 这些需要显式打开：

```bash
JASMINE_CUDA_STRESS=1 ./build-cuda/tests/cuda_tests
```

实测这两个压力用例合计约 0.5 秒，跑完 45℃。**只要不是反复循环跑，就不会有问题。**

### 10.3 TF32：目标机上才会出现的精度漂移（关键）

Ampere 及以后的卡上，**单精度 GEMM 可以选择走 TF32 张量核** —— 指数位仍是 8 位，但尾数只剩 **10 位**，相对误差约 1e-3。而开不开取决于 **math mode + `NVIDIA_TF32_OVERRIDE` 环境变量**，也就是说**同一份代码在不同机器上会给出不同数值**。

对本项目这是个直接的陷阱：整套测试的价值就在于「设备结果与主机参考逐元素对齐」，float 对拍用的是 1e-5 量级的相对容差，TF32 的 1e-3 一冲就没了。更糟的是它在测试机上**看不出来**（P4 是 Pascal，根本没有 TF32），到目标机上才发现基准线不对。

所以默认把精度钉死：

```cpp
enum class gemm_math
{
    precise, // CUBLAS_PEDANTIC_MATH：真 FP32/FP64，跨机器可复现  ← 默认
    tf32,    // CUBLAS_TF32_TENSOR_OP_MATH：Ampere+ 走 TF32，快但尾数少 13 位
};
```

- 句柄创建时就把 math mode 设成 `CUBLAS_PEDANTIC_MATH`，不依赖环境变量；
- 想要速度显式调 `cuda::set_gemm_math(cuda::gemm_math::tf32)`；
- 在**不支持 TF32 的卡上请求 `tf32` 会抛异常**，而不是静默降级成 FP32 —— 静默降级会让你在测试机上"验证过"一个在目标机上根本没生效的加速，却毫无提示；
- `double` 的 `Dgemm` 不受 TF32 影响，两种模式下都是真双精度。

用例 `CudaKvCacheTest.GemmMathModeDefaultsToPrecise` 在 Pascal 上断言"请求 TF32 必须抛异常"，在 Ampere 上则断言切换生效 —— 同一份测试覆盖两种机器。

### 10.4 后续 GPU 选型的含义

目标机是 Ampere 及以后，于是：显存不再是 8 GB，**量化/分页 KV cache 的紧迫性下降**（当前实现是 `reserve` + 翻倍扩容，够用）；而 TF32 / bf16 / fp16 的混精度路径是真实可选项 —— 设备叶子是模板、`gemm` 目前只特化 `float`/`double`，混精度要动的是那里。

### 10.5 Pascal 的算力特性（测试机）

P4 是 Pascal（sm_61），没有 Tensor Core，也没有 TF32。所以：

- GEMM 走 FP32 的 `cublasSgemm` / `cublasDgemm`，**不要指望 Tensor Core 级别的吞吐**；
- 逐元素融合是**访存受限**的（P4 显存带宽约 192 GB/s），融合省下的是中间结果的显存往返，这在带宽受限的卡上恰恰是最大的收益来源。

---

## 12. 测试

```bash
# 主机端回归（必须仍然全绿）
cmake --build build -j && ./build/tests/unit_tests

# CUDA 后端
cmake --build build-cuda -j
./build-cuda/tests/cuda_tests                    # 快，不发热
JASMINE_CUDA_STRESS=1 ./build-cuda/tests/cuda_tests   # 含算力型用例
```

`cuda_tests` 覆盖：

- **类型契约**（零发热，纯编译期）：设备叶子平凡可拷贝、`mat_t`/`mat_view_t` 不可上设备、设备叶子按值拥有、`dot`/`softmax` 不被误标、广播叶子满足 `is_matrix`、**具名（左值）设备表达式节点也自持**、**主机树仍然借引用**（防回归）。
- **前导维**（零发热）：逻辑列数与存储步长可不等、`operator()` 走 `m_ld` 而不是 `col_num()`（用「每个元素填自己的下标」的缓冲区让读值等价于地址）、`view()` 只改逻辑形状、转置不改进步长、叶子仍平凡可拷贝。
- **融合求值**：四种算术、标量（左/右/两侧/整型）、`exp`/`sigmoid`、比较、五层深链、转置视图、float32、不同 block 大小、单元素/单行/单列边界。
- **cuBLAS**：方阵、非方阵（M/N/K 互不相等）、转置 A、转置 B（`Q·Kᵀ`）、双转置、`beta` 累加、形状不匹配报错、float32、融合段与 GEMM 拼接。
- **归约**：`row_sum`/`row_max`/`col_sum`/`col_max`/`sum_all`/`max_all` 与主机端逐元素对拍，含**轴不得搞反**的专项用例（形状不对称 + 行列和各不相同）、融合归约（`row_sum(exp(x*2))`）、float32、单行/单列/单元素边界。
- **softmax**：与主机 `hsoftmax` 对拍、每行和为 1、大输入（1000 量级）的数值稳定性、**`-inf` 因果掩码**（未来位置必须恰好为 0 且整行不出现 nan）、表达式输入、具名表达式输入。**两条路径（单趟 / 三趟回退）各跑一遍同一份输入对拍**，并用启动计数器断言「该走快路径时确实走了」；另有「恰好用满共享内存预算的行仍走单趟」与「超预算自动回退、不启动失败」两条边界。
- **RoPE**（`test_cuda_rope.cu`）：以主机 `RoPE_net_t` 为参照（而不是另写一份公式 —— 测的是**口径一致**而非公式复述），覆盖两种配对约定、`start_pos != 0`（decode 的命门）、单列/多列、float32、转置输入、表达式输入融合、原地与外置结果逐位相同。另有两条**不依赖参考实现**的性质检查：第 0 列在 `start_pos = 0` 时必须恒等、旋转必须保持每对特征的模长。`row_slice` 的地址运算用主机缓冲零发热验算（含转置视图）。最后一条端到端用例把 **RoPE + KV cache + attention** 串起来与主机侧同一条链对拍。
- **归一化层**：`layer_norm` / `rms_norm` 与主机 `layer_norm_net_t` / `rms_norm_net_t` 对拍（含非平凡 gamma/beta）、每列零均值、RMSNorm 的尺度不变性与非平移不变性、表达式输入。
- **`dot` 分派**（`test_cuda_dot.cu`）：`.dot()` 在设备端立即求值（编译期断言返回类型是拥有者而非节点）、`mat_dot_t` 仍被显式拒绝上设备、`matmul` 三入口（叶子/拥有者/表达式操作数）、四种转置组合、链式 `.dot()`、`alpha` 缩放、结果回灌表达式、内维不匹配报错、**转置后形状参与校验**、float32。
- **注意力端到端**：`GEMM → 缩放/加掩码 → 逐行 softmax → GEMM` 拼起来与主机算式对拍（两份：`gemm_to_host` 版与自然 `.dot()` 版），外加一条不依赖参考实现的**因果性**行为检查（改动第 t 个 token 之后的内容，前 t 个位置的输出必须一字不变）。
- **KV cache**（`test_cuda_kv_cache.cu`）：以主机 `kv_cache_t` 为参照做**容器级对拍**（同一串 append 后 `keys()`/`values()` 的长度与内容逐元素一致 —— 这直接验证前导维被用对了）、`keys()` 的 `leading_dim()` 等于容量且未扩容时指针稳定、翻倍扩容跨过重分配后内容不错位、`static_fixed` 越界抛异常且长度不变、形状/尺寸校验；注意力侧覆盖 **12 步 decode 逐步对拍**、**带因果掩码的 prefill 对拍**（`q_pos != 0`）、掩码形状错误必须报错、head_dim 不匹配报错、**GQA（4 Q 头共享 2 KV 头）**下各 KV 头长度恒等 + 每头内容 + 每个 Q 头输出对拍、非法配置（不整除/下标越界/行数不符）报错。另有 float32 decode 与 GEMM 精度模式用例。
- **反向传播**（`test_cuda_backward.cu`）：更新器 `sgd`/`adam`/`nadam`（各 6~12 步）与梯度累积器逐元素对拍主机；`silu`/`layer_norm`/`rms_norm`/`linear`/`rope`/`softmax` 的反向**同时**对主机解析解与**有限差分**（两条互不相干的裁判，理由见 9.4）；RoPE 另有「转置旋回原值」（两种配对约定 × `start_pos ∈ {0, 3}`）；参数梯度用「初始参数 − 更新后参数，除以学习率」反推，顺带验证更新语义；最后一条端到端用例把 `LayerNorm → Linear → SiLU → Linear → MSE` 整栈训练 6 步，**逐步**与主机同构栈对拍损失与参数（第 0 步容差 1e-12，之后放宽到 1e-7 —— 训练是反馈过程，容差不放宽测的就是混沌而不是正确性），外加一条「与主机无关」的自检：60 步后损失确实下降。
- **设备端 MHA / Embedding**（`test_cuda_mha.cu`）：单头前向/反向/`attend_cached` 三项与主机
  `mat_head_gen_t` 逐元素对拍（两种 RoPE 约定 × 有无掩码）；多头外壳在 MHA / GQA / MQA 三种配置下
  对拍主机；反向既比输入梯度（`dq/dk/dv`）也比**参数梯度**（用「初始 − 更新后，除以学习率」反推，
  覆盖 `W_Q/W_K/W_V/W_O` 与 `b_Q`）；外加一条**独立于主机**的有限差分；decode 路径
  （prefill + 增量）与主机对拍。Embedding 前后向对拍（含**重复 id** 的原子累加）与越界 id 必须报错。
- **设备端整模型**（`test_cuda_llama.cu`）：
  1. 参数搬运后**逐层**对拍主机 `forward_stages` 与最终 logits（搬错一个矩阵立刻红）；
  2. 增量解码（prefill + 逐 token）对拍主机，以及一条**不依赖主机**的自洽检查：
     整段前向 == 逐 token 前向后取最后一列；
  3. 反向用**有限差分**验收 —— 主机 `llama_model_t` 是纯推理实现、没有 backward 可以对照，
     所以这一条是唯一的裁判。差分**覆盖每一个参数矩阵**（2 层 × 9 个 + `wte` + `ln_f` + `lm_head`），
     而不是只钉头/中/尾三处（理由见 9.7 那条 GQA 归并）；
  4. 训练收敛自检：同批数据跑 60 步，损失 0.69 → 3e-5。
- **压力**（默认跳过）：4096² 融合、256³ GEMM。

数值对拍一律用**相对容差**，原因见 5.4。

---

## 13. 文件一览

| 文件 | 职责 |
| --- | --- |
| `jas_cuda_compat.hpp` | `JAS_HD` / `JAS_DEV` 宏、设备安全数学（`device_exp` / `device_max` / `device_sqrt`）、`device_evaluable` 探测。**不依赖 CUDA 运行时**，纯 CPU 构建也能 include |
| `jas_cuda_leaf.hpp` | 设备叶子 `dev_mat_t`（薄壳、转置视图、**独立前导维 + `view()` 零拷贝子视图**、**`row_slice()` 行子块**）、按值拥有定制点、`is_dev_leaf` 判别、`.dot()` 的声明。同样不依赖运行时 |
| `jas_cuda_buffer.hpp` | 运行时基础设施：错误检查、设备查询、`dev_buf_t`、`pinned_buf_t`、`sync()`、**动态共享内存上限查询** |
| `jas_cuda_matrix.hpp` | `dev_matrix_t`：把「管内存的 buf」和「薄壳 leaf」绑成所有者；`const_leaf()` |
| `jas_cuda_fused.hpp` | 融合逐元素 kernel 与启动器（含「不可上设备」的编译期断言） |
| `jas_cuda_gemm.hpp` | cuBLAS GEMM、`matmul` 三入口、`.dot()` 的定义（含行优先映射与转置组合）、**精度的 math mode 控制（`gemm_math` / `set_gemm_math`）** |
| `jas_cuda_reduce.hpp` | 广播叶子、`dev_colvec_t`/`dev_rowvec_t`（含**不广播的 `flat_leaf()`**，更新器要用）、归约 kernel、`softmax_rows`（**单趟共享内存 + 三趟回退**）/ `softmax_backward` / `layer_norm` / `layer_norm_backward` / `rms_norm` / `rms_norm_backward`、**反向所需的缓存结构 `layer_norm_cache_t` / `rms_norm_cache_t`** |
| `jas_cuda_kv_cache.hpp` | `dev_kv_cache_t`（单头，零拷贝视图 + 翻倍扩容）、`dev_kv_caches_t`（多 KV 头 + GQA 映射，`append_all` 是唯一写入入口）、`attend_cached` |
| `jas_cuda_rope.hpp` | `dev_rope_t`：cos/sin 表上设备、两种配对约定（`rope_pair_layout`）、`start_pos` 偏移、表达式输入融合、原地旋转、**反向（同一个 kernel 的 `Inverse` 分支）** |
| `jas_cuda_updator.hpp` | 设备端优化器：`dev_sgd_t` / `dev_adam_t` / `dev_nadam_t` / `dev_cache_updator_t`。参数更新走**原地** kernel（不是 `eval_fused`），一趟算完动量、偏差修正与写入 |
| `jas_cuda_net.hpp` | 设备端层库：`dev_linear_t` / `dev_layer_norm_t` / `dev_rms_norm_t` / `dev_silu_t` / `dev_gated_t` / `dev_residual_t` / `dev_chain_t` / `dev_mse_loss_t`，全部 forward + backward，协议与主机同形；`detail::materialize_input` 统一吸收「主机 `mat_t` / 设备叶子 / 拥有者 / 表达式」四种输入 |
| `jas_cuda_mha.hpp` | 设备端注意力：`dev_head_gen_t`（单头，含 RoPE、就地掩码、`attend_cached`）与 `dev_mha_t`（多头外壳，GQA 梯度归并、`forward` / `forward_one` / `backward`） |
| `jas_cuda_embedding.hpp` | `dev_embedding_t`：离散 gather + 按 id `atomicAdd` 的稠密梯度缓冲（跨步复用，`gradient_bytes()` 供调用方估算显存代价） |
| `jas_cuda_llama.hpp` | `dev_llama_t` / `dev_llama_block_t`：整模型的设备端实现（`forward` / `forward_stages` / `forward_one` / `prefill` / `backward`），权重经 `upload_from(host)` 从主机搬运 |

主机端被改动的文件：三处加注 `JAS_HD`、`jas_mat_express_t.hpp` 里的标量叶子（`scalar_leaf_t`）、「按值拥有」判据（见 3.3）、`mat_dot_t` 显式标注 `device_evaluable = false`、以及 `is_mat_dot` 判别 trait。

---

## 14. 下一步

1. ~~归约 kernel~~ ✅ 已完成（第 5 节）：`row_*` / `col_*` / `sum_all` / `max_all`，以及建立其上的 `softmax_rows` / `layer_norm` / `rms_norm`。
2. ~~`dot` 接上 cuBLAS 调度~~ ✅ 已完成（第 6 节）。
3. ~~设备端 KV cache（零拷贝视图 + GQA + decode/prefill）~~ ✅ 已完成（第 7 节）。
4. ~~设备端 RoPE~~ ✅ 已完成（第 8 节）：cos/sin 表上设备、两种配对约定（`interleaved` / `half_split`）、逐头行子块旋转、表达式输入融合、原地旋转。**「设备端自包含的多轮 decode」到这一轮才闭环** —— `test_cuda_rope.cu` 里有一条 RoPE + KV cache + attention 串起来的端到端对拍。
5. ~~softmax 专用 kernel（性能）~~ ✅ 已完成（5.6）：单趟共享内存路径把「读 3 遍 + `exp` 2 次」降到「读 1 遍 + `exp` 1 次」，整行放不进共享内存时回退三趟。这里刻意**没有**用 online-softmax，理由见 5.6。
6. **fused attention（Flash Attention 那一类）**：attention 现在仍是 `GEMM → softmax（物化概率矩阵）→ GEMM` 三步，中间那个 `(q_len × len)` 概率矩阵要落一遍显存。5.6 里解释过 online-softmax 真正不可替代的场合正是这里 —— 把 softmax 与 `V` 的乘融进一趟，中间矩阵根本不物化。代价是要绕过 cuBLAS 自己写注意力 kernel，并处理分块与在线重标定，属于另一个量级的改动；长上下文收益最大。
7. ~~反向传播的设备路径~~ ✅ 已完成（第 9 节）：`softmax` / `layer_norm` / `rms_norm` / `RoPE` 的反向，加上设备端层库（`jas_cuda_net.hpp`）与优化器（`jas_cuda_updator.hpp`）。归约的反向不需要新算子 —— `hsum` 的反向就是广播，`dev_colvec_t::leaf()` 那个广播叶子已经在做这件事。**「能在 GPU 上训练」到这一轮才闭环**：`test_cuda_backward.cu` 里有一条 `LayerNorm → Linear → SiLU → Linear → MSE` 整栈训练 6 步、逐步与主机同构栈对拍的端到端用例。
8. ~~把模型类真正搬上设备~~ ✅ 已完成（9.7–9.9）：`dev_mha_t` / `dev_embedding_t` / `dev_llama_t`，
   含 GQA 梯度归并、decode 路径、参数搬运（`upload_from`）与整栈反向。加载器仍留在主机端 ——
   设备端只搬数值，这样不必维护第二份权重解析逻辑。
   **仍未做的**：`gpt2_model_t` 的设备端对应物（结构同 9.9，工作量在「按 GPT-2 的层序再拼一遍」，
   `dev_embedding_t` 可直接复用：GPT-2 的 `wte` 与 `lm_head` 共享权重这一点要显式处理）。
9. **混精度（可选）**：目标机是 Ampere 及以后，TF32/bf16/fp16 是真实可选项。设备叶子已经是模板，要动的是 `gemm` 的类型特化与 math mode 的选择策略（见 10.3）。
