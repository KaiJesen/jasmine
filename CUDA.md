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
| 归约（`row_sum` / `col_sum` / `sum_all` / …） | ✅ 已实现（第 5 节） |
| 逐行 softmax（含 `-inf` 因果掩码） | ✅ 已实现（`softmax_rows`） |
| LayerNorm / RMSNorm | ✅ 已实现 |
| 单头注意力前向端到端 | ✅ 已实现（GEMM + 掩码 + softmax + GEMM，有对拍用例） |
| 反向传播的设备路径 | ❌ 尚未实现 |
| 整个模型的端到端 GPU 推理 | ❌ 尚未实现（缺 `mat_dot_t` 的 cuBLAS 调度与 KV cache） |

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

using namespace jasmine;

cuda::dev_matrix_t<double> a(rows, cols, ha);   // 分配 + 上传
cuda::dev_matrix_t<double> b(rows, cols, hb);
cuda::dev_matrix_t<double> c(rows, cols, hc);

// 逐元素：整条链一次 launch
auto y = cuda::eval_fused_to_host((a.leaf() + b.leaf()) * c.leaf());

// 矩阵乘：走 cuBLAS
auto s = cuda::gemm_to_host(q.leaf(), k.leaf().t());   // S = Q·Kᵀ

// 归约：整棵表达式树都能塞进去，exp 融进求和那一趟
auto row_totals = cuda::row_sum(exp(x.leaf() * 2.0));

// 逐行 softmax：掩码（0 / -inf）直接加进表达式
auto weights = cuda::softmax_rows(scores.leaf() / scale + mask.leaf());

// 归一化层：gamma 是 (d_model × 1) 的列向量，沿列广播
cuda::dev_colvec_t<double> gamma(d_model);
gamma.buffer().upload(host_gamma.data(), d_model);
auto normed = cuda::rms_norm(x.leaf(), gamma, 1e-5f);
```

归约结果的两种用法：`.download()` 回主机对拍，或 `.leaf()` 取广播叶子继续参与表达式运算。

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

这两类节点因此刻意**不标** `device_evaluable`，含它们的表达式会在编译期被 `static_assert` 挡住。

> 注意区分：**`mat_softmax_t` 这个节点**不能用于设备（理由如上），但「在设备上做 softmax」是另一回事 —— 见第 5 节，走归约 kernel 而不是靠节点融合。

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

---

## 6. 行优先 ↔ 列优先

cuBLAS 只认列优先，而本项目一律行优先。做法是不复制数据，靠「转置同一块内存」对齐：

> 行优先存储的 W（前导维 `m_cols`）在 cuBLAS 眼里就是一个列优先的 Wᵀ。

于是要算行优先的 `C(M×N) = opA(A) · opB(B)`，转到列优先就是算 `Cᵀ = Bᵀ·Aᵀ`，也就是 **A 和 B 要交换位置、M 和 N 也要交换**：

```
cublasXgemm(handle, opB, opA, N, M, K, ..., B, ldb, A, lda, ..., C, ldc)
```

- 不转置 → `op = CUBLAS_OP_N`（那个列优先视图正好就是要的转置）
- 转置 → `op = CUBLAS_OP_T`（再转回来）
- 前导维一律用**未经转置解释的存储步长** `leading_dim()` —— 转置只改索引解释，内存布局没动

四种转置组合都有对拍测试，其中 `Q·Kᵀ`（注意力打分）单独一个用例，因为它是最容易搞反的一处。

---

## 7. 这台机器上的注意事项

### 7.1 必须用 CUDA 12.4，不能用 13.x

**CUDA 13 起移除了 Pascal 支持**，而本机 `/usr/local/cuda` 默认指向 13.2。用它编译 `sm_61` 会直接失败。

CMake 已经处理：`JASMINE_USE_CUDA=ON` 时会主动去找一个还支持目标架构的 toolkit（按 `12.6 → 12.5 → 12.4 → 12 → /usr/local/cuda` 顺序），并显式设置 `CMAKE_CUDA_COMPILER`。需要覆盖时：

```bash
cmake -S . -B build-cuda -DJASMINE_USE_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
      -DJASMINE_CUDA_ARCHITECTURES=61
```

### 7.2 散热（重要）

本机只有一张 **Tesla P4，没有风扇**，靠机箱风道散热，TDP 75W。长时间满负载会持续升温。

针对这一点，CUDA 测试做了三层保护：

1. **默认用例都很小**：逐元素最大 512×512，GEMM 最大 256³。单次 kernel 亚毫秒级，整个 `cuda_tests` 跑完约 0.8 秒，结温纹丝不动（实测 42℃）。
2. **温度守门**：每个用例前后测温，超过 80℃ 直接 `skip` 而不是硬跑。温度也会打印出来，方便观察趋势。
3. **算力型用例默认关闭**：4096×4096 融合、256³ GEMM 这些需要显式打开：

```bash
JASMINE_CUDA_STRESS=1 ./build-cuda/tests/cuda_tests
```

实测这两个压力用例合计约 0.5 秒，跑完 45℃。**只要不是反复循环跑，就不会有问题。**

### 7.3 Pascal 的算力特性

P4 是 Pascal（sm_61），没有 Tensor Core，也没有 TF32。所以：

- GEMM 走 FP32 的 `cublasSgemm` / `cublasDgemm`，**不要指望 Tensor Core 级别的吞吐**；
- 逐元素融合是**访存受限**的（P4 显存带宽约 192 GB/s），融合省下的是中间结果的显存往返，这在带宽受限的卡上恰恰是最大的收益来源。

---

## 8. 测试

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
- **融合求值**：四种算术、标量（左/右/两侧/整型）、`exp`/`sigmoid`、比较、五层深链、转置视图、float32、不同 block 大小、单元素/单行/单列边界。
- **cuBLAS**：方阵、非方阵（M/N/K 互不相等）、转置 A、转置 B（`Q·Kᵀ`）、双转置、`beta` 累加、形状不匹配报错、float32、融合段与 GEMM 拼接。
- **归约**：`row_sum`/`row_max`/`col_sum`/`col_max`/`sum_all`/`max_all` 与主机端逐元素对拍，含**轴不得搞反**的专项用例（形状不对称 + 行列和各不相同）、融合归约（`row_sum(exp(x*2))`）、float32、单行/单列/单元素边界。
- **softmax**：与主机 `hsoftmax` 对拍、每行和为 1、大输入（1000 量级）的数值稳定性、**`-inf` 因果掩码**（未来位置必须恰好为 0 且整行不出现 nan）、表达式输入、具名表达式输入。
- **归一化层**：`layer_norm` / `rms_norm` 与主机 `layer_norm_net_t` / `rms_norm_net_t` 对拍（含非平凡 gamma/beta）、每列零均值、RMSNorm 的尺度不变性与非平移不变性、表达式输入。
- **注意力端到端**：`GEMM → 缩放/加掩码 → 逐行 softmax → GEMM` 拼起来与主机算式对拍，外加一条不依赖参考实现的**因果性**行为检查（改动第 t 个 token 之后的内容，前 t 个位置的输出必须一字不变）。
- **压力**（默认跳过）：4096² 融合、256³ GEMM。

数值对拍一律用**相对容差**，原因见 5.4。

---

## 9. 文件一览

| 文件 | 职责 |
| --- | --- |
| `jas_cuda_compat.hpp` | `JAS_HD` / `JAS_DEV` 宏、设备安全数学（`device_exp` / `device_max` / `device_sqrt`）、`device_evaluable` 探测。**不依赖 CUDA 运行时**，纯 CPU 构建也能 include |
| `jas_cuda_leaf.hpp` | 设备叶子 `dev_mat_t`（薄壳、转置视图）、按值拥有定制点。同样不依赖运行时 |
| `jas_cuda_buffer.hpp` | 运行时基础设施：错误检查、设备查询、`dev_buf_t`、`pinned_buf_t`、`sync()` |
| `jas_cuda_matrix.hpp` | `dev_matrix_t`：把「管内存的 buf」和「薄壳 leaf」绑成所有者 |
| `jas_cuda_fused.hpp` | 融合逐元素 kernel 与启动器 |
| `jas_cuda_gemm.hpp` | cuBLAS GEMM（含行优先映射与转置组合） |
| `jas_cuda_reduce.hpp` | 广播叶子、`dev_colvec_t`/`dev_rowvec_t`、归约 kernel、`softmax_rows` / `layer_norm` / `rms_norm` |

主机端被改动的文件只有三处加注 `JAS_HD`，以及 `jas_mat_express_t.hpp` 里的标量叶子（`scalar_leaf_t`）与「按值拥有」判据（见 3.3）。

---

## 10. 下一步

1. ~~归约 kernel~~ ✅ 已完成（第 5 节）：`row_*` / `col_*` / `sum_all` / `max_all`，以及建立其上的 `softmax_rows` / `layer_norm` / `rms_norm`。
2. **softmax 专用 kernel（性能）**：现在 `softmax_rows` 是**三趟**（求最大值 → 求指数和 → 归一化），每趟都融合。合成一趟 online-softmax（边扫边修正 `m` 与 `s`）可以把三遍读 x 降到一遍，对带宽受限的 P4 收益明显。当前实现优先保证与主机端语义易于对拍。
3. **把 `mat_dot_t` 接上 cuBLAS 调度**：让 `a.dot(b)` 在设备叶子上自动落到 GEMM，而不是只在编译期被挡住。这是让整个 `mat_mha_t` / `mat_llama_t` 能直接跑在 GPU 上的最后一块拼图。
4. **KV cache 与 attention**：P4 只有 8 GB 显存，量化或分页 KV cache 是跑长上下文的前提。
5. **反向传播的设备路径**：目前只做了前向。训练要跑在 GPU 上还需要归约的梯度（`hsum` 的反向是广播）。
