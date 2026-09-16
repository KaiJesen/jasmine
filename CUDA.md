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
| `softmax` / 归约（`sum` / `hsum`） | ❌ 尚未实现（需要专用 kernel） |
| 整个模型的端到端 GPU 推理 | ❌ 尚未实现 |

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

using namespace jasmine;

cuda::dev_matrix_t<double> a(rows, cols, ha);   // 分配 + 上传
cuda::dev_matrix_t<double> b(rows, cols, hb);
cuda::dev_matrix_t<double> c(rows, cols, hc);

// 逐元素：整条链一次 launch
auto y = cuda::eval_fused_to_host((a.leaf() + b.leaf()) * c.leaf());

// 矩阵乘：走 cuBLAS
auto s = cuda::gemm_to_host(q.leaf(), k.leaf().t());   // S = Q·Kᵀ
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

### 3.4 一条编译期防线

`device_evaluable` 沿类型树递归传播：叶子写 `true`，节点与操作数做与运算，`mat_t` 这类主机独占类型连这个成员都没有，自然是 `false`。启动 kernel 前 `static_assert` 挡一道，把「拿主机表达式直接上设备」这类错误在**编译期**问出来，而不是运行时段错误。

---

## 4. 为什么 `dot` / `softmax` 不能融合

融合逐元素 kernel 的模型是「一个线程算一个输出元素」。这要求每个输出只依赖同位置的输入。

矩阵乘不是这样：每个输出元素要跨一整行/一列求和，必须多线程协作 + 分块复用数据。硬塞进「每线程算一个元素」的 kernel，等于让每个线程自己走一遍内积，访存完全无法复用，性能会塌掉。

所以正确的做法是**在 `dot` 处切分**：逐元素链 → 融合 kernel，`dot` → cuBLAS。这正是主流框架张量表达式的调度方式。

`softmax` 另有一层原因：它的 `max` / `sum` 统计量在**构造时于主机上算好**，而 `max()`/`sum()` 是主机独占的归约。对一个设备矩阵调 `softmax()`，会在主机上解引用设备指针。所以它既不能融合，也不该被构造出来。

这两类节点因此刻意**不标** `device_evaluable`，含它们的表达式会在编译期被 `static_assert` 挡住。

---

## 5. 行优先 ↔ 列优先

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

## 6. 这台机器上的注意事项

### 6.1 必须用 CUDA 12.4，不能用 13.x

**CUDA 13 起移除了 Pascal 支持**，而本机 `/usr/local/cuda` 默认指向 13.2。用它编译 `sm_61` 会直接失败。

CMake 已经处理：`JASMINE_USE_CUDA=ON` 时会主动去找一个还支持目标架构的 toolkit（按 `12.6 → 12.5 → 12.4 → 12 → /usr/local/cuda` 顺序），并显式设置 `CMAKE_CUDA_COMPILER`。需要覆盖时：

```bash
cmake -S . -B build-cuda -DJASMINE_USE_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
      -DJASMINE_CUDA_ARCHITECTURES=61
```

### 6.2 散热（重要）

本机只有一张 **Tesla P4，没有风扇**，靠机箱风道散热，TDP 75W。长时间满负载会持续升温。

针对这一点，CUDA 测试做了三层保护：

1. **默认用例都很小**：逐元素最大 512×512，GEMM 最大 256³。单次 kernel 亚毫秒级，整个 `cuda_tests` 跑完约 0.8 秒，结温纹丝不动（实测 42℃）。
2. **温度守门**：每个用例前后测温，超过 80℃ 直接 `skip` 而不是硬跑。温度也会打印出来，方便观察趋势。
3. **算力型用例默认关闭**：4096×4096 融合、256³ GEMM 这些需要显式打开：

```bash
JASMINE_CUDA_STRESS=1 ./build-cuda/tests/cuda_tests
```

实测这两个压力用例合计约 0.5 秒，跑完 45℃。**只要不是反复循环跑，就不会有问题。**

### 6.3 Pascal 的算力特性

P4 是 Pascal（sm_61），没有 Tensor Core，也没有 TF32。所以：

- GEMM 走 FP32 的 `cublasSgemm` / `cublasDgemm`，**不要指望 Tensor Core 级别的吞吐**；
- 逐元素融合是**访存受限**的（P4 显存带宽约 192 GB/s），融合省下的是中间结果的显存往返，这在带宽受限的卡上恰恰是最大的收益来源。

---

## 7. 测试

```bash
# 主机端回归（必须仍然全绿）
cmake --build build -j && ./build/tests/unit_tests

# CUDA 后端
cmake --build build-cuda -j
./build-cuda/tests/cuda_tests                    # 快，不发热
JASMINE_CUDA_STRESS=1 ./build-cuda/tests/cuda_tests   # 含算力型用例
```

`cuda_tests` 覆盖：

- **类型契约**（零发热，纯编译期）：设备叶子平凡可拷贝、`mat_t`/`mat_view_t` 不可上设备、设备叶子按值拥有、`dot`/`softmax` 不被误标。
- **融合求值**：四种算术、标量（左/右/两侧/整型）、`exp`/`sigmoid`、比较、五层深链、转置视图、float32、不同 block 大小、单元素/单行/单列边界。
- **cuBLAS**：方阵、非方阵（M/N/K 互不相等）、转置 A、转置 B（`Q·Kᵀ`）、双转置、`beta` 累加、形状不匹配报错、float32、融合段与 GEMM 拼接。
- **压力**（默认跳过）：4096² 融合、256³ GEMM。

数值对拍用**相对容差**：主机走 glibc 的 `std::exp`，设备走 CUDA 的 `::exp`，两者都是高精度实现但**不保证逐位一致**。另外 `exp()` 会把结果量级指数放大（`exp(13)` 就是 7e5），此时一个 ulp 的差异就有 1e-10 量级，用绝对容差等于在比「完全相等」，纯属误报。

---

## 8. 文件一览

| 文件 | 职责 |
| --- | --- |
| `jas_cuda_compat.hpp` | `JAS_HD` 宏、设备安全数学（`device_exp` / `device_max`）、`device_evaluable` 探测。**不依赖 CUDA 运行时**，纯 CPU 构建也能 include |
| `jas_cuda_leaf.hpp` | 设备叶子 `dev_mat_t`（薄壳、转置视图）、按值拥有定制点。同样不依赖运行时 |
| `jas_cuda_buffer.hpp` | 运行时基础设施：错误检查、设备查询、`dev_buf_t`、`pinned_buf_t`、`sync()` |
| `jas_cuda_matrix.hpp` | `dev_matrix_t`：把「管内存的 buf」和「薄壳 leaf」绑成所有者 |
| `jas_cuda_fused.hpp` | 融合逐元素 kernel 与启动器 |
| `jas_cuda_gemm.hpp` | cuBLAS GEMM（含行优先映射与转置组合） |

---

## 9. 下一步

1. **归约 kernel**：`sum` / `hsum` / `max` 的设备实现，这是 softmax、LayerNorm、RMSNorm 的前置。
2. **softmax 专用 kernel**：两趟扫描（求最大值 → 求指数和 → 归一化），并让 `mat_softmax_t` 在设备叶子上有对应的设备路径。
3. **把 `mat_dot_t` 接上 cuBLAS 调度**：让 `a.dot(b)` 在设备叶子上自动落到 GEMM，而不是只在编译期被挡住。
4. **KV cache 与 attention**：P4 只有 8 GB 显存，量化或分页 KV cache 是跑长上下文的前提。
