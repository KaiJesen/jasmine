#ifndef __JAS_CUDA_LEAF_HPP__
#define __JAS_CUDA_LEAF_HPP__

/**
 * 设备叶子：让表达式模板可以在 GPU 上求值的矩阵句柄。
 *
 * 设计要点（这几点合起来才让「表达式树上设备」成立）：
 *
 *  1. **薄壳**：`dev_mat_t` 只有「设备指针 + 行列数」，不管内存生死。
 *     内存归 `dev_buf_t`（见 jas_cuda_buffer.hpp）管。分开之后，树里存的只是薄壳，
 *     拷贝一个叶子就是拷三个字段。
 *
 *  2. **POD 且平凡可拷贝**：CUDA kernel 的参数就是**按值**传的，而且必须是平凡可拷贝、
 *     有大小上限的类型。薄壳天然满足，于是整棵表达式树可以直接当 kernel 参数递进去。
 *
 *  3. **JAS_HD 访问器**：`row_num` / `col_num` / `operator()` 在主机和设备上都能调。
 *     表达式节点的 `work()` 只依赖这三个接口，所以同一个 `work()` 在两个世界通用。
 *
 *  4. **不做取模**：`mat_t::operator()` 里有 `r % row_num()`，那是为了容忍越界索引。
 *     在设备上每个线程都要重算一遍模运算纯属浪费，还会挡住合并访存分析，所以这里直接算地址。
 *
 *  5. **按值拥有**：注册到 `operand_owned_by_value`，于是建树时叶子被拷进树里而不是借引用。
 *     设备端不存在"主机对象的地址"，引用毫无意义。
 *
 *  6. **逻辑列数与前导维分开**：`m_cols` 是逻辑列数，`m_ld` 是存储步长，两者可以不等。
 *     这样「按 cap 分配、只暴露前 len 列」这类子视图（KV cache）能零拷贝地表达 ——
 *     否则每步都得拷成紧凑缓冲，正好废掉 KV cache 的意义。`view()` 与
 *     `make_dev_leaf_strided()` 是它的两个入口。
 *
 * 这个头不依赖 CUDA 运行时，纯 CPU 构建也能 include（`JAS_HD` 会展开为空），
 * 方便在主机上对类型特征写编译期断言。
 */

#include "jas_mat_express_t.hpp"

namespace jasmine {

namespace cuda {
// 前向声明：dot() 要返回拥有者类型，但它的定义在 jas_cuda_gemm.hpp（那里才有 cuBLAS）。
// 只在类里声明、在 GEMM 头里定义，是为了让本头继续不依赖 CUDA 运行时。
template <typename T>
class dev_matrix_t;
} // namespace cuda

/**
 * 行优先存储的设备矩阵句柄，带可选的转置视图。
 *
 * 只读语义：设备上的表达式求值是「读若干叶子、写一个输出」，叶子不需要可写接口。
 * （GEMM 的输出走另一个显式可写的叶子，见 jas_cuda_gemm.hpp。）
 */
template <typename T>
struct dev_mat_t
{
    using ele_type = T;

    T* m_data = nullptr;
    int m_rows = 0;          // 转置【之前】的行数
    int m_cols = 0;          // 转置【之前】的列数（逻辑列数）
    int m_ld = 0;            // 存储前导维（一行占多少个元素）
    bool m_transposed = false;

    static constexpr bool device_evaluable = true;

    dev_mat_t() = default;
    JAS_HD dev_mat_t(T* data, int rows, int cols, bool transposed = false)
        : m_data(data), m_rows(rows), m_cols(cols), m_ld(cols), m_transposed(transposed)
    {
    }

    /**
     * 逻辑列数与前导维**分开**的构造：缓冲区比实际用到的宽。
     *
     * 这不是可有可无的灵活性。KV cache 就是按 `cap` 分配缓冲、只对外暴露前 `len` 列；
     * 一旦「列数」和「前导维」是同一个字段，这种视图就只能靠**拷贝**成紧凑缓冲来实现 ——
     * 而 KV cache 存在的全部意义恰恰是不做这件事。
     *
     * cuBLAS 本身完全支持 `ld > cols`，`gemm` 读的也是 `leading_dim()`，
     * 所以补上这个字段之后零拷贝视图直接就能用。
     */
    JAS_HD dev_mat_t(T* data, int rows, int cols, int leading_dim, bool transposed)
        : m_data(data), m_rows(rows), m_cols(cols), m_ld(leading_dim), m_transposed(transposed)
    {
    }

    JAS_HD int row_num() const { return m_transposed ? m_cols : m_rows; }
    JAS_HD int col_num() const { return m_transposed ? m_rows : m_cols; }

    /**
     * 存储步长（一行有多少个元素），**可以与 col_num() 不同**。
     *
     * 注意它不随转置改变 —— 转置只是换了索引的解释方式，内存布局没动。
     * GEMM 需要这个原始步长来算 lda/ldb/ldc，所以单独暴露，别用 col_num() 代替。
     */
    JAS_HD int leading_dim() const { return m_ld; }

    /** 行优先线性寻址；转置时交换两个下标。无取模、无多余分支。 */
    JAS_HD T operator()(int r, int c) const
    {
        return m_transposed ? m_data[static_cast<std::ptrdiff_t>(c) * m_ld + r]
                            : m_data[static_cast<std::ptrdiff_t>(r) * m_ld + c];
    }

    /**
     * 左上角 (rows × cols) 子块：**指针与存储步长都不动**，只改逻辑形状。
     *
     * 同样为 KV cache 而设：`(d × cap)` 的缓冲要能直接当 `(d × len)` 用。
     * 在**未转置的坐标系**里解释，所以别对转置视图调它后假设形状（要转置就 `view(...).t()`）。
     */
    JAS_HD dev_mat_t view(int rows, int cols) const
    {
        dev_mat_t r = *this;
        r.m_rows = rows;
        r.m_cols = cols;
        return r;
    }

    /** 转置视图：只翻一个标志位，不碰内存。 */
    JAS_HD dev_mat_t t() const
    {
        dev_mat_t r = *this;
        r.m_transposed = !m_transposed;
        return r;
    }

    JAS_HD bool transposed() const { return m_transposed; }
    JAS_HD bool valid() const { return m_data != nullptr; }

    /**
     * 设备端矩阵乘 `this · other`（转置视图自动生效，见 jas_cuda_gemm.hpp）。
     *
     * 与主机端的 `.dot()` 有个本质区别：**这里立即求值**，返回一个拥有显存的结果，
     * 而不是构造一个 `mat_dot_t` 节点。
     *
     * 理由是 GEMM 根本无法融合。主机端的 `.dot()` 返回惰性节点是有意义的 ——
     * 它还能被并进更大的表达式树、由 `work()` 逐元素求值；而设备端的
     * `mat_dot_t::operator()` 是「每个输出元素自己走一遍 K 循环」，融进逐元素 kernel
     * 等于把访存复用完全丢掉。既然它必然要单独执行，那就不该假装它是个惰性节点：
     * 显式地立即落成 cuBLAS，语义与性能都对得上。
     *
     * 也正因如此，`mat_dot_t` 刻意不标 `device_evaluable`（见 jas_mat_express_t.hpp），
     * 于是「误把主机 dot 节点丢给融合 kernel」在编译期就被挡住。
     */
    cuda::dev_matrix_t<T> dot(const dev_mat_t<T>& other) const;
    cuda::dev_matrix_t<T> dot(const cuda::dev_matrix_t<T>& other) const;
};

/** 判断是不是设备叶子。用来给「叶子直通、表达式先物化」的重载分流。 */
template <typename T>
struct is_dev_leaf : std::false_type {};
template <typename T>
struct is_dev_leaf<dev_mat_t<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_dev_leaf_v = is_dev_leaf<std::remove_cvref_t<T>>::value;

/**
 * 设备叶子按值拥有：即便调用方传的是左值，树里也存一份薄壳。
 * 理由见本文件顶部第 5 条。
 */
template <typename T>
inline constexpr bool operand_owned_by_value<dev_mat_t<T>> = true;

/** 从裸设备指针造一个叶子，省得每次写全字段。 */
template <typename T>
JAS_HD dev_mat_t<T> make_dev_leaf(T* data, int rows, int cols)
{
    return dev_mat_t<T>(data, rows, cols);
}

/** 前导维大于列数的叶子（缓冲区比用到的宽，如 KV cache 的空闲容量）。 */
template <typename T>
JAS_HD dev_mat_t<T> make_dev_leaf_strided(T* data, int rows, int cols, int leading_dim)
{
    return dev_mat_t<T>(data, rows, cols, leading_dim, false);
}

/**
 * 零拷贝取**行子块**：从第 `row0` 行起的 `rows` 行，列数与存储步长都不动。
 *
 * 是 `view()` 的行方向补充：`view()` 切左上角子块（行列都动、起点固定在左上角），
 * 本函数只切行、起点任意。
 *
 * 为**逐头 RoPE** 而设：QKV 打包成 `(num_heads * d_head × seq)` 时，
 * 每个头正好是一个行子块，切出来直接喂给 `dev_rope_t::rotate_into`。
 * 转置视图下也成立（那时逻辑行对应存储的列，偏移退化成 `row0` 个元素）。
 */
template <typename T>
JAS_HD dev_mat_t<T> row_slice(const dev_mat_t<T>& leaf, int row0, int rows)
{
    dev_mat_t<T> r = leaf;
    r.m_data += leaf.transposed()
                    ? static_cast<std::ptrdiff_t>(row0)
                    : static_cast<std::ptrdiff_t>(row0) * leaf.leading_dim();
    // m_rows / m_cols 是**转置之前**的维度，所以要看情况改哪一个
    if (leaf.transposed())
        r.m_cols = rows;
    else
        r.m_rows = rows;
    return r;
}
/**
 * 零拷贝取**列子块**：从第 `col0` 列起的 `cols` 列，行数与存储步长都不动。
 *
 * 是 `row_slice` 的列方向对称物（`view()` 起点的列方向补充）。为**取序列的最后一列**
 * 而设：decode 时只要最后一个位置的 logits，而隐藏状态是 `(d_model × T)`，
 * 需要的正是 `(d_model × 1)` 的尾列 —— 拷一份紧凑副本纯属浪费。
 *
 * 转置视图下逻辑列对应存储的行，偏移退化成 `col0 * leading_dim()`。
 */
template <typename T>
JAS_HD dev_mat_t<T> col_slice(const dev_mat_t<T>& leaf, int col0, int cols)
{
    dev_mat_t<T> r = leaf;
    r.m_data += leaf.transposed() ? static_cast<std::ptrdiff_t>(col0) * leaf.leading_dim()
                                  : static_cast<std::ptrdiff_t>(col0);
    // m_rows / m_cols 是**转置之前**的维度，所以要看情况改哪一个
    if (leaf.transposed())
        r.m_rows = cols;
    else
        r.m_cols = cols;
    return r;
}

} // namespace jasmine

#endif
