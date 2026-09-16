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
 * 这个头不依赖 CUDA 运行时，纯 CPU 构建也能 include（`JAS_HD` 会展开为空），
 * 方便在主机上对类型特征写编译期断言。
 */

#include "jas_mat_express_t.hpp"

namespace jasmine {

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
    int m_cols = 0;          // 转置【之前】的列数，同时也是存储前导维
    bool m_transposed = false;

    static constexpr bool device_evaluable = true;

    dev_mat_t() = default;
    JAS_HD dev_mat_t(T* data, int rows, int cols, bool transposed = false)
        : m_data(data), m_rows(rows), m_cols(cols), m_transposed(transposed)
    {
    }

    JAS_HD int row_num() const { return m_transposed ? m_cols : m_rows; }
    JAS_HD int col_num() const { return m_transposed ? m_rows : m_cols; }

    /**
     * 存储步长（一行有多少个元素）。
     *
     * 注意它**不随转置改变** —— 转置只是换了索引的解释方式，内存布局没动。
     * GEMM 需要这个原始步长来算 lda/ldb/ldc，所以单独暴露，别用 col_num() 代替。
     */
    JAS_HD int leading_dim() const { return m_cols; }

    /** 行优先线性寻址；转置时交换两个下标。无取模、无多余分支。 */
    JAS_HD T operator()(int r, int c) const
    {
        return m_transposed ? m_data[static_cast<std::ptrdiff_t>(c) * m_cols + r]
                            : m_data[static_cast<std::ptrdiff_t>(r) * m_cols + c];
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
};

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
} // namespace jasmine

#endif
