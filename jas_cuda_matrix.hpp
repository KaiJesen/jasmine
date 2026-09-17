#ifndef __JAS_CUDA_MATRIX_HPP__
#define __JAS_CUDA_MATRIX_HPP__

/**
 * 拥有显存的设备矩阵：把「管内存的 dev_buf_t」和「薄壳 dev_mat_t」绑在一起。
 *
 * 为什么要分成两个类型：
 *   - `dev_mat_t` 只是「指针 + 维度」，平凡可拷贝，所以能按值塞进 kernel 参数；
 *   - 但它不管内存生死查。真让它管（加析构函数）就不再平凡可拷贝，也就传不进 kernel 了。
 *
 * 折中办法就是这里：`dev_matrix_t` 当所有者，需要喂给表达式/kernel 时用 `leaf()` 现取薄壳。
 * `leaf()` 刻意**每次现算**而不是缓存成员 —— 否则对象一被移动/交换，缓存的那个叶子
 * 就会指向旧的缓冲区地址。
 *
 * 这个头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <cstddef>
#include <utility>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_mat_t.hpp"

namespace jasmine {
namespace cuda {

/**
 * 这个元素类型有没有对应的主机矩阵类型。
 *
 * `mat_t` 的约束是算术类型，而降精度类型（`bf16` / `fp16`）不是 —— 它们既没有
 * 主机侧的算术（`mat_t` 的成员里到处在用），也没有必要有：降精度的用法是
 * 「在主机上产生、在设备上算、算完取回参考」，走原始字节就够了。
 *
 * 所以这里把它做成一个显式开关，而不是让 `mat_t<bf16>` 硬生生编译过去：
 * 后者会把"主机矩阵"这个概念稀释掉，而且 `mat_t` 里任何一个用到算术的成员
 * 都会在某个没人预料到的地方炸开。降精度的搬运接口见
 * `jas_cuda_precision.hpp` 的 `to_host_double` / `from_host_double`。
 *
 * 之所以做成模板参数上的约束（而不是把这几个成员删掉），是因为
 * `dev_matrix_t<bf16>` 这个类本身必须能实例化 —— 成员是模板时，只有被调用才会实例化。
 */
template <typename T>
inline constexpr bool has_host_matrix_v = std::is_arithmetic_v<T>;

template <typename T>
class dev_matrix_t
{
public:
    using ele_type = T;

    dev_matrix_t() = default;

    dev_matrix_t(int rows, int cols)
    {
        allocate(rows, cols);
    }

    /**
     * 分配并上传一份主机矩阵。
     *
     * 参数写成 `std::type_identity_t<...>` 是**刻意**的：`U` 必须落在非推导语境里，
     * 否则模板推导会要求实参恰好是 `mat_t<U>`，而 `mat_view_t`（`t()` 的返回类型）
     * 转成 `mat_t` 是用户定义转换、推导阶段不考虑 —— 于是 `dev_matrix_t(r, c, m.t())`
     * 会突然编不过。非推导语境下 `U` 取默认值 `T`，参数类型仍是 `mat_t<T>`，转换照旧生效。
     */
    template <typename U = T>
    requires has_host_matrix_v<U>
    dev_matrix_t(int rows, int cols, const std::type_identity_t<mat_t<U>>& host)
    {
        allocate(rows, cols);
        upload(host);
    }

    void allocate(int rows, int cols)
    {
        m_rows = rows;
        m_cols = cols;
        m_buf.allocate(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    }

    /** 把主机矩阵拷上来；形状必须已经一致（用 allocate 定形状）。 */
    template <typename U = T>
    requires has_host_matrix_v<U>
    void upload(const std::type_identity_t<mat_t<U>>& host)
    {
        if (host.row_num() != m_rows || host.col_num() != m_cols)
            throw std::invalid_argument("dev_matrix_t::upload: 形状不匹配");
        m_buf.upload(host.data(), static_cast<std::size_t>(m_rows) * m_cols);
    }

    /** 拷回主机，形状按当前设备矩阵。 */
    template <typename U = T>
    requires has_host_matrix_v<U>
    mat_t<U> download() const
    {
        mat_t<U> host(m_rows, m_cols);
        if (!m_buf.empty())
            m_buf.download(host.data(), m_buf.size());
        return host;
    }

    /**
     * 取薄壳。现算而不是缓存 —— 见文件顶部说明。
     *
     * 刻意不加 const：叶子带着可写指针（GEMM 的输出要写它），
     * 而 const 的 `dev_buf_t` 只能给出 `const T*`。既然调用方拿到的叶子可写，
     * 就不该用 const 方法假装它只读。
     */
    dev_mat_t<T> leaf() { return dev_mat_t<T>(m_buf.data(), m_rows, m_cols); }

    /** 转置视图的薄壳。 */
    dev_mat_t<T> leaf_transposed() { return dev_mat_t<T>(m_buf.data(), m_rows, m_cols, true); }

    /**
     * 只读叶子。仅用于作 GEMM 的**输入**。
     *
     * `dev_mat_t::m_data` 是 `T*`（GEMM 的输出要写它），所以从 const 的
     * `dev_matrix_t` 里取不出 `dev_mat_t<T>`。这里用一次显式 `const_cast` 把
     * 「语义上只读」这件事表达出来 —— `gemm` 的 A/B 形参本就是
     * `const dev_mat_t<T>&`，只读 `m_data`，从不写。
     *
     * **拿到它只应传给 GEMM 这类只读接口**；要写请用非 const 的 `leaf()`。
     */
    dev_mat_t<T> const_leaf() const
    {
        return dev_mat_t<T>(const_cast<T*>(m_buf.data()), m_rows, m_cols);
    }

    /** 矩阵乘 `this · other`；立即求值，定义在 jas_cuda_gemm.hpp。 */
    dev_matrix_t<T> dot(const dev_mat_t<T>& other) const;
    dev_matrix_t<T> dot(const dev_matrix_t<T>& other) const;

    dev_buf_t<T>& buffer() { return m_buf; }
    const dev_buf_t<T>& buffer() const { return m_buf; }

    int row_num() const { return m_rows; }
    int col_num() const { return m_cols; }

private:
    dev_buf_t<T> m_buf;
    int m_rows = 0;
    int m_cols = 0;
};

} // namespace cuda
} // namespace jasmine

#endif
