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

template <typename T>
class dev_matrix_t
{
public:
    dev_matrix_t() = default;

    dev_matrix_t(int rows, int cols)
    {
        allocate(rows, cols);
    }

    /** 分配并上传一份主机矩阵。 */
    dev_matrix_t(int rows, int cols, const mat_t<T>& host)
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
    void upload(const mat_t<T>& host)
    {
        if (host.row_num() != m_rows || host.col_num() != m_cols)
            throw std::invalid_argument("dev_matrix_t::upload: 形状不匹配");
        m_buf.upload(host.data(), static_cast<std::size_t>(m_rows) * m_cols);
    }

    /** 拷回主机，形状按当前设备矩阵。 */
    mat_t<T> download() const
    {
        mat_t<T> host(m_rows, m_cols);
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
