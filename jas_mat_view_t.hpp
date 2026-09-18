#ifndef __JAS_MAT_VIEW_T_HPP__
#define __JAS_MAT_VIEW_T_HPP__ 
#include "jas_mat_concepts.hpp"
#include "jas_mat_t.hpp"
#include "jas_mat_utility.hpp"
#include "jas_cuda_compat.hpp"

namespace jasmine {

template <typename agent_type>
class mat_view_t
{ 
public:
    using val_type = typename agent_type::ele_type;
    using ele_type = val_type;
private:
    agent_type &m_mat;
    int m_row_offset;
    int m_col_offset;
    int m_row_size;
    int m_col_size;
    bool m_transposed;      // 是否转置了
public:
    mat_view_t(agent_type& m, int row_offset = 0, int col_offset = 0, int row_size = -1, int col_size = -1) noexcept
        : m_mat(m), m_row_offset(row_offset), m_col_offset(col_offset), m_row_size(row_size), m_col_size(col_size), m_transposed(false)
    {
        m_row_size = (row_size == -1) ? m_mat.row_num() - row_offset : row_size;
        m_col_size = (col_size == -1) ? m_mat.col_num() - col_offset : col_size;
    }

    inline JAS_HD val_type operator()(int row, int col) const noexcept
    {
        if (m_transposed)
        {
            // 转置后的 (row,col) 对应原视图的 (col,row)
            return m_mat(m_row_offset + col, m_col_offset + row);
        }
        else
        {
            return m_mat(row + m_row_offset, col + m_col_offset);
        }
    }

    inline JAS_HD val_type& operator()(int row, int col) noexcept
    {
        if (m_transposed)
        {
            return m_mat(m_row_offset + col, m_col_offset + row);
        }
        else
        {
            return m_mat(row + m_row_offset, col + m_col_offset);
        }
    }

    template<typename other_type>
    void assign(const other_type& other)
    {
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    inline JAS_HD int row_num() const noexcept
    {
        if (m_transposed)
            return m_col_size;
        else
            return m_row_size;
    }

    inline JAS_HD int col_num() const noexcept
    {
        if (m_transposed)
            return m_row_size;
        else
            return m_col_size;
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "matrix_view(" << row_num() << ", " << col_num() << ")";
        for (int i = 0; i < row_num(); ++i)
        {
            ss << "\n[ ";
            for (int j = 0; j < col_num(); ++j)
            {
                ss << (*this)(i, j) << " ";
            }
            ss << "]";
        }
        return ss.str();
    }

    inline mat_view_t<agent_type> t() const noexcept
    {
        mat_view_t<agent_type> mv(*this);
        mv.m_transposed = !m_transposed;
        return mv;
    }

    /** 存储序继承自底层矩阵（视图不改变存储顺序，只改索引方式） */
    bool row_first() const noexcept
    {
        return m_mat.row_first();
    }

    /**
     * 只有当本视图在一维展平后与内存顺序一致时才算紧凑：
     *   - 行优先底层：只有一行（行内连续），或横跨整行（行间无空隙）
     *   - 列优先底层：只有一列，或纵跨整列
     *   - 转置视图：行方向在内存里跨步，一律不算紧凑
     * 这是 reshape_view() 的前提：跨步子视图线性展平出来的顺序不是内存顺序。
     */
    bool densely_packed() const noexcept
    {
        if (!m_mat.densely_packed() || m_transposed)
            return false;
        if (m_mat.row_first())
            return m_row_size == 1 || m_col_size == m_mat.col_num();
        return m_col_size == 1 || m_row_size == m_mat.row_num();
    }

    /**
     * 形状视图：把本视图的元素按内存顺序重新解释成 (rows, cols)；不紧凑时抛异常。
     *
     * 同样只允许左值调用：`x.view(...).reshape_view(...)` 里那个临时视图在整表达式结束时
     * 就析构了，而 reshape 视图持有它的引用 —— 先具名再 reshape。
     */
    mat_reshape_view_t<mat_view_t<agent_type>> reshape_view(int const& rows, int const& cols) &
    {
        return mat_reshape_view_t<mat_view_t<agent_type>>(*this, rows, cols);
    }

    mat_reshape_view_t<mat_view_t<agent_type>> reshape_view(int const& rows, int const& cols) && = delete;

    /**
     * GEMM 操作数描述符：子视图仍然是一段「行距固定、列连续」的稠密矩阵，
     * 所以只要底层能给出描述，本视图也能零拷贝交给 BLAS：
     *   - 未转置：ptr = 基址 + row_offset*ld + col_offset，ld 用底层的前导维
     *   - 转置  ：存储上就是「基址偏移处的 (col_size x row_size) 行优先矩阵」，
     *             交给 BLAS 时带 CblasTrans（要求 ld >= row_size）
     * 底层列优先、或底层本身已经是转置视图（嵌套转置）时给不出，返回 invalid 退回物化。
     */
    detail::gemm_buffer<val_type> gemm_view() const noexcept
    {
        const detail::gemm_buffer<val_type> base = m_mat.gemm_view();
        if (!base.valid || base.transposed || !m_mat.row_first())
            return {};
        const std::size_t offset = static_cast<std::size_t>(m_row_offset) * base.ld + m_col_offset;
        if (base.ld < (m_transposed ? m_row_size : m_col_size))
            return {};
        return {base.ptr + offset, base.ld, m_transposed, true};
    }

    // 同 mat_t::dot：按接收者值类别分派，否则 `m.t().dot(x)` 这类写法存下来会悬垂
    // 实现放到jas_mat_express_t.hpp中，因为此时还没有定义全局的dot函数
    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(other_type&& m) const &;

    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(other_type&& m) const &&;

    operator agent_type() const
    {
        using unconst_type = std::remove_const_t<agent_type>;
        unconst_type m(row_num(), col_num());
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(i, j) = (*this)(i, j);
            }
        }
        return m;
    }

    mat_t<val_type> clone() const
    {
        mat_t<val_type> m(row_num(), col_num());
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(i, j) = (*this)(i, j);
            }
        }
        return std::move(m);
    }

};

/**
 * 形状视图：把同一段存储按新的 (rows, cols) 重新解释，**不拷贝、不分配**。
 *
 * 与 `mat_view_t` 的分工：
 *   - `mat_view_t`      → 取子区域（子块/行/列/转置），元素是原矩阵的子集；
 *   - `mat_reshape_view_t` → 换形状，元素与原矩阵一一对应，只是换了索引方式。
 *
 * 约定：
 *   - `rows * cols` 必须等于底层元素总数，否则构造时抛 `std::invalid_argument`；
 *   - 索引按**与底层相同的存储顺序**解释：底层行优先时 view(i,j) 就是展平后的第
 *     `i*cols + j` 个元素；底层列优先时是第 `j*rows + i` 个（等价于 numpy 的 order='C'/'F'）；
 *   - 因此它是**别名**而不是副本：通过视图写入会直接改到底层数据（反向亦然）；
 *   - `t()` 返回带转置标志的同类视图，仍然零拷贝。
 *
 * 用途：`(n, t)` 与 `(t, n)` 之间的重解释（1D 卷积/patchify 把输入切成窗口）、
 * 以及把展平的张量按任意二维形状喂给 `dot` / BLAS，全程不动数据。
 */
template <typename agent_type>
class mat_reshape_view_t
{
public:
    using val_type = typename agent_type::ele_type;
    using ele_type = val_type;
private:
    agent_type& m_mat;
    int m_rows;
    int m_cols;
    bool m_transposed = false;

    /** 按底层的存储顺序取第 k 个元素 */
    inline val_type& flat(int const k) noexcept
    {
        if (m_mat.row_first())
            return m_mat(k / m_mat.col_num(), k % m_mat.col_num());
        return m_mat(k % m_mat.row_num(), k / m_mat.row_num());
    }

    inline const val_type& flat(int const k) const noexcept
    {
        if (m_mat.row_first())
            return m_mat(k / m_mat.col_num(), k % m_mat.col_num());
        return m_mat(k % m_mat.row_num(), k / m_mat.row_num());
    }

    /** 未转置视图的 (i,j) */
    inline val_type& at(int const i, int const j) noexcept
    {
        return flat(m_mat.row_first() ? i * m_cols + j : j * m_rows + i);
    }

    inline const val_type& at(int const i, int const j) const noexcept
    {
        return flat(m_mat.row_first() ? i * m_cols + j : j * m_rows + i);
    }

public:
    mat_reshape_view_t(agent_type& m, int const rows, int const cols)
        : m_mat(m), m_rows(rows), m_cols(cols)
    {
        if (rows < 0 || cols < 0 || rows * cols != m.row_num() * m.col_num())
            throw std::invalid_argument("mat_reshape_view_t: rows*cols must equal the element count");
        // 跨步子视图不能线性展平（展平顺序 ≠ 内存顺序），这里的下标计算会读错元素。
        // mat_t 恒为紧凑；mat_view_t 只在横跨整行（行优先）/整列（列优先）时才算紧凑。
        if constexpr (requires(const agent_type& a) { a.densely_packed(); })
        {
            if (!m.densely_packed())
                throw std::invalid_argument("mat_reshape_view_t: source is not densely packed");
        }
    }

    inline int row_num() const noexcept
    {
        return m_transposed ? m_cols : m_rows;
    }

    inline int col_num() const noexcept
    {
        return m_transposed ? m_rows : m_cols;
    }

    std::tuple<int, int> shape() const noexcept
    {
        return std::make_tuple(row_num(), col_num());
    }

    inline val_type operator()(int const row, int const col) const noexcept
    {
        return m_transposed ? at(col, row) : at(row, col);
    }

    inline val_type& operator()(int const row, int const col) noexcept
    {
        return m_transposed ? at(col, row) : at(row, col);
    }

    inline mat_reshape_view_t<agent_type> t() const noexcept
    {
        mat_reshape_view_t<agent_type> mv(*this);
        mv.m_transposed = !m_transposed;
        return mv;
    }

    /**
     * GEMM 操作数描述符：reshape 之后每行的长度就是新的 `cols`（底层元素连续），
     * 所以 ld = cols；转置时存储矩阵是 (cols x rows)、行距仍是 cols，正好满足 BLAS 的 ldb >= K。
     */
    detail::gemm_buffer<val_type> gemm_view() const noexcept
    {
        const detail::gemm_buffer<val_type> base = m_mat.gemm_view();
        if (!base.valid || base.transposed || !m_mat.row_first())
            return {};
        return {base.ptr, m_cols, m_transposed, true};
    }

    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(other_type&& m) const &;

    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(other_type&& m) const &&;

    template<typename other_type>
    requires is_matrix<other_type>
    void assign(const other_type& other)
    {
        for (int i = 0; i < row_num(); ++i)
            for (int j = 0; j < col_num(); ++j)
                (*this)(i, j) = other(i, j);
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "matrix_reshape_view(" << row_num() << ", " << col_num() << ")";
        for (int i = 0; i < row_num(); ++i)
        {
            ss << "\n[ ";
            for (int j = 0; j < col_num(); ++j)
                ss << (*this)(i, j) << " ";
            ss << "]";
        }
        return ss.str();
    }

    mat_t<val_type> clone() const
    {
        mat_t<val_type> m(row_num(), col_num());
        for (int i = 0; i < row_num(); ++i)
            for (int j = 0; j < col_num(); ++j)
                m(i, j) = (*this)(i, j);
        return m;
    }

    operator mat_t<val_type>() const
    {
        return clone();
    }
};

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::t() noexcept
{
    mat_view_t<mat_t<val_type>> mv(*this);
    return mv.t();
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_reshape_view_t<mat_t<val_type>> mat_t<val_type>::reshape_view(int const& rows, int const& cols) &
{
    return mat_reshape_view_t<mat_t<val_type>>(*this, rows, cols);
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::view(int const& row_offset, int const& col_offset, int const& row_size, int const& col_size) noexcept
{
    mat_view_t<mat_t<val_type>> mv(*this, row_offset, col_offset, row_size, col_size);
    return mv;
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<const mat_t<val_type>> mat_t<val_type>::view(int const& row_offset, int const& col_offset, int const& row_size, int const& col_size) const noexcept
{
    mat_view_t<const mat_t<val_type>> mv(*this, row_offset, col_offset, row_size, col_size);
    return mv;
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::col(int const& idx) noexcept
{
    return mat_view_t<mat_t<val_type>>(*this, 0, idx, row_num(), 1);
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<const mat_t<val_type>> mat_t<val_type>::col(int const& idx) const noexcept
{
    return mat_view_t<const mat_t<val_type>>(*this, 0, idx, row_num(), 1);
}    

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::row(int const& idx) noexcept
{
    return mat_view_t<mat_t<val_type>>(*this, idx, 0, 1, col_num());
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<const mat_t<val_type>> mat_t<val_type>::row(int const& idx) const noexcept
{
    return mat_view_t<const mat_t<val_type>>(*this, idx, 0, 1, col_num());
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::back_col() noexcept
{
    return mat_view_t<mat_t<val_type>>(*this, 0, col_num() - 1, row_num(), 1);
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<const mat_t<val_type>> mat_t<val_type>::back_col() const noexcept
{
    return mat_view_t<const mat_t<val_type>>(*this, 0, col_num() - 1, row_num(), 1);
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::back_row() noexcept
{
    return mat_view_t<mat_t<val_type>>(*this, row_num() - 1, 0, 1, col_num());
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<const mat_t<val_type>> mat_t<val_type>::back_row() const noexcept
{
    return mat_view_t<const mat_t<val_type>>(*this, row_num() - 1, 0, 1, col_num());
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::front_col() noexcept
{
    return mat_view_t<mat_t<val_type>>(*this, 0, 0, row_num(), 1);
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<const mat_t<val_type>> mat_t<val_type>::front_col() const noexcept
{
    return mat_view_t<const mat_t<val_type>>(*this, 0, 0, row_num(), 1);
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::front_row() noexcept
{
    return mat_view_t<mat_t<val_type>>(*this, 0, 0, 1, col_num());
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<const mat_t<val_type>> mat_t<val_type>::front_row() const noexcept
{
    return mat_view_t<const mat_t<val_type>>(*this, 0, 0, 1, col_num());
}



} // namespace jasmine
#endif