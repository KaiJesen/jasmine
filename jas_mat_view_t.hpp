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
    // whether this view is transposed
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
            // after transposing, (row,col) maps to (col,row) of the original view
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

    /** The storage order is inherited from the base (a view changes indexing, not layout) */
    bool row_first() const noexcept
    {
        return m_mat.row_first();
    }

    /**
     * A view counts as dense only when flattening it linearly matches memory order:
     *   - row-major base: a single row (contiguous within it), or a full-width block
     *   - column-major base: a single column, or a full-height block
     *   - a transposed view: its rows stride through memory, so never dense
     * This is the precondition of reshape_view(): flattening a strided sub-view does not produce
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
     * Shape view: reinterpret this view's elements in memory order as (rows, cols); throws when the
     *
     * view is not dense. Only lvalues may call it here either: in `x.view(...).reshape_view(...)` the
     * temporary view dies at the end of the full expression while the reshape view references it.
     */
    mat_reshape_view_t<mat_view_t<agent_type>> reshape_view(int const& rows, int const& cols) &
    {
        return mat_reshape_view_t<mat_view_t<agent_type>>(*this, rows, cols);
    }

    mat_reshape_view_t<mat_view_t<agent_type>> reshape_view(int const& rows, int const& cols) && = delete;

    /**
     * GEMM operand descriptor: a sub-view is still dense (fixed row stride, contiguous columns), so
     * as long as the base can describe itself the view reaches BLAS without a copy:
     *   - not transposed: ptr = base + row_offset*ld + col_offset, ld from the base's leading dim
     *   - transposed    : in storage this is the (col_size x row_size) row-major matrix at that offset,
     *                     handed to BLAS with CblasTrans (requires ld >= row_size)
     * A column-major base, or a base that is itself a transposed view (nested transpose), cannot
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

    // Same as mat_t::dot: dispatch on the receiver's value category, otherwise a stored
    // `m.t().dot(x)` would dangle. The implementation lives in jas_mat_express_t.hpp because the
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
 * Shape view: reinterpret the same storage as a new (rows, cols), **without copying or allocating**.
 *
 * How it differs from `mat_view_t`:
 *   - `mat_view_t`         -> take a sub-region (block / row / column / transpose); its elements
 *   - `mat_reshape_view_t` -> change the shape; elements correspond one to one, only indexing changes.
 *
 * Rules:
 *   - `rows * cols` must equal the base element count, otherwise the constructor throws
 *   - indexing follows the **same storage order as the base**: for a row-major base, view(i,j)
 *     `i*cols + j`-th element of the flattening; for a column-major base it is `j*rows + i`
 *   - it is therefore an **alias**, not a copy: writing through it changes the base data (and back);
 *   - `t()` returns a view of the same kind with the transpose flag set, still without copying.
 *
 * Use cases: reinterpreting between `(n, t)` and `(t, n)` (1-D convolution / patchify), and
 * feeding a flattened tensor as any 2-D shape into `dot` / BLAS without touching the data.
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

    /** Take the k-th element in the base's storage order */
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

    /** (i,j) of the untransposed view */
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
        // A strided sub-view cannot be flattened linearly (flattening order != memory order) and the
        // indexing below would read the wrong elements. mat_t is always dense; mat_view_t is dense only
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
     * GEMM operand descriptor: after reshaping one row is exactly the new `cols` (base elements are
     * contiguous), so ld = cols; when transposed the stored matrix is (cols x rows) with the same row
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