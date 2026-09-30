#ifndef _JAS_MAT_T_HPP_
#define _JAS_MAT_T_HPP_
#include <cstring>
#include <cassert>
#include <tuple>
#include <string>
#include <sstream>

#include <iostream>

#include "jas_mat_utility.hpp"
#include "jas_cuda_compat.hpp"

namespace jasmine {

template<typename val_type>
class mat_view_t;               // declared first: the transpose function below needs it
template<typename agent_type>
class mat_reshape_view_t;       // shape view: same storage, new (rows, cols), no copy

namespace detail {

/**
 * GEMM operand descriptor: tells the GEMM about "one contiguous buffer + a leading dimension +
 *
 * whether it is transposed". With it, `mat_t` / `mat_view_t` / `mat_reshape_view_t` can all reach
 * BLAS without a copy:
 *   - transposed == false: ptr points at row-major storage, `ld` elements per row
 *   - transposed == true : the logical matrix is the transpose of the stored matrix; the stored
 *                          matrix is (logical cols x logical rows) row-major with `ld` elements
 * per row (requires ld >= logical rows). Anything that cannot describe itself (column-major
 * storage, a nested transposed view, ...) returns valid == false and the GEMM materialises.
 */
template <typename val_type>
struct gemm_buffer
{
    using ele_type = val_type;
    const val_type* ptr = nullptr;
    int ld = 0;
    bool transposed = false;
    bool valid = false;
};

} // namespace detail

template <typename val_type>
requires std::is_arithmetic_v<val_type>
class mat_t
{
    template <typename U>
    requires std::is_arithmetic_v<U>
    friend class mat_t;
public:
    using ele_type = val_type;
private:
    int m_dims[2];          // dimensions: 0 = inner dim, 1 = outer dim
    val_type *m_data;       // laid out inner-first then outer, tightly packed
    bool m_row_first;    // whether the storage is row-major
    bool m_scalar = false;  // whether this is a scalar
    val_type m_scalar_val{};
    void destroy() noexcept
    {
        if (m_scalar)       // a scalar matrix owns no buffer to free
        {
            m_scalar = false;
            return;
        }
        if (m_data)
        {
            delete[] m_data;
        }
        m_data = nullptr;
    }

    inline void init_scalar(const ele_type& val) noexcept
    {
        m_scalar = true;
        m_scalar_val = val;
        m_data = &m_scalar_val;
        m_dims[0] = 1;
        m_dims[1] = 1;
    }

    inline void init_matrix(const int& rows, const int& cols, bool row_first) noexcept
    {
        m_scalar = false;
        if (row_first)
        {
            m_dims[0] = cols;
            m_dims[1] = rows;
        }
        else
        {
            m_dims[0] = rows;
            m_dims[1] = cols;
        }
        m_row_first = row_first;
        m_data = new val_type[cols * rows];
        memset(m_data, 0, sizeof(val_type) * cols * rows);
    }
public:

    mat_t() noexcept
        : m_dims{0, 0}, m_data(nullptr), m_row_first(true), m_scalar(false), m_scalar_val(0)
    {}

    mat_t(const ele_type& val) noexcept
        : m_dims{1, 1}, m_data(nullptr), m_row_first(true), m_scalar(true), m_scalar_val(val)
    {
        init_scalar(val);
    }


    mat_t(int rows, int cols, bool row_first = true) noexcept
        : m_row_first(row_first)
    {
        if (rows != 1 || cols != 1)
        {
            init_matrix(rows, cols, row_first);
        }
        else
        {
            init_scalar(val_type{});
        }
    }

    mat_t(const int& rows, const int& cols, std::initializer_list<val_type> l) noexcept
        : m_row_first(true)
    {
        if (rows != 1 || cols != 1)
        {
            init_matrix(rows, cols, true);
            std::copy(l.begin(), l.end(), m_data);
        }
        else
            init_scalar(*(l.begin()));
    }

    mat_t(mat_t<val_type>&& m) noexcept
        : m_dims{0, 0}, m_data(m.m_data), m_row_first(m.m_row_first), m_scalar(false), m_scalar_val(0)
    {
        if (m.is_scalar())
        {
            init_scalar(m.m_scalar_val);
        }
        else
        {
            m_scalar = false;
            m_dims[0] = m.m_dims[0];
            m_dims[1] = m.m_dims[1];
            m.m_data = nullptr;
        }
    }

    mat_t& operator=(mat_t<val_type>&& m) noexcept
    {
        if (this != &m)
        {
            if (m.is_scalar())
            {
                destroy();
                init_scalar(m.m_scalar_val);
            }
            else
            {
                destroy();
                m_scalar = false;
                m_dims[0] = m.m_dims[0];
                m_dims[1] = m.m_dims[1];
                m_row_first = m.m_row_first;
                m_data = m.m_data;
                m.m_data = nullptr;
            }
        }
        return *this;
    }

    mat_t(const mat_t<val_type>& m) noexcept
        : m_row_first(m.m_row_first)
    { 
        if (m.is_scalar())
        {
            init_scalar(m.m_scalar_val);
            return;
        }
        else
        {
            init_matrix(m.row_num(), m.col_num(), m.m_row_first);
            std::copy(m.m_data, m.m_data + (m_dims[0] * m_dims[1]), m_data);
        }
    }

    template<typename scalar_type>
    requires std::is_arithmetic_v<scalar_type>
    mat_t& operator=(scalar_type s) noexcept
    {
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                (*this)(i, j) = static_cast<val_type>(s);
            }
        }
        return *this;
    }

    mat_t& operator=(const mat_t<val_type>& m) noexcept
    { 
        if (this != &m)
        {
            if (m.is_scalar())
            {
                destroy();
                init_scalar(m.m_scalar_val);
                return *this;
            }
            else
            {
                destroy();
                init_matrix(m.row_num(), m.col_num(), m.m_row_first);
                std::copy(m.m_data, m.m_data + (m_dims[0] * m_dims[1]), m_data);
            }
        }
        return *this;
    }

    template <typename other_val_type>
    requires std::is_convertible_v<other_val_type, val_type>
    mat_t(const mat_t<other_val_type>& m) noexcept
        : m_row_first(m.m_row_first)
    {
        if (m.is_scalar())
        {
            init_scalar(m.m_scalar_val);
            return;
        }
        else
        {
            init_matrix(m.row_num(), m.col_num(), m.m_row_first);
            for (int i = 0; i < row_num(); ++i)
            {
                for (int j = 0; j < col_num(); ++j)
                {
                    (*this)(i, j) = static_cast<val_type>(m(i, j));
                }
            }
        }
    }

    template <typename other_val_type>
    requires std::is_convertible_v<other_val_type, val_type>
    mat_t& operator=(const mat_t<other_val_type>& m) noexcept
    {
        if constexpr (std::is_same_v<other_val_type, val_type>)
        {
            if (this == &m)
                return *this;
        }
        if (m.is_scalar())
        {
            destroy();
            init_scalar(m.m_scalar_val);
            return *this;
        }
        else
        {
            destroy();
            init_matrix(m.row_num(), m.col_num(), m.m_row_first);
            for (int i = 0; i < row_num(); ++i)
            {
                for (int j = 0; j < col_num(); ++j)
                {
                    (*this)(i, j) = static_cast<val_type>(m(i, j));
                }
            }
        }
        return *this;
    }

    ~mat_t() noexcept
    {
        destroy();
    }

    JAS_HD bool is_scalar() const noexcept
    {
        return m_scalar;
    }

    JAS_HD bool row_first() const noexcept
    {
        return m_row_first;
    }

    /**
     * Whether the storage is as dense as the shape (no stride-induced gaps).
     * A mat_t is always one dense array, so this is always true; only views can be strided.
     * A reshape view flattens by index, so only a dense matrix/view can be reinterpreted safely.
     */
    JAS_HD bool densely_packed() const noexcept
    {
        return true;
    }

    /** mat_t::reshape_view is defined in jas_mat_view_t.hpp (the view types live there) */

    JAS_HD val_type* data() noexcept
    {
        return m_data;
    }

    JAS_HD const val_type* data() const noexcept
    {
        return m_data;
    }

    JAS_HD int row_num() const noexcept
    {
        if (m_row_first)
            return m_dims[1];
        else
            return m_dims[0];
    }

    JAS_HD int col_num() const noexcept
    {
        if (m_row_first)
            return m_dims[0];
        else
            return m_dims[1];
    }

    std::tuple<int, int> shape() const noexcept
    {
        return std::make_tuple(row_num(), col_num());
    }

    void reshape(int rows, int cols) noexcept
    {
        if (rows == 1 && cols == 1)
        {
            if (!m_scalar)
            {
                val_type scalar_val = (*this)(0, 0);
                destroy();
                init_scalar(scalar_val);
            }
        }
        else
        {
            if (rows * cols != row_num() * col_num())
            {
                destroy();
                m_row_first = true;
                init_matrix(rows, cols, true);
            }
        }
    }

    JAS_HD val_type& operator()(int r, int c) noexcept
    {
        /* Index folding with `%` is deliberate: the bottom layer stays permissive so
         * that the expression layer can broadcast ([R,1] or [1,C]) and so that
         * periodic indexing (a kernel sliding over a larger input) needs no special
         * case. See the design principle in doc/OPTIMIZATION_CHECKLIST.md.
         *
         * The catch is cost: an integer division/modulo is ~20-40 cycles on x86 and
         * does not pipeline, and this accessor is called once per element per
         * operand, so an elementwise loop paid 2 divisions per array touched. Guard
         * with a range check first, which turns the common in-range case into a
         * compare; an out-of-range index (including a negative one, for which the
         * unsigned compare fails) still takes the original modulo, so behaviour is
         * unchanged for every input. Measured on a 768x1024 pass: 6.8 -> 22.0 GB/s.
         *
         * row_num()/col_num() are read once because they depend on m_row_first -- m_dims[0]
         * is the inner (contiguous) dimension and is the row dimension only when the
         * storage is column-major.
         */
        const int rows = row_num();
        const int cols = col_num();
        const int i = static_cast<unsigned>(r) < static_cast<unsigned>(rows) ? r : r % rows;
        const int j = static_cast<unsigned>(c) < static_cast<unsigned>(cols) ? c : c % cols;
        if (m_row_first)
            return m_data[i * m_dims[0] + j];
        else
            return m_data[j * m_dims[0] + i];
    }

    JAS_HD const val_type& operator()(int r, int c) const noexcept
    {
        const int rows = row_num();
        const int cols = col_num();
        const int i = static_cast<unsigned>(r) < static_cast<unsigned>(rows) ? r : r % rows;
        const int j = static_cast<unsigned>(c) < static_cast<unsigned>(cols) ? c : c % cols;
        if (m_row_first)
            return m_data[i * m_dims[0] + j];
        else
            return m_data[j * m_dims[0] + i];
    }

    /**
     * Accessor for loops whose bounds are already known to be in range. Behaviour is
     * undefined outside [0, row_num()) x [0, col_num()); the assert catches that in
     * debug builds and it compiles away under NDEBUG.
     *
     * Why this exists next to operator(): operator() folds out-of-range indices with
     * `%`, which is what lets the expression layer broadcast a [R,1] or [1,C] operand
     * and what makes periodic indexing work. That folding is a data-dependent branch
     * and a division, so the compiler cannot vectorize any loop that uses it --
     * measured, an (a - b) pass runs at 22 GB/s through operator() versus 49 GB/s
     * through a plain pointer loop. Keeping the promise in the hot paths instead of
     * weakening the guarantee is the whole point: broadcast semantics stay exactly as
     * they were, and loops that already validated their shapes get the vectorizable
     * accessor.
     */
    JAS_HD val_type& unchecked(int r, int c) noexcept
    {
        assert(r >= 0 && r < row_num() && c >= 0 && c < col_num() &&
               "mat_t::unchecked: index out of range; use operator() for folded access");
        return m_row_first ? m_data[r * m_dims[0] + c] : m_data[c * m_dims[0] + r];
    }

    JAS_HD const val_type& unchecked(int r, int c) const noexcept
    {
        assert(r >= 0 && r < row_num() && c >= 0 && c < col_num() &&
               "mat_t::unchecked: index out of range; use operator() for folded access");
        return m_row_first ? m_data[r * m_dims[0] + c] : m_data[c * m_dims[0] + r];
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "matrix(" << row_num() << ", " << col_num() << ")";
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

    bool valid() const noexcept
    {
        return m_data != nullptr;
    }

    /**
     * GEMM operand descriptor: row-major storage reports (data, col_num, not transposed) directly.
     * Column-major cannot (the fast path only understands row-major): returns invalid, caller materialises.
     */
    detail::gemm_buffer<val_type> gemm_view() const noexcept
    {
        if (!m_row_first || m_data == nullptr)
            return {};
        return {m_data, col_num(), false, true};
    }

    mat_view_t<mat_t<val_type>> t() noexcept;
    /**
     * Shape view: reinterpret the same storage as (rows, cols) without copying (rows*cols must equal
     *
     * the element count). The view holds a **reference** to this matrix, so only lvalues may call it:
     * with `mat_t(...).reshape_view(..)` the source dies at the end of the full expression and the view
     * dangles. The rvalue overload is deleted so those forms fail to compile (TESTING.md section 9).
     */
    mat_reshape_view_t<mat_t<val_type>> reshape_view(int const& rows, int const& cols) &;
    mat_reshape_view_t<mat_t<val_type>> reshape_view(int const& rows, int const& cols) && = delete;
    mat_view_t<mat_t<val_type>> view(int const& row_offset = 0, int const& col_offset = 0, int const& row_size = -1, int const& col_size = -1) noexcept;
    mat_view_t<const mat_t<val_type>> view(int const& row_offset = 0, int const& col_offset = 0, int const& row_size = -1, int const& col_size = -1) const noexcept;
    mat_view_t<mat_t<val_type>> col(int const& col_num) noexcept;
    mat_view_t<const mat_t<val_type>> col(int const& col_num) const noexcept;
    mat_view_t<mat_t<val_type>> row(int const& row_num) noexcept;
    mat_view_t<const mat_t<val_type>> row(int const& row_num) const noexcept;
    mat_view_t<mat_t<val_type>> back_col() noexcept;
    mat_view_t<const mat_t<val_type>> back_col() const noexcept;
    mat_view_t<mat_t<val_type>> back_row() noexcept;
    mat_view_t<const mat_t<val_type>> back_row() const noexcept;
    mat_view_t<mat_t<val_type>> front_col() noexcept;
    mat_view_t<const mat_t<val_type>> front_col() const noexcept;
    mat_view_t<mat_t<val_type>> front_row() noexcept;
    mat_view_t<const mat_t<val_type>> front_row() const noexcept;

    mat_t<val_type>& t_() noexcept
    {
        mat_t<val_type> m(col_num(), row_num());
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(j, i) = (*this)(i, j);
            }
        }
        *this = std::move(m);
        return *this;
    }

    mat_t<val_type> clone() const noexcept
    {
        return mat_t<val_type>(*this);
    }

    // Inside a member function `*this` is always an lvalue, so dispatch on the value category of the
    // *receiver*: const&  -> lvalue matrix, borrow a reference (zero copy)
    //            const&& -> temporary matrix (e.g. `make().dot(x)`), own by value, otherwise the stored
//            expression tree would dangle
    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(other_type&& m) const &;

    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(other_type&& m) const &&;
};




} // namespace jasmine
#endif
