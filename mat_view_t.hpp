#ifndef __MAT_VIEW_T_HPP__
#define __MAT_VIEW_T_HPP__ 
#include "mat_concepts.hpp"
#include "mat_t.hpp"
#include "mat_utility.hpp"

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

    inline val_type operator()(int row, int col) const noexcept
    {
        if (m_transposed)
        {
            return m_mat(col + m_col_offset, row + m_row_offset);
        }
        else
        {
            return m_mat(row + m_row_offset, col + m_col_offset);
        }
    }

    inline val_type& operator()(int row, int col) noexcept
    {
        //return m_mat(row + m_row_offset, col + m_col_offset);
        if (m_transposed)
        {
            return m_mat(col + m_col_offset, row + m_row_offset);
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

    inline int row_num() const noexcept
    {
        if (m_transposed)
            return m_col_size;
        else
            return m_row_size;
    }

    inline int col_num() const noexcept
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

    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(const other_type& m) const;    // 实现放到mat_express_t.hpp中，因为此时还没有定义全局的dot函数

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

template<typename val_type>
requires std::is_arithmetic_v<val_type>
mat_view_t<mat_t<val_type>> mat_t<val_type>::t() noexcept
{
    mat_view_t<mat_t<val_type>> mv(*this);
    return mv.t();
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

void test_mat_view_t()
{ 
    mat_t<double> m(3, 3, {1.1, 1.2, 1.3,
                             2.1, 2.2, 2.3,
                             3.1, 3.2, 3.3});
    mat_view_t<mat_t<double>> mv(m, 1, 1, 2, 1);
    std::cout << mv << "transposed view \n" << mv.t() << std::endl;
}

void test_mat_t()
{
    mat_t<double> m0(1.);
    mat_t<int> m(2, 3, {1, 2, 3, 4, 5, 6});
    std::cout << m.to_string() << std::endl;
    std::cout << m.t_() << std::endl << m.t() << std::endl;
    mat_t<int> m2(1);
    std::cout << m2 << std::endl;
}


#endif