#ifndef _MAT_T_HPP_
#define _MAT_T_HPP_
#include <cstring>
#include <tuple>
#include <string>
#include <sstream>

#include <iostream>

#include "mat_utility.hpp"

template<typename val_type>
class mat_view_t;           // 这里先声明，因为后面要用到这个来声明转置函数

template <typename val_type>
requires std::is_arithmetic_v<val_type>
class mat_t
{
    template <typename U>
    requires std::is_arithmetic_v<U>
    friend class mat_t;
public:
    using return_type = val_type;
private:
    int m_dims[2];          // 维度数组，0为内层维度，1为外层维度
    val_type *m_data;       // 由先内层后外层紧密排列
    bool m_row_first;    // 是否为行优先存储
    bool m_scalar = false;  // 是否为标量
    val_type m_scalar_val;
    void destroy() noexcept
    {
        if (m_scalar)       // 标量矩阵，不进行释放
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

    inline void init_scalar(const return_type& val) noexcept
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

    mat_t(const return_type& val) noexcept
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
                m_scalar = false;
                destroy();
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

    bool is_scalar() const noexcept
    {
        return m_scalar;
    }   

    int row_num() const noexcept
    {
        if (m_row_first)
            return m_dims[1];
        else
            return m_dims[0];
    }

    int col_num() const noexcept
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

    val_type& operator()(int r, int c) noexcept
    {
        int i = r % row_num();
        int j = c % col_num();
        if (m_row_first)
            return m_data[i * m_dims[0] + j];
        else
            return m_data[j * m_dims[0] + i];
    }

    const val_type& operator()(int r, int c) const noexcept
    {
        int i = r % row_num();
        int j = c % col_num();
        if (m_row_first)
            return m_data[i * m_dims[0] + j];
        else
            return m_data[j * m_dims[0] + i];
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

    mat_view_t<mat_t<val_type>> t() noexcept;
    mat_view_t<mat_t<val_type>> view(int const& row_offset = 0, int const& col_offset = 0, int const& row_size = -1, int const& col_size = -1) noexcept;
    mat_view_t<const mat_t<val_type>> view(int const& row_offset = 0, int const& col_offset = 0, int const& row_size = -1, int const& col_size = -1) const noexcept;

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

    template<typename other_type>
    requires is_matrix<other_type>
    auto dot(const other_type& m) const;
};



#endif
