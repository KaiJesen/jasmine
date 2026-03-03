#ifndef __MAT_ROPE_HPP__
#define __MAT_ROPE_HPP__

#include "mat_view_t.hpp"

// 一个可以变动的缓存矩阵，用于RoPE的计算
template<typename val_type>
class mat_cache_t
{
private:
    mat_t<val_type> m_cache;        
    int m_cache_rows;
    int m_cache_cols;

    static constexpr int EXPAND_SIZE = 1024;

    void expand_cache(int rows, int cols)
    {
        if (rows > m_cache.row_num() || cols > m_cache.col_num())
        {
            int new_rows = m_cache.row_num();
            if (rows > m_cache.row_num())
                new_rows = std::max(rows, m_cache.row_num() + EXPAND_SIZE);
            int new_cols = m_cache.col_num();
            if (cols > m_cache.col_num())
                new_cols = std::max(cols, m_cache.col_num() + EXPAND_SIZE);
            //std::cout << "expand cache from " << m_cache.row_num() << "x" << m_cache.col_num() << " to " << new_rows << "x" << new_cols << std::endl;
            mat_t<val_type> new_cache(new_rows, new_cols);
            for (int i = 0; i < m_cache.row_num(); ++i)
            {
                for (int j = 0; j < m_cache.col_num(); ++j)
                {
                    new_cache(i, j) = m_cache(i, j);
                }
            }
            m_cache = std::move(new_cache);
        }
    }
public:
    mat_cache_t() : m_cache(EXPAND_SIZE, EXPAND_SIZE), m_cache_rows(0), m_cache_cols(0) {}

    val_type& operator()(int row, int col)
    {
        if (row >= m_cache_rows || col >= m_cache_cols)
        {
            expand_cache(row + 1, col + 1);
            m_cache_rows = row + 1;
            m_cache_cols = col + 1;
        }
        return m_cache(row, col);
    }

    val_type& operator()(int row, int col) const
    {
        if (row >= m_cache_rows || col >= m_cache_cols)
        {
            throw std::out_of_range("Cache access out of range");
        }
        return m_cache(row, col);
    }

    mat_view_t<mat_t<val_type>> range(int row, int col, int rows, int cols)
    {
        if (row + rows > m_cache_rows || col + cols > m_cache_cols)
        {
            expand_cache(row + rows, col + cols);
            m_cache_rows = row + rows;
            m_cache_cols = col + cols;
        }
        return m_cache.view(row, col, rows, cols);
    }

};

template <typename val_type>
class mat_RoPE_t
{
private:
    mat_cache_t<val_type> m_cache;    // 用于存储sin和cos的缓存矩阵
    int m_enable_rows;              // 当前缓存中可用的行数，也是指向尚未进行初始化的行的指针
    int m_enable_cols;              // 当前缓存中可用的列数，也是指向尚未进行初始化的列的指针
    int m_d;
public:
    mat_RoPE_t(int const& d) : m_enable_rows(0), m_enable_cols(0), m_d(d) {}

    void set_d(int const& d)
    {
        m_d = d;
        m_enable_rows = 0;   // 重新设置d后需要重置缓存状态
        m_enable_cols = 0;
    }

    val_type get_d() const
    {
        return m_d;
    }

    // 一次性先初始化一块
    void init(int const& row_beg, int const& rows, int const& col_beg, int const& cols)
    {
        // 初始化正向举证
        /**!SECTION
         * | cos(m * A_i), -sin(m * A_i) |
         * | sin(m * A_i),  cos(m * A_i) |
         * 其中A_i = 1 / 10000^(2i/d)，m是位置索引，i是维度索引（维度按照2个1组，也就是说元素第n个成员的i=n/2），d是模型维度。对于每个位置m，我们需要为每个维度i计算对应的cos和sin值，并且按照上述方式存储在缓存矩阵中。这样，在前向传播时，我们可以直接从缓存矩阵中获取对应位置和维度的cos和sin值，进行旋转操作，而不需要每次都计算这些值，从而提高效率。
         */
        for (int i = 0; i < row_beg + rows; ++i)
        {
            for (int j = 0; j < col_beg + cols; ++j)
            {
                if (i < m_enable_rows && j < m_enable_cols)
                    continue;   // 已经初始化过了
                val_type angle = j / std::pow(10000.0, (i / 2) / static_cast<val_type>(m_d));
                if (i % 2 == 0 && j % 2 == 0)    // cos
                {
                    m_cache(i, j) = std::cos(angle);
                }
                else if (i % 2 == 1 && j % 2 == 1)   // cos
                {
                    m_cache(i, j) = std::cos(angle);
                }
                else if (i % 2 == 0 && j % 2 == 1)   // -sin
                {
                    m_cache(i, j) = -std::sin(angle);
                }
                else                            // sin
                {
                    m_cache(i, j) = std::sin(angle);
                }
            }
        }
        m_enable_rows = row_beg + rows;
        m_enable_cols = col_beg + cols;
    }

    mat_view_t<mat_t<val_type>> range(int row_beg, int row_num, int col_beg, int col_num)
    {
        if (row_beg + row_num >= m_enable_rows || col_beg + col_num >= m_enable_cols)
        {
            init(row_beg, row_num, col_beg, col_num);
        }
        return m_cache.range(row_beg, col_beg, row_num, col_num);
    }

    mat_view_t<mat_t<val_type>> forward_unite(int const& i, int const& m)
    {
        return range(i * 2, 2, m * 2, 2);
    }

};

template <typename input_type>
class RoPE_net_t
{
private:
    using val_type = typename input_type::ele_type;
    mat_RoPE_t<val_type> m_rope;
public:
    RoPE_net_t(int const& d_model) : m_rope(d_model) {}

    void set_param(int const& d_model)
    {
        m_rope.set_d(d_model);
    }

    mat_t<val_type> forward(input_type const& x)
    {
        // 正向传播
        int seq_len = x.col_num();
        int d_model = x.row_num();
        if (d_model != m_rope.get_d())
        {
            throw std::runtime_error("Input dimension does not match RoPE dimension");
        }
        // 获得一个与输入同等大小的矩阵
        mat_t<val_type> ret(d_model, seq_len);
        for (int m = 0; m < seq_len; ++m)
        {
            for (int i = 0; i < d_model / 2; ++i)
            {
                auto ret_view = ret.view(i * 2, m, 2, 1);
                auto input_view = x.view(i * 2, m, 2, 1);
                auto rope_mat = m_rope.forward_unite(i, m);
                ret_view.assign(rope_mat.dot(input_view));
            }
        }
        return ret;
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        // 反向传播
        int seq_len = delta.col_num();
        int d_model = delta.row_num();
        if (d_model != m_rope.get_d())
        {
            throw std::runtime_error("Input dimension does not match RoPE dimension");
        }
        mat_t<val_type> ret(d_model, seq_len);
        for (int m = 0; m < seq_len; ++m)
        {
            for (int i = 0; i < d_model / 2; ++i)
            {
                auto delta_view = delta.view(i * 2, m, 2, 1);
                auto ret_view = ret.view(i * 2, m, 2, 1);
                auto rope_mat = m_rope.forward_unite(i, m);
                ret_view.assign(rope_mat.t().dot(delta_view));
            }
        }
        return ret;
    }

    template<typename>
    void init_weight()
    {
        // 什么也不做
    }

    void step()
    {
        // 什么也不做
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "RoPE_net_t:(d_model:" << m_rope.get_d() << ")";
        return ss.str();
    }
};

void test_RoPE()
{
    mat_RoPE_t<float> rope(64);
    std::cout << rope.forward_unite(0, 0) << std::endl;
    std::cout << rope.forward_unite(0, 1) << std::endl;
    std::cout << rope.forward_unite(0, 2) << std::endl;

    RoPE_net_t<mat_t<double>> rope_net(4);

    std::cout << rope_net.forward(mat_t<double>(4, 4, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6})) << std::endl;

}


#endif