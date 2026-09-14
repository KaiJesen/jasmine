#ifndef __MAT_ROPE_HPP__
#define __MAT_ROPE_HPP__

#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "mat_view_t.hpp"
#include "mat_express_t.hpp"

namespace jasmine {

/**
 * dynamic: 按需扩容（可能替换底层存储，多线程共享时不安全）
 * static_fixed: 预先 reserve，运行期禁止扩容（适合多线程只读共享）
 */
enum class rope_cache_mode
{
    dynamic,
    static_fixed
};

// 一个可以变动的缓存矩阵，用于RoPE的计算
template<typename val_type>
class mat_cache_t
{
private:
    mat_t<val_type> m_cache;
    int m_cache_rows;
    int m_cache_cols;
    rope_cache_mode m_mode;

    static constexpr int EXPAND_SIZE = 1024;

    void ensure_capacity(int rows, int cols)
    {
        if (rows <= m_cache.row_num() && cols <= m_cache.col_num())
            return;

        if (m_mode == rope_cache_mode::static_fixed)
        {
            throw std::runtime_error(
                "RoPE static cache capacity exceeded; call reserve() with a larger max_seq_len");
        }

        int new_rows = m_cache.row_num();
        if (rows > m_cache.row_num())
            new_rows = std::max(rows, m_cache.row_num() + EXPAND_SIZE);
        int new_cols = m_cache.col_num();
        if (cols > m_cache.col_num())
            new_cols = std::max(cols, m_cache.col_num() + EXPAND_SIZE);

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

public:
    mat_cache_t()
        : m_cache(EXPAND_SIZE, EXPAND_SIZE), m_cache_rows(0), m_cache_cols(0),
          m_mode(rope_cache_mode::dynamic)
    {
    }

    void set_mode(rope_cache_mode mode)
    {
        m_mode = mode;
    }

    rope_cache_mode mode() const
    {
        return m_mode;
    }

    int capacity_rows() const { return m_cache.row_num(); }
    int capacity_cols() const { return m_cache.col_num(); }
    int logical_rows() const { return m_cache_rows; }
    int logical_cols() const { return m_cache_cols; }

    /** 预分配容量；static 模式下运行期不再替换底层存储 */
    void reserve(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
            throw std::invalid_argument("RoPE cache reserve size must be positive");

        if (m_mode == rope_cache_mode::static_fixed)
        {
            // 静态模式允许在 reserve 时（一次性）分配/重分配
            if (rows != m_cache.row_num() || cols != m_cache.col_num())
            {
                m_cache = mat_t<val_type>(rows, cols);
            }
            m_cache_rows = 0;
            m_cache_cols = 0;
            return;
        }

        ensure_capacity(rows, cols);
    }

    val_type& operator()(int row, int col)
    {
        if (row >= m_cache_rows || col >= m_cache_cols)
        {
            ensure_capacity(row + 1, col + 1);
            m_cache_rows = std::max(m_cache_rows, row + 1);
            m_cache_cols = std::max(m_cache_cols, col + 1);
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
            ensure_capacity(row + rows, col + cols);
            m_cache_rows = std::max(m_cache_rows, row + rows);
            m_cache_cols = std::max(m_cache_cols, col + cols);
        }
        return m_cache.view(row, col, rows, cols);
    }

    /** 只读视图：不扩容，越界抛异常（供 static 热路径） */
    mat_view_t<mat_t<val_type>> range_readonly(int row, int col, int rows, int cols) const
    {
        if (row + rows > m_cache_rows || col + cols > m_cache_cols)
        {
            throw std::out_of_range("RoPE cache range out of initialized region");
        }
        // const_cast：view 需要非 const mat 引用，但调用方承诺只读
        return const_cast<mat_t<val_type>&>(m_cache).view(row, col, rows, cols);
    }

    void reset_logical()
    {
        m_cache_rows = 0;
        m_cache_cols = 0;
    }
};

template <typename val_type>
class mat_RoPE_t
{
private:
    mat_cache_t<val_type> m_cache;    // 用于存储sin和cos的缓存矩阵
    int m_enable_rows;              // 当前已填充 RoPE 值的行数
    int m_enable_cols;              // 当前已填充 RoPE 值的列数
    int m_d;
    int m_max_seq_len;              // static 模式下预留的最大序列长；dynamic 为 0 表示未限制
public:
    mat_RoPE_t(int const& d)
        : m_enable_rows(0), m_enable_cols(0), m_d(d), m_max_seq_len(0)
    {
    }

    void set_cache_mode(rope_cache_mode mode)
    {
        m_cache.set_mode(mode);
        m_enable_rows = 0;
        m_enable_cols = 0;
        m_cache.reset_logical();
        m_max_seq_len = 0;
    }

    rope_cache_mode cache_mode() const
    {
        return m_cache.mode();
    }

    /**
     * 静态缓存：按 d × (2 * max_seq_len) 预分配并填满。
     * 之后 forward 只读，不再替换底层存储。
     */
    void reserve(int max_seq_len)
    {
        if (max_seq_len <= 0)
            throw std::invalid_argument("max_seq_len must be positive");
        if (m_d <= 0 || m_d % 2 != 0)
            throw std::invalid_argument("RoPE d must be positive even");

        m_max_seq_len = max_seq_len;
        const int rows = m_d;
        const int cols = max_seq_len * 2;
        m_cache.reserve(rows, cols);
        m_enable_rows = 0;
        m_enable_cols = 0;
        init(0, rows, 0, cols);
    }

    void set_d(int const& d)
    {
        m_d = d;
        m_enable_rows = 0;
        m_enable_cols = 0;
        m_cache.reset_logical();
        if (m_cache.mode() == rope_cache_mode::static_fixed && m_max_seq_len > 0)
        {
            reserve(m_max_seq_len);
        }
    }

    int get_d() const
    {
        return m_d;
    }

    int max_seq_len() const
    {
        return m_max_seq_len;
    }

    // 一次性先初始化一块
    void init(int const& row_beg, int const& rows, int const& col_beg, int const& cols)
    {
        // 初始化正向矩阵
        /**!SECTION
         * | cos(m * A_i), -sin(m * A_i) |
         * | sin(m * A_i),  cos(m * A_i) |
         * 其中A_i = 1 / 10000^(2i/d)，m是位置索引，i是维度对索引（特征下标 n 对应 i=n/2），d是模型维度。
         * 缓存布局：行按特征二维对展开，列按位置展开为 2x2 块；同一块共用 θ = m / 10000^(2i/d)。
         */
        const int row_end = row_beg + rows;
        const int col_end = col_beg + cols;
        for (int i = 0; i < row_end; ++i)
        {
            for (int j = 0; j < col_end; ++j)
            {
                if (i < m_enable_rows && j < m_enable_cols)
                    continue;   // 已经初始化过了
                const int pair = i / 2;
                const int pos = j / 2;
                const val_type angle = static_cast<val_type>(pos)
                    / std::pow(static_cast<val_type>(10000.0),
                               (static_cast<val_type>(2 * pair)) / static_cast<val_type>(m_d));
                const val_type c = std::cos(angle);
                const val_type s = std::sin(angle);
                if (i % 2 == 0 && j % 2 == 0)    // cos
                {
                    m_cache(i, j) = c;
                }
                else if (i % 2 == 1 && j % 2 == 1)   // cos
                {
                    m_cache(i, j) = c;
                }
                else if (i % 2 == 0 && j % 2 == 1)   // -sin
                {
                    m_cache(i, j) = -s;
                }
                else                            // sin
                {
                    m_cache(i, j) = s;
                }
            }
        }
        m_enable_rows = std::max(m_enable_rows, row_end);
        m_enable_cols = std::max(m_enable_cols, col_end);
    }

    mat_view_t<mat_t<val_type>> range(int row_beg, int row_num, int col_beg, int col_num)
    {
        if (m_cache.mode() == rope_cache_mode::static_fixed)
        {
            if (row_beg + row_num > m_enable_rows || col_beg + col_num > m_enable_cols)
            {
                throw std::out_of_range(
                    "RoPE static cache miss; increase reserve(max_seq_len)");
            }
            return m_cache.range_readonly(row_beg, col_beg, row_num, col_num);
        }

        if (row_beg + row_num > m_enable_rows || col_beg + col_num > m_enable_cols)
        {
            init(0, std::max(m_enable_rows, row_beg + row_num),
                 0, std::max(m_enable_cols, col_beg + col_num));
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

    void set_cache_mode(rope_cache_mode mode)
    {
        m_rope.set_cache_mode(mode);
    }

    rope_cache_mode cache_mode() const
    {
        return m_rope.cache_mode();
    }

    /** static 模式：预分配并填充到 max_seq_len；dynamic 模式也可提前预热 */
    void reserve(int max_seq_len)
    {
        m_rope.reserve(max_seq_len);
    }

    mat_t<val_type> forward(input_type const& x)
    {
        // 正向传播，逆时针旋转输入token中的向量
        int seq_len = x.col_num();
        int d_model = x.row_num();
        if (d_model != m_rope.get_d())
        {
            throw std::runtime_error("Input dimension does not match RoPE dimension");
        }
        // 获得一个与输入同等大小的矩阵，这个矩阵作为输出矩阵，用于存储RoPE计算后的结果
        mat_t<val_type> ret(d_model, seq_len);
        // 2层循环，外层遍历序列中的每个位置m，内层遍历每个位置中的每组特征（2个特征1组，组索引i），取出缓存位置的2x2的矩阵，与输入的2x1的矩阵进行点积，得到输出矩阵的2x1的矩阵。从原理上解释，缓存矩阵的每2列2行使用相同的旋转角度，属于1组旋转。实际缓存矩阵存储的就是这样一组一组的2x2小旋转矩阵
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
        // 反向传播，与正向传播相反，顺时针将误差旋转回来
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

    int get_d() const
    {
        return m_rope.get_d();
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "RoPE_net_t:(d_model:" << m_rope.get_d()
           << ", cache:" << (m_rope.cache_mode() == rope_cache_mode::static_fixed ? "static" : "dynamic")
           << ")";
        return ss.str();
    }
};

/**
 * 按旋转维度 d（通常是 d_head）共享 RoPE。
 * 同一 d 的多个 MHA head / 层复用同一份 cache。
 */
template <typename val_type>
class rope_registry_t
{
public:
    using rope_net_type = RoPE_net_t<mat_t<val_type>>;
    using rope_ptr = std::shared_ptr<rope_net_type>;

    static rope_registry_t& instance()
    {
        static rope_registry_t reg;
        return reg;
    }

    void set_default_mode(rope_cache_mode mode)
    {
        m_default_mode = mode;
    }

    void set_default_max_seq(int max_seq_len)
    {
        if (max_seq_len <= 0)
            throw std::invalid_argument("default max_seq_len must be positive");
        m_default_max_seq = max_seq_len;
    }

    rope_cache_mode default_mode() const { return m_default_mode; }
    int default_max_seq() const { return m_default_max_seq; }

    /** 获取（或创建）维度为 d 的 RoPE；可选预热 max_seq_len */
    rope_ptr get(int d, int max_seq_len = 0)
    {
        if (d <= 0 || d % 2 != 0)
            throw std::invalid_argument("RoPE registry d must be positive even");

        auto it = m_pool.find(d);
        if (it != m_pool.end())
        {
            if (max_seq_len > 0)
                it->second->reserve(max_seq_len);
            return it->second;
        }

        auto rope = std::make_shared<rope_net_type>(d);
        rope->set_cache_mode(m_default_mode);
        const int reserve_len = max_seq_len > 0 ? max_seq_len : m_default_max_seq;
        if (m_default_mode == rope_cache_mode::static_fixed || max_seq_len > 0)
            rope->reserve(reserve_len);

        m_pool.emplace(d, rope);
        return rope;
    }

    bool contains(int d) const
    {
        return m_pool.find(d) != m_pool.end();
    }

    void clear()
    {
        m_pool.clear();
    }

    std::size_t size() const
    {
        return m_pool.size();
    }

private:
    rope_registry_t() = default;

    std::unordered_map<int, rope_ptr> m_pool;
    rope_cache_mode m_default_mode = rope_cache_mode::dynamic;
    int m_default_max_seq = 1024;
};

} // namespace jasmine
#endif
