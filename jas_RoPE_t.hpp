#ifndef __JAS_ROPE_HPP__
#define __JAS_ROPE_HPP__

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include "jas_mat_view_t.hpp"
#include "jas_mat_express_t.hpp"

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

/**
 * RoPE 的特征配对约定：θ_i = m / 10000^(2i/d) 两种约定完全一致，
 * 区别只在于「第 i 个旋转角作用在哪两个特征上」。
 *
 * - interleaved：作用于第 (2i, 2i+1) 对特征（原始 RoPE 论文 / GPT-NeoX 写法），
 *   也是本仓库既有的实现，保持默认以兼容已有模型与测例。
 * - half_split ：作用于第 (i, i + d/2) 对特征（GPT-J 写法），
 *   HuggingFace 的 `LlamaRotaryEmbedding` + `rotate_half` 用的就是它。
 *
 * 注意：位置 m = 0 时旋转恒为恒等变换，两种约定输出完全相同，
 * 因此只比对单 token、位置 0 的输出无法区分二者；多 token 序列上必须显式指定。
 * 导出 LLaMA 系权重（HF 实现）时必须选 half_split。
 */
enum class rope_pair_layout
{
    interleaved = 0,
    half_split = 1
};

inline const char* rope_pair_layout_name(rope_pair_layout layout)
{
    return layout == rope_pair_layout::half_split ? "half_split" : "interleaved";
}

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
    rope_pair_layout m_layout = rope_pair_layout::interleaved;

    /** 既有实现：旋转第 (2i, 2i+1) 对特征。列 j 用位置 start_pos + j。 */
    template <typename in_t>
    mat_t<val_type> interleaved_rotate_at(in_t const& x, int start_pos)
    {
        int seq_len = x.col_num();
        int d_model = x.row_num();
        mat_t<val_type> ret(d_model, seq_len);
        // 外层：序列位置；内层：特征二维对。缓存每 2×2 块共用 θ = m / 10000^(2i/d)
        for (int j = 0; j < seq_len; ++j)
        {
            const int m = start_pos + j;
            for (int i = 0; i < d_model / 2; ++i)
            {
                auto ret_view = ret.view(i * 2, j, 2, 1);
                auto input_view = x.view(i * 2, j, 2, 1);
                auto rope_mat = m_rope.forward_unite(i, m);
                ret_view.assign(rope_mat.dot(input_view));
            }
        }
        return ret;
    }

    /** 既有实现的反向：按 2×2 旋转矩阵的转置旋转回来。 */
    template <typename in_t>
    mat_t<val_type> interleaved_backward(in_t const& delta)
    {
        int seq_len = delta.col_num();
        int d_model = delta.row_num();
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

    /**
     * half_split 与 interleaved 只差一个「特征行重排」：
     * half_split 的第 (i, i + d/2) 两个特征，正对应 interleaved 的第 (2i, 2i+1) 两个特征，
     * 而且两者的角 θ_i 定义完全相同。所以把行搬成 interleaved 顺序 → 复用既有旋转 → 搬回去，
     * 就得到 half_split 的结果（搬运不改变任何 θ）。
     */
    mat_t<val_type> gather_to_interleaved(input_type const& x)
    {
        const int d = x.row_num(), seq_len = x.col_num(), half = d / 2;
        mat_t<val_type> out(d, seq_len);
        for (int j = 0; j < seq_len; ++j)
        {
            for (int i = 0; i < half; ++i)
            {
                out(i * 2, j) = x(i, j);
                out(i * 2 + 1, j) = x(i + half, j);
            }
        }
        return out;
    }

    mat_t<val_type> scatter_from_interleaved(const mat_t<val_type>& y)
    {
        const int d = y.row_num(), seq_len = y.col_num(), half = d / 2;
        mat_t<val_type> out(d, seq_len);
        for (int j = 0; j < seq_len; ++j)
        {
            for (int i = 0; i < half; ++i)
            {
                out(i, j) = y(i * 2, j);
                out(i + half, j) = y(i * 2 + 1, j);
            }
        }
        return out;
    }

public:
    RoPE_net_t(int const& d_model) : m_rope(d_model) {}

    void set_param(int const& d_model)
    {
        m_rope.set_d(d_model);
    }

    /** 选择特征配对约定，见 rope_pair_layout。改变了要重新 bind_rope/预热。 */
    void set_pair_layout(rope_pair_layout layout)
    {
        m_layout = layout;
    }

    rope_pair_layout pair_layout() const
    {
        return m_layout;
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
        // 默认：列下标即绝对位置 0..seq_len-1
        return forward_at(x, 0);
    }

    /**
     * 从绝对位置 start_pos 起旋转：列 j 使用位置 start_pos + j。
     * forward_one / KV cache 路径需要：单列新 token 的位置 = cache.length()。
     */
    mat_t<val_type> forward_at(input_type const& x, int start_pos)
    {
        int d_model = x.row_num();
        if (d_model != m_rope.get_d())
        {
            throw std::runtime_error("Input dimension does not match RoPE dimension");
        }
        if (start_pos < 0)
            throw std::invalid_argument("RoPE forward_at start_pos must be >= 0");

        if (m_layout == rope_pair_layout::half_split)
        {
            auto permuted = gather_to_interleaved(x);
            return scatter_from_interleaved(interleaved_rotate_at(permuted, start_pos));
        }
        return interleaved_rotate_at(x, start_pos);
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        // 反向传播，与正向传播相反，顺时针将误差旋转回来
        int d_model = delta.row_num();
        if (d_model != m_rope.get_d())
        {
            throw std::runtime_error("Input dimension does not match RoPE dimension");
        }
        if (m_layout == rope_pair_layout::half_split)
        {
            // 正向是 S·R·G，G/S 互为逆置换，故反向为 S·R^T·G（同一个搬运方向）
            auto permuted = gather_to_interleaved(delta);
            return scatter_from_interleaved(interleaved_backward(permuted));
        }
        return interleaved_backward(delta);
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
           << ", layout:" << rope_pair_layout_name(m_layout)
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

    /** 获取（或创建）维度为 d、配对约定为 layout 的 RoPE；可选预热 max_seq_len */
    rope_ptr get(int d, int max_seq_len = 0,
                 rope_pair_layout layout = rope_pair_layout::interleaved)
    {
        if (d <= 0 || d % 2 != 0)
            throw std::invalid_argument("RoPE registry d must be positive even");

        const auto key = std::make_pair(d, static_cast<int>(layout));
        auto it = m_pool.find(key);
        if (it != m_pool.end())
        {
            if (max_seq_len > 0)
                it->second->reserve(max_seq_len);
            return it->second;
        }

        auto rope = std::make_shared<rope_net_type>(d);
        rope->set_pair_layout(layout);
        rope->set_cache_mode(m_default_mode);
        const int reserve_len = max_seq_len > 0 ? max_seq_len : m_default_max_seq;
        if (m_default_mode == rope_cache_mode::static_fixed || max_seq_len > 0)
            rope->reserve(reserve_len);

        m_pool.emplace(key, rope);
        return rope;
    }

    bool contains(int d, rope_pair_layout layout = rope_pair_layout::interleaved) const
    {
        return m_pool.find(std::make_pair(d, static_cast<int>(layout))) != m_pool.end();
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

    std::map<std::pair<int, int>, rope_ptr> m_pool;
    rope_cache_mode m_default_mode = rope_cache_mode::dynamic;
    int m_default_max_seq = 1024;
};

} // namespace jasmine
#endif
