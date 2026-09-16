#ifndef __JAS_ROPE_HPP__
#define __JAS_ROPE_HPP__

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include "jas_mat_view_t.hpp"
#include "jas_mat_express_t.hpp"

namespace jasmine {

/**
 * dynamic: 按需扩容 —— 补齐与扩容都在锁内完成，且判「已填充」的范围永远是一个
 *          真被写满的矩形；扩容是「拷进新对象 + 原子换指针」，已发布的缓冲对象
 *          不再改动，所以读者拿到的视图在并发扩容下依然稳定（读路径只做两次
 *          acquire 读 + 一次指针 acquire 读，不加锁）。
 *          代价是扩容时旧对象不释放（已交给别人的视图还指着它），见 mat_cache_t::m_buffers。
 * static_fixed: 预先 reserve 一次填满，运行期不写、不扩容，最强约束。
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
    /**
     * 所有缓冲对象：当前的一个 + 退休的若干。所有权都在这里。
     *
     * 为什么不是「一个 mat_t 成员，扩容时原地换掉它」：
     * mat_view_t 里存的是 `mat_t*`，**每次取元素都会重新解引用那个对象**
     * （读它的 data 指针与维度，见 jas_mat_view_t.hpp）。于是只要扩容原地替换了
     * 那个成员，任何一个已经交出去的视图在并发扩容下就会读到一个正在被改写的
     * 对象 —— 这不是「值偏差」，是 UB：实测会段错误，TSan 报在
     * `mat_t::col_num()`（读者）与 `ensure_capacity()`（写者）之间。
     * （之前之所以看不见：构造函数预分配了 1024×1024，小规模用例压根走不到扩容。）
     *
     * 现在的约定是「对象一经发布就不再改动」：
     *   扩容 = 拷进一个新对象，再把 m_cur 原子换过去；
     *   旧对象留在 m_buffers 里不析构，已经发出去的视图继续有效。
     * 于是读者只需一次 acquire 载入指针，之后拿到的维度与 data 都是稳定的；
     * 写者只写「已发布范围之外」的格子，与读者读的格子不重叠，故无数据竞争。
     *
     * 代价是内存：扩容按 1.5 倍几何增长，退休总量收敛 —— 在职 + 退休 ≤ 3 倍最终容量。
     */
    std::vector<std::unique_ptr<mat_t<val_type>>> m_buffers;
    /** 当前缓冲对象。读方用 acquire，写方（持锁）用 relaxed */
    std::atomic<mat_t<val_type>*> m_cur;

    int m_cache_rows;      // 写方记账：grow_to 扩到过的范围
    int m_cache_cols;
    rope_cache_mode m_mode;

    mat_t<val_type>* cache_write()
    {
        return m_cur.load(std::memory_order_relaxed);
    }

    mat_t<val_type>* cache_read() const
    {
        return m_cur.load(std::memory_order_acquire);
    }

    void install(std::unique_ptr<mat_t<val_type>> fresh)
    {
        m_buffers.push_back(std::move(fresh));
        m_cur.store(m_buffers.back().get(), std::memory_order_release);
    }

    void ensure_capacity(int rows, int cols)
    {
        mat_t<val_type>* cur = cache_write();
        if (rows <= cur->row_num() && cols <= cur->col_num())
            return;

        if (m_mode == rope_cache_mode::static_fixed)
        {
            throw std::runtime_error(
                "RoPE static cache capacity exceeded; call reserve() with a larger max_seq_len");
        }

        // 纯几何增长：够用就行，其次按 1.5 倍扩（避免逐位置增长时 O(n) 次重分配）。
        // 刻意不设「绝对行/列数下限」——RoPE 的行数是 d_head，几个到上百个，
        // 给它加个 1024 的下限等于每次都为小维度白分配 1024 行：
        // d_head=4 时是 8 MiB 对 32 KB 的差别（这就是之前构造函数预分配
        // EXPAND_SIZE×EXPAND_SIZE 的代价，也与「精确预留」的目标正好相反）。
        int new_rows = cur->row_num();
        if (rows > new_rows)
            new_rows = std::max(rows, cur->row_num() * 3 / 2);
        int new_cols = cur->col_num();
        if (cols > new_cols)
            new_cols = std::max(cols, cur->col_num() * 3 / 2);

        auto fresh = std::make_unique<mat_t<val_type>>(new_rows, new_cols);
        for (int i = 0; i < cur->row_num(); ++i)
            for (int j = 0; j < cur->col_num(); ++j)
                (*fresh)(i, j) = (*cur)(i, j);
        install(std::move(fresh));
    }

public:
    mat_cache_t()
        : m_cache_rows(0), m_cache_cols(0), m_mode(rope_cache_mode::dynamic)
    {
        install(std::make_unique<mat_t<val_type>>());   // 空对象起步，不预分配
    }

    void set_mode(rope_cache_mode mode)
    {
        m_mode = mode;
    }

    rope_cache_mode mode() const
    {
        return m_mode;
    }

    int capacity_rows() const { return cache_read()->row_num(); }
    int capacity_cols() const { return cache_read()->col_num(); }
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
            mat_t<val_type>* cur = cache_write();
            if (rows != cur->row_num() || cols != cur->col_num())
                install(std::make_unique<mat_t<val_type>>(rows, cols));
            m_cache_rows = 0;
            m_cache_cols = 0;
            return;
        }

        ensure_capacity(rows, cols);
    }

    /**
     * 只把容量和「已准备区间」扩到 rows × cols，不写值、不判断哪些格子需要填。
     * 写值的记账由 mat_RoPE_t 负责（它必须精确到「哪些格子真写过」）。
     */
    void grow_to(int rows, int cols)
    {
        ensure_capacity(rows, cols);
        m_cache_rows = std::max(m_cache_rows, rows);
        m_cache_cols = std::max(m_cache_cols, cols);
    }

    /** 写入用：容量已由 grow_to 保证，这里不做扩容、不改记账 */
    val_type& cell(int row, int col)
    {
        return (*cache_write())(row, col);
    }

    /**
     * 共享读用：不加锁、不改任何状态，直接给视图。
     * 调用方（mat_RoPE_t）必须自己保证 [row, row+rows) × [col, col+cols) 已经写满。
     *
     * 这里只做一次 acquire 载入当前对象——之后视图解引用的就是这个对象，
     * 而对象一经发布不再改动（扩容是换新对象），所以并发扩容不会让视图读到半个对象。
     */
    mat_view_t<mat_t<val_type>> raw_view(int row, int col, int rows, int cols)
    {
        return cache_read()->view(row, col, rows, cols);
    }

    /** 只读视图：不扩容，越界抛异常（供 static 热路径） */
    mat_view_t<mat_t<val_type>> range_readonly(int row, int col, int rows, int cols) const
    {
        mat_t<val_type>* cur = cache_read();
        if (row + rows > cur->row_num() || col + cols > cur->col_num())
        {
            throw std::out_of_range("RoPE cache range out of capacity");
        }
        return cur->view(row, col, rows, cols);
    }

    /** 已退休（非当前）缓冲对象占用的字节数；static 预留路径应当为 0（不经历扩容） */
    std::size_t retired_bytes() const
    {
        mat_t<val_type>* cur = cache_read();
        std::size_t total = 0;
        for (const auto& m : m_buffers)
        {
            if (m.get() == cur)
                continue;
            total += static_cast<std::size_t>(m->row_num())
                   * static_cast<std::size_t>(m->col_num()) * sizeof(val_type);
        }
        return total;
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
    /**
     * 已填充范围的上限：保证 [0, m_filled_rows) × [0, m_filled_cols) 每个格子都真写过。
     * 读路径只用它做「够不够」的判断，所以发布必须用 release（写完再置位），
     * 读用 acquire；这样别的线程一旦看到上限，对应的值一定已经可见。
     */
    std::atomic<int> m_filled_rows;
    std::atomic<int> m_filled_cols;
    /** 串行化「扩容 + 补齐」；读命中已填充区域时不进去，所以不会成为读的瓶颈 */
    std::mutex m_fill_mutex;
    int m_d;
    int m_max_seq_len;              // static 模式下预留的最大序列长；dynamic 为 0 表示未限制
public:
    mat_RoPE_t(int const& d)
        : m_filled_rows(0), m_filled_cols(0), m_d(d), m_max_seq_len(0)
    {
    }

    void set_cache_mode(rope_cache_mode mode)
    {
        m_cache.set_mode(mode);
        m_filled_rows.store(0, std::memory_order_relaxed);
        m_filled_cols.store(0, std::memory_order_relaxed);
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
        m_filled_rows.store(0, std::memory_order_relaxed);
        m_filled_cols.store(0, std::memory_order_relaxed);
        init(0, rows, 0, cols);
    }

    void set_d(int const& d)
    {
        m_d = d;
        m_filled_rows.store(0, std::memory_order_relaxed);
        m_filled_cols.store(0, std::memory_order_relaxed);
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

    /** 退休存储占用的字节数；static 预留路径应当为 0（不经历扩容） */
    std::size_t retired_bytes() const
    {
        return m_cache.retired_bytes();
    }

    /**
     * 补齐缓存：保证 [0, row_beg+rows) × [0, col_beg+cols) 里每个格子都已写入。
     *
     * 缓存内容的布局：
     * | cos(m * A_i), -sin(m * A_i) |
     * | sin(m * A_i),  cos(m * A_i) |
     * 其中 A_i = 1 / 10000^(2i/d)，m 是位置索引，i 是维度对索引（特征下标 n 对应 i = n/2），
     * d 是模型维度。缓存行按特征二维对展开，列按位置展开为 2x2 块；同一块共用 θ = m / 10000^(2i/d)。
     *
     * 只增不减、只补齐不重算，并且永远从原点补齐（调用方都传 row_beg = col_beg = 0）。
     * 需要补的因此总是「L 形带」：新来的行配全部列 + 老行配新来的列。
     *
     * 不能在请求到的矩形里逐格判断「已初始化」再按两轴的 max 记账：先补 (d, 2) 再补 (2, 8)
     * 时第二次只遍历 2 行，却把列数记到 8，于是 [2, d) × [2, 8) 成了「自称已填、其实从没写过」
     * 的格子 —— 读到的是一律 0 的旋转矩阵（mat_t 分配时 memset 过），数值会静默偏掉。
     */
    void init(int const& row_beg, int const& rows, int const& col_beg, int const& cols)
    {
        std::lock_guard<std::mutex> lk(m_fill_mutex);
        fill_from_origin(row_beg + rows, col_beg + cols);
    }

    mat_view_t<mat_t<val_type>> range(int row_beg, int row_num, int col_beg, int col_num)
    {
        const int row_end = row_beg + row_num;
        const int col_end = col_beg + col_num;

        // 快路径：只做两次 acquire 读，不碰任何共享状态。命中已填充区时多线程可以随便读。
        if (row_end <= m_filled_rows.load(std::memory_order_acquire) &&
            col_end <= m_filled_cols.load(std::memory_order_acquire))
        {
            if (m_cache.mode() == rope_cache_mode::static_fixed)
                return m_cache.range_readonly(row_beg, col_beg, row_num, col_num);
            return m_cache.raw_view(row_beg, col_beg, row_num, col_num);
        }

        // 慢路径：扩容 + 补齐，串行化（fill_from_origin 里会重新读一次上限并取 max，
        // 所以两个线程同时要走慢路径时不会丢更新）
        init(0, row_end, 0, col_end);

        if (m_cache.mode() == rope_cache_mode::static_fixed)
        {
            if (row_end > m_filled_rows.load(std::memory_order_acquire) ||
                col_end > m_filled_cols.load(std::memory_order_acquire))
            {
                throw std::out_of_range(
                    "RoPE static cache miss; increase reserve(max_seq_len)");
            }
            return m_cache.range_readonly(row_beg, col_beg, row_num, col_num);
        }
        return m_cache.raw_view(row_beg, col_beg, row_num, col_num);
    }

    mat_view_t<mat_t<val_type>> forward_unite(int const& i, int const& m)
    {
        return range(i * 2, 2, m * 2, 2);
    }

private:
    /** 前置：已持有 m_fill_mutex。把已填充矩形扩到至少 row_end × col_end */
    void fill_from_origin(int row_end, int col_end)
    {
        // 静态模式：容量就是上限，越界直接报「没预留够」（与旧行为一致，且绝不在运行期扩容）
        if (m_cache.mode() == rope_cache_mode::static_fixed &&
            (m_max_seq_len <= 0 || row_end > m_cache.capacity_rows() ||
             col_end > m_cache.capacity_cols()))
        {
            throw std::out_of_range(
                "RoPE static cache miss; increase reserve(max_seq_len)");
        }

        const int have_rows = m_filled_rows.load(std::memory_order_relaxed);
        const int have_cols = m_filled_cols.load(std::memory_order_relaxed);
        row_end = std::max(row_end, have_rows);
        col_end = std::max(col_end, have_cols);

        if (row_end > have_rows || col_end > have_cols)
        {
            // 扩容先行：旧存储会退休而不是析构，别人手里的视图不会悬空
            m_cache.grow_to(row_end, col_end);
            fill_rect(have_rows, row_end, 0, col_end);       // 新行 × 全部列
            fill_rect(0, have_rows, have_cols, col_end);     // 老行 × 新列
        }

        // 先写完再发布：读者 acquire 到上限时，值一定已经就位
        m_filled_rows.store(row_end, std::memory_order_release);
        m_filled_cols.store(col_end, std::memory_order_release);
    }

    /** 前置：已持有 m_fill_mutex，且容量已经够 */
    void fill_rect(int row_beg, int row_end, int col_beg, int col_end)
    {
        for (int i = row_beg; i < row_end; ++i)
        {
            for (int j = col_beg; j < col_end; ++j)
            {
                const int pair = i / 2;
                const int pos = j / 2;
                const val_type angle = static_cast<val_type>(pos)
                    / std::pow(static_cast<val_type>(10000.0),
                               (static_cast<val_type>(2 * pair)) / static_cast<val_type>(m_d));
                const val_type c = std::cos(angle);
                const val_type s = std::sin(angle);
                if (i % 2 == 0 && j % 2 == 0)    // cos
                {
                    m_cache.cell(i, j) = c;
                }
                else if (i % 2 == 1 && j % 2 == 1)   // cos
                {
                    m_cache.cell(i, j) = c;
                }
                else if (i % 2 == 0 && j % 2 == 1)   // -sin
                {
                    m_cache.cell(i, j) = -s;
                }
                else                            // sin
                {
                    m_cache.cell(i, j) = s;
                }
            }
        }
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

    /** static 模式下已预留的序列长；dynamic 或未预留时为 0 */
    int cache_max_seq_len() const
    {
        return m_rope.max_seq_len();
    }

    /** 扩容时退休的旧存储字节数（static 预留路径应为 0） */
    std::size_t cache_retired_bytes() const
    {
        return m_rope.retired_bytes();
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
