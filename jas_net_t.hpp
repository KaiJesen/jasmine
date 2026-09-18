#ifndef __JAS_NET_T_HPP__
#define __JAS_NET_T_HPP__

#include <string>
#include <sstream>

#include "jas_mat_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_storage.hpp"

#include "jas_updator_t.hpp"

namespace jasmine {

template <typename input_type, template<typename> class updator_type>
class weight_net_t
{
public:
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_weight;
    updator_type<val_type> m_weight_updator;
    mat_t<val_type> m_bias;
    updator_type<val_type> m_bias_updator;

    mat_t<val_type> m_input;
public:
    weight_net_t(int const& input_size = 1, int const& output_size = 1)
        : m_weight(output_size, input_size), m_weight_updator(), m_bias(output_size, 1), m_bias_updator()
    {
        // 初始化权重和偏置
    }

    void reinit(std::vector<int> const& container)      // 初始化权重矩阵的维度，以为权重初始化准备
    {
        m_weight.reshape(container[1], container[0]);
        m_bias.reshape(container[1], 1);
    }

    /** 权重矩阵 [out, in]；供权重加载器直接写入 */
    mat_t<val_type>& weight() { return m_weight; }
    mat_t<val_type> const& weight() const { return m_weight; }
    /** 偏置 [out, 1]；供权重加载器直接写入或置零 */
    mat_t<val_type>& bias() { return m_bias; }
    mat_t<val_type> const& bias() const { return m_bias; }

    // m_input 持久化供 backward；mat rvalue 在层间移动，表达式只物化一次
    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        // 偏置必须分两步加：`W.dot(x) + b` 会让 mat_add_t::clone() 逐元素求值，
        // 每个输出元素自己去扫一遍 K，整次矩阵乘退回朴素循环、**碰不到 BLAS**。
        // 实测（M=64,N=1024,K=288）：表达式 ~100ms，拆两句 7.2ms。详见 TESTING.md 第 11 节。
        mat_t<val_type> out = m_weight.dot(m_input);
        out += m_bias;
        return out;
    }

    /** 无状态层：单列输入与整段 forward 相同 */
    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename init_type>
    void init_weight()
    {
        init_matrix<init_type>(m_weight);
        init_matrix<init_type>(m_bias);
    }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_weight_updator.set(std::forward<upr_arg_types>(args)...);
        m_bias_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_weight_updator.set_lr(lr);
        m_bias_updator.set_lr(lr);
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        mat_t<val_type> delta_weight = delta.dot(m_input.t());
        auto delta_bias = hsum(delta);
        mat_t<val_type> ret = m_weight.t().dot(delta);
        // 更新权重和偏置
        m_weight_updator.update(delta_weight, m_weight);
        m_bias_updator.update(delta_bias, m_bias);
        return ret;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "weight_net_t:(in:" << m_weight.col_num() << ", out:" << m_weight.row_num() << ")";
        return ss.str();
    }

    void step()
    {
        m_weight_updator.step();
        m_bias_updator.step();
    }
};

template <typename input_type>
class sigmoid_net_t
{
public:
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_output;                // 前一次的输入，反向传播时用于快捷计算
public:
    sigmoid_net_t() = default;

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        m_output = sigmoid(std::forward<Src>(input));
        return m_output;
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    auto backward(const mat_t<val_type>& delta)
    {
        if (delta.row_num() != m_output.row_num() || delta.col_num() != m_output.col_num())
        {
            throw std::runtime_error("delta size does not match input size");
        }
        return mat_t<val_type>(delta * (static_cast<val_type>(1) - m_output) * m_output);
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "sigmoid_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 什么也不做
    }

    void step()
    {
        // 什么也不做
    }
};

template <typename input_type>
class relu_net_t
{
private:
    using val_type = typename input_type::ele_type;
    mat_t<val_type> m_input;                // 前一次的输入，反向传播时用于快捷计算
public:
    relu_net_t() = default;

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        return (m_input > 0) * m_input;
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    auto backward(const other_type& delta)
    {
        if (delta.row_num() != m_input.row_num() || delta.col_num() != m_input.col_num())
        {
            throw std::runtime_error("delta size does not match input size");
        }
        return mat_t<val_type>(delta * (m_input > 0));
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "relu_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 什么也不做
    }

    void step()
    {
        // 什么也不做
    }
};

/**
 * Dropout 层（inverted dropout）。
 *
 *   forward : 每个元素以 keep = 1-p 的概率保留，保留下来的乘以 1/keep —— 这样输出的期望不变，
 *             推理时直接恒等即可，不需要额外补偿（这就是 "inverted" 的含义）。
 *   backward: delta ⊙ mask（forward 被丢掉的位置梯度为 0）
 *
 * 与 flatten / pool / mean_pool 一样是无参数静态层：没有 updator，`init_weight` / `step` 空实现，
 * 也没有 `reinit`（不占 complex_net_t::reinit 的容器槽位）。
 *
 * **训练/推理开关是显式的**：`set_enabled(false)` 让 forward 变成恒等（评估/推理时用）。
 * 之所以不做成「infer 时自动跳过」：那要求链上每一层都有 `forward_one`（本库的 `encoder_t`
 * 没有），会限制它出现在哪些链里；显式开关最简单，也让评估路径一目了然，并且保持了
 * 训练时前向的可复现性（随机数取自 `g_random_engine`）。
 */
template <typename input_type>
class dropout_net_t
{
public:
    // 公开：允许该层位于 complex_net 链首（complex_net_t 从首个成员取 val_type）
    using val_type = typename input_type::ele_type;
private:
    val_type m_p = val_type(0);         // 丢弃概率
    bool m_enabled = true;              // false = 恒等（评估/推理）
    mat_t<val_type> m_input;            // forward 缓存
    mat_t<val_type> m_mask;             // forward 缓存：保留处 = 1/(1-p)，丢弃处 = 0；空 = 恒等
public:
    dropout_net_t() = default;
    explicit dropout_net_t(val_type const& p) : m_p(p) {}

    void set_param(val_type const& p) { m_p = p; }
    val_type drop_probability() const { return m_p; }
    void set_enabled(bool const on) { m_enabled = on; }
    bool enabled() const { return m_enabled; }

    template <typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        const int rows = m_input.row_num();
        const int cols = m_input.col_num();

        if (!m_enabled || m_p <= val_type(0))
        {
            m_mask = mat_t<val_type>();                  // 标记为「恒等」
            return m_input;
        }
        if (m_p >= val_type(1))
            throw std::invalid_argument("dropout_net_t::forward: drop probability must be < 1");

        const val_type keep = val_type(1) - m_p;
        const val_type scale = val_type(1) / keep;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        mat_t<val_type> out(rows, cols);
        m_mask = mat_t<val_type>(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                const bool survive = uni(g_random_engine) < static_cast<double>(keep);
                m_mask(i, j) = survive ? scale : val_type(0);
                out(i, j) = m_input(i, j) * m_mask(i, j);
            }
        }
        return out;
    }

    /** 无状态层：单列输入与整段 forward 相同 */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        if (delta.row_num() != m_input.row_num() || delta.col_num() != m_input.col_num())
            throw std::runtime_error("dropout_net_t::backward: delta size does not match input size");
        if (!m_mask.valid())                             // forward 是恒等 → 梯度原样回传
            return mat_t<val_type>(delta);
        mat_t<val_type> out(m_input.row_num(), m_input.col_num());
        for (int i = 0; i < out.row_num(); ++i)
            for (int j = 0; j < out.col_num(); ++j)
                out(i, j) = delta(i, j) * m_mask(i, j);
        return out;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "dropout_net_t:(p:" << m_p
           << ", " << (m_enabled ? "train" : "eval") << ")";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 无权重
    }

    void step()
    {
        // 无权重
    }
};

/**
 * 序列均值池化层：把 [d_model, T] 的 token 序列压成 [d_model, 1]。
 *
 * 用途是 Transformer 分类头的前半段（encoder → **mean pool** → linear → CE）：对 T 个 token
 * 取平均，得到一个与序列长度无关的向量。
 *
 * forward: out(i,0) = (1/T) * Σ_t input(i,t)
 * backward: dL/dinput(i,t) = delta(i,0) / T —— 梯度平均分摊回每一列（与 mean 的定义一致）
 *
 * 与 flatten / pooling / relu 一样是无参数静态层：不持有 updator，`init_weight` / `step`
 * 空实现，也没有 `reinit`（不占 complex_net_t::reinit 的容器槽位）。
 */
template <typename input_type>
class mean_pool_net_t
{
public:
    // 公开：允许该层位于 complex_net 链首（complex_net_t 从首个成员取 val_type）
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_input;    // forward 缓存（只需要列数，但保持与其它层一致的语义）

public:
    mean_pool_net_t() = default;

    template <typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        const int rows = m_input.row_num();
        const int cols = m_input.col_num();
        mat_t<val_type> out(rows, 1);
        for (int i = 0; i < rows; ++i)
        {
            val_type s = val_type(0);
            for (int t = 0; t < cols; ++t)
                s += m_input(i, t);
            out(i, 0) = s / static_cast<val_type>(cols);
        }
        return out;
    }

    /** 无状态层：单列输入与整段 forward 相同 */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        if (delta.row_num() != m_input.row_num() || delta.col_num() != 1)
            throw std::runtime_error("mean_pool_net_t::backward: delta must be [d_model, 1]");
        const int cols = m_input.col_num();
        mat_t<val_type> out(m_input.row_num(), cols);
        for (int i = 0; i < out.row_num(); ++i)
            for (int t = 0; t < cols; ++t)
                out(i, t) = static_cast<val_type>(delta(i, 0)) / static_cast<val_type>(cols);
        return out;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "mean_pool_net_t:(tokens:" << m_input.col_num() << ")";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 无权重
    }

    void step()
    {
        // 无权重
    }
};

/**
 * 展平层：把 [C, W] 的特征图按行优先展平成 [C*W, 1] 的单列向量，喂给全连接/编码器。
 *
 * 用途是 CNN 尾部「特征图 → 向量」这一步（conv→relu→pool→**flatten**→encoder）。
 * 语义上就是 `reshape_view`，但它必须是一个**层**才能参与 `complex_net_t` 的静态层堆叠：
 * forward 缓存输入形状、backward 把 [C*W, 1] 的梯度还原成 [C, W]。
 *
 * 与 relu / pooling 一样是无参数静态层：不持有 updator，`init_weight` / `step` 为空实现
 * （`complex_net_t::init_weight` / `step` 会对所有成员无条件调用它们），也没有 `reinit`
 * （形状由 `set_param` 给），所以不占用 `complex_net_t::reinit` 的容器槽位。
 *
 * 注意：本层按「单样本一列」的约定工作（T == 1），这与 conv/pool 只认单张图一致；
 * 多列批处理请走 `cache_updator_t` 的梯度累加，而不是把多个样本塞进一列。
 */
template <typename input_type>
class flatten_net_t
{
public:
    // 公开：允许该层位于 complex_net 链首（complex_net_t 从首个成员取 val_type）
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_input;    // forward 缓存，backward 用它还原形状
    int m_rows = 0;             // 期望的输入形状 [rows, cols]
    int m_cols = 0;

public:
    flatten_net_t() = default;

    /** 显式指定输入特征图形状（不指定则按首次 forward 的输入懒初始化） */
    void set_param(int const& rows, int const& cols)
    {
        m_rows = rows;
        m_cols = cols;
    }

    template <typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        if (m_rows == 0 || m_cols == 0)
        {
            m_rows = input.row_num();
            m_cols = input.col_num();
        }
        if (input.row_num() != m_rows || input.col_num() != m_cols)
            throw std::invalid_argument("flatten_net_t::forward: input must be [rows, cols] of set_param");

        detail::store_for_backward(m_input, std::forward<Src>(input));
        mat_t<val_type> out(m_rows * m_cols, 1);
        for (int i = 0; i < m_rows; ++i)
            for (int j = 0; j < m_cols; ++j)
                out(i * m_cols + j, 0) = m_input(i, j);     // 行优先展平，与 conv/pool 的布局一致
        return out;
    }

    /** 无状态层：单列输入与整段 forward 相同 */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        if (delta.row_num() != m_rows * m_cols || delta.col_num() != 1)
            throw std::runtime_error("flatten_net_t::backward: delta must be [rows*cols, 1]");
        mat_t<val_type> out(m_rows, m_cols);
        for (int i = 0; i < m_rows; ++i)
            for (int j = 0; j < m_cols; ++j)
                out(i, j) = delta(i * m_cols + j, 0);
        return out;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "flatten_net_t:(" << m_rows << "x" << m_cols
           << " -> " << m_rows * m_cols << "x1)";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 无权重
    }

    void step()
    {
        // 无权重
    }
};

// 纵向的标准化层，即对每一列（每个 token）在特征维上做 LayerNorm
template <typename input_type, template<typename> class updator_type>
class layer_norm_net_t
{
public:
    // 公开：complex_net_t 从首个成员取 val_type 推断整链类型（LN 可能位于链首，如 pre-norm 分支）
    using val_type = typename input_type::ele_type;
private:
    static constexpr val_type eps = static_cast<val_type>(1e-5);
    mat_t<val_type> m_hx;
    mat_t<val_type> m_mean;
    mat_t<val_type> m_std;
    mat_t<val_type> m_gama;     // 缩放参数 [d_model, 1]
    updator_type<val_type> m_gama_updator;
    mat_t<val_type> m_beta;     // 平移参数 [d_model, 1]
    updator_type<val_type> m_beta_updator;
public:
    layer_norm_net_t() = default;

    /**
     * 显式分配 gamma/beta 并置为恒等变换（gamma=1, beta=0）。
     *
     * 默认走懒初始化：首次 forward 依据输入行数分配。权重加载器需要在 forward 之前写入
     * gamma/beta，因此必须先调用本函数完成分配，否则访问到的是未分配的无效矩阵。
     *
     * 刻意不叫 reinit：is_reinitable_net 靠 `requires { net.reinit(std::vector<int>()); }`
     * 判定（见 jas_mat_concepts.hpp），加 reinit 会改变所有含 LayerNorm 的复杂网络的 reinit 语义。
     */
    void set_param(int const& d_model)
    {
        m_gama.reshape(d_model, 1);
        m_beta.reshape(d_model, 1);
        m_gama = val_type(1);
        m_beta = val_type(0);
    }

    /** 缩放参数 gamma [d_model, 1]；供加载器写入 */
    mat_t<val_type>& gama() { return m_gama; }
    mat_t<val_type> const& gama() const { return m_gama; }
    /** 平移参数 beta [d_model, 1]；供加载器写入 */
    mat_t<val_type>& beta() { return m_beta; }
    mat_t<val_type> const& beta() const { return m_beta; }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_gama_updator.set(std::forward<upr_arg_types>(args)...);
        m_beta_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_gama_updator.set_lr(lr);
        m_beta_updator.set_lr(lr);
    }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        // 对每一列在 row（特征）维上标准化
        m_mean = vmean(input);
        mat_t<val_type> centered = (input - m_mean).clone();
        mat_t<val_type> var = (vmean(pow(centered, 2.0)) + eps).clone();
        m_std = sqrt(var);
        m_hx = (centered / m_std).clone();
        if (m_gama.valid() == false)
        {
            m_gama = mat_t<val_type>(input.row_num(), 1);
            m_beta = mat_t<val_type>(input.row_num(), 1);
            m_gama = val_type(1);
            m_beta = val_type(0);
        }
        return (m_gama * m_hx + m_beta).clone();
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    auto backward(const other_type& delta)
    {
        // gamma/beta 梯度：沿序列维（列）累加
        auto L_gama = hsum(delta * m_hx);
        auto L_beta = hsum(delta);

        // 输入梯度：沿特征维（行）归约，需与 forward 的 vmean/vsum 一致
        val_type m = static_cast<val_type>(delta.row_num());
        auto dx_norm = delta * m_gama;
        auto sum_dx_norm = vsum(dx_norm);
        auto sum_dx_norm_x_hx = vsum(dx_norm * m_hx);
        mat_t<val_type> L_input =
            ((dx_norm * m - sum_dx_norm - m_hx * sum_dx_norm_x_hx) / m / m_std).clone();

        m_gama_updator.update(L_gama, m_gama);
        m_beta_updator.update(L_beta, m_beta);
        return L_input;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "layer_norm_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // LayerNorm 仿射参数在首次 forward 时懒初始化为 gamma=1, beta=0
    }

    void step()
    {
        m_gama_updator.step();
        m_beta_updator.step();
    }
};

/**
 * RMSNorm（Root Mean Square Normalization）：LayerNorm 的"去掉减均值"版本。
 *
 *     LayerNorm(x) = gamma ⊙ (x - mean(x)) / sqrt(var(x) + eps) + beta
 *     RMSNorm(x)   = gamma ⊙  x            / sqrt(mean(x²) + eps)
 *
 * 相比 layer_norm_net_t 少了三样东西：**不减均值**、**没有平移参数 beta**、
 * eps 加在"均方值"上而不是"方差"上。无参数外只有 gamma，且 gamma 恒为正缩放。
 *
 * 为什么删掉减均值还能work：归一化的关键作用是 **缩放不变性**（RMSNorm(c·x) = RMSNorm(x)，c>0），
 * 它让梯度不依赖激活值的绝对幅度，从而抑制爆炸/消失。重新中心化（减均值）是最不重要的一环：
 * 紧跟其后的仿射层本来就会重新引入偏置。代价是丢掉平移不变性（RMSNorm(x + c) ≠ RMSNorm(x)）。
 *
 * 实现代价：省掉一次行归约（不用算 mean），提速约 5~15%（主要省访存）。
 * 反向也精确地"少一项"：delta 里不需要减去 mean(g)，其余与 LayerNorm 同形。
 *
 * 与 layer_norm_net_t 一样是 stable 网络（仿射参数按 d_model 固定，不随输入形状变化），
 * 因此初始化接口叫 set_param 而不是 reinit —— 否则 is_reinitable_net 判定会变。
 */
template <typename input_type, template<typename> class updator_type>
class rms_norm_net_t
{
public:
    // 公开：complex_net_t 从首个成员取 val_type 推断整链类型（RMSNorm 常位于 pre-norm 链首）
    using val_type = typename input_type::ele_type;

    /** 默认 eps。注意：对齐开源权重时必须从模型 config 读（LLaMA 系 1e-5 / 1e-6 都有用） */
    static constexpr val_type kDefaultEps = static_cast<val_type>(1e-5);

private:
    val_type m_eps = kDefaultEps;
    mat_t<val_type> m_hx;       // 归一化后的 x/rms（不含 gamma），供反向使用
    mat_t<val_type> m_rms;      // 每列的均方根 [1, T]
    mat_t<val_type> m_gama;     // 缩放参数 [d_model, 1]（没有 beta）
    updator_type<val_type> m_gama_updator;

public:
    rms_norm_net_t() = default;

    /**
     * 显式分配 gamma 并置为恒等（gamma=1）。权重加载器需在 forward 之前写入 gamma，
     * 因此必须先调用本函数，否则访问到的是未分配的无效矩阵（同 layer_norm_net_t）。
     */
    void set_param(int const& d_model, val_type const& eps = kDefaultEps)
    {
        m_eps = eps;
        m_gama.reshape(d_model, 1);
        m_gama = val_type(1);
    }

    /** 单独设置 eps（对齐参考实现时用） */
    void set_eps(val_type const& eps) { m_eps = eps; }
    val_type eps() const { return m_eps; }

    /** 缩放参数 gamma [d_model, 1]；供加载器写入 */
    mat_t<val_type>& gama() { return m_gama; }
    mat_t<val_type> const& gama() const { return m_gama; }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_gama_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_gama_updator.set_lr(lr);
    }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        // 对每一列在 row（特征）维上求均方根 —— 不减均值，这是与 LayerNorm 的唯一实质差别
        // 注意：sqrt 要求实参已是 mat_t（它内部会按元素赋值），故先 clone 物化
        mat_t<val_type> ms = (vmean(pow(input, 2.0)) + m_eps).clone();
        m_rms = sqrt(ms);
        m_hx = (input / m_rms).clone();
        if (m_gama.valid() == false)
        {
            m_gama = mat_t<val_type>(input.row_num(), 1);
            m_gama = val_type(1);
        }
        return (m_gama * m_hx).clone();
    }

    /** 无 KV 状态：单列与整段等价 */
    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    auto backward(const other_type& delta)
    {
        // gamma 梯度与 LayerNorm 完全一致：沿序列维（列）累加
        auto L_gama = hsum(delta * m_hx);

        /* 输入梯度：与 layer_norm_net_t::backward 同形，但**没有** mean(g) 那一项。
         *   LayerNorm: (g·m - sum(g) - hx·sum(g⊙hx)) / m / std
         *   RMSNorm  : (g    -          hx·mean(g⊙hx))     / rms
         * 推导：u = x/r，r = sqrt(mean(x²)+eps)
         *   ∂u_j/∂x_i = δ_ij/r - x_j·x_i/(r³·d)
         *   ∂L/∂x_i   = g_i/r - x_i·Σ_j(g_j·x_j)/(r³·d)
         *             = (g_i - hx_i·mean(g⊙hx)) / r        （因 Σ_j g_j x_j / d = r·mean(g⊙hx)）
         */
        val_type const m = static_cast<val_type>(delta.row_num());
        auto dx_norm = delta * m_gama;
        auto sum_dx_norm_x_hx = vsum(dx_norm * m_hx);
        mat_t<val_type> L_input =
            ((dx_norm * m - m_hx * sum_dx_norm_x_hx) / m / m_rms).clone();

        m_gama_updator.update(L_gama, m_gama);
        return L_input;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "rms_norm_net_t(eps:" << m_eps << ")";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 仿射参数在首次 forward 时懒初始化为 gamma=1（同 LayerNorm）
    }

    void step()
    {
        m_gama_updator.step();
    }
};

template <typename input_type>
class hsoftmax_net_t
{
public:
    using val_type = typename input_type::ele_type;
    mat_t<val_type> m_output;

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        m_output = hsoftmax(std::forward<Src>(input));
        return m_output;
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        return mat_t<val_type>(m_output * (delta - hsum(m_output * delta)));
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "hsoftmax_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 不需要初始化权重
    }

    void step()
    {}
};

template <typename base_net_type>
class residual_net_t
{
public:
    using val_type = typename base_net_type::val_type;
    base_net_type m_net;

    residual_net_t()        // 入参没有什么作用，仅仅用于表示这是一个需要reinit的网络
        : m_net()
    {
    }

    base_net_type& base_net()
    {
        return m_net;
    }

    base_net_type const& base_net() const
    {
        return m_net;
    }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        mat_t<val_type> skip(std::forward<Src>(input));
        return m_net.forward(skip) + skip;
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        mat_t<val_type> skip(std::forward<Src>(input));
        return m_net.forward_one(skip) + skip;
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        return mat_t<val_type>(m_net.backward(delta) + delta);
    }
    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "residual_net_t:\n" << m_net.net_type(indent + 2);
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        m_net.template init_weight<init_type>();
    }

    template <size_t...nums>
    decltype(auto) get()
    {
        return m_net.template get<nums...>();
    }

    template <size_t...nums>
    decltype(auto) get() const
    {
        return m_net.template get<nums...>();
    }

    void step()
    {
        m_net.step();
    }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_net.set_updator(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_net.set_lr(lr);
    }
};

/**
 * 门控容器：两个分支共享同一输入，各自前向，结果逐元素相乘汇合。
 *
 *     out = gate_branch(x) ⊙ up_branch(x)
 *
 * 这就是 SwiGLU 的中间部分：gate 分支 = Linear→SiLU，up 分支 = Linear，
 * 对应 HuggingFace 的 `silu(gate_proj(x)) * up_proj(x)`。
 * **down_proj 不在本容器内**，它作为普通层接在容器后面 —— 与 residual_net_t 只负责
 * 加 skip、不负责分支内部的层一样：容器只管拓扑（分叉 + 汇合），层的内容由分支决定。
 *
 * 与 residual_net_t 的差别：residual 是「单分叉 + 加法合并」，本容器是「双分叉 +
 * 逐元素乘合并」。合并算子不同，且两分支不对称（通常只有一个带激活）。
 * 把门控函数换成 GELU/ReLU 即得 GEGLU/ReGLU，所以本容器不依赖任何具体激活。
 *
 * 反向要点：两条分支读的是**同一个 x**，由多变量链式法则 ∂L/∂x 是两条路径之和
 *     ∂L/∂x = ∂L/∂gate · ∂gate/∂x + ∂L/∂up · ∂up/∂x
 * 而逐元素乘的梯度就是「乘对方」：
 *     ∂L/∂gate = delta ⊙ up,   ∂L/∂up = delta ⊙ gate
 * 所以 forward 必须缓存两个分支的输出（容器本身只多存这两份矩阵）。
 */
template <typename gate_net_type, typename up_net_type>
class gated_net_t
{
public:
    using val_type = typename gate_net_type::val_type;
private:
    gate_net_type m_gate;
    up_net_type m_up;
    // 前向输出缓存，供 backward 计算 ∂L/∂gate = delta ⊙ up、∂L/∂up = delta ⊙ gate
    mat_t<val_type> m_gate_out;
    mat_t<val_type> m_up_out;

public:
    gated_net_t() = default;

    /** gate 分支（通常为 Linear→激活） */
    gate_net_type& gate_branch() { return m_gate; }
    gate_net_type const& gate_branch() const { return m_gate; }
    /** up 分支（通常为纯 Linear，不过激活） */
    up_net_type& up_branch() { return m_up; }
    up_net_type const& up_branch() const { return m_up; }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        // 物化一份：两个分支都要用同一个 x（与 residual_net_t 先存 skip 同理）
        mat_t<val_type> x(std::forward<Src>(input));
        m_gate_out = m_gate.forward(x);
        m_up_out = m_up.forward(x);
        return m_gate_out * m_up_out;       // 逐元素乘（Hadamard），非矩阵乘
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        mat_t<val_type> x(std::forward<Src>(input));
        m_gate_out = m_gate.forward_one(x);
        m_up_out = m_up.forward_one(x);
        return m_gate_out * m_up_out;
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        if (delta.row_num() != m_gate_out.row_num() || delta.col_num() != m_gate_out.col_num())
        {
            throw std::runtime_error("delta size does not match output size");
        }
        mat_t<val_type> d(delta);
        mat_t<val_type> delta_gate = d * m_up_out;      // ∂L/∂gate
        mat_t<val_type> delta_up = d * m_gate_out;      // ∂L/∂up
        // 两条分支梯度相加：它们共享同一个输入 x
        return m_gate.backward(delta_gate) + m_up.backward(delta_up);
    }

    /**
     * 用同一个 {in, out} 配置两个分支。
     * 仅当两分支形状相同时才成立 —— SwiGLU 的 gate_proj/up_proj 都是 d_model→d_ff，正合此约定。
     * 形状不同的分支请用 gate_branch()/up_branch() 各自 reinit。
     * requires 子句保证：两分支都无权重时本层被视为静态层，不占用 reinit 的容器槽位。
     */
    void reinit(std::vector<int> const& container)
        requires (is_reinitable_net<gate_net_type> || is_reinitable_net<up_net_type>)
    {
        if constexpr (is_reinitable_net<gate_net_type>)
            m_gate.reinit(container);
        if constexpr (is_reinitable_net<up_net_type>)
            m_up.reinit(container);
    }

    template<typename init_type>
    void init_weight()
    {
        m_gate.template init_weight<init_type>();
        m_up.template init_weight<init_type>();
    }

    void step()
    {
        m_gate.step();
        m_up.step();
    }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        if constexpr (is_updatable_net<gate_net_type>)
            m_gate.set_updator(std::forward<upr_arg_types>(args)...);
        if constexpr (is_updatable_net<up_net_type>)
            m_up.set_updator(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        if constexpr (requires(gate_net_type& net, val_type v) { net.set_lr(v); })
            m_gate.set_lr(lr);
        if constexpr (requires(up_net_type& net, val_type v) { net.set_lr(v); })
            m_up.set_lr(lr);
    }

    /** 分支内的 KV cache 清理（FFN 用不到，但让容器对含注意力的分支也成立） */
    void infer_reset()
    {
        if constexpr (requires { m_gate.infer_reset(); })
            m_gate.infer_reset();
        else if constexpr (requires { m_gate.clear_kv_cache(); })
            m_gate.clear_kv_cache();
        if constexpr (requires { m_up.infer_reset(); })
            m_up.infer_reset();
        else if constexpr (requires { m_up.clear_kv_cache(); })
            m_up.clear_kv_cache();
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "gated_net_t:\n"
           << print_indent(indent + 2) << "gate:\n" << m_gate.net_type(indent + 4) << "\n"
           << print_indent(indent + 2) << "up:\n" << m_up.net_type(indent + 4);
        return ss.str();
    }

    /** get<0> = gate 分支，get<1> = up 分支；多级下标透传给对应分支（与 residual_net_t 对称） */
    template <size_t N, size_t...nums>
    decltype(auto) get()
    {
        if constexpr (N == 0)
        {
            if constexpr (sizeof...(nums) == 0) return (m_gate);
            else return m_gate.template get<nums...>();
        }
        else
        {
            static_assert(N == 1, "gated_net_t 只有两个分支：get<0> = gate, get<1> = up");
            if constexpr (sizeof...(nums) == 0) return (m_up);
            else return m_up.template get<nums...>();
        }
    }

    template <size_t N, size_t...nums>
    decltype(auto) get() const
    {
        if constexpr (N == 0)
        {
            if constexpr (sizeof...(nums) == 0) return (m_gate);
            else return m_gate.template get<nums...>();
        }
        else
        {
            static_assert(N == 1, "gated_net_t 只有两个分支：get<0> = gate, get<1> = up");
            if constexpr (sizeof...(nums) == 0) return (m_up);
            else return m_up.template get<nums...>();
        }
    }
};

template <typename... net_types>
class complex_net_t
{
private:
    std::tuple<net_types...> m_nets;
public:
    using val_type = typename std::tuple_element_t<0, std::tuple<net_types...>>::val_type;

    template <typename input_type>
    auto forward(input_type&& input)
    {
        return std::apply([&input](auto&&... nets) {
            return net_forward(std::forward<input_type>(input), nets...);
        }, m_nets);
    }

    /**
     * 推理前向：与 forward 同一套 net 顺序；对 skip_on_infer 层（通常为 loss）跳过不调用，
     * 输入原样继续后续层。各层走 forward_one（有 KV 的层可增量，其余默认 ≡ forward）。
     */
    template <typename input_type>
    auto infer(input_type&& input)
    {
        return infer_chain<0>(std::forward<input_type>(input));
    }

    /** 新序列推理前：递归清除子网中的 KV cache（若存在） */
    void infer_reset()
    {
        infer_reset_chain<0>();
    }

    /** 可选：为含 reserve_kv_cache 的子网预分配容量 */
    void infer_prepare(int max_kv_seq = 0)
    {
        infer_prepare_chain<0>(max_kv_seq);
    }

    /** decoder 等无 loss 尾的子网：整链 forward_one；含 loss 尾时请用 infer() */
    template <typename input_type>
    auto forward_one(input_type&& input)
    {
        return std::apply([&input](auto&&... nets) {
            return net_forward_one(std::forward<input_type>(input), nets...);
        }, m_nets);
    }

    template <typename input_type>
    auto backward(const input_type& delta)
    {
        return std::apply([&delta](auto&&... nets) {return net_backward(delta, nets...); }, m_nets);
    }

    constexpr size_t size()
    {
        return sizeof...(net_types);
    }

    template <typename container_type, size_t N = 0, size_t I = 0>
    void reinit(container_type const& container)      // 如果是稳定网络则不需要重新初始化
    {
        using mbr_net_type = std::tuple_element_t<N, std::tuple<net_types...>>;
        if constexpr (is_reinitable_net<mbr_net_type>)    // 有状态的网络层需要初始化权重
        {
            if (I + 1 >= container.size())
            {
                throw std::runtime_error("container size does not match net size");
            }
            std::get<N>(m_nets).reinit({container[I], container[I + 1]});   // 初始化权重
            if constexpr (N + 1 < sizeof...(net_types))
            {
                reinit<container_type, N + 1, I + 1>(container);
            }
        }
        else                                            // 无状态的网络层不需要初始化权重
        {
            if constexpr (N + 1 < sizeof...(net_types))
            {
                reinit<container_type, N + 1, I>(container);
            }
        }
    }

    template <size_t N, typename...upr_arg_types>
    void set_updator__(upr_arg_types&&... args)
    {
        using mbr_net_type = std::tuple_element_t<N, std::tuple<net_types...>>;
        if constexpr (is_updatable_net<mbr_net_type>)
        {
            std::get<N>(m_nets).set_updator(std::forward<upr_arg_types>(args)...);
        }
        if constexpr (N + 1 < sizeof...(net_types))
        {
            set_updator__<N + 1, upr_arg_types...>(std::forward<upr_arg_types>(args)...);
        }
    }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        set_updator__<0, upr_arg_types...>(std::forward<upr_arg_types>(args)...);
    }

    template <size_t N>
    void set_lr__(val_type lr)
    {
        using mbr_net_type = std::tuple_element_t<N, std::tuple<net_types...>>;
        if constexpr (requires(mbr_net_type& net, val_type v) { net.set_lr(v); })
        {
            std::get<N>(m_nets).set_lr(lr);
        }
        if constexpr (N + 1 < sizeof...(net_types))
        {
            set_lr__<N + 1>(lr);
        }
    }

    void set_lr(val_type lr)
    {
        set_lr__<0>(lr);
    }

    template <typename init_type>
    void init_weight()
    {
        std::apply([](auto&&... nets) {((nets.template init_weight<init_type>()),...); }, m_nets);
    }

    void step()
    {
        std::apply([](auto&&... nets) {((nets.step()),...); }, m_nets);
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "complex_net_t = [";
        std::apply([&ss, indent](auto&&... nets) {((ss << std::endl << nets.net_type(indent + 2)),...); }, m_nets);
        ss << std::endl
        << print_indent(indent) << "]";
        return ss.str();
    }

    auto back()
    {
        return std::get<sizeof...(net_types) - 1>(m_nets);
    }

    template<size_t N, size_t...nums>
    decltype(auto) get()
    {
        if constexpr (sizeof...(nums) == 0)
        {
            return std::get<N>(m_nets);
        }
        else
        {
            return std::get<N>(m_nets).template get<nums...>();
        }
    }

    template<size_t N, size_t...nums>
    decltype(auto) get() const
    {
        if constexpr (sizeof...(nums) == 0)
        {
            return std::get<N>(m_nets);
        }
        else
        {
            return std::get<N>(m_nets).template get<nums...>();
        }
    }

private:
    template <size_t I, typename Input>
    auto infer_chain(Input&& input)
    {
        if constexpr (I >= sizeof...(net_types))
            return std::forward<Input>(input);
        else
        {
            using net_type_at_i = std::tuple_element_t<I, std::tuple<net_types...>>;
            if constexpr (is_infer_skipped_net<net_type_at_i>::value)
                return infer_chain<I + 1>(std::forward<Input>(input));
            else
            {
                auto& net = std::get<I>(m_nets);
                return infer_chain<I + 1>(net.forward_one(std::forward<Input>(input)));
            }
        }
    }

    template <size_t I>
    void infer_reset_chain()
    {
        if constexpr (I >= sizeof...(net_types))
            return;
        else
        {
            auto& net = std::get<I>(m_nets);
            if constexpr (requires { net.infer_reset(); })
                net.infer_reset();
            else if constexpr (requires { net.clear_kv_cache(); })
                net.clear_kv_cache();
            infer_reset_chain<I + 1>();
        }
    }

    template <size_t I>
    void infer_prepare_chain(int max_kv_seq)
    {
        if constexpr (I >= sizeof...(net_types))
            return;
        else
        {
            auto& net = std::get<I>(m_nets);
            if constexpr (requires { net.infer_prepare(max_kv_seq); })
                net.infer_prepare(max_kv_seq);
            else if constexpr (requires { net.clear_kv_cache(); })
                net.clear_kv_cache();
            if constexpr (requires { net.reserve_kv_cache(max_kv_seq); })
            {
                if (max_kv_seq > 0)
                    net.reserve_kv_cache(max_kv_seq);
            }
            infer_prepare_chain<I + 1>(max_kv_seq);
        }
    }
};

/** 输出投影：d_model → vocab logits（weight_net 语义别名） */
template <typename input_type, template <typename> class updator_type>
using output_proj_net_t = weight_net_t<input_type, updator_type>;

/*
 * 复杂网络构造器存在意义说明：如果直接构建复杂网络需要一次性输入各层的网络实例，不够灵活，且不够清晰。复杂网络构造器则提供了一套接口，可以逐步构建复杂网络的结构，并且在构建过程中可以清晰地看到每一步的网络结构变化，同时也可以在构建过程中设置每一层的参数，最后一步才生成复杂网络实例。
 * 并且可以不用为每层网络设置val_type参数，复杂网络构造器会自动推断出每层网络的val_type参数，避免了重复输入参数的麻烦。
 */
template <typename val_type, typename...net_types>
struct complex_net_builder_t
{
    template<template<typename, template<typename> class> class cur_net_tpl, template<typename> class updator_tpl>
    using push_back_updatable = complex_net_builder_t<val_type, net_types..., cur_net_tpl<mat_t<val_type>, updator_tpl>>;

    template<template<typename> class cur_net_tpl>
    using push_back_staticnet = complex_net_builder_t<val_type, net_types..., cur_net_tpl<mat_t<val_type>>>;

    template<typename new_net_type>
    using push_back_impl = complex_net_builder_t<val_type, net_types..., new_net_type>;

    using type = complex_net_t<net_types...>;

};

/**
 * 门控 FFN 的两条分支：gate = Linear→act_net_tpl，up = Linear。
 * 直接可用作 gated_net_t 的模板实参，激活类型作为参数传入：
 *     SwiGLU: act_net_tpl = silu_net_t    GEGLU: gelu_net_t    ReGLU: relu_net_t
 * 完整的前馈还要把本容器接一个 down_proj（d_ff→d_model）在后面。
 * 注意 gate/up 形状相同（都是 d_model→d_ff），因此 gated_net_t::reinit 传一份 {d_model, d_ff} 即可。
 */
template <typename val_type, template<typename> class updator_type, template<typename> class act_net_tpl>
using gated_ffn_branches_t = gated_net_t<
    typename complex_net_builder_t<val_type>
        ::template push_back_updatable<weight_net_t, updator_type>
        ::template push_back_staticnet<act_net_tpl>
        ::type,
    weight_net_t<mat_t<val_type>, updator_type>>;

} // namespace jasmine
#endif
