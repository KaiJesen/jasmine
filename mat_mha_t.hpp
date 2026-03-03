#ifndef __MAT_MHA_T_HPP__
#define __MAT_MHA_T_HPP__

#include "mat_t.hpp"
#include "mat_view_t.hpp"
#include "mat_net_t.hpp"
#include "mat_express_t.hpp"

template <typename input_type, template<typename> class updator_type>
class mat_head_gen_t
{
public:
    using val_type = typename input_type::return_type;
    using return_type = typename input_type::return_type;

private:
    using ffn_type = weight_net_t<input_type, updator_type>;
    using softmax_type = hsoftmax_net_t<input_type>;

    ffn_type m_q_net;
    ffn_type m_k_net;
    ffn_type m_v_net;
    softmax_type m_softmax;
    mat_t<val_type> m_q, m_k, m_v;
    bool m_mask;

public:
    mat_head_gen_t(int const& d_model = 1, bool const& mask = false, int const& seq_len = 1)
        : m_q_net(d_model, d_model), m_k_net(d_model, d_model), m_v_net(d_model, d_model),
          m_q(d_model, seq_len), m_k(d_model, seq_len), m_v(d_model, seq_len), m_mask(mask)
    {
    }

    void set_param(int const& d_model, bool const& mask = false, int const& seq_len = 1)
    {
        m_q_net.reinit(std::vector<int>(d_model, d_model));
        m_k_net.reinit(std::vector<int>(d_model, d_model));
        m_v_net.reinit(std::vector<int>(d_model, d_model));
        m_q.reshape(d_model, seq_len);
        m_k.reshape(d_model, seq_len);
        m_v.reshape(d_model, seq_len);
        m_mask = mask;
    }
    
    template<typename mat_type>
    requires std::is_same_v<std::decay_t<mat_type>, mat_t<val_type>>
    mat_t<val_type> forward(const mat_view_t<mat_type>& input)
    {
        m_q = m_q_net.forward(input); // Q 为 [d_model, seq_len]
        m_k = m_k_net.forward(input);
        m_v = m_v_net.forward(input);

        auto attn_scores = (m_q.t().dot(m_k) / sqrt(m_q.row_num())).clone();
        /*!ANCHOR 掩码规则说明
        * 由于scores=Q'K，也就是说scores中的i行j列元素表示的是Q序列中第i个值与K序列中第j个值之间的分数；
        * Q是表示的是当前的查询，K是可关注的历史。那么就需要就针对每个Q让他只能看到之前发生的K。也就是j > i的都设置为无效的
        */
        if (m_mask)
        {
            for (int i = 0; i < attn_scores.row_num(); ++i)
            {
                for (int j = i + 1; j < attn_scores.col_num(); ++j)
                {
                    attn_scores(i, j) = -std::numeric_limits<val_type>::infinity();
                }
            }
        }
        auto attn_weights = m_softmax.forward(attn_scores);
        auto output = m_v.dot(attn_weights.t()).clone();

        return output;
    }

    template<typename mat_type>
    requires std::is_same_v<std::decay_t<mat_type>, mat_t<val_type>>
    mat_t<val_type> forward(const mat_view_t<mat_type>& decoder_input, const mat_view_t<mat_type>& encoder_input)
    {
        m_q = m_q_net.forward(decoder_input); // Q 为 [d_model, seq_len]
        m_k = m_k_net.forward(encoder_input);
        m_v = m_v_net.forward(encoder_input);

        auto attn_scores = (m_q.t().dot(m_k) / sqrt(m_q.row_num())).clone();
        // 较差注意力不需要mask层
        auto attn_weights = m_softmax.forward(attn_scores);
        auto output = m_v.dot(attn_weights.t()).clone();

        return output;
    }

    mat_t<val_type> backward(const mat_view_t<mat_t<val_type>>& delta)
    {
        mat_t<val_type> delta_v = delta.dot(m_softmax.m_output);
        mat_t<val_type> delta_attn_weights = delta.t().dot(m_v);
        mat_t<val_type> delta_qt_k = m_softmax.backward(delta_attn_weights);
        /*!LINK - 链接规则说明
        * 由于scores=Q'K，也就是说scores中的i行j列元素表示的是Q序列中第i个值与K序列中第j个值之间的分数；
        * Q是表示的是当前的查询，K是可关注的历史。那么就需要就针对每个Q让他只能看到之前发生的K。也就是j > i的都设置为无效的
        */
        if (m_mask)
        {
            for (int i = 0; i < delta_qt_k.row_num(); ++i)
            {
                for (int j = i + 1; j < delta_qt_k.col_num(); ++j)
                {
                    delta_qt_k(i, j) = 0;
                }
            }
        }

        mat_t<val_type> delta_q = m_k.dot(delta_qt_k.t()) / static_cast<val_type>(sqrt(m_q.row_num()));
        mat_t<val_type> delta_k = m_q.dot(delta_qt_k) / static_cast<val_type>(sqrt(m_q.row_num()));

        mat_t<val_type> delta_input =
            m_q_net.backward(delta_q) + m_k_net.backward(delta_k) + m_v_net.backward(delta_v);

        return delta_input;
    }

    mat_t<val_type> backward(const mat_view_t<mat_t<val_type>>& delta, mat_view_t<mat_t<val_type>>& encoder_delta)
    {
        mat_t<val_type> delta_v = delta.dot(m_softmax.m_output);
        mat_t<val_type> delta_attn_weights = delta.t().dot(m_v);
        mat_t<val_type> delta_qt_k = m_softmax.backward(delta_attn_weights);
        mat_t<val_type> delta_q = m_k.dot(delta_qt_k.t()) / static_cast<val_type>(sqrt(m_q.row_num()));
        mat_t<val_type> delta_k = m_q.dot(delta_qt_k) / static_cast<val_type>(sqrt(m_q.row_num()));
        encoder_delta += ((m_k_net.backward(delta_k) + m_v_net.backward(delta_v)));
        mat_t<val_type> delta_input = m_q_net.backward(delta_q);
        return delta_input;
    }

    template<typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_q_net.set_updator(std::forward<upr_arg_types>(args)...);
        m_k_net.set_updator(std::forward<upr_arg_types>(args)...);
        m_v_net.set_updator(std::forward<upr_arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        m_q_net.template init_weight<init_type>();
        m_k_net.template init_weight<init_type>();
        m_v_net.template init_weight<init_type>();
    }

    std::string net_type(int const& indent = 0) const
    {
        return print_indent(indent) + "mat_head_gen_t";
    }

    void step()
    {
        m_q_net.step();
        m_k_net.step();
        m_v_net.step();
    }

};

// 垂直拼接多个 mat_t 对象（按行拼接）
template <typename val_type>
mat_t<val_type> vconcat(const std::vector<mat_t<val_type>>& mats)
{
    if (mats.empty())
        throw std::runtime_error("Cannot concatenate empty vector");

    int cols = mats[0].col_num();
    int total_rows = 0;

    for (const auto& mat : mats)
    {
        if (mat.col_num() != cols)
            throw std::runtime_error("All matrices must have the same number of columns");
        total_rows += mat.row_num();
    }

    mat_t<val_type> result(total_rows, cols);
    int row_offset = 0;

    for (const auto& mat : mats)
    {
        for (int i = 0; i < mat.row_num(); ++i)
        {
            for (int j = 0; j < mat.col_num(); ++j)
            {
                result(row_offset + i, j) = mat(i, j);
            }
        }
        row_offset += mat.row_num();
    }

    return result;
}

// 垂直分割 mat_t 为多个子矩阵（按行分割，返回视图）
template <typename mat_type>
std::vector<mat_view_t<mat_type>> vsplit(mat_type& mat, int num_splits)
{
    if (num_splits <= 0 || mat.row_num() % num_splits != 0)
        throw std::runtime_error("Invalid number of splits or row count mismatch");

    int rows_per_split = mat.row_num() / num_splits;
    std::vector<mat_view_t<mat_type>> views;

    for (int i = 0; i < num_splits; ++i)
    {
        int start_row = i * rows_per_split;
        views.emplace_back(mat, start_row, 0, rows_per_split, mat.col_num());
    }

    return views;
}

/*!SECTION: MHA
*   多头注意力机制和线型变换网络不太一样，线型变换网络初始化时传入的是输入的维度和输出的维度，但是MHA传入的是头的数目以及模型的维度。另外，在transformer中，每一层的MHA的输入和输出都是相等维度的，另外，编码器和解码器的输出维度相等，但是序列长度可能不同。
所以，总体上来说，就维度而言，transformer实际只需要1个整形来表示输入和输出的维度，另外需要1个整形来表示头数量。因此，我们可以将mha认为是一种stable的网络。而stable的网络不能定义reinit函数。
*/
template <typename input_type, template<typename> class updator_type>
class mat_mha_t
{
public:
    using val_type = typename input_type::return_type;

private:
    std::vector<mat_head_gen_t<input_type, updator_type>> m_heads;
    weight_net_t<mat_t<val_type>, updator_type> m_output_proj;
    int m_num_heads;
    int m_d_model;
    int m_d_head;

public:
    mat_mha_t(int num_heads = 1, int d_model = 1, bool mask = false, int seq_len = 1)
        : m_output_proj(d_model, d_model), m_num_heads(num_heads), m_d_model(d_model), m_d_head(d_model / num_heads)
    {
        /* 设置默认构造参数目的是让其可以没有参数进行构造，以便放入其他复杂网络结构中 */
        if (d_model % num_heads != 0)
            throw std::runtime_error("d_model must be divisible by num_heads");

        for (int i = 0; i < num_heads; ++i)
        {
            m_heads.emplace_back(m_d_head, mask, seq_len);
        }
    }

    void set_param(int num_heads, int d_model, bool mask = false, int seq_len = 1)
    {
        m_num_heads = num_heads;
        m_d_model = d_model;
        m_d_head = d_model / num_heads;

        if (d_model % num_heads != 0)
            throw std::runtime_error("d_model must be divisible by num_heads");

        m_heads.resize(num_heads);
        for (int i = 0; i < num_heads; ++i)
        {
            m_heads[i].set_param(m_d_head, mask, seq_len);
        }
        m_output_proj.reinit(std::vector<int>(d_model, d_model));
    }

    mat_t<val_type> forward(const input_type& input)
    {
        // Step 1: 检查输入维度是否合法
        if (input.row_num() != m_d_model)
            throw std::runtime_error("Input row dimension must match d_model");

        // Step 2: 按头数分割输入矩阵
        auto input_splits = vsplit(input, m_num_heads);

        // Step 3: 每个头独立进行正向传播
        std::vector<mat_t<val_type>> head_outputs;
        head_outputs.reserve(m_num_heads);

        for (int i = 0; i < m_num_heads; ++i)
        {
            auto head_output = m_heads[i].forward(input_splits[i]);
            head_outputs.push_back(head_output);
        }

        // Step 4: 拼接所有头的输出
        mat_t<val_type> concatenated_output = vconcat(head_outputs);

        // Step 5: 通过输出映射网络得到最终结果
        auto final_output = m_output_proj.forward(concatenated_output);

        return final_output;
    }

    mat_t<val_type> forward(const input_type& input, const input_type& encoder_input)
    {
        // Step 1: 检查输入维度是否合法
        if (input.row_num() != m_d_model)
            throw std::runtime_error("Input row dimension must match d_model");
        if (encoder_input.row_num() != m_d_model)
            throw std::runtime_error("Encoder input row dimension must match d_model");

        // Step 2: 按头数分割输入矩阵
        auto input_splits = vsplit(input, m_num_heads);
        auto encoder_input_splits = vsplit(encoder_input, m_num_heads);

        // Step 3: 每个头独立进行正向传播
        std::vector<mat_t<val_type>> head_outputs;
        head_outputs.reserve(m_num_heads);

        for (int i = 0; i < m_num_heads; ++i)
        {
            auto head_output = m_heads[i].forward(input_splits[i], encoder_input_splits[i]);
            head_outputs.push_back(head_output);
        }

        // Step 4: 拼接所有头的输出
        mat_t<val_type> concatenated_output = vconcat(head_outputs);

        // Step 5: 通过输出映射网络得到最终结果
        auto final_output = m_output_proj.forward(concatenated_output);

        return final_output;
    }

    mat_t<val_type> backward(const input_type& delta)
    {
        // 反向传播到线性变换层
        auto delta_concat = m_output_proj.backward(delta);

        // 按行分割 delta_concat 为多个子矩阵（视图）
        auto deltas = vsplit(delta_concat, m_num_heads);

        // 创建一个与输入大小相同的零矩阵，用于累加所有头的梯度
        mat_t<val_type> total_delta(delta.row_num(), delta.col_num(), 0.0);

        // 为 total_delta 创建视图，方便按块赋值
        std::vector<mat_view_t<mat_t<val_type>>> total_delta_views = vsplit(total_delta, m_num_heads);

        // 对每个头执行反向传播，并将结果赋值到对应的视图中
        for (int i = 0; i < m_num_heads; ++i)
        {
            auto delta_input = m_heads[i].backward(deltas[i]);
            total_delta_views[i].assign(delta_input); // 使用 assign 将结果写入对应位置
        }

        return total_delta;
    }

    // 交叉注意力机制使用，encoder_delta会在各层使用累加的方式计算，但是要记得最后除以层数
    mat_t<val_type> backward(const input_type& delta, mat_t<val_type>& encoder_delta)
    {
        // 反向传播到线性变换层
        auto delta_concat = m_output_proj.backward(delta);

        // 按行分割 delta_concat 为多个子矩阵（视图）
        auto deltas = vsplit(delta_concat, m_num_heads);

        // 创建一个与输入大小相同的零矩阵，用于累加所有头的梯度
        mat_t<val_type> total_delta(delta.row_num(), delta.col_num(), 0.0);

        // 为 total_delta 创建视图，方便按块赋值
        std::vector<mat_view_t<mat_t<val_type>>> total_delta_views = vsplit(total_delta, m_num_heads);
        std::vector<mat_view_t<mat_t<val_type>>> encoder_delta_views = vsplit(encoder_delta, m_num_heads);

        // 对每个头执行反向传播，并将结果赋值到对应的视图中
        for (int i = 0; i < m_num_heads; ++i)
        {
            auto delta_input = m_heads[i].backward(deltas[i], encoder_delta_views[i]);
            total_delta_views[i].assign(delta_input); // 使用 assign 将结果写入对应位置
        }

        return total_delta;
    }

    template<typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        for (auto& head : m_heads)
        {
            head.set_updator(std::forward<upr_arg_types>(args)...);
        }
        m_output_proj.set_updator(std::forward<upr_arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        for (auto& head : m_heads)
        {
            head.template init_weight<init_type>();
        }
        m_output_proj.template init_weight<init_type>();
    }

    std::string net_type(int const& indent = 0) const
    {
        return print_indent(indent) + "MHA(heads:" + std::to_string(m_num_heads) + ", d_model:" + std::to_string(m_d_model) + ")";
    }

    void step()
    {
        for (auto& head: m_heads)
        {
            head.step();
        }
        m_output_proj.step();
    }

};

template <typename input_type, template<typename> class updator_type>
class mat_mhca_t:public mat_mha_t<input_type, updator_type>
{
public:
    using base_type = mat_mha_t<input_type, updator_type>;
    using val_type = typename input_type::return_type;
private:
    mat_t<val_type>* m_encoder_output;   // 用于保存编码器的输出，以便交叉注意力机制使用
    mat_t<val_type>* m_encoder_delta;    // 用于保存编码器的梯度，以便交叉注意力机制使用
public:
    mat_mhca_t(int num_heads = 1, int d_model = 1, bool mask = false, int seq_len = 1): base_type(num_heads, d_model, mask, seq_len)
    {
        m_encoder_output = nullptr;
        m_encoder_delta = nullptr;
    }

    void set_encoder_param(mat_t<val_type>& encoder_output, mat_t<val_type>& encoder_delta)
    {
        m_encoder_output = &encoder_output;
        m_encoder_delta = &encoder_delta;
    }

    mat_t<val_type> forward(const input_type& input)
    {
        if (m_encoder_output == nullptr)
            throw std::runtime_error("Encoder output not set for cross attention");

        return base_type::forward(input, *m_encoder_output);
    }

    mat_t<val_type> backward(const input_type& delta)
    {
        if (m_encoder_delta == nullptr)
            throw std::runtime_error("Encoder delta not set for cross attention");

        return base_type::backward(delta, *m_encoder_delta);
    }

    void step()
    {
        base_type::step();
    }

};


#include "mat_updator_t.hpp"

void test_mat_head_gen_t()
{
    mat_head_gen_t<mat_view_t<mat_t<double>>, adam_t> mha(4);
    mha.init_weight<xavier_gaussian_t>();
    mha.set_updator(0.01);
    using head_gen_type = mat_head_gen_t<mat_view_t<mat_t<double>>, adam_t>;
    std::vector<head_gen_type> heads;

    mat_t<double> input(4, 2, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
    mat_t<double> expected(4, 2, {0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});

    std::cout << "input: \n" << input << std::endl;
    std::cout << "expected: \n" << expected << std::endl;

    std::cout << "before training mha output: \n" << mha.forward(input.view()) << std::endl;

    for (int i = 0; i < 10000; ++i)
    {
        auto output = mha.forward(input.view());
        auto delta = output - expected;
        mha.backward(delta.clone().view());
    }

    std::cout << "after training mha output: \n" << mha.forward(input.view()) << std::endl;
}

void test_mha_tools()
{
    std::vector<mat_t<double>> inputs;
    for (double i = 0; i < 4; ++i)
    {
        inputs.emplace_back(mat_t<double>(2, 2, {i, i + 1, i + 2, i + 3}));
    }
    mat_t<double> output = vconcat(inputs);
    std::cout << output << std::endl;
    std::vector<mat_view_t<mat_t<double>>> views = vsplit(output, 4);
    for (auto& view : views)
    {
        std::cout << view << std::endl;
    }
}

void test_multi_head_attention()
{
    int num_heads = 4;
    int d_model = 8;
    int seq_len = 2;

    mat_mha_t<mat_t<double>, nadam_t> mha(num_heads, d_model, seq_len);
    mha.init_weight<xavier_gaussian_t>();
    mha.set_updator(0.01);

    mat_t<double> input(d_model, seq_len, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                           0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6});
    auto output = mha.forward(input);
    std::cout << "MHA Output:\n" << output << std::endl;
    for (int i = 0; i < 1000; ++i)
    {
        auto output = mha.forward(input);
        auto delta = output - 0.5;
        mha.backward(delta.clone().view());
    }

    output = mha.forward(input);
    std::cout << "MHA Last Output:\n" << output << std::endl;
}

#endif
