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

public:
    mat_head_gen_t(int const& token_len, int const& d_model, int const& seq_len = 1)
        : m_q_net(token_len, d_model), m_k_net(token_len, d_model), m_v_net(token_len, d_model),
          m_q(d_model, seq_len), m_k(d_model, seq_len), m_v(d_model, seq_len)
    {
    }

    template<typename mat_type>
    requires std::is_same_v<std::decay_t<mat_type>, mat_t<val_type>>
    mat_t<val_type> forward(const mat_view_t<mat_type>& input)
    {
        m_q = m_q_net.forward(input); // Q 为 [d_model, seq_len]
        m_k = m_k_net.forward(input);
        m_v = m_v_net.forward(input);

        auto attn_scores = (m_q.t().dot(m_k) / sqrt(m_q.row_num())).clone();
        auto attn_weights = m_softmax.forward(attn_scores);
        auto output = m_v.dot(attn_weights.t()).clone();

        return output;
    }

    mat_t<val_type> backward(const mat_view_t<mat_t<val_type>>& delta)
    {
        mat_t<val_type> delta_v = delta.dot(m_softmax.m_output);
        mat_t<val_type> delta_attn_weights = delta.t().dot(m_v);
        mat_t<val_type> delta_qt_k = m_softmax.backward(delta_attn_weights);

        mat_t<val_type> delta_q = m_k.dot(delta_qt_k.t()) / sqrt(m_q.row_num());
        mat_t<val_type> delta_k = m_q.dot(delta_qt_k) / sqrt(m_q.row_num());

        mat_t<val_type> delta_input =
            m_q_net.backward(delta_q) + m_k_net.backward(delta_k) + m_v_net.backward(delta_v);

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

    std::string net_type() const
    {
        return "mat_head_gen_t";
    }

    void reinit(std::vector<int> const& container)
    {
        m_q_net.reinit(container);
        m_k_net.reinit(container);
        m_v_net.reinit(container);
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

template <typename input_type, template<typename> class updator_type>
class multi_head_attention_t
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
    multi_head_attention_t(int num_heads, int d_model, int seq_len = 1)
        : m_output_proj(d_model, d_model), m_num_heads(num_heads), m_d_model(d_model), m_d_head(d_model / num_heads)
    {
        if (d_model % num_heads != 0)
            throw std::runtime_error("d_model must be divisible by num_heads");

        for (int i = 0; i < num_heads; ++i)
        {
            m_heads.emplace_back(m_d_head, m_d_head, seq_len);
        }
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

    mat_t<val_type> backward(const mat_view_t<mat_t<val_type>>& delta)
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
};

#include "mat_updator_t.hpp"

void test_mat_head_gen_t()
{
    mat_head_gen_t<mat_view_t<mat_t<double>>, adam_t> mha(4, 2);
    mha.init_weight<xavier_gaussian_t>();
    mha.set_updator(0.01);
    using head_gen_type = mat_head_gen_t<mat_view_t<mat_t<double>>, adam_t>;
    std::vector<head_gen_type> heads;

    mat_t<double> input(4, 2, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
    mat_t<double> expected(2, 2, {0.5, 0.2, 0.3, 0.4});

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

#if 1
void test_multi_head_attention()
{
    int num_heads = 4;
    int d_model = 8;
    int seq_len = 2;

    multi_head_attention_t<mat_t<double>, nadam_t> mha(num_heads, d_model, seq_len);
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

#endif
