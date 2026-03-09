#ifndef __TRANSFORMER_TEST_HPP__
#define __TRANSFORMER_TEST_HPP__

#include "mat_t.hpp"
#include "mat_view_t.hpp"
#include "mat_transformer_t.hpp"
#include "mat_transformer_kernel_t.hpp"
#include "mat_net_t.hpp"

/*!SECTION
 * 测试场景：
 * 1. 定义一个给输入数据增加SOS和EOS标签的工具类，即在输入数据的末尾增加2个标志位，分别表示SOS和EOS标签；
 * 2. 构造编码器和解码器的测试数据，并且定义transformer结构，设置参数；
 * 3. 将测试数据通过标签增加工具增加标签作为训练的数据对transformer进行训练；
 * 4. 编码器输入测试数据，解码器输入SOS标签，观察输出结果是否正确。
*/

mat_t<double> sos(int const& input_row_num)
{
    mat_t<double> ret(input_row_num + 2, 1);
    ret(input_row_num, 0) = 1.0;
    return ret;
}

mat_t<double> add_sos(const mat_t<double>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<double> new_mat(row_num + 2, col_num + 1);
    mat_view_t<mat_t<double>> origin(new_mat, 0, 1, row_num, col_num);
    origin.assign(input);
    new_mat(row_num, 0) = 1.;     // SOS标签
    return new_mat;
}

mat_t<double> add_eos(const mat_t<double>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<double> new_mat(row_num + 2, col_num + 1);
    mat_view_t<mat_t<double>> origin(new_mat, 0, 0, row_num, col_num);       // 原来的值复制
    origin.assign(input);
    new_mat(row_num + 1, col_num) = 1.;     // EOS标签
    return new_mat;
}

// 编码器不需要标志位，但是为了保持d_model一致，所以需要加2行空的标志位
mat_t<double> expand_encoder_input(const mat_t<double>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<double> new_mat(row_num + 2, col_num);
    mat_view_t<mat_t<double>> origin(new_mat, 0, 0, row_num, col_num);
    origin.assign(input);
    return new_mat;
}

inline mat_view_t<mat_t<double>> back_col(mat_t<double>& mat)
{
    return mat.view(0, mat.col_num() - 1, mat.row_num(), 1);
}

inline bool is_eos(mat_t<double>& mat)
{
    return back_col(mat)(mat.row_num() - 1, 0) > 0.8; // 判断是否为EOS标签 
}

class test_transformer_t
{

template<typename val_type>
using test_updator_t = cache_updator_t<val_type, nadam_t>;
using val_type = double;

using net_type = complex_net_builder_t<val_type>
    ::template push_back_updatable<transformer_base_t, base_upr_tpl>
    ::template push_back_updatable<weight_net_t, base_upr_tpl>
    ::template push_back_staticnet<sigmoid_net_t>
    ::template push_back_staticnet<mse_loss_t>
    ::type
    ;


    net_type m_net;
public:

    static constexpr int en_layers = 2;
    static constexpr int de_layers = 3;
    static constexpr int head_num = 3;
    static constexpr int d_model = 6;
    static constexpr int input_dim = d_model - 2;   // 需要预留2个标志位

    auto& tf_base()
    {
        return m_net.template get<0>();
    }

    auto& ffn()
    {
        return m_net.template get<1>();
    }

    void init(double lr)
    {
        tf_base().set_param(en_layers, de_layers, head_num, d_model);
        ffn().reinit({d_model, d_model});
        m_net.init_weight<xavier_gaussian_t>();
        m_net.set_updator(lr);
    }


    void train(mat_t<double> const& en_input, mat_t<double> const& de_input, mat_t<double> const& label, int train_times)
    {
        auto en_input_expand = expand_encoder_input(en_input);
        auto input_sos = add_sos(de_input);
        auto label_eos = add_eos(label);
        // 编码器先进行编码
        tf_base().encoder_forward(en_input_expand);
        for (int i = 0; i < train_times; i++)
        {
            auto output = m_net.forward(input_sos);
            m_net.backward(label_eos);
            m_net.step();
        }
        std::cout << "label: \n" << label_eos << std::endl;
        std::cout << "train output: \n" << m_net.forward(input_sos) << std::endl;;
    }

    mat_t<double> remove_flags(const mat_t<double>& mat)
    {
        return mat.view(0, 0, mat.row_num() - 2, mat.col_num()).clone();
    }

    void predict(mat_t<double>& en_input)
    {
        // 构建只有1个SOS标志的矩阵，然后递归得出预测结果
        
        auto en_input_expand = expand_encoder_input(en_input);
        mat_t<double> de_input(d_model, 1024);
        int seq_len = 0;
        de_input(input_dim, seq_len++) = 1.;
        tf_base().encoder_forward(en_input_expand);
        while (true)
        {
            if (seq_len >= 10)
            {
                break;
            }
            auto output = m_net.forward(de_input.view(0, 0, d_model, seq_len));
            if (is_eos(output))
            {
                break;
            }
            de_input.view(0, seq_len++, d_model, 1).assign(back_col(output));
        }
        auto final_output = remove_flags(de_input).view(0, 1, input_dim, seq_len - 1).clone();
        std::cout << "final output: \n" << final_output << std::endl;

        std::cout << "origin output: \n" << de_input.view(0, 0, d_model, seq_len) << std::endl;
    }

};


void test_transformer()
{
    #if 0
    // 测试标签增加是否有效
    mat_t<double> input(2, 2, {0.5, 0.8, 0.3, 0.7});
    mat_t<double> label(3, 2, {0.2, 0.4, 0.6, 0.8, 0.1, 0.9});
    auto input_sos = add_sos(input);
    auto label_eos = add_eos(label);
    std::cout << "input with sos \n" << input_sos << "\nlabel with eos \n" << label_eos << std::endl;
    #endif
    test_transformer_t net;
    net.init(3e-4);
    int input_dim = test_transformer_t::input_dim;
    mat_t<double> en_input(input_dim, 3, {0.5, 0.8, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8, 0.1, 0.9, 0.3, 0.7});
    mat_t<double> de_input(input_dim, 3, {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
    mat_t<double> label(input_dim, 3, {0.3, 0.2, 0.1, 0.8, 0.5, 0.1, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2});
    net.train(en_input, de_input, label, 10000);
    net.predict(en_input);
}

#endif
