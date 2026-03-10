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
using test_val_type = float;

mat_t<test_val_type> sos(int const& input_row_num)
{
    mat_t<test_val_type> ret(input_row_num + 2, 1);
    ret(input_row_num, 0) = 1.0;
    return ret;
}

mat_t<test_val_type> add_sos(const mat_t<test_val_type>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<test_val_type> new_mat(row_num + 2, col_num + 1);
    mat_view_t<mat_t<test_val_type>> origin(new_mat, 0, 1, row_num, col_num);
    origin.assign(input);
    new_mat(row_num, 0) = 1.;     // SOS标签
    return new_mat;
}

mat_t<test_val_type> add_eos(const mat_t<test_val_type>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<test_val_type> new_mat(row_num + 2, col_num + 1);
    mat_view_t<mat_t<test_val_type>> origin(new_mat, 0, 0, row_num, col_num);       // 原来的值复制
    origin.assign(input);
    new_mat(row_num + 1, col_num) = 1.;     // EOS标签
    return new_mat;
}

// 编码器不需要标志位，但是为了保持d_model一致，所以需要加2行空的标志位
mat_t<test_val_type> expand_encoder_input(const mat_t<test_val_type>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<test_val_type> new_mat(row_num + 2, col_num);
    mat_view_t<mat_t<test_val_type>> origin(new_mat, 0, 0, row_num, col_num);
    origin.assign(input);
    return new_mat;
}

inline mat_view_t<mat_t<test_val_type>> back_col(mat_t<test_val_type>& mat)
{
    return mat.view(0, mat.col_num() - 1, mat.row_num(), 1);
}

inline bool is_eos(mat_t<test_val_type>& mat)
{
    return back_col(mat)(mat.row_num() - 1, 0) > 0.8; // 判断是否为EOS标签 
}

class test_transformer_t
{

template<typename test_val_type>
using test_updator_t = cache_updator_t<test_val_type, nadam_t>;

using net_type = complex_net_builder_t<test_val_type>
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

    void init(test_val_type lr)
    {
        tf_base().set_param(en_layers, de_layers, head_num, d_model);
        ffn().reinit({d_model, d_model});
        m_net.init_weight<xavier_gaussian_t>();
        m_net.set_updator(lr);
    }


    void train(mat_t<test_val_type> const& en_input, mat_t<test_val_type> const& de_input, mat_t<test_val_type> const& label, int train_times)
    {
        auto en_input_expand = expand_encoder_input(en_input);
        std::cout << "expanded encoder input: \n" << en_input_expand << std::endl;
        mat_t<test_val_type> input_sos = add_sos(de_input);
        std::cout << "input with flags: \n" << input_sos << std::endl;
        mat_t<test_val_type> label_eos = add_eos(label);
        std::cout << "label with flags: \n" << label_eos << std::endl;
        // 编码器先进行编码
        tf_base().encoder_forward(en_input_expand);
        for (int i = 0; i < train_times; i++)
        {
            auto output = m_net.forward(input_sos);
            m_net.backward(label_eos);
            m_net.step();
        }
        std::cout << "label: \n" << label_eos << std::endl;
        std::cout << "train output: \n" << m_net.forward(input_sos) << std::endl;
        std::cout << "test input: \n" << input_sos.front_col() << std::endl;
        std::cout << "test first: \n" << m_net.forward(input_sos.front_col()) << std::endl;
        
        /* 逐次将解码器的输入进行输入，获得输出 */
        mat_t<test_val_type> pred_input(input_dim, 1023);
        auto pred_input_sos = add_sos(pred_input);
        int len = 1;
        while (true)
        {
            auto cur_input = pred_input_sos.view(0, 0, d_model, len);
            auto output = m_net.forward(cur_input);
            //std::cout << "---- seq_idx: " << len << std::endl << "input: \n" << cur_input << std::endl << " output: \n" << output.back_col() << std::endl;
            pred_input_sos.col(len++).assign(output.back_col());
            if (is_eos(output))break;
            if (len >= 10)break;
        }
        std::cout << "final input with flags: \n" << pred_input_sos.view(0, 0, d_model, len) << std::endl;
        std::cout << "label: \n" << label_eos << std::endl;

    }

    mat_t<test_val_type> remove_flags(const mat_t<test_val_type>& mat)
    {
        return mat.view(0, 0, mat.row_num() - 2, mat.col_num()).clone();
    }

    auto predict(mat_t<test_val_type>& en_input)
    {
        // 构建只有1个SOS标志的矩阵，然后递归得出预测结果
        
        auto en_input_expand = expand_encoder_input(en_input);
        std::cout << "predict expanded encoder input: \n" << en_input_expand << std::endl;
        mat_t<test_val_type> de_input(d_model, 1024);
        int seq_len = 0;
        de_input(input_dim, seq_len++) = 1.;
        tf_base().encoder_forward(en_input_expand);
        while (true)
        {
            if (seq_len >= 10)
            {
                break;
            }
            std::cout << "decoder input: \n" << de_input.view(0, 0, d_model, seq_len) << std::endl;
            auto output = m_net.forward(de_input.view(0, 0, d_model, seq_len));
            std::cout << "seq_idx: " << seq_len << " output: \n" << output.back_col() << std::endl;
            if (is_eos(output))
            {
                break;
            }
            de_input.col(seq_len++).assign(back_col(output));
        }
        std::cout << "final decoder input with flags: \n" << de_input.view(0, 0, d_model, seq_len) << std::endl;
        auto final_output = remove_flags(de_input).view(0, 1, input_dim, seq_len - 1).clone();
        //std::cout << "final output: \n" << final_output << std::endl;
        //std::cout << "origin output: \n" << de_input.view(0, 0, d_model, seq_len) << std::endl;
        return final_output;
    }

};

void test_sequence_mha()
{
    /*!SECTION
    * 按照原理来讲对于有mask的mha，输入的长度是1和输入长度是2的序列，其第1个位置输出应该是一样的，本示例就是要验证这个功能
    */
    mat_mha_t<mat_t<test_val_type>, nadam_t> mha(2, 6, true, 10);
    mha.init_weight<xavier_gaussian_t>();
    mat_t<test_val_type> input1(6, 1, {0.5
                                , 0.3
                                , 0.2
                                , 0.1
                                , 0.3
                                , 0.5});
    mat_t<test_val_type> input2(6, 2, {0.5, 0.8
                                , 0.3, 0.7
                                , 0.2, 0.4
                                , 0.1, 0.2
                                , 0.3, 0.4
                                , 0.5, 0.6});
    std::cout << "input1: \n" << input1 << std::endl;
    std::cout << "input2: \n" << input2 << std::endl;
    std::cout << "output1: \n" << mha.forward(input1) << std::endl;
    std::cout << "output2: \n" << mha.forward(input2) << std::endl;
    std::cout << ((input2.front_col() - input1) < 0.0001) << std::endl;
    // 对于transformer也是一样，保持编码器输入不变，如果保持解码器输入的第一个位置不变，那么输出的第一个位置也应该是一样的
    transformer_base_t<mat_t<test_val_type>, nadam_t> tf_base(2, 3, 2, 6);
    tf_base.init_weight<xavier_gaussian_t>();
    mat_t<test_val_type> en_input(6, 3, {0.5, 0.8, 0.3
                                , 0.7, 0.2, 0.4
                                , 0.6, 0.8, 0.1
                                , 0.9, 0.3, 0.7
                                , 0.3, 0.7, 0.2
                                , 0.4, 0.5, 0.6});
    tf_base.encoder_forward(en_input);
    std::cout << "tf_base output1: \n" << tf_base.forward(input1) << std::endl;
    std::cout << "tf_base output2: \n" << tf_base.forward(input2) << std::endl;
}


void test_transformer()
{
    #if 0
    // 测试标签增加是否有效
    mat_t<test_val_type> input(2, 2, {0.5, 0.8, 0.3, 0.7});
    mat_t<test_val_type> label(3, 2, {0.2, 0.4, 0.6, 0.8, 0.1, 0.9});
    auto input_sos = add_sos(input);
    auto label_eos = add_eos(label);
    std::cout << "input with sos \n" << input_sos << "\nlabel with eos \n" << label_eos << std::endl;
    #endif
    test_transformer_t net;
    net.init(1e-4);
    int input_dim = test_transformer_t::input_dim;
    mat_t<test_val_type> en_input(input_dim, 3,    { 0.5, 0.8, 0.3
                                            , 0.7, 0.2, 0.4
                                            , 0.6, 0.8, 0.1
                                            , 0.9, 0.3, 0.7});
    mat_t<test_val_type> de_input(input_dim, 3,    { 0.4, 0.5, 0.6
                                            , 0.7, 0.8, 0.9
                                            , 0.1, 0.2, 0.3
                                            , 0.4, 0.5, 0.6});
    mat_t<test_val_type> label(input_dim, 3,       { 0.3, 0.2, 0.1
                                            , 0.8, 0.5, 0.1
                                            , 0.7, 0.8, 0.9
                                            , 0.4, 0.3, 0.2});
    net.train(en_input, de_input, label, 50000);
    //std::cout << net.predict(en_input) << std::endl;
}

#endif
