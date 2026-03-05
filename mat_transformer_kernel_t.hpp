#ifndef __MAT_TRANSFORMER_KERNEL_T_HPP__
#define __MAT_TRANSFORMER_KERNEL_T_HPP__

#include "mat_net_t.hpp"
#include "mat_mha_t.hpp"

// 定义基本的堆叠单元
template<typename val_type, template<typename> class updator_type>
using res_mha_t = residual_net_t<mat_mha_t<mat_t<val_type>, updator_type>>;

template<typename val_type, template<typename> class updator_type>
using res_mhca_t = residual_net_t<mat_mhca_t<mat_t<val_type>, updator_type>>;

template<typename val_type, template<typename> class updator_type>
using res_mha_norm_t = complex_net_builder_t<val_type>
    ::template push_back_impl<res_mha_t<val_type, updator_type>>
    ::template push_back_updatable<layer_norm_net_t, updator_type>
    ::type;

template<typename val_type, template<typename> class updator_type>
using res_mhca_norm_t = complex_net_builder_t<val_type>
    ::template push_back_impl<res_mhca_t<val_type, updator_type>>
    ::template push_back_updatable<layer_norm_net_t, updator_type>
    ::type;

template<typename val_type, template<typename> class updator_type>
using base_ffn_t = complex_net_builder_t<val_type>
    ::template push_back_updatable<weight_net_t, updator_type>
    ::template push_back_staticnet<relu_net_t>
    ::type;

template<typename val_type, template<typename> class updator_type>
using res_ffn_t = residual_net_t<base_ffn_t<val_type, updator_type>>;

template<typename val_type, template<typename> class updator_type>
using res_ffn_norm_t = complex_net_builder_t<val_type>
    ::template push_back_impl<res_ffn_t<val_type, updator_type>>
    ::template push_back_updatable<layer_norm_net_t, updator_type>
    ::type;

template<typename val_type, template<typename> class updator_type>
using encoder_layer_t = complex_net_builder_t<val_type>
    ::template push_back_impl<res_mha_norm_t<val_type, updator_type>>
    ::template push_back_impl<res_ffn_norm_t<val_type, updator_type>>
    ::type;

template<typename val_type, template<typename> class updator_type>
using decoder_layer_t = complex_net_builder_t<val_type>
    ::template push_back_impl<res_mha_norm_t<val_type, updator_type>>
    ::template push_back_impl<res_mhca_norm_t<val_type, updator_type>>
    ::template push_back_impl<res_ffn_norm_t<val_type, updator_type>>
    ::type;


/**!SECTION
 * 编码器是由多层编码层组成，每层编码层都包含1个残差多头注意力层，1个add&norm层，1个残差ffn层，1个add&norm层。编码器的输入是一个矩阵，输出也是一个矩阵，输入输出维度相同。
 * 0. res_mha_norm_t: 包含一个残差多头注意力层和一个add&norm层，输入输出维度相同
 * {
 *      0. res_mha_t: 包含一个残差多头注意力层，输入输出维度相同
 *      {
 *          0. mha_t: 残差多头注意力层，输入输出维度相同
 *      }
 *      1. layer_norm_net_t: 包含一个层归一化层，输入输出维度相同
 * }
 * 1. res_ffn_norm_t: 包含一个残差ffn层和一个add&norm层，输入输出维度相同
 * {
 *      0. res_ffn_t: 包含一个残差ffn层，输入输出维度相同
 *      {
 *           0. weight_net_t: 包含一个线性层，输入输出维度相同
 *           1. relu_net_t: 包含一个ReLU激活层，输入输出维度相同
 *      }
 *      1. layer_norm_net_t: 包含一个层归一化层，输入输出维度相同
 * }
 */
template<typename val_type_, template<typename> class updator_type>
class encoder_t
{
public:
    using val_type = val_type_;
private:
    std::vector<encoder_layer_t<val_type, updator_type>> m_layers;
public:
    encoder_t(int const& n_layers = 1, int const& head_num = 1, int const& d_model = 1, int const& seq_len = 1)
    {
        m_layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            get_mha(i).set_param(head_num, d_model, false, seq_len);
            get_ffn(i).reinit(std::vector<int>{d_model, d_model});
        }
    }

    void set_param(int const& n_layers, int const& head_num, int const& d_model, int const& seq_len)
    {
        m_layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            get_mha(i).set_param(head_num, d_model, false, seq_len);
            get_ffn(i).reinit(std::vector<int>{d_model, d_model});
        }
    }

    mat_t<val_type> forward(mat_t<val_type> input)
    { 
        for (auto& layer: m_layers)
        {
            input = layer.forward(input);
        }
        return input;
    }

    mat_t<val_type> backward(mat_t<val_type> delta)
    {
        for (int i = m_layers.size() - 1; i >= 0; --i)
        {
            delta = m_layers[i].backward(delta);
        }
        return delta;
    }

    auto& get_mha(int const& i)
    {
        return m_layers[i].template get<0, 0>().base_net();
    }

    auto& get_mha_norm(int const& i)
    {
        return m_layers[i].template get<0, 1>();
    }

    auto& get_ffn(int const& i)
    {
        return m_layers[i].template get<1, 0, 0>();
    }

    auto& get_ffn_norm(int const& i)
    {
        return m_layers[i].template get<1, 1>();
    }

    auto& get_mha(int const& i) const
    {
        return m_layers[i].template get<0, 0>().base_net();
    }

    auto& get_mha_norm(int const& i) const
    {
        return m_layers[i].template get<0, 1>().base_net();
    }
    
    auto& get_ffn(int const& i) const
    {
        return m_layers[i].template get<1, 0, 0>();
    }

    auto& get_ffn_norm(int const& i) const
    {
        return m_layers[i].template get<1, 1>().base_net();
    }

    size_t size() const
    {
        return m_layers.size();
    }

    template<typename init_type>
    void init_weight()
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).template init_weight<init_type>();       // 初始化每一层的权重
            get_mha_norm(i).template init_weight<init_type>();
            get_ffn(i).template init_weight<init_type>();       
            get_ffn_norm(i).template init_weight<init_type>();
        }
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "encoder = [";
        for (size_t i = 0; i < size(); ++i)
        {
            ss << "\n" << print_indent(indent + 2) << "Layer " << i << " mha: " << get_mha(i).net_type() << "-->";
            ss << "ffn: " << get_ffn(i).net_type();
        }
        ss << std::endl << print_indent(indent) << "]";
        return ss.str();
    }

    template<typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).template set_updator(std::forward<upr_arg_types>(args)...);
            get_mha_norm(i).template set_updator(std::forward<upr_arg_types>(args)...);
            get_ffn(i).template set_updator(std::forward<upr_arg_types>(args)...);
            get_ffn_norm(i).template set_updator(std::forward<upr_arg_types>(args)...);
        }
    }

    void step()
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).step();
            get_mha_norm(i).step();
            get_ffn(i).step();
            get_ffn_norm(i).step();
        }
    }

};

/*!SECTION
 * 解码器是由多层解码层组成，每层解码层都包含1个残差多头注意力层，1个add&norm层，1个残差交叉多头注意力层，1个add&norm层，1个残差ffn层，1个add&norm层。解码器的输入是一个矩阵，输出也是一个矩阵，输入输出维度相同。
 * 0. res_mha_norm_t: 包含一个残差多头注意力层和一个add&norm层，输入输出维度相同
 * {
 *      0. res_mha_t: 包含一个残差多头注意力层，输入输出维度相同
 *      {
 *          base_net. mha_t: 残差多头注意力层，输入输出维度相同
 *      }
 *      1. layer_norm_net_t: 包含一个层归一化层，输入输出维度相同
 * }
 * 1. res_mhca_norm_t: 包含一个残差交叉多头注意力层和一个add&norm层，输入输出维度相同
 * {
 *      0. res_mhca_t: 包含一个残差交叉多头注意力层，输入输出维度相同
 *      {
 *          base_net. mhca_t: 残差交叉多头注意力层，输入输出维度相同
 *      }
 *      1. layer_norm_net_t: 包含一个层归一化层，输入输出维度相同
 * }
 * 2. res_ffn_norm_t: 包含一个残差ffn层和一个add&norm层，输入输出维度相同
 * {
 *      0. res_ffn_t: 包含一个残差ffn层，输入输出维度相同
 *      {
 *         0. weight_net_t: 包含一个线性层，输入输出维度相同
 *         1. relu_net_t: 包含一个ReLU激活层，输入输出维度相同
 *      }
 *      1. layer_norm_net_t: 包含一个层归一化层，输入输出维度相同
 * }
 */
template<typename val_type_, template<typename> class updator_type>
class decoder_t
{
public:
    using val_type = val_type_;
private:
    mat_t<val_type> m_encoder_output;   // 用于保存编码器的输出，以便交叉注意力机制使用
    mat_t<val_type> m_encoder_delta;    // 用于保存编码器的梯度，以便交叉注意力机制使用
    encoder_t<val_type, updator_type>* m_encoder;   
    std::vector<decoder_layer_t<val_type, updator_type>> m_layers;
public:
    decoder_t(int const& n_layers = 1, int const& head_num = 1, int const& d_model = 1, int const& seq_len = 1): m_encoder(nullptr)
    {
        m_layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            get_mha(i).set_param(head_num, d_model, true, seq_len);
            get_mhca(i).set_param(head_num, d_model, false, seq_len);
            get_mhca(i).set_encoder_param(m_encoder_output, m_encoder_delta);
            get_ffn(i).reinit(std::vector<int>{d_model, d_model});
        }
    }

    void set_param(int const& n_layers, int const& head_num, int const& d_model, int const& seq_len = 1)
    {
        m_layers.resize(n_layers);
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).set_param(head_num, d_model, true, seq_len);
            get_mhca(i).set_param(head_num, d_model, false, seq_len);
            get_mhca(i).set_encoder_param(m_encoder_output, m_encoder_delta);   // 设置编码器输出和梯度的引用，以便交叉注意力机制使用
            get_ffn(i).reinit(std::vector<int>{d_model, d_model});
        }
    }

    mat_t<val_type> forward(const mat_t<val_type>& input)
    {
        mat_t<val_type> output = input;
        for (auto& layer: m_layers)
        {
            output = layer.forward(output);
        }
        return output;
    }

    mat_t<val_type> backward(const mat_t<val_type>& input)
    {
        if (m_encoder_delta.row_num() != input.row_num() || m_encoder_delta.col_num() != input.col_num())
        {
            m_encoder_delta.reshape(input.row_num(), input.col_num());
        }
        m_encoder_delta = 0.0;

        mat_t<val_type> grad = input;
        for (int i = m_layers.size() - 1; i >= 0; --i)
        {
            grad = m_layers[i].backward(grad);
            if (m_encoder)
            {
                m_encoder->backward(m_encoder_delta);   // 将编码器的梯度传递给编码器
                m_encoder_delta = 0.0; // 清空编码器的梯度，以便下一层使用
            }
        }
        return grad;
    }

    mat_t<val_type> get_encoder_delta() const
    {
        return m_encoder_delta;
    }

    void set_encoder(encoder_t<val_type, updator_type>& encoder)
    {
        m_encoder = &encoder;
        for (size_t i = 0; i < size(); ++i)
        {
            get_mhca(i).set_encoder_param(m_encoder_output, m_encoder_delta);   // 设置编码器输出和梯度的引用，以便交叉注意力机制使用
        }
    }

    void set_encoder_output(const mat_t<val_type>& encoder_output)
    {
        m_encoder_output = encoder_output;
        m_encoder_delta.reshape(encoder_output.row_num(), encoder_output.col_num());
    }

    mat_t<val_type> get_encoder_output() const
    {
        return m_encoder_output;
    }

    size_t size() const
    {
        return m_layers.size();
    }

    template <typename init_type>
    void init_weight()
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).template init_weight<init_type>();       // 初始化每一层的多头注意力机制
            get_mha_norm(i).template init_weight<init_type>();  // 初始化每一层的多头注意力机制的归一化层
            get_mhca(i).template init_weight<init_type>();      // 初始化每一层的交叉多头注意力机制
            get_mhca_norm(i).template init_weight<init_type>(); // 初始化每一层的交叉多头注意力机制的归一化层
            get_ffn(i).template init_weight<init_type>();       // 初始化每一层的前馈网络
            get_ffn_norm(i).template init_weight<init_type>();  // 初始化每一层的前馈网络的归一化层
        }
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "decoder = [";
        for (size_t i = 0; i < size(); ++i)
        {
            ss << "\n" << print_indent(indent + 2) << "Layer " << i << " mha: " << get_mha(i).net_type() << "-->";
            ss << "mhca: " << get_mhca(i).net_type() << "-->";
            ss << "ffn: " << get_ffn(i).net_type();
        }
        ss << std::endl << print_indent(indent) << "]";
        return ss.str();
    }

    template<typename... upr_param_types>
    void set_updator(upr_param_types&&... params)
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).template set_updator(std::forward<upr_param_types>(params)...);
            get_mha_norm(i).template set_updator(std::forward<upr_param_types>(params)...);
            get_mhca(i).template set_updator(std::forward<upr_param_types>(params)...);
            get_mhca_norm(i).template set_updator(std::forward<upr_param_types>(params)...);
            get_ffn(i).template set_updator(std::forward<upr_param_types>(params)...);
            get_ffn_norm(i).template set_updator(std::forward<upr_param_types>(params)...);
        }
    }

    void step()
    {
        for (size_t i = 0; i < size(); ++i)
        {
            m_layers[i].step();
        }
    }

    auto& get_mha(int const& i)
    {
        return m_layers[i].template get<0, 0>().base_net();
    }

    auto& get_mha_norm(int const& i)
    {
        return m_layers[i].template get<0, 1>();
    }

    auto& get_mhca(int const& i)
    {
        return m_layers[i].template get<1, 0>().base_net();
    }

    auto& get_mhca_norm(int const& i)
    {
        return m_layers[i].template get<1, 1>();
    }

    auto& get_ffn(int const& i)
    {
        return m_layers[i].template get<2, 0, 0>();
    }

    auto& get_ffn_norm(int const& i)
    {
        return m_layers[i].template get<2, 1>();
    }

    auto& get_mha(int const& i) const
    {
        return m_layers[i].template get<0, 0>().base_net();
    }

    auto& get_mha_norm(int const& i) const
    {
        return m_layers[i].template get<0, 1>().base_net();
    }

    auto& get_mhca(int const& i) const
    {
        return m_layers[i].template get<1, 0>().base_net();
    }

    auto& get_mhca_norm(int const& i) const
    {
        return m_layers[i].template get<1, 1>().base_net();
    }

    auto& get_ffn(int const& i) const
    {
        return m_layers[i].template get<2, 0, 0>();
    }

    auto& get_ffn_norm(int const& i) const
    {
        return m_layers[i].template get<2, 1>().base_net();
    }
};

template<typename val_type_, template<typename> class updator_type>
class transformer_kernel_t
{
public:
    using val_type = val_type_;
    using encoder_type = encoder_t<val_type, updator_type>;
    using decoder_type = decoder_t<val_type, updator_type>;
private:
    encoder_type m_encoder;
    decoder_type m_decoder;

public:
    transformer_kernel_t(int const& en_layers = 1, int const& de_layers = 1, int const& head_num = 1, int const& d_model = 1, int const& seq_len = 1)
        : m_encoder(en_layers, head_num, d_model, seq_len), m_decoder(de_layers, head_num, d_model, seq_len)
    {
        m_decoder.set_encoder(m_encoder);   // 将编码器的引用传递给解码器，以便交叉注意力机制使用
    }

    void encoder_forward(const mat_t<val_type>& input)
    {
        m_decoder.set_encoder_output(m_encoder.forward(input));
    }

    // 必须在encoder_forward之后调用，因为decoder的一部分输入是encoder的输出，可以在encoder_forward调用一次后多次调用forward以生成序列信息
    mat_t<val_type> forward(const mat_t<val_type>& input)
    {
        return m_decoder.forward(input);
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        return m_decoder.backward(delta);
    }

    #if 0 
    // 这部分删除，由解码器自动逐层调用编码器的backward。但是需要使用带缓存的优化器，这样比较符合实际
    // 编码器反向传播，必须在backward之后调用，因为编码器的梯度是在decoder的backward中计算的
    mat_t<val_type> encoder_backward()
    {
        auto encoder_delta = m_decoder.get_encoder_delta();
        return m_encoder.backward(encoder_delta);
    }
    #endif

    template<typename...upr_param_types>
    void set_updator(upr_param_types&&... params)
    {
        m_encoder.set_updator(std::forward<upr_param_types>(params)...);
        m_decoder.set_updator(std::forward<upr_param_types>(params)...);
    }

    template<typename init_type>
    void init_weight()
    {
        m_encoder.template init_weight<init_type>();
        m_decoder.template init_weight<init_type>();
    }

    // 编解码器序列长度可能不一样，因此不设置，而且序列长度仅影响初始化时候的qkv缓存长度，实际不影响运行
    void set_param(int const& en_layers, int const& de_layers, int const& head_num, int const& d_model)
    {
        m_encoder.set_param(en_layers, head_num, d_model, 1);
        m_decoder.set_param(de_layers, head_num, d_model, 1);
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "transformer_kernel = [\n" 
        << m_encoder.net_type(indent + 2) << " \n" 
        << m_decoder.net_type(indent + 2)
        << std::endl
        << print_indent(indent) << "]";
        return ss.str();
    }

    void step()
    {
        m_encoder.step();
        m_decoder.step();
    }

};

#include <iostream>
#include "mat_updator_t.hpp"
#include "mat_init_t.hpp"
#include "mat_loss_t.hpp"

void test_base_type()
{
    using val_type = double;
    mat_t<val_type> input(4, 2, {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4});
    mat_t<val_type> target(4, 2, {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4}); 
    int train_steps = 20000;

    res_mha_t<val_type, nadam_t> res_mha;
    res_mha.base_net().set_param(2, 4, false, 1);
    res_mha.base_net().set_updator(0.01);
    res_mha.init_weight<xavier_gaussian_t>();
    for (int i = 0; i < train_steps; ++i)
    {        
        auto output = res_mha.forward(input);
        res_mha.backward(output - target);
        res_mha.step();
    }
    std::cout << "res_mha output: \n" << res_mha.forward(input) << std::endl;

    res_mha_norm_t<val_type, nadam_t> res_mha_norm;
    res_mha_norm.get<0>().base_net().set_param(2, 4, false, 1);
    res_mha_norm.get<0>().base_net().set_updator(0.01);
    res_mha_norm.init_weight<xavier_gaussian_t>();
    for (int i = 0; i < train_steps; ++i)
    {
        auto output = res_mha_norm.forward(input);
        res_mha_norm.backward(output - target);
        res_mha_norm.step();
    }
    std::cout << "res_mha_norm output: \n" << res_mha_norm.forward(input) << std::endl;
    res_ffn_t<val_type, nadam_t> res_ffn;
    res_ffn.base_net().reinit(std::vector<int>{4, 4});
    res_ffn.base_net().set_updator(0.01);
    res_ffn.init_weight<xavier_gaussian_t>();
    for (int i = 0; i < train_steps; ++i)
    {
        auto output = res_ffn.forward(input);
        res_ffn.backward(output - target);
        res_ffn.step();
    }
    std::cout << "res_ffn output: \n" << res_ffn.forward(input) << std::endl;

    res_ffn_norm_t<val_type, nadam_t> res_ffn_norm;
    res_ffn_norm.get<0>().base_net().reinit(std::vector<int>{4, 4});
    res_ffn_norm.get<0>().base_net().set_updator(0.01);
    res_ffn_norm.get<1>().set_updator(0.01);
    res_ffn_norm.init_weight<xavier_gaussian_t>();
    for (int i = 0; i < train_steps; ++i)
    {
        auto output = res_ffn_norm.forward(input);
        res_ffn_norm.backward(output - target);
        res_ffn_norm.step();
    }
    std::cout << "res_ffn_norm output: \n" << res_ffn_norm.forward(input) << std::endl;
}

void test_encoder()
{
    //encoder_t<float, nadam_t> encoder(2, 2, 4, 1);
    //mat_t<float> input(4, 1, {0.1f, 0.2f, 0.3f, 0.4f});
    //encoder.init_weight<xavier_gaussian_t>();
    //std::cout << encoder.net_type() << std::endl;
    //std::cout << encoder.forward(input) << std::endl;
    using encoder_type = encoder_t<double, adam_t>;
    using net_type = complex_net_builder_t<double>
        ::push_back_impl<encoder_type>
        ::push_back_staticnet<mse_loss_t>
        ::type;
    net_type cnet;
    cnet.template get<0>().set_param(2, 2, 4, 1);           // 设置编码器的参数，注意，这里要单独设置，因为和权重网络不一样
    std::cout << "input lr:";
    double lr = 0.01;
    std::cin >> lr;
    cnet.set_updator(lr);                  // 设置学习率，set_updator会自动递归调用下面的所有层并且设置相同的参数
    cnet.init_weight<xavier_gaussian_t>();
    std::cout << cnet.net_type() << std::endl;
    mat_t<double> input(4, 2, {   0.1, 0.4
                                , 0.2, 0.3
                                , 0.3, 0.2
                                , 0.4, 0.1});
    mat_t<double> target(4, 2, {  0.1, 0.4
                                , 0.2, 0.3
                                , 0.3, 0.2
                                , 0.4, 0.1});
    std::cout << "before training encoder output: \n" << cnet.forward(input) << std::endl;
    for (int i = 0; i < 2000; ++i)
    {
        cnet.forward(input);
        cnet.backward(target);
        cnet.step();
    }
    std::cout << "after training encoder output: \n" << cnet.forward(input) << std::endl;
}

void test_decoder()
{
    using decoder_type = decoder_t<double, nadam_t>;
    using net_type = complex_net_builder_t<double>
        ::push_back_impl<decoder_type>
        ::push_back_staticnet<mse_loss_t>
        ::type;
    net_type cnet;
    cnet.template get<0>().set_param(2, 2, 4, 1);           // 设置解码器的参数，注意，这里要单独设置，因为和权重网络不一样
    std::cout << "input lr:";
    double lr = 0.01;
    std::cin >> lr;
    cnet.set_updator(lr);                  // 设置学习率，set_updator会自动递归调用下面的所有层并且设置相同的参数
    cnet.init_weight<xavier_gaussian_t>();
    std::cout << cnet.net_type() << std::endl;
    mat_t<double> input(4, 1, {0.1, 0.2, 0.3, 0.4});
    mat_t<double> target(4, 1, {0.5, 0.6, 0.7, 0.8});
    mat_t<double> encoder_output(4, 1, {0.9, 0.8, 0.7, 0.6});       // 假设编码器的输出就是这个，实际使用中应该是通过编码器的forward函数得到的
    cnet.template get<0>().set_encoder_output(encoder_output);   // 设置编码器的输出，以便交叉注意力机制使用
    std::cout << "before training decoder output: \n" << cnet.forward(input) << std::endl;
    for (int i = 0; i < 2000; ++i)
    {
        cnet.forward(input);
        cnet.backward(target);
    }
    std::cout << "after training decoder output: \n" << cnet.forward(input) << std::endl;
}

// 定义优化器
template<typename val_type>
using upr_tpl = cache_updator_t<val_type, nadam_t>;

void test_tf_kernel()
{
    using val_type = double;
    /* 测试数据准备 */
    mat_t<val_type> encoder_input(4, 1, {0.1, 0.2, 0.3, 0.4});
    // 设置解码器输入和目标输出
    mat_t<val_type> decoder_input(4, 1, {0.1, 0.2, 0.3, 0.4});
    mat_t<val_type> decoder_target(4, 1, {0.1, 0.2, 0.3, 0.4});


    /* 定义网络类型，设置网络参数 */
    using tf_type = transformer_kernel_t<val_type, upr_tpl>;
    using net_type = complex_net_builder_t<val_type>
        ::push_back_impl<tf_type>
        ::push_back_staticnet<mse_loss_t>
        ::type;
    net_type cnet;
    tf_type& tf = cnet.template get<0>();
    // 编码器层数、解码器层数、注意力头数、模型维度、序列长度
    tf.set_param(3, 2, 2, 4);           // 设置编码器和解码器的参数，注意，这里要单独设置，因为和权重网络不一样

    /* 设置学习率，初始化权重 */
    std::cout << cnet.net_type() << std::endl;
    std::cout << "input lr:";
    double lr = 0.01;
    cnet.init_weight<xavier_gaussian_t>();
    std::cin >> lr;
    cnet.set_updator(lr);                  // 设置学习率，set_updator会自动递归调用下面的所有层并且设置相同的参数

    /**!SECTION
     * 测试场景：编码器输入一次，解码器通过这次编码器的输入进行多次训练，因为后期在编码器输入之后解码器需要循环生成序列，
     * 也就是说解码器需要在一次编码器的输入情况下多次进行正向传播。在训练时是在一次解码器输入的情况下进行多次的训练
     */
    // 设置编码器输入
    tf.encoder_forward(encoder_input);
    // 解码器正向传播，然后反向传播
    auto output = tf.forward(decoder_input);
    std::cout << "Before training: " << output.to_string() << std::endl;
    for (int i = 0; i < 150; ++i)
    {
        output = cnet.forward(decoder_input);
        cnet.backward(decoder_target);    // 这里可以得到解码器的回传误差，并且会计算出编码器的梯度
        //tf.encoder_backward();  // 这里可以得到编码器的回传误差
        cnet.step();                // 更新权重，注意，这里是整个网络一起更新的，因为编码器的梯度是在解码器的backward中计算的，所以只要调用cnet.step()就可以了，不需要单独调用编码器的step()
    }
    std::cout << "After training: " << output.to_string() << std::endl;
}


#endif