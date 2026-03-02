#ifndef __TRANSFORMER_BASE_HPP__
#define __TRANSFORMER_BASE_HPP__

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
using res_ffn_t = residual_net_t<weight_net_t<mat_t<val_type>, updator_type>>;

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
        return m_layers[i].template get<1, 0>().base_net();
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
        return m_layers[i].template get<1, 0>().base_net();
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

    std::string net_type() const
    {
        std::stringstream ss;
        ss << "encoder[";
        for (size_t i = 0; i < size(); ++i)
        {
            ss << "Layer " << i << " mha: " << get_mha(i).net_type() << "-->";
            ss << "ffn: " << get_ffn(i).net_type();
            if (i != size() - 1)
                ss << " >> ";
        }
        ss << "]";
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
};

template<typename val_type_, template<typename> class updator_type>
class decoder_t
{
public:
    using val_type = val_type_;
private:
    mat_t<val_type> m_encoder_output;   // 用于保存编码器的输出，以便交叉注意力机制使用
    mat_t<val_type> m_encoder_delta;    // 用于保存编码器的梯度，以便交叉注意力机制使用
    std::vector<decoder_layer_t<val_type, updator_type>> m_layers;
public:
    decoder_t(int const& n_layers = 1, int const& head_num = 1, int const& d_model = 1, int const& seq_len = 1)
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
        mat_t<val_type> grad = input;
        for (int i = m_layers.size() - 1; i >= 0; --i)
        {
            grad = m_layers[i].backward(grad);
        }
        return grad;
    }

    mat_t<val_type> get_encoder_delta() const
    {
        return m_encoder_delta;
    }

    void set_encoder_output(const mat_t<val_type>& encoder_output)
    {
        m_encoder_output = encoder_output;
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

    std::string net_type() const
    {
        std::stringstream ss;
        ss << "decoder[";
        for (size_t i = 0; i < size(); ++i)
        {
            ss << "Layer " << i << " mha: " << get_mha(i).net_type() << "-->";
            ss << "mhca: " << get_mhca(i).net_type() << "-->";
            ss << "ffn: " << get_ffn(i).net_type();
            if (i != size() - 1)
                ss << " >> ";
        }
        ss << "]";
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
        return m_layers[i].template get<2, 0>().base_net();
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
        return m_layers[i].template get<2, 0>().base_net();
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
    transformer_kernel_t(int const& n_layers = 1, int const& head_num = 1, int const& d_model = 1, int const& seq_len = 1)
        : m_encoder(n_layers, head_num, d_model, seq_len), m_decoder(n_layers, head_num, d_model, seq_len)
    {
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

    // 编码器反向传播，必须在backward之后调用，因为编码器的梯度是在decoder的backward中计算的
    mat_t<val_type> encoder_backward()
    {
        auto encoder_delta = m_decoder.get_encoder_delta();
        return m_encoder.backward(encoder_delta);
    }

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
    void set_param(int const& n_layers, int const& head_num, int const& d_model)
    {
        m_encoder.set_param(n_layers, head_num, d_model);
        m_decoder.set_param(n_layers, head_num, d_model);
    }

};

#include <iostream>
#include "mat_updator_t.hpp"
#include "mat_init_t.hpp"
#include "mat_loss_t.hpp"

void test_encoder()
{
    //encoder_t<float, nadam_t> encoder(2, 2, 4, 1);
    //mat_t<float> input(4, 1, {0.1f, 0.2f, 0.3f, 0.4f});
    //encoder.init_weight<xavier_gaussian_t>();
    //std::cout << encoder.net_type() << std::endl;
    //std::cout << encoder.forward(input) << std::endl;
    using encoder_type = encoder_t<double, nadam_t>;
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
    mat_t<double> input(4, 1, {0.1, 0.2, 0.3, 0.4});
    mat_t<double> target(4, 1, {0.5, 0.6, 0.7, 0.8});
    std::cout << "before training encoder output: \n" << cnet.forward(input) << std::endl;
    for (int i = 0; i < 2000; ++i)
    {
        cnet.forward(input);
        cnet.backward(target);
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


#endif