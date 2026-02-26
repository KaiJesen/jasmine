#ifndef __TRANSFORMER_HPP__
#define __TRANSFORMER_HPP__

#include "mat_net_t.hpp"
#include "mat_mha_t.hpp"

// 定义基本的堆叠单元
template<typename val_type, template<typename> class updator_type>
using res_mha_t = residual_net_t<mat_mha_t<mat_t<val_type>, updator_type>>;

template<typename val_type, template<typename> class updator_type>
using res_mha_norm_t = complex_net_builder_t<val_type>
    ::template push_back_impl<res_mha_t<val_type, updator_type>>
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
using coder_layer_t = complex_net_builder_t<val_type>
    ::template push_back_impl<res_mha_norm_t<val_type, updator_type>>
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
    std::vector<coder_layer_t<val_type, updator_type>> m_layers;
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

    auto& get_ffn(int const& i)
    {
        return m_layers[i].template get<1, 0>().base_net();
    }

    auto& get_mha(int const& i) const
    {
        return m_layers[i].template get<0, 0>().base_net();
    }

    auto& get_ffn(int const& i) const
    {
        return m_layers[i].template get<1, 0>().base_net();
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
            get_ffn(i).template init_weight<init_type>();       
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
            get_ffn(i).template set_updator(std::forward<upr_arg_types>(args)...);
        }
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
    cnet.set_updator(0.1);                  // 设置学习率，set_updator会自动递归调用下面的所有层并且设置相同的参数
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


#endif