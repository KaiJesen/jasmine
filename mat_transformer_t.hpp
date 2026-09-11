#ifndef __MAT_TRANSFORMER_T_HPP__
#define __MAT_TRANSFORMER_T_HPP__

/* 
 * @brief: 组装transformer编解码器和RoPE的接口，提供一个统一的接口来调用编码器和解码器的前向传播和反向传播，同时提供一个接口来设置优化器的参数，进行训练。这个类的设计目的是为了简化transformer的使用，让用户可以更方便地进行训练和推理。
*/

#include "mat_net_t.hpp"
#include "mat_transformer_kernel_t.hpp"
#include "mat_RoPE_t.hpp"
#include "mat_loss_t.hpp"

namespace jasmine {

/*
 * transformer的基座，可以在上面增加各种识别层，比如：softmax层用于分类，线性层用于回归，或者其他的层。这个类的设计目的是为了提供一个统一的接口来调用transformer的前向传播和反向传播，同时提供一个接口来设置优化器的参数，进行训练。
 * 这个类的设计原则是：尽量简化接口，让用户可以更方便地进行训练和推理，同时提供足够的灵活性，让用户可以根据自己的需求来定制transformer的结构和参数。
 * 这个类的设计原则是：尽量简化接口，让用户可以更方便地进行训练和推理，同时提供足够的灵活性，让用户可以根据自己的需求来定制transformer的结构和参数。
 * 前层可以套接各种类型的embedding层，甚至是词典也可以，但是要和输出反向递归的一致。
*/
template <typename input_type, template <typename> class updator_type>
class transformer_base_t
{
public:
    using val_type = typename input_type::ele_type;
    using RoPE_type = RoPE_net_t<input_type>;
    using kernel_type = transformer_kernel_t<val_type, updator_type>;

private:
    kernel_type m_kernel;
    RoPE_type m_rope;

public:
    transformer_base_t(size_t en_layers = 1, size_t de_layers = 1, size_t head_num = 1, int d_model = 1, int d_ff = 0)
    : m_kernel(en_layers, de_layers, head_num, d_model, d_ff ? d_ff : d_model * 4), m_rope(d_model)
    {
    }

    void encoder_forward(const mat_t<val_type>& input)
    {
        m_kernel.encoder_forward(m_rope.forward(input));    // 经过RoPE处理后输入编码器，生成解码器的输入
    }

    mat_t<val_type> forward(const mat_t<val_type>& input)
    {
        return m_kernel.forward(m_rope.forward(input));    // 经过RoPE处理后输入解码器，生成输出
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        return m_rope.backward(m_kernel.backward(delta));    // 先反向传播得到解码器的输入梯度，再经过RoPE的反向传播得到编码器的输入梯度
    }

    template<typename...upr_param_types>
    void set_updator(upr_param_types&&... params)
    {
        m_kernel.set_updator(std::forward<upr_param_types>(params)...);
    }

    void set_lr(val_type lr)
    {
        m_kernel.set_lr(lr);
    }

    template<typename init_type>
    void init_weight()
    {
        m_kernel.template init_weight<init_type>();
    }

    void set_param(size_t en_layers, size_t de_layers, size_t head_num, int d_model, int d_ff = 0)
    {
        m_kernel.set_param(en_layers, de_layers, head_num, d_model, d_ff ? d_ff : d_model * 4);
        m_rope.set_param(d_model);
    }

    void step()
    {
        m_kernel.step();
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "transformer_base_t = [\n" 
        << m_kernel.net_type(indent + 2) << " \n" 
        << m_rope.net_type(indent + 2)
        << std::endl
        << print_indent(indent) << "]";
        return ss.str();
    }

};


template<typename val_type>
using base_upr_tpl = cache_updator_t<val_type, nadam_t>;



} // namespace jasmine
#endif
