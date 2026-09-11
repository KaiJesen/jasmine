#ifndef __MAT_TRANSFORMER_KERNEL_T_HPP__
#define __MAT_TRANSFORMER_KERNEL_T_HPP__

#include "mat_net_t.hpp"
#include "mat_mha_t.hpp"

namespace jasmine {

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
    ::template push_back_updatable<weight_net_t, updator_type>
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
 *           0. weight_net_t: 包含一个线性层，输入d_model，输出d_ff
 *           1. relu_net_t: 包含一个ReLU激活层，输入输出维度相同
 *           2. weight_net_t: 包含一个线性层，输入d_ff，输出d_model
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
    encoder_t(int const& n_layers = 1, int const& head_num = 1, int const& d_model = 1, int d_ff = 0, int const& seq_len = 1)
    {
        if (d_ff == 0) d_ff = d_model * 4;
        m_layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            get_mha(i).set_param(head_num, d_model, false, seq_len);
            get_ffn_front(i).reinit(std::vector<int>{d_model, d_ff});
            get_ffn_back(i).reinit(std::vector<int>{d_ff, d_model});
        }
    }

    void set_param(int const& n_layers, int const& head_num, int const& d_model, int const& d_ff, int const& seq_len)
    {
        m_layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            get_mha(i).set_param(head_num, d_model, false, seq_len);
            get_ffn_front(i).reinit(std::vector<int>{d_model, d_ff});
            get_ffn_back(i).reinit(std::vector<int>{d_ff, d_model});
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

    auto& get_ffn_front(int const& i)
    {
        return m_layers[i].template get<1, 0, 0>();
    }

    auto& get_ffn_back(int const& i)
    {
        return m_layers[i].template get<1, 0, 2>();
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
    
    auto& get_ffn_front(int const& i) const
    {
        return m_layers[i].template get<1, 0, 0>();
    }
    
    auto& get_ffn_back(int const& i) const
    {
        return m_layers[i].template get<1, 0, 2>();
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
            get_ffn_front(i).template init_weight<init_type>();       
            get_ffn_back(i).template init_weight<init_type>();       
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
            ss << "ffn: " << get_ffn_front(i).net_type() << "-->" << get_ffn_back(i).net_type();
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
            get_ffn_front(i).template set_updator(std::forward<upr_arg_types>(args)...);
            get_ffn_back(i).template set_updator(std::forward<upr_arg_types>(args)...);
            get_ffn_norm(i).template set_updator(std::forward<upr_arg_types>(args)...);
        }
    }

    void set_lr(val_type lr)
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).set_lr(lr);
            get_mha_norm(i).set_lr(lr);
            get_ffn_front(i).set_lr(lr);
            get_ffn_back(i).set_lr(lr);
            get_ffn_norm(i).set_lr(lr);
        }
    }

    void step()
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).step();
            get_mha_norm(i).step();
            get_ffn_front(i).step();
            get_ffn_back(i).step();
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
 *         0. weight_net_t: 包含一个线性层，输入d_model，输出d_ff
 *         1. relu_net_t: 包含一个ReLU激活层，输入输出维度相同
 *         2. weight_net_t: 包含一个线性层，输入d_ff，输出d_model
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
    decoder_t(int const& n_layers = 1, int const& head_num = 1, int const& d_model = 1, int d_ff = 0, int const& seq_len = 1): m_encoder(nullptr)
    {
        if (d_ff == 0) d_ff = d_model * 4;
        m_layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            get_mha(i).set_param(head_num, d_model, true, seq_len);
            get_mhca(i).set_param(head_num, d_model, false, seq_len);
            get_mhca(i).set_encoder_param(m_encoder_output, m_encoder_delta);
            get_ffn_front(i).reinit(std::vector<int>{d_model, d_ff});
            get_ffn_back(i).reinit(std::vector<int>{d_ff, d_model});
        }
    }

    void set_param(int const& n_layers, int const& head_num, int const& d_model, int const& d_ff, int const& seq_len = 1)
    {
        m_layers.resize(n_layers);
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).set_param(head_num, d_model, true, seq_len);
            get_mhca(i).set_param(head_num, d_model, false, seq_len);
            get_mhca(i).set_encoder_param(m_encoder_output, m_encoder_delta);   // 设置编码器输出和梯度的引用，以便交叉注意力机制使用
            get_ffn_front(i).reinit(std::vector<int>{d_model, d_ff});
            get_ffn_back(i).reinit(std::vector<int>{d_ff, d_model});
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
        if (m_encoder_delta.row_num() != m_encoder_output.row_num()
            || m_encoder_delta.col_num() != m_encoder_output.col_num())
        {
            m_encoder_delta.reshape(m_encoder_output.row_num(), m_encoder_output.col_num());
        }
        m_encoder_delta = 0.0;

        mat_t<val_type> grad = input;
        for (int i = m_layers.size() - 1; i >= 0; --i)
        {
            grad = m_layers[i].backward(grad);
            // 各层 cross-attn 的 encoder 梯度累加到 m_encoder_delta，循环结束后再统一回传
        }
        if (m_encoder)
        {
            m_encoder->backward(m_encoder_delta);
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
            get_ffn_front(i).template init_weight<init_type>();       // 初始化每一层的前馈网络
            get_ffn_back(i).template init_weight<init_type>();       // 初始化每一层的前馈网络
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
            ss << "ffn: " << get_ffn_front(i).net_type() << "-->" << get_ffn_back(i).net_type();
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
            get_ffn_front(i).template set_updator(std::forward<upr_param_types>(params)...);
            get_ffn_back(i).template set_updator(std::forward<upr_param_types>(params)...);
            get_ffn_norm(i).template set_updator(std::forward<upr_param_types>(params)...);
        }
    }

    void set_lr(val_type lr)
    {
        for (size_t i = 0; i < size(); ++i)
        {
            get_mha(i).set_lr(lr);
            get_mha_norm(i).set_lr(lr);
            get_mhca(i).set_lr(lr);
            get_mhca_norm(i).set_lr(lr);
            get_ffn_front(i).set_lr(lr);
            get_ffn_back(i).set_lr(lr);
            get_ffn_norm(i).set_lr(lr);
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

    auto& get_ffn_front(int const& i)
    {
        return m_layers[i].template get<2, 0, 0>();
    }
    
    auto& get_ffn_back(int const& i)
    {
        return m_layers[i].template get<2, 0, 2>();
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

    auto& get_ffn_front(int const& i) const
    {
        return m_layers[i].template get<2, 0, 0>();
    }
    
    auto& get_ffn_back(int const& i) const
    {
        return m_layers[i].template get<2, 0, 2>();
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
    transformer_kernel_t(int const& en_layers = 1, int const& de_layers = 1, int const& head_num = 1, int const& d_model = 1, int d_ff = 0, int const& seq_len = 1)
        : m_encoder(en_layers, head_num, d_model, d_ff ? d_ff : d_model * 4, seq_len), m_decoder(de_layers, head_num, d_model, d_ff ? d_ff : d_model * 4, seq_len)
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

    void set_lr(val_type lr)
    {
        m_encoder.set_lr(lr);
        m_decoder.set_lr(lr);
    }

    template<typename init_type>
    void init_weight()
    {
        m_encoder.template init_weight<init_type>();
        m_decoder.template init_weight<init_type>();
    }

    // 编解码器序列长度可能不一样，因此不设置，而且序列长度仅影响初始化时候的qkv缓存长度，实际不影响运行
    void set_param(int const& en_layers, int const& de_layers, int const& head_num, int const& d_model, int const& d_ff)
    {
        m_encoder.set_param(en_layers, head_num, d_model, d_ff, 1);
        m_decoder.set_param(de_layers, head_num, d_model, d_ff, 1);
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



} // namespace jasmine
#endif
