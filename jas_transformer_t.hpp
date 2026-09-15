#ifndef __JAS_TRANSFORMER_T_HPP__
#define __JAS_TRANSFORMER_T_HPP__

/* 
 * @brief: 组装transformer编解码器接口。位置编码采用 RoPE，作用在各层 MHA/MHCA 的 Q/K 上
 *         （经 rope_registry 按 d_head 共享），而不是在进 encoder/decoder 前旋转整段输入。
*/

#include "jas_net_t.hpp"
#include "jas_transformer_kernel_t.hpp"
#include "jas_RoPE_t.hpp"
#include "jas_loss_t.hpp"

namespace jasmine {

/*
 * transformer的基座，可以在上面增加各种识别层，比如：softmax层用于分类，线性层用于回归，或者其他的层。这个类的设计目的是为了提供一个统一的接口来调用transformer的前向传播和反向传播，同时提供一个接口来设置优化器的参数，进行训练。
 * 这个类的设计原则是：尽量简化接口，让用户可以更方便地进行训练和推理，同时提供足够的灵活性，让用户可以根据自己的需求来定制transformer的结构和参数。
 * 前层可以套接各种类型的embedding层，甚至是词典也可以，但是要和输出反向递归的一致。
*/
template <typename input_type, template <typename> class updator_type>
class transformer_base_t
{
public:
    using val_type = typename input_type::ele_type;
    using kernel_type = transformer_kernel_t<val_type, updator_type>;

private:
    kernel_type m_kernel;
    int m_head_num = 1;
    int m_d_model = 1;

public:
    transformer_base_t(size_t en_layers = 1, size_t de_layers = 1, size_t head_num = 1, int d_model = 1, int d_ff = 0)
    : m_kernel(en_layers, de_layers, head_num, d_model, d_ff ? d_ff : d_model * 4),
      m_head_num(static_cast<int>(head_num)), m_d_model(d_model)
    {
        ensure_rope_registry();
    }

    /**
     * RoPE 作用位置：各注意力头在得到 Q/K 之后、算 score 之前旋转（见 mat_head_gen_t）。
     * 此处不再对 encoder/decoder 输入做整段 RoPE。
     */
    void encoder_forward(const mat_t<val_type>& input)
    {
        m_kernel.encoder_forward(input);
    }

    mat_t<val_type> forward(const mat_t<val_type>& input)
    {
        return m_kernel.forward(input);
    }

    /**
     * 推理单列/增量：decoder self-attn 使用 KV cache。
     * 入口在 transformer 子模块，不经过 complex_net 顶层的 loss 头。
     */
    mat_t<val_type> forward_one(const mat_t<val_type>& input)
    {
        return m_kernel.forward_one(input);
    }

    void clear_kv_cache()
    {
        m_kernel.clear_kv_cache();
    }

    void reserve_kv_cache(int max_seq)
    {
        m_kernel.reserve_kv_cache(max_seq);
    }

    void set_kv_cache_mode(kv_cache_mode mode)
    {
        m_kernel.set_kv_cache_mode(mode);
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        return m_kernel.backward(delta);
    }

    /** 供外层 embedding：encoder 输入上的梯度（需先完成 decoder backward） */
    mat_t<val_type> const& encoder_input_delta() const
    {
        return m_kernel.encoder_input_delta();
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
        m_head_num = static_cast<int>(head_num);
        m_d_model = d_model;
        m_kernel.set_param(en_layers, de_layers, head_num, d_model, d_ff ? d_ff : d_model * 4);
        ensure_rope_registry();
    }

    /** 预热 RoPE 静态/动态缓存到给定最大序列长（按 d_head 注册） */
    void reserve_rope(int max_seq_len)
    {
        const int d_head = m_d_model / m_head_num;
        if (d_head > 0 && d_head % 2 == 0)
            rope_registry_t<val_type>::instance().get(d_head, max_seq_len);
    }

    void set_rope_cache_mode(rope_cache_mode mode)
    {
        auto& reg = rope_registry_t<val_type>::instance();
        reg.set_default_mode(mode);
        // 已存在的条目也切模式并按需重绑
        ensure_rope_registry();
    }

    void step()
    {
        m_kernel.step();
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        const int d_head = (m_head_num > 0) ? (m_d_model / m_head_num) : 0;
        ss << print_indent(indent) << "transformer_base_t = [\n"
           << m_kernel.net_type(indent + 2) << "\n"
           << print_indent(indent + 2) << "rope: Q/K via registry(d_head=" << d_head << ")\n"
           << print_indent(indent) << "]";
        return ss.str();
    }

private:
    void ensure_rope_registry()
    {
        if (m_head_num <= 0 || m_d_model % m_head_num != 0)
            return;
        const int d_head = m_d_model / m_head_num;
        if (d_head > 0 && d_head % 2 == 0)
            rope_registry_t<val_type>::instance().get(d_head);
    }

};


template<typename val_type>
using base_upr_tpl = cache_updator_t<val_type, nadam_t>;



} // namespace jasmine
#endif
