#ifndef __MAT_NET_T_HPP__
#define __MAT_NET_T_HPP__

#include <string>
#include <sstream>

#include "mat_t.hpp"
#include "mat_express_t.hpp"
#include "mat_storage.hpp"

#include "mat_updator_t.hpp"

namespace jasmine {

template <typename input_type, template<typename> class updator_type>
class weight_net_t
{
public:
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_weight;
    updator_type<val_type> m_weight_updator;
    mat_t<val_type> m_bias;
    updator_type<val_type> m_bias_updator;

    mat_t<val_type> m_input;
public:
    weight_net_t(int const& input_size = 1, int const& output_size = 1)
        : m_weight(output_size, input_size), m_weight_updator(), m_bias(output_size, 1), m_bias_updator()
    {
        // 初始化权重和偏置
    }

    void reinit(std::vector<int> const& container)      // 初始化权重矩阵的维度，以为权重初始化准备
    {
        m_weight.reshape(container[1], container[0]);
        m_bias.reshape(container[1], 1);
    }

    // m_input 持久化供 backward；mat rvalue 在层间移动，表达式只物化一次
    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        return m_weight.dot(m_input) + m_bias;
    }

    /** 无状态层：单列输入与整段 forward 相同 */
    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename init_type>
    void init_weight()
    {
        init_matrix<init_type>(m_weight);
        init_matrix<init_type>(m_bias);
    }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_weight_updator.set(std::forward<upr_arg_types>(args)...);
        m_bias_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_weight_updator.set_lr(lr);
        m_bias_updator.set_lr(lr);
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        mat_t<val_type> delta_weight = delta.dot(m_input.t());
        auto delta_bias = hsum(delta);
        mat_t<val_type> ret = m_weight.t().dot(delta);
        // 更新权重和偏置
        m_weight_updator.update(delta_weight, m_weight);
        m_bias_updator.update(delta_bias, m_bias);
        return ret;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "weight_net_t:(in:" << m_weight.col_num() << ", out:" << m_weight.row_num() << ")";
        return ss.str();
    }

    void step()
    {
        m_weight_updator.step();
        m_bias_updator.step();
    }
};

template <typename input_type>
class sigmoid_net_t
{
public:
    using val_type = typename input_type::ele_type;
private:
    mat_t<val_type> m_output;                // 前一次的输入，反向传播时用于快捷计算
public:
    sigmoid_net_t() = default;

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        m_output = sigmoid(std::forward<Src>(input));
        return m_output;
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    auto backward(const mat_t<val_type>& delta)
    {
        if (delta.row_num() != m_output.row_num() || delta.col_num() != m_output.col_num())
        {
            throw std::runtime_error("delta size does not match input size");
        }
        return mat_t<val_type>(delta * (static_cast<val_type>(1) - m_output) * m_output);
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "sigmoid_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 什么也不做
    }

    void step()
    {
        // 什么也不做
    }
};

template <typename input_type>
class relu_net_t
{
private:
    using val_type = typename input_type::ele_type;
    mat_t<val_type> m_input;                // 前一次的输入，反向传播时用于快捷计算
public:
    relu_net_t() = default;

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        return (m_input > 0) * m_input;
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    auto backward(const other_type& delta)
    {
        if (delta.row_num() != m_input.row_num() || delta.col_num() != m_input.col_num())
        {
            throw std::runtime_error("delta size does not match input size");
        }
        return mat_t<val_type>(delta * (m_input > 0));
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "relu_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 什么也不做
    }

    void step()
    {
        // 什么也不做
    }
};

// 纵向的标准化层，即对每一列（每个 token）在特征维上做 LayerNorm
template <typename input_type, template<typename> class updator_type>
class layer_norm_net_t
{
private:
    using val_type = typename input_type::ele_type;
    static constexpr val_type eps = static_cast<val_type>(1e-5);
    mat_t<val_type> m_hx;
    mat_t<val_type> m_mean;
    mat_t<val_type> m_std;
    mat_t<val_type> m_gama;     // 缩放参数 [d_model, 1]
    updator_type<val_type> m_gama_updator;
    mat_t<val_type> m_beta;     // 平移参数 [d_model, 1]
    updator_type<val_type> m_beta_updator;
public:
    layer_norm_net_t() = default;

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_gama_updator.set(std::forward<upr_arg_types>(args)...);
        m_beta_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_gama_updator.set_lr(lr);
        m_beta_updator.set_lr(lr);
    }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        // 对每一列在 row（特征）维上标准化
        m_mean = vmean(input);
        mat_t<val_type> centered = (input - m_mean).clone();
        mat_t<val_type> var = (vmean(pow(centered, 2.0)) + eps).clone();
        m_std = sqrt(var);
        m_hx = (centered / m_std).clone();
        if (m_gama.valid() == false)
        {
            m_gama = mat_t<val_type>(input.row_num(), 1);
            m_beta = mat_t<val_type>(input.row_num(), 1);
            m_gama = val_type(1);
            m_beta = val_type(0);
        }
        return (m_gama * m_hx + m_beta).clone();
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
    }

    template <typename other_type>
    auto backward(const other_type& delta)
    {
        // gamma/beta 梯度：沿序列维（列）累加
        auto L_gama = hsum(delta * m_hx);
        auto L_beta = hsum(delta);

        // 输入梯度：沿特征维（行）归约，需与 forward 的 vmean/vsum 一致
        val_type m = static_cast<val_type>(delta.row_num());
        auto dx_norm = delta * m_gama;
        auto sum_dx_norm = vsum(dx_norm);
        auto sum_dx_norm_x_hx = vsum(dx_norm * m_hx);
        mat_t<val_type> L_input =
            ((dx_norm * m - sum_dx_norm - m_hx * sum_dx_norm_x_hx) / m / m_std).clone();

        m_gama_updator.update(L_gama, m_gama);
        m_beta_updator.update(L_beta, m_beta);
        return L_input;
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "layer_norm_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // LayerNorm 仿射参数在首次 forward 时懒初始化为 gamma=1, beta=0
    }

    void step()
    {
        m_gama_updator.step();
        m_beta_updator.step();
    }
};

template <typename input_type>
class hsoftmax_net_t
{
public:
    using val_type = typename input_type::ele_type;
    mat_t<val_type> m_output;

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        m_output = hsoftmax(std::forward<Src>(input));
        return m_output;
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        return mat_t<val_type>(m_output * (delta - hsum(m_output * delta)));
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "hsoftmax_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 不需要初始化权重
    }

    void step()
    {}
};

template <typename base_net_type>
class residual_net_t
{
public:
    using val_type = typename base_net_type::val_type;
    base_net_type m_net;

    residual_net_t()        // 入参没有什么作用，仅仅用于表示这是一个需要reinit的网络
        : m_net()
    {
    }

    base_net_type& base_net()
    {
        return m_net;
    }

    base_net_type const& base_net() const
    {
        return m_net;
    }

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        mat_t<val_type> skip(std::forward<Src>(input));
        return m_net.forward(skip) + skip;
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        mat_t<val_type> skip(std::forward<Src>(input));
        return m_net.forward_one(skip) + skip;
    }

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        return mat_t<val_type>(m_net.backward(delta) + delta);
    }
    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "residual_net_t:\n" << m_net.net_type(indent + 2);
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        m_net.template init_weight<init_type>();
    }

    template <size_t...nums>
    decltype(auto) get()
    {
        return m_net.template get<nums...>();
    }

    template <size_t...nums>
    decltype(auto) get() const
    {
        return m_net.template get<nums...>();
    }

    void step()
    {
        m_net.step();
    }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_net.set_updator(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_net.set_lr(lr);
    }
};

template <typename... net_types>
class complex_net_t
{
private:
    std::tuple<net_types...> m_nets;
public:

    using val_type = typename std::tuple_element_t<0, std::tuple<net_types...>>::val_type;

    template <typename input_type>
    auto forward(input_type&& input)
    {
        return std::apply([&input](auto&&... nets) {
            return net_forward(std::forward<input_type>(input), nets...);
        }, m_nets);
    }

    /**
     * 推理前向：与 forward 同一套 net 顺序；对 skip_on_infer 层（通常为 loss）跳过不调用，
     * 输入原样继续后续层。各层走 forward_one（有 KV 的层可增量，其余默认 ≡ forward）。
     */
    template <typename input_type>
    auto infer(input_type&& input)
    {
        return infer_chain<0>(std::forward<input_type>(input));
    }

    /** 新序列推理前：递归清除子网中的 KV cache（若存在） */
    void infer_reset()
    {
        infer_reset_chain<0>();
    }

    /** 可选：为含 reserve_kv_cache 的子网预分配容量 */
    void infer_prepare(int max_kv_seq = 0)
    {
        infer_prepare_chain<0>(max_kv_seq);
    }

    /** decoder 等无 loss 尾的子网：整链 forward_one；含 loss 尾时请用 infer() */
    template <typename input_type>
    auto forward_one(input_type&& input)
    {
        return std::apply([&input](auto&&... nets) {
            return net_forward_one(std::forward<input_type>(input), nets...);
        }, m_nets);
    }

    template <typename input_type>
    auto backward(const input_type& delta)
    {
        return std::apply([&delta](auto&&... nets) {return net_backward(delta, nets...); }, m_nets);
    }

    constexpr size_t size()
    {
        return sizeof...(net_types);
    }

    template <typename container_type, size_t N = 0, size_t I = 0>
    void reinit(container_type const& container)      // 如果是稳定网络则不需要重新初始化
    {
        using mbr_net_type = std::tuple_element_t<N, std::tuple<net_types...>>;
        if constexpr (is_reinitable_net<mbr_net_type>)    // 有状态的网络层需要初始化权重
        {
            if (I + 1 >= container.size())
            {
                throw std::runtime_error("container size does not match net size");
            }
            std::get<N>(m_nets).reinit({container[I], container[I + 1]});   // 初始化权重
            if constexpr (N + 1 < sizeof...(net_types))
            {
                reinit<container_type, N + 1, I + 1>(container);
            }
        }
        else                                            // 无状态的网络层不需要初始化权重
        {
            if constexpr (N + 1 < sizeof...(net_types))
            {
                reinit<container_type, N + 1, I>(container);
            }
        }
    }

    template <size_t N, typename...upr_arg_types>
    void set_updator__(upr_arg_types&&... args)
    {
        using mbr_net_type = std::tuple_element_t<N, std::tuple<net_types...>>;
        if constexpr (is_updatable_net<mbr_net_type>)
        {
            std::get<N>(m_nets).set_updator(std::forward<upr_arg_types>(args)...);
        }
        if constexpr (N + 1 < sizeof...(net_types))
        {
            set_updator__<N + 1, upr_arg_types...>(std::forward<upr_arg_types>(args)...);
        }
    }

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        set_updator__<0, upr_arg_types...>(std::forward<upr_arg_types>(args)...);
    }

    template <size_t N>
    void set_lr__(val_type lr)
    {
        using mbr_net_type = std::tuple_element_t<N, std::tuple<net_types...>>;
        if constexpr (requires(mbr_net_type& net, val_type v) { net.set_lr(v); })
        {
            std::get<N>(m_nets).set_lr(lr);
        }
        if constexpr (N + 1 < sizeof...(net_types))
        {
            set_lr__<N + 1>(lr);
        }
    }

    void set_lr(val_type lr)
    {
        set_lr__<0>(lr);
    }

    template <typename init_type>
    void init_weight()
    {
        std::apply([](auto&&... nets) {((nets.template init_weight<init_type>()),...); }, m_nets);
    }

    void step()
    {
        std::apply([](auto&&... nets) {((nets.step()),...); }, m_nets);
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "complex_net_t = [";
        std::apply([&ss, indent](auto&&... nets) {((ss << std::endl << nets.net_type(indent + 2)),...); }, m_nets);
        ss << std::endl
        << print_indent(indent) << "]";
        return ss.str();
    }

    auto back()
    {
        return std::get<sizeof...(net_types) - 1>(m_nets);
    }

    template<size_t N, size_t...nums>
    decltype(auto) get()
    {
        if constexpr (sizeof...(nums) == 0)
        {
            return std::get<N>(m_nets);
        }
        else
        {
            return std::get<N>(m_nets).template get<nums...>();
        }
    }

    template<size_t N, size_t...nums>
    decltype(auto) get() const
    {
        if constexpr (sizeof...(nums) == 0)
        {
            return std::get<N>(m_nets);
        }
        else
        {
            return std::get<N>(m_nets).template get<nums...>();
        }
    }

private:
    template <size_t I, typename Input>
    auto infer_chain(Input&& input)
    {
        if constexpr (I >= sizeof...(net_types))
            return std::forward<Input>(input);
        else
        {
            using net_type_at_i = std::tuple_element_t<I, std::tuple<net_types...>>;
            if constexpr (is_infer_skipped_net<net_type_at_i>::value)
                return infer_chain<I + 1>(std::forward<Input>(input));
            else
            {
                auto& net = std::get<I>(m_nets);
                return infer_chain<I + 1>(net.forward_one(std::forward<Input>(input)));
            }
        }
    }

    template <size_t I>
    void infer_reset_chain()
    {
        if constexpr (I >= sizeof...(net_types))
            return;
        else
        {
            auto& net = std::get<I>(m_nets);
            if constexpr (requires { net.infer_reset(); })
                net.infer_reset();
            else if constexpr (requires { net.clear_kv_cache(); })
                net.clear_kv_cache();
            infer_reset_chain<I + 1>();
        }
    }

    template <size_t I>
    void infer_prepare_chain(int max_kv_seq)
    {
        if constexpr (I >= sizeof...(net_types))
            return;
        else
        {
            auto& net = std::get<I>(m_nets);
            if constexpr (requires { net.infer_prepare(max_kv_seq); })
                net.infer_prepare(max_kv_seq);
            else if constexpr (requires { net.clear_kv_cache(); })
                net.clear_kv_cache();
            if constexpr (requires { net.reserve_kv_cache(max_kv_seq); })
            {
                if (max_kv_seq > 0)
                    net.reserve_kv_cache(max_kv_seq);
            }
            infer_prepare_chain<I + 1>(max_kv_seq);
        }
    }
};

/*
 * 复杂网络构造器存在意义说明：如果直接构建复杂网络需要一次性输入各层的网络实例，不够灵活，且不够清晰。复杂网络构造器则提供了一套接口，可以逐步构建复杂网络的结构，并且在构建过程中可以清晰地看到每一步的网络结构变化，同时也可以在构建过程中设置每一层的参数，最后一步才生成复杂网络实例。
 * 并且可以不用为每层网络设置val_type参数，复杂网络构造器会自动推断出每层网络的val_type参数，避免了重复输入参数的麻烦。
 */
template <typename val_type, typename...net_types>
struct complex_net_builder_t
{
    template<template<typename, template<typename> class> class cur_net_tpl, template<typename> class updator_tpl>
    using push_back_updatable = complex_net_builder_t<val_type, net_types..., cur_net_tpl<mat_t<val_type>, updator_tpl>>;

    template<template<typename> class cur_net_tpl>
    using push_back_staticnet = complex_net_builder_t<val_type, net_types..., cur_net_tpl<mat_t<val_type>>>;

    template<typename new_net_type>
    using push_back_impl = complex_net_builder_t<val_type, net_types..., new_net_type>;

    using type = complex_net_t<net_types...>;

};



} // namespace jasmine
#endif
