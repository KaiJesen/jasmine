#ifndef __MAT_NET_T_HPP__
#define __MAT_NET_T_HPP__

#include <string>
#include <sstream>

#include "mat_t.hpp"
#include "mat_express_t.hpp"

#include "mat_updator_t.hpp"

template <typename input_type, template<typename> class updator_type>
class weight_net_t
{
public:
    using val_type = typename input_type::return_type;
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

    // 返回类型是一个表达式模板类的对象，与相关参数进行了引用绑定，可以惰性地获得指定位置的值（调用时进行计算）
    mat_t<val_type> forward(const input_type& input) 
    {
        m_input = input.clone();
        auto ret = (m_weight.dot(m_input) + m_bias).clone();  // 这里用m_input避免重复计算，同时必须在这里全量计算，否则被引用的临时变量会失效
        //std::cout << "Weight forward: input \n" << input << " \noutput \n" << ret << std::endl;
        return ret;
        /*
        auto a = m_weight.dot(m_input);     // 输出结果a引用了m_input和m_weight
        auto b = a + m_bias;            // 输出结果b引用了a和m_bias
        return b.clone();             // 返回b，引用链条为b->a->m_input/m_weight/m_bias，返回后a失效，因此b失效，因此必须在这个时候计算所有的值，否则失效
        */
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

    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        auto delta_weight = delta.dot(m_input.t());
        auto delta_bias = hsum(delta);
        mat_t<val_type> ret = m_weight.t().dot(delta);
        // 更新权重和偏置
        m_weight_updator.update(delta_weight, m_weight);
        m_bias_updator.update(delta_bias, m_bias);
        return ret;
    }

    std::string net_type() const
    {
        std::stringstream ss;
        ss << "weight_net_t:(in:" << m_weight.col_num() << ", out:" << m_weight.row_num() << ")";
        return ss.str();
    }
};

template <typename input_type>
class sigmoid_net_t
{
public:
    using val_type = typename input_type::return_type;
private:
    mat_t<val_type> m_output;                // 前一次的输入，反向传播时用于快捷计算
public:
    sigmoid_net_t() = default;

    mat_t<val_type> forward(const input_type& input)
    {
        m_output = sigmoid(input).clone();
        //std::cout << "Sigmoid forward: input \n" << input << " \noutput \n" << ret << std::endl;
        return m_output;
    }

    auto backward(const mat_t<val_type>& delta)
    {
        if (delta.row_num() != m_output.row_num() || delta.col_num() != m_output.col_num())
        {
            throw std::runtime_error("delta size does not match input size");
        }
        auto delta_sigmoid = delta * (1 - m_output) * m_output;
        return delta_sigmoid.clone();
    }

    std::string net_type() const
    {
        std::stringstream ss;
        ss << "sigmoid_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 什么也不做
    }
};

template <typename input_type>
class relu_net_t
{
private:
    using val_type = typename input_type::return_type;
    mat_t<val_type> m_input;                // 前一次的输入，反向传播时用于快捷计算
public:
    relu_net_t() = default;

    mat_t<val_type> forward(const input_type& input)
    {
        m_input = input.clone();
        //return relu(m_input).clone();
        auto ret = (m_input > 0) * m_input;     // 直接用表达式模板计算，避免中间变量
        //std::cout << "ReLu forward: input \n" << input << " \noutput \n" << ret << std::endl;
        return ret;
    }

    template <typename other_type>
    auto backward(const other_type& delta)
    {
        if (delta.row_num() != m_input.row_num() || delta.col_num() != m_input.col_num())
        {
            throw std::runtime_error("delta size does not match input size");
        }
        auto delta_relu = delta.clone() * (m_input > 0);
        return delta_relu.clone();
    }
    std::string net_type() const
    {
        std::stringstream ss;
        ss << "relu_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 什么也不做
    }
};

// 纵向的标准化层，即对每一列进行标准化
template <typename input_type, template<typename> class updator_type>
class layer_norm_net_t
{
private:
    using val_type = typename input_type::return_type;
    mat_t<val_type> m_hx;
    mat_t<val_type> m_mean;
    mat_t<val_type> m_std;
    mat_t<val_type> m_gama;     // 缩放参数
    updator_type<val_type> m_gama_updator;
    mat_t<val_type> m_beta;     // 平移参数
    updator_type<val_type> m_beta_updator;
public:
    layer_norm_net_t() = default;

    template <typename...upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_gama_updator.set(std::forward<upr_arg_types>(args)...);
        m_beta_updator.set(std::forward<upr_arg_types>(args)...);
    }

    mat_t<val_type> forward(const input_type& input)
    {
        m_mean = vmean(input);
        auto delta = input - m_mean;
        m_std = sqrt(vmean(pow(delta.clone(), 2.0)));
        m_hx = delta / m_std;
        if (m_gama.valid() == false)
        {
            m_gama = mat_t<val_type>(input.row_num(), 1);
            m_beta = mat_t<val_type>(input.row_num(), 1);
            init_matrix<he_gaussian_t>(m_gama);
            init_matrix<he_gaussian_t>(m_beta);
        }
        return (m_gama * m_hx + m_beta).clone();
    }

    template <typename other_type>
    auto backward(const other_type& delta)
    { 
        auto L_gama = hsum(delta * m_hx);
        auto L_beta = hsum(delta);
        double m = static_cast<double>(delta.row_num());
        auto L_input = (delta - (m_hx * vsum(m_hx * delta) + vsum(delta))/m) * m_gama / m_std;
        m_gama_updator.update(L_gama, m_gama);
        m_beta_updator.update(L_beta, m_beta);
        return L_input.clone();
    }

    std::string net_type() const
    {
        std::stringstream ss;
        ss << "layer_norm_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        init_matrix<init_type>(m_gama);
        init_matrix<init_type>(m_beta);
    }
};

template <typename input_type>
class hsoftmax_net_t
{
public:
    using val_type = typename input_type::return_type;
    mat_t<val_type> m_output;

    mat_t<val_type> forward(const input_type& input)
    {
        //m_input = input.clone();
        //return hsoftmax(m_input);
        m_output = hsoftmax(input).clone();
        return m_output;
    }

    mat_t<val_type> backward(const mat_t<val_type>& delta)
    {
        auto grad = (m_output * (delta - hsum(m_output * delta))).clone();
        return grad;
    }

    std::string net_type() const
    {
        std::stringstream ss;
        ss << "hsoftmax_net_t";
        return ss.str();
    }

    template<typename init_type>
    void init_weight()
    {
        // 不需要初始化权重
    }
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

    mat_t<val_type> forward(const mat_t<val_type>& input)
    {
        return m_net.forward(input) + input;
    }
    template <typename other_type>
    auto backward(const other_type& delta)
    {
        return m_net.backward(delta) + delta;
    }
    std::string net_type() const
    {
        std::stringstream ss;
        ss << "residual_net_t:" << m_net.net_type();
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

};

template <typename... net_types>
class complex_net_t
{
private:
    std::tuple<net_types...> m_nets;
public:

    using val_type = typename std::tuple_element_t<0, std::tuple<net_types...>>::val_type;

    template <typename input_type>
    auto forward(const input_type& input)
    {
        return std::apply([&input](auto&&... nets) {return net_forward(input, nets...); }, m_nets);
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
        if constexpr (is_unstable_net<mbr_net_type>)    // 有状态的网络层需要初始化权重
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

    template <typename init_type>
    void init_weight()
    {
        std::apply([](auto&&... nets) {((nets.template init_weight<init_type>()),...); }, m_nets);
    }

    std::string net_type() const
    {
        std::stringstream ss;
        ss << "complex_net_t = [";
        std::apply([&ss](auto&&... nets) {((ss << nets.net_type() << " >> "),...); }, m_nets);
        ss << "$output]";
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

};

template <typename val_type, typename...net_types>
struct complex_net_builder_t
{
    template<template<typename, template<typename> class> class cur_net_tpl, template<typename> class updator_tpl>
    using push_back_updatable = complex_net_builder_t<val_type, net_types..., cur_net_tpl<mat_t<val_type>, updator_tpl>>;

    template<template<typename> class cur_net_tpl>
    using push_back_stablenet = complex_net_builder_t<val_type, net_types..., cur_net_tpl<mat_t<val_type>>>;

    template<typename new_net_type>
    using push_back_impl = complex_net_builder_t<val_type, net_types..., new_net_type>;

    using type = complex_net_t<net_types...>;

};


template<size_t N>
struct test_net_t
{
    using val_type = int;
    int forward(int x)
    {
        std::cout << "testnet " << N << " forward" << std::endl;
        return x + 1;
    }

    int backward(int delta)
    {
        std::cout << "testnet " << N << " backward" << std::endl;
        return delta - 1;
    }
};

#include "mat_loss_t.hpp"

void test_mat_net_t()
{ 
    std::cout << "mat_net_t test" << std::endl;
    auto cnet2 = complex_net_t<test_net_t<1>, test_net_t<2>, test_net_t<3>>();
    auto out2 = cnet2.forward(0);
    std::cout << "final out2: " << out2 << std::endl;
    auto out3 = cnet2.backward(3);
    std::cout << "final out3: " << out3 << std::endl;
    using base_net_type = complex_net_builder_t<double> // 残差网络的基础网络，包含一个线性层和一个激活层
        ::push_back_updatable<weight_net_t, nadam_t>
        ::push_back_stablenet<sigmoid_net_t>
        ::type;


    using res_net_type = residual_net_t<base_net_type>;

    using cpx_net_type = complex_net_builder_t<double>
        ::push_back_impl<res_net_type>           // 增加一个残差网络，基础网络是上面定义的base_net_type
        ::push_back_updatable<layer_norm_net_t, nadam_t>
        ::push_back_updatable<weight_net_t, nadam_t>
        ::push_back_stablenet<sigmoid_net_t>
        ::push_back_updatable<weight_net_t, nadam_t>
        ::push_back_stablenet<sigmoid_net_t>
        ::push_back_stablenet<mse_loss_t>
        ::type;
    
    cpx_net_type cnet;
    cnet.get<0>().base_net().reinit(std::vector<int>{3, 4});        // 初始化残差网络的基础网络的维度，用于布置好内部的矩阵
    cnet.get<0, 0>().set_updator(0.1);                  // 设置残差网络内部的线性层的权重更新器的参数
    cnet.get<2>().set_updator(0.1);
    //cnet.get<0>().set_updator(0.1);
    //cnet.get<3>().set_updator(0.1);
    cnet.reinit(std::vector<int>{4, 3, 1});
    cnet.init_weight<xavier_gaussian_t>();
    std::cout << cnet.net_type() << std::endl;
    mat_t<double> input(3, 1, {0.5, 0.5});
    std::cout << "first output: " << cnet.forward(input) << std::endl;
    while (true)
    {
        std::cout << "train times: ";
        int train_times;
        std::cin >> train_times;
        if (train_times <= 0)
        {
            break;
        }
        for (int i = 0; i < train_times; i++)
        {
            auto output = cnet.forward(input);
            //auto delta = output - 0.8;
            //std::cout << " delta: " << delta << std::endl;
            cnet.backward(0.8);
        }
        std::cout << "\tloss: " << cnet.back().loss(0.8) << std::endl;
    }
    std::cout << "final output: " << cnet.forward(input) << "\nloss:" << cnet.back().loss(0.8) << std::endl;
}

#endif
