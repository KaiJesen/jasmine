#ifndef __MAT_LOSS_T_HPP__
#define __MAT_LOSS_T_HPP__

#include <sstream>
#include <string>

#include "mat_concepts.hpp"
#include "mat_express_t.hpp"

// 损失函数做成和网络一样的形式，但是只有反向传播有意义，正向传播直接透传
template <typename input_type>
class mat_loss_t
{ 
public:
    using val_type = typename input_type::ele_type;
    mat_loss_t() = default;

    mat_t<val_type> m_input;

    mat_t<val_type> forward(const input_type& input)
    {
        m_input = input.clone();
        return input;
    }
};

template <typename input_type>
class mse_loss_t:public mat_loss_t<input_type>
{
public:
    using base_type = mat_loss_t<input_type>;
    using val_type = typename base_type::val_type;

    mse_loss_t() = default;

    mat_t<val_type> backward(const mat_t<val_type>& target) const
    {
        return base_type::m_input - target;
    }

    val_type loss(const mat_t<val_type>& target) const
    {
        return mean(pow(base_type::m_input - target));
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "mse_loss_t";
        return ss.str();
    }

    template <typename init_type>
    void init_weight(){}

    void step(){}
};


#endif
