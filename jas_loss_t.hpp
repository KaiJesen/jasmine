#ifndef __JAS_LOSS_T_HPP__
#define __JAS_LOSS_T_HPP__

#include <sstream>
#include <string>

#include "jas_mat_concepts.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_storage.hpp"

namespace jasmine {

// 损失函数做成和网络一样的形式，但是只有反向传播有意义，正向传播直接透传
template <typename input_type>
class mat_loss_t
{ 
public:
    using val_type = typename input_type::ele_type;
    /** complex_net::infer 跳过本层，输入原样传给后续层（训练仍走 forward 透传） */
    static constexpr bool skip_on_infer = true;
    mat_loss_t() = default;

    mat_t<val_type> m_input;

    template<typename Src>
    mat_t<val_type> forward(Src&& input)
    {
        detail::store_for_backward(m_input, std::forward<Src>(input));
        return m_input;
    }

    template<typename Src>
    mat_t<val_type> forward_one(Src&& input)
    {
        return forward(std::forward<Src>(input));
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

    // 根据目标计算误差，输入是目标值，输出是误差值
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



} // namespace jasmine
#endif
