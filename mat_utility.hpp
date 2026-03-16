#ifndef __MAT_UTILITY_HPP__
#define __MAT_UTILITY_HPP__
#include <math.h>

#include "mat_concepts.hpp"

template <typename val_type>
requires is_serializable<val_type>
std::ostream& operator<<(std::ostream& os, const val_type& m)
{
    os << m.to_string();
    return os;
}

template<typename net_type, typename input_type>
requires is_forwardable<net_type, input_type>
auto operator>>(input_type const& input, net_type& net)
{
    return net.forward(input);
}

template<typename input_type, typename... net_types>
auto net_forward(input_type const& input, net_types&&... nets)
{
    return (input >> ... >> nets);
}

template<typename net_type, typename input_type>
requires is_backwardable<net_type, input_type>
auto operator<<(net_type& net, input_type const& input)
{
    return net.backward(input);
}

template<typename... net_type, typename input_type>
auto net_backward(input_type const& input, net_type&&... nets)
{
    return (nets << ... << input);
}

template<size_t...nums>
struct index_sequence{};

template<size_t N, typename cur_seq>
struct make_index_sequence_impl;

template<size_t N, size_t...nums, template<size_t...> class seq>
struct make_index_sequence_impl<N, seq<nums...>>
{
    using type = make_index_sequence_impl<N - 1, seq<N - 1, nums...>>::type;
};

template<size_t...nums, template<size_t...> class seq>
struct make_index_sequence_impl<0, seq<nums...>>
{
    using type = seq<nums...>;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_impl<N, index_sequence<>>::type;

template<size_t N, typename cur_seq>
struct make_reverse_index_sequence_impl;

template<size_t N, size_t...nums, template<size_t...> class seq>
struct make_reverse_index_sequence_impl<N, seq<nums...>>
{
    using type = make_reverse_index_sequence_impl<N - 1, seq<nums..., N - 1>>::type;
};
template<size_t...nums, template<size_t...> class seq>
struct make_reverse_index_sequence_impl<0, seq<nums...>>
{
    using type = seq<nums...>;
};

template<size_t N>
using make_reverse_index_sequence = typename make_reverse_index_sequence_impl<N, index_sequence<>>::type;

template<typename _Fn, typename _Tuple, size_t..._Indices>
decltype(auto) __rapply_impl(_Fn&& __f, _Tuple&& __t, index_sequence<_Indices...>)
{
    return std::forward<_Fn>(__f)(std::get<_Indices>(std::forward<_Tuple>(__t))...);
}

template<typename _Fn, typename _Tuple>
decltype(auto) rapply(_Fn&& __f, _Tuple&& __t)
{
    using _Indices = make_reverse_index_sequence<std::tuple_size_v<std::remove_reference_t<_Tuple>>>;
    return __rapply_impl(std::forward<_Fn>(__f),
                         std::forward<_Tuple>(__t),
                         _Indices{});
}

std::string print_indent(int const indent)
{
    std::string ret;
    for (int i = 0; i < indent; ++i)
        ret += "  ";
    return ret;
}

// 余弦退火学习率衰减【包含热重启】
struct cosine_annealing_decay
{
    double max_lr; // 最大学习率
    double min_lr; // 最小学习率
    int init_decay_epoch; // 初始衰减epoch，后期根据warm restart规则会每次epoch翻倍
    int decay_total_epoches; // 当前衰减的总步数
    int decay_curr_epoch; // 当前步数

    double warmup_rate; // 预热率，默认设置20%的预热
    double warmup_steps; // 预热步数

    // warm restart相关参数
    double T_multiplier; // T_multiplier用于热重启的倍增，默认为2.0
    double lr_decay_rate; // 学习率衰减率，默认为0.8
    int total_epochs; // 总的epoch数
    int current_epoch; // 当前执行的所有epoch数

    int total_cycle; // 总周期数
    int cur_cycle; // 当前周期数


    cosine_annealing_decay(
        int epoch_max = 1000
        , int init_decay_steps = 100
        , double max_lr = 1e-4
        , double min_lr = 1e-6
        , double warmup_rate = 0.2
        , double T_multiplier = 2.0
        , double lr_decay_rate = 0.8
    )
        : max_lr(max_lr)
        , min_lr(min_lr)
        , decay_total_epoches(init_decay_steps)
        , decay_curr_epoch(0)
        , warmup_rate(warmup_rate)
        , warmup_steps(static_cast<int>(decay_total_epoches * warmup_rate))
        , T_multiplier(T_multiplier)
        , lr_decay_rate(lr_decay_rate)
        , total_epochs(epoch_max)
        , current_epoch(0)
        , total_cycle(0)
        , cur_cycle(0)
        {
            init_decay_epoch = init_decay_steps; // 初始化衰减epoch
            total_cycle = static_cast<int>(log2(static_cast<double>(total_epochs) / init_decay_epoch) / log2(T_multiplier)) + 1;
        }

    void reset_config(int epoch_max
        , int init_decay_steps = 100
        , double max_lr = 1e-4
        , double min_lr = 1e-6
        , double warmup_rate = 0.2
        , double T_multiplier = 2.0
        , double lr_decay_rate = 0.8)
    {
        this->max_lr = max_lr; // 设置最大学习率
        this->min_lr = min_lr; // 设置最小学习率
        this->init_decay_epoch = init_decay_steps; // 设置初始衰减epoch
        this->decay_total_epoches = init_decay_steps; // 重置衰减总步数
        this->warmup_rate = warmup_rate; // 设置预热率
        this->warmup_steps = static_cast<int>(decay_total_epoches * warmup_rate); // 更新预热步数
        this->T_multiplier = T_multiplier; // 设置T_multiplier
        this->lr_decay_rate = lr_decay_rate; // 设置学习率衰减率
        this->total_epochs = epoch_max; // 设置总的epoch数
        this->current_epoch = 0; // 重置当前epoch数
        this->decay_curr_epoch = 0; // 重置当前步数
        this->total_cycle = static_cast<int>(log2(static_cast<double>(total_epochs) / init_decay_epoch) / log2(T_multiplier)) + 1; // 重置总周期数
        this->cur_cycle = 0; // 重置当前周期数
    }
    
    int get_total_cycle() const
    {
        // 根据开始周期和T_multiplier计算总周期数
        return total_cycle; // 返回总周期数
    }

    int get_current_cycle() const
    {
        // 返回当前周期数
        return cur_cycle;
    }

    void set_init_decay_epoch(int init_epoch)
    {
        init_decay_epoch = init_epoch; // 设置初始衰减epoch
        decay_total_epoches = init_decay_epoch; // 重置衰减总步数
        warmup_steps = static_cast<int>(decay_total_epoches * warmup_rate); // 更新预热步数
        total_cycle = static_cast<int>(log2(static_cast<double>(decay_total_epoches) / init_decay_epoch) / log2(T_multiplier)) + 1;
    }

    double get_lr()
    {
        if (decay_curr_epoch < warmup_steps)
        {
            return max_lr * (decay_curr_epoch / static_cast<double>(warmup_steps)); // 预热阶段线性增加学习率
        }
        double progress = static_cast<double>(decay_curr_epoch - warmup_steps) / (decay_total_epoches - warmup_steps);
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(M_PI * progress)); // 余弦退火公式
    }

    bool step()
    {
        decay_curr_epoch++; // 更新当前步数
        current_epoch++;
        if (current_epoch >= total_epochs) // 如果当前epoch数超过总的epoch数
        {
            return false;                   // 停止训练
        }
        if (decay_curr_epoch > decay_total_epoches)     // 本次衰减结束，进行warm restart
        {
            warm_restart(); // 进行warm restart
            return true;    // 继续执行
        }
        return true; // 继续执行
    }

    void warm_restart()
    {
        decay_curr_epoch = 0; // 重置当前步数
        decay_total_epoches = static_cast<int>(decay_total_epoches * T_multiplier); // 更新衰减步数
        max_lr *= lr_decay_rate; // 更新最大学习率
        min_lr *= lr_decay_rate; // 更新最小学习率
        cur_cycle++; // 更新当前周期数
    }

    void reset()
    {
        decay_curr_epoch = 0; // 重置当前步数
        current_epoch = 0; // 重置当前epoch数
        decay_total_epoches = init_decay_epoch; // 重置衰减步数
        max_lr = 1e-4; // 重置最大学习率
        min_lr = 1e-6; // 重置最小学习率
        warmup_steps = static_cast<int>(decay_total_epoches * warmup_rate); // 重置预热步数
        total_cycle = static_cast<int>(log2(static_cast<double>(decay_total_epoches) / init_decay_epoch) / log2(T_multiplier)) + 1; // 重置总周期数
        cur_cycle = 0; // 重置当前周期数
    }
};

#endif