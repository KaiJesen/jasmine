#ifndef __MAT_UTILITY_HPP__
#define __MAT_UTILITY_HPP__
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

#endif