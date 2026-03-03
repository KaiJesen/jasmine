#include "mat_t.hpp"
#include "mat_express_t.hpp"
#include "mat_view_t.hpp"
#include "mat_net_t.hpp"
#include "mat_init_t.hpp"
#include "mat_mha_t.hpp"
#include "mat_transformer_base.hpp"

template<typename T>
void foo(T x)
{
    std::cout << "foo: " << typeid(x).name() << std::endl;
}


template <typename... Args>
void variadic_func(Args... args)
{
    (foo(args), ...);
}


template<typename...T>
int sum_from_zero(T...args)
{
    return (0 + ... + args);
}

template<typename T>
class node
{
public:
    T value;
    node(T val) : value(val) {}

    template<typename other_t>
    T forward(other_t const& input) const
    {
        return value + input;
    }
};

template<typename t1, typename t2>
auto operator<<(node<t1> const& n1, t2 const& n2)
{
    return n1.forward(n2);
}

template<typename...arg_ts>
auto foo(arg_ts...args)
{
    return (args << ... << 0);
}


template<typename T>
struct father
{
    //using type = typename T::type;
    void hit_son()
    {
        std::cout << "father hit son" << std::endl;
        static_cast<T*>(this)->cry();
    }
};

struct son: father<son>
{
    using type = int;
    void cry()
    {
        std::cout << "son cry" << std::endl;
    }
};

#if 0

template<size_t...nums>
struct index_sequence
{
};

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

template<size_t N, typename seq>
constexpr size_t get_seq;

template<size_t N, size_t first, size_t...nums>
constexpr size_t get_seq<N, index_sequence<first, nums...>> = get_seq<N - 1, index_sequence<nums...>>;

template<size_t first, size_t...nums>
constexpr size_t get_seq<0, index_sequence<first, nums...>> = first;

void print_012(size_t i1, size_t i2, size_t i3)
{
    std::cout << i1 << ", " << i2 << ", " << i3 << std::endl;
}

template<typename seq, size_t...nums>
void print_seq(seq, index_sequence<nums...>)
{
    print_012(get_seq<nums, seq>...);
}

template<typename _Fn, typename _Tuple, size_t..._Indices>
decltype(auto) __rapply_impl(_Fn&& __f, _Tuple&& __t, index_sequence<_Indices...>)
{
    return __f(std::get<_Indices>(std::forward<_Tuple>(__t))...);
}

template<typename _Fn, typename _Tuple>
decltype(auto) rapply(_Fn&& __f, _Tuple&& __t)
{
    using _Indices = make_reverse_index_sequence<std::tuple_size_v<std::remove_reference_t<_Tuple>>>;
    return __rapply_impl(std::forward<_Fn>(__f),
                         std::forward<_Tuple>(__t),
                         _Indices{});
}

template<typename... Args>
void print_all(Args&&... args)
{
    (std::cout << ... << args) << std::endl;
}
#endif

template<size_t ... ns>
void print()
{
    ((std::cout << " " << ns),...);
}

template<int...ns>
int minus()
{
    return (ns - ...);
}

#include <vector>
int main()
{
    //test_mat_t();
    //test_mat_express_t();
    //test_mat_view_t();
    //test_mat_net_t();
    //test_mat_init_t();
    //test_mat_head_gen_t();
    //test_mha_tools();
    //test_multi_head_attention();
    //test_encoder();
    //test_decoder();
    test_tf_kernel();
    return 0;
}