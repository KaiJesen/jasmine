#ifndef __MAT_CONCEPTS_HPP__
#define __MAT_CONCEPTS_HPP__
#include <vector>
#include <type_traits>

template<typename val_type>
concept is_matrix = 
    requires(val_type m, int i, int j) 
    {
        { m.row_num() } -> std::convertible_to<int>;
        { m.col_num() } -> std::convertible_to<int>;
        { m(i, j) } ;
    };

template<typename...val_types>
concept is_caculable = ((std::is_arithmetic_v<val_types> || is_matrix<val_types>) && ...);

template<typename val_type>
concept is_serializable = 
    requires (val_type m)
    {
        { m.to_string() } -> std::convertible_to<std::string>;
    };

template<typename net_type, typename val_type>
concept is_forwardable = 
    requires (net_type net, val_type m)
    {
        { net.forward(m) } ;
    };

template<typename net_type, typename val_type>
concept is_backwardable = 
    requires (net_type net, val_type m)
    {
        { net.backward(m) } ;
    };

template<typename net_type>
concept is_reinitable_net =
    requires (net_type net)
    {
        net.reinit(std::vector<int>());
    };

template<typename net_type>
concept is_unreinitable_net = !is_reinitable_net<net_type>;

template<typename net_type>
concept is_updatable_net = 
    requires (net_type net)
    {
        net.set_updator(0.01);
    };

template<typename net_type>
concept is_unupdatable_net = !is_updatable_net<net_type>;

#endif
