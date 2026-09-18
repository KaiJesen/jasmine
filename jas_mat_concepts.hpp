#ifndef __JAS_MAT_CONCEPTS_HPP__
#define __JAS_MAT_CONCEPTS_HPP__
#include <vector>
#include <type_traits>

namespace jasmine {

template<typename val_type>
concept is_matrix = 
    requires(val_type m, int i, int j) 
    {
        { m.row_num() } -> std::convertible_to<int>;
        { m.col_num() } -> std::convertible_to<int>;
        { m(i, j) } ;
    };

/**
 * Can take part in a matrix expression: a scalar or a matrix.
 *
 * Reference and const qualifiers have to be stripped before testing for a scalar: the binary
 * operators now take their operands by forwarding reference, so an lvalue scalar (e.g.
 * `double s; m / s;`) deduces to `double&`, and `is_arithmetic_v<double&>` is false -- which would
 * silently remove the whole overload.
 */
template<typename...val_types>
concept is_caculable =
    ((std::is_arithmetic_v<std::remove_cvref_t<val_types>> || is_matrix<val_types>) && ...);

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

/** Skipped during inference (usually a loss layer): it is not called, its input is passed on unchanged */
template<typename T, typename = void>
struct is_infer_skipped_net : std::false_type {};

template<typename T>
struct is_infer_skipped_net<T, std::void_t<decltype(T::skip_on_infer)>>
    : std::bool_constant<(T::skip_on_infer)> {};


} // namespace jasmine
#endif
