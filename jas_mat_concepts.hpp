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
 * 可参与矩阵表达式运算：标量或矩阵。
 *
 * 判标量前必须先剥掉引用 / const —— 二元运算符如今用转发引用接收操作数，
 * 左值标量（例如 `double s; m / s;`）推导出的类型是 `double&`，
 * 直接 `is_arithmetic_v<double&>` 是 false，会让整个重载悄悄消失。
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

/** 推理时跳过该层（通常为 loss）：不调用本层，输入原样传给后续层 */
template<typename T, typename = void>
struct is_infer_skipped_net : std::false_type {};

template<typename T>
struct is_infer_skipped_net<T, std::void_t<decltype(T::skip_on_infer)>>
    : std::bool_constant<(T::skip_on_infer)> {};


} // namespace jasmine
#endif
