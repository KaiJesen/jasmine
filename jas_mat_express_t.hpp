#ifndef __JAS_MAT_EXPRESS_T_HPP__
#define __JAS_MAT_EXPRESS_T_HPP__
#include <cassert>
#include <cmath>
#include <sstream>
#include <tuple>
#include <string>
#include <utility>
#include <vector>
#include <numeric>

#include "jas_mat_concepts.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_mat_gemm.hpp"

namespace jasmine {

template<typename lval_type, typename rval_type>
requires is_matrix<lval_type> && is_matrix<rval_type>
auto dot(lval_type&& lval, rval_type&& rval);

/**
 * 表达式操作数的存储方式，按【值类别】决定：
 *
 *  - 标量        → 存 `mat_t<T>`：包成 1×1 矩阵，才能和矩阵共用 row_num/col_num/operator() 接口
 *  - 左值（具名变量）→ 存 `T const&`：借引用，零拷贝。这是表达式模板能避免临时量的前提，
 *                      代价是调用方必须保证它比表达式活得久（与 Eigen/Blaze 同一套约定）
 *  - 右值（临时量）  → 存 `T`：按值拥有。右值往往就是临时的子表达式或临时矩阵
 *                      （`(a+b)*c` 里的 `a+b`、`f() + x` 里的 `f()`），借引用必然悬垂
 *
 * 「右值按值拥有」这一条是表达式树能成为【可安全拷贝/传递的值】的关键。
 * 之前一律按引用存，于是 `auto tree = (a+b)*c;` 会在语句结束时丢掉 `a+b` 这个临时量，
 * tree 里的引用随即悬垂（ASan 报 stack-use-after-scope）。按值类别分派后，
 * 这类树全程自持结构，拷贝/传参都安全。
 *
 * 仍然存在的边界：左值操作数、以及"视图所引用的矩阵"，其生命周期依旧由调用方负责。
 * 前者是有意为之（否则每次构造表达式都要深拷贝整块矩阵），后者见下方 mat_view_t 的说明。
 */
template <typename T, bool is_scalar>
struct storage_of;
template <typename T>
struct storage_of<T, true>
{
    using type = mat_t<std::remove_cvref_t<T>>;
};
template <typename T>
struct storage_of<T, false>
{
    using type = std::conditional_t<std::is_lvalue_reference_v<T>,
                                    std::remove_cvref_t<T> const&,   // 左值：借引用
                                    std::remove_cvref_t<T>>;         // 右值：按值拥有
};

template <typename T>
using storage_type = typename storage_of<
    T, std::is_arithmetic_v<std::remove_cvref_t<T>>>::type;

template <typename lval_type, typename rval_type, template<typename,typename> class tpl>
class mat_express_2_param_stable_t
{
public:
    using lval_storage_type = storage_type<lval_type>;
    using rval_storage_type = storage_type<rval_type>;
    using lval_base_type = typename std::decay_t<lval_storage_type>::ele_type;
    using rval_base_type = typename std::decay_t<rval_storage_type>::ele_type;
    using derived_type = tpl<lval_type, rval_type>;
    using ele_type = std::common_type_t<lval_base_type, rval_base_type>;

    // 形参直接取「存储类型」而不是 lval_type/rval_type：
    //  - 左值操作数 → 形参是 `T const&`，只是再绑一次引用，不拷贝
    //  - 右值操作数 → 形参是 `T`（值），配合 std::forward 就是移动构造，不深拷贝临时量
    // 另外，这样写出来的构造函数首参类型不可能是本节点自身，因此不会顶掉隐式拷贝/移动构造，
    // 表达式树本身仍可正常拷贝（这是它能被安全传递的前提）。
    mat_express_2_param_stable_t(lval_storage_type left, rval_storage_type right)
        : m_left(std::move(left)), m_right(std::move(right))
    {
    }

    int row_num() const
    {
        return std::max(m_left.row_num(), m_right.row_num());
    }

    int col_num() const
    {
        return  std::max(m_left.col_num(), m_right.col_num());
    }

    std::tuple<int, int> shape() const
    {
        return std::make_tuple(row_num(), col_num());
    }

    auto operator()(int i, int j) const
    {
        return static_cast<
                tpl<lval_type, rval_type> const*
            >(this)->work(m_left(i, j), m_right(i, j));
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << tpl<lval_type, rval_type>::type_name() << "(" << row_num() << ", " << col_num() << ")";
        for (int i = 0; i < row_num(); ++i)
        {
            ss << "\n[ ";
            for (int j = 0; j < col_num(); ++j)
            {
                ss << (*this)(i, j) << " ";
            }
            ss << " ]";
        }
        return ss.str();
    }

    /**
     * `.dot()` 必须按【接收者】的值类别分派，原因很反直觉：
     * 成员函数里的 `*this` 永远是左值，哪怕对象本身是临时量。
     * 于是 `(a+b).dot(c)`、`a.t().dot(c)` 里的接收者会被当成左值借引用，
     * 而它其实是临时量 —— 表达式树一旦被存下来就是悬垂引用。
     * 这里用 ref-qualifier 区分：
     *   const&  → 左值接收者，借引用（零拷贝）
     *   const&& → 右值接收者，按值拥有（视图/节点都很小，拷贝极廉价；矩阵则必须拥有）
     */
    template<typename other_type>
    auto dot(other_type&& m) const &
    {
        return jasmine::dot(*reinterpret_cast<derived_type const*>(this),
                            std::forward<other_type>(m));
    }

    template<typename other_type>
    auto dot(other_type&& m) const &&
    {
        return jasmine::dot(std::move(*reinterpret_cast<derived_type const*>(this)),
                            std::forward<other_type>(m));
    }

    // 计算所有元素的值，并赋值给一个新的mat_t对象并返回
    mat_t<ele_type> clone() const
    {
        mat_t<ele_type> m(row_num(), col_num());
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(i, j) = (*this)(i, j);
            }
        }
        return m;
    }

    operator mat_t<ele_type>() const
    {
        return clone();
    }

protected:
    lval_storage_type m_left;
    rval_storage_type m_right;
};

template <typename lval_type, typename rval_type>
class mat_greater_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_greater_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_greater_t>;
    using lval_storage_type = typename base_type::lval_storage_type;
    using rval_storage_type = typename base_type::rval_storage_type;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using ele_type = std::common_type_t<lval_base_type, rval_base_type>;
    mat_greater_t(lval_storage_type left, rval_storage_type right)
        : base_type(std::move(left), std::move(right))
    {
    }

    auto work(lval_base_type i, rval_base_type j) const
    {
        return i > j;
    }

    static std::string type_name()
    {
        return "mat_greater_t";
    }
};

template <typename lval_type, typename rval_type>
requires is_caculable<lval_type, rval_type>
auto operator>(lval_type&& left, rval_type&& right)
{
    return mat_greater_t<lval_type&&, rval_type&&>(
        std::forward<lval_type>(left), std::forward<rval_type>(right));
}

template <typename lval_type, typename rval_type>
class mat_less_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_less_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_less_t>;
    using lval_storage_type = typename base_type::lval_storage_type;
    using rval_storage_type = typename base_type::rval_storage_type;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using ele_type = std::common_type_t<lval_base_type, rval_base_type>;

    mat_less_t(lval_storage_type left, rval_storage_type right)
        : base_type(std::move(left), std::move(right))
    {
    }

    auto work(lval_base_type i, rval_base_type j) const
    {
        return i < j;
    }

    static std::string type_name()
    {
        return "mat_less_t";
    }
};

template <typename lval_type, typename rval_type>
requires is_caculable<lval_type, rval_type>
auto operator<(lval_type&& left, rval_type&& right)
{
    return mat_less_t<lval_type&&, rval_type&&>(
        std::forward<lval_type>(left), std::forward<rval_type>(right));
}

template <typename lval_type, typename rval_type>
class mat_add_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_add_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_add_t>;
    using lval_storage_type = typename base_type::lval_storage_type;
    using rval_storage_type = typename base_type::rval_storage_type;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using ele_type = std::common_type_t<lval_base_type, rval_base_type>;


    mat_add_t(lval_storage_type left, rval_storage_type right)
        : base_type(std::move(left), std::move(right))
    {
    }

    auto work(lval_base_type i, rval_base_type j) const
    {
        return i + j;
    }

    static std::string type_name()
    {
        return "mat_add_t";
    }

};

template<typename lval_type, typename rval_type>
requires is_caculable<lval_type, rval_type>
auto operator+(lval_type&& left, rval_type&& right)
{
    return mat_add_t<lval_type&&, rval_type&&>(
        std::forward<lval_type>(left), std::forward<rval_type>(right));
}

template<typename lval_type, typename rval_type>
requires is_matrix<lval_type> && is_matrix<rval_type>
decltype(auto) operator+=(lval_type& left, rval_type const& right)
{
    for (int i = 0; i < left.row_num(); ++i)
    {
        for (int j = 0; j < left.col_num(); ++j)
        {
            left(i, j) += right(i, j);
        }
    }
    return left;
}

template<typename lval_type, typename rval_type>
requires is_matrix<lval_type> && std::is_arithmetic_v<rval_type>
decltype(auto) operator+=(lval_type& left, rval_type const& right)
{
    for (int i = 0; i < left.row_num(); ++i)
    {
        for (int j = 0; j < left.col_num(); ++j)
        {
            left(i, j) += right;
        }
    }
    return left;
}

template <typename lval_type, typename rval_type>
class mat_sub_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_sub_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_sub_t>;
    using lval_storage_type = typename base_type::lval_storage_type;
    using rval_storage_type = typename base_type::rval_storage_type;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using ele_type = std::common_type_t<lval_base_type, rval_base_type>;

    mat_sub_t(lval_storage_type left, rval_storage_type right)
        : base_type(std::move(left), std::move(right))
    {
    }

    auto work(lval_base_type i, rval_base_type j) const
    {
        return i - j;
    }

    static std::string type_name()
    {
        return "mat_sub_t";
    }
};

template<typename lval_type, typename rval_type>
requires is_caculable<lval_type, rval_type>
auto operator-(lval_type&& left, rval_type&& right)
{
    return mat_sub_t<lval_type&&, rval_type&&>(
        std::forward<lval_type>(left), std::forward<rval_type>(right));
}

template <typename lval_type, typename rval_type>
class mat_mul_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_mul_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_mul_t>;
    using lval_storage_type = typename base_type::lval_storage_type;
    using rval_storage_type = typename base_type::rval_storage_type;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using ele_type = std::common_type_t<lval_base_type, rval_base_type>;

    mat_mul_t(lval_storage_type left, rval_storage_type right)
        : base_type(std::move(left), std::move(right))
    {
    }

    auto work(lval_base_type i, rval_base_type j) const
    {
        return i * j;
    }

    static std::string type_name()
    {
        return "mat_mul_t";
    }
};

template<typename lval_type, typename rval_type>
requires is_caculable<lval_type, rval_type>
auto operator*(lval_type&& left, rval_type&& right)
{
    return mat_mul_t<lval_type&&, rval_type&&>(
        std::forward<lval_type>(left), std::forward<rval_type>(right));
}

template <typename lval_type, typename rval_type>
class mat_div_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_div_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_div_t>;
    using lval_storage_type = typename base_type::lval_storage_type;
    using rval_storage_type = typename base_type::rval_storage_type;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using ele_type = std::common_type_t<lval_base_type, rval_base_type>;


    mat_div_t(lval_storage_type left, rval_storage_type right)
        : base_type(std::move(left), std::move(right))
    {
    }

    auto work(lval_base_type i, rval_base_type j) const
    {
        return i / j;
    }

    static std::string type_name()
    {
        return "mat_div_t";
    }

};

template<typename lval_type, typename rval_type>
requires is_caculable<lval_type, rval_type>
auto operator/(lval_type&& left, rval_type&& right)
{
    return mat_div_t<lval_type&&, rval_type&&>(
        std::forward<lval_type>(left), std::forward<rval_type>(right));
}

template<typename val_type, template<typename> class tpl>
class mat_express_1_param_stable_t
{
public:
    using derived_type = tpl<val_type>;
    using val_storage_type = storage_type<val_type>;
    using val_base_type = typename std::decay_t<val_storage_type>::ele_type;
    using ele_type = val_base_type;

    // 与二元基类同理：形参取存储类型，左值零拷贝、右值移入（见 storage_type 的说明）
    mat_express_1_param_stable_t(val_storage_type val)
        : m_val(std::move(val))
    {
    }

    int row_num() const
    {
        return m_val.row_num();
    }

    int col_num() const
    {
        return m_val.col_num();
    }

    std::tuple<int, int> shape() const
    {
        return std::make_tuple(row_num(), col_num());
    }

    auto operator()(int i, int j) const
    {
        return static_cast<tpl<val_type> const*>(this)->work(m_val(i, j));
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << tpl<val_type>::type_name() << "(" << row_num() << ", " << col_num() << ")";
        for (int i = 0; i < row_num(); ++i)
        {
            ss << "\n[ ";
            for (int j = 0; j < col_num(); ++j)
            {
                ss << (*this)(i, j) << " ";
            }
            ss << " ]";
        }
        return ss.str();
    }

    // 同二元基类：`*this` 在成员函数里恒为左值，必须用 ref-qualifier 区分接收者是否为临时量
    template<typename other_type>
    auto dot(other_type&& m) const &
    {
        return jasmine::dot(*reinterpret_cast<derived_type const*>(this),
                            std::forward<other_type>(m));
    }

    template<typename other_type>
    auto dot(other_type&& m) const &&
    {
        return jasmine::dot(std::move(*reinterpret_cast<derived_type const*>(this)),
                            std::forward<other_type>(m));
    }

    mat_t<ele_type> clone() const
    {
        mat_t<ele_type> m(row_num(), col_num());
        // 不在 clone 里开 parallel：训练中会频繁物化表达式，并行区开销远大于收益
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(i, j) = (*this)(i, j);
            }
        }
        return m;
    }

    operator mat_t<ele_type>() const
    {
        return clone();
    }

protected:
    val_storage_type m_val;
};


template <typename val_type>
class mat_exp_t:public mat_express_1_param_stable_t<val_type, mat_exp_t>
{
public:
    using base_type = mat_express_1_param_stable_t<val_type, mat_exp_t>;
    using val_storage_type = typename base_type::val_storage_type;
    // val_type 可能是 `T&`（左值操作数）或 `T&&`，取 ele_type 前必须先剥掉引用
    using ele_type = typename std::remove_cvref_t<val_type>::ele_type;
    using val_base_type = typename base_type::val_base_type;

    mat_exp_t(val_storage_type val)
        : mat_express_1_param_stable_t<val_type, mat_exp_t>(std::move(val))
    {
    }

    auto work(val_base_type i) const
    {
        return std::exp(i);
    }

    static std::string type_name()
    {
        return "mat_exp_t";
    }
};


template <typename val_type>
requires is_matrix<val_type>
auto exp(val_type&& val)
{
    return mat_exp_t<val_type&&>(std::forward<val_type>(val));
}

template <typename val_type>
requires std::is_arithmetic_v<val_type>
val_type sigmoid(val_type const& val)
{
    return 1.0 / (1.0 + std::exp(-val));
}


template <typename val_type>
requires is_matrix<val_type>
class mat_sigmoid_t:public mat_express_1_param_stable_t<val_type, mat_sigmoid_t>
{
public:
    using base_type = mat_express_1_param_stable_t<val_type, mat_sigmoid_t>;
    using val_storage_type = typename base_type::val_storage_type;
    using val_base_type = typename base_type::val_base_type;
    // val_type 可能是 `T&`（左值操作数）或 `T&&`，取 ele_type 前必须先剥掉引用
    using ele_type = typename std::remove_cvref_t<val_type>::ele_type;

    mat_sigmoid_t(val_storage_type val)
        : mat_express_1_param_stable_t<val_type, mat_sigmoid_t>(std::move(val))
    {
    }

    auto work(val_base_type i) const
    {
        return 1.0 / (1.0 + std::exp(-i));
    }

    static std::string type_name()
    {
        return "mat_sigmoid_t";
    }
};

template <typename val_type>
requires is_matrix<val_type>
auto sigmoid(val_type&& val)
{
    return mat_sigmoid_t<val_type&&>(std::forward<val_type>(val));
}

template<typename val_type>
requires std::is_arithmetic_v<val_type>
auto sum(val_type const& val)
{
    return val;
}

template <typename val_type>
requires is_matrix<val_type>
auto sum(val_type const& val)
{ 
    using ele_type = typename val_type::ele_type;
    ele_type s = 0.;
    for (int i = 0; i < val.row_num(); ++i)
        for (int j = 0; j < val.col_num(); ++j)
            s += val(i, j);
    return s;
}

template <typename val_type>
requires is_matrix<val_type>
auto vsum(val_type const& val)           // 每一列的和，返回一个1行col_num列的矩阵
{
    using ele_type = typename val_type::ele_type;
    mat_t<ele_type> result(1, val.col_num());
    for (int j = 0; j < val.col_num(); ++j)
    {
        ele_type s = 0.;
        for (int i = 0; i < val.row_num(); ++i)
        {
            s += val(i, j);
        }
        result(0, j) = s;
    }
    return result;
}

template <typename val_type>
requires is_matrix<val_type>
auto hsum(val_type const& val)           // 每一行的和，返回一个row_num行1列的矩阵
{
    using ele_type = typename val_type::ele_type;
    mat_t<ele_type> result(val.row_num(), 1);
    for (int i = 0; i < val.row_num(); ++i)
    {
        ele_type s = 0.;
        for (int j = 0; j < val.col_num(); ++j)
        {
            s += val(i, j);
        }
        result(i, 0) = s;
    }
    return result;
}

template <typename val_type>
requires is_matrix<val_type>
auto vmean(val_type const& val)
{
    return (vsum(val) / val.row_num()).clone();
}

template <typename val_type>
requires is_matrix<val_type>
auto hmean(val_type const& val)
{
    return (hsum(val) / val.col_num()).clone();
}

template <typename val_type>
requires is_matrix<val_type>
auto mean(val_type const& val)
{
    return sum(val) / (val.row_num() * val.col_num());
}

template <typename input_type>
auto pow(input_type const& val, double const p = 2.)
{
    using val_type = typename std::decay_t<input_type>::ele_type;
    mat_t<val_type> ret(val.row_num(), val.col_num());
    for (int i = 0; i < val.row_num(); ++i)
        for (int j = 0; j < val.col_num(); ++j)
            ret(i, j) = std::pow(val(i, j), p);
    return ret;
}

template<typename input_type>
requires is_matrix<input_type>
auto max(input_type const& val)
{ 
    using ele_type = typename input_type::ele_type;
    ele_type ret = std::numeric_limits<ele_type>::lowest();
    for (int i = 0; i < val.row_num(); ++i)
        for (int j = 0; j < val.col_num(); ++j)
            if (val(i, j) > ret)
                ret = val(i, j);
    return ret;
}

template<typename input_type>
requires is_matrix<input_type>
auto hmax(input_type const& val)
{
    using ele_type = typename input_type::ele_type;
    mat_t<ele_type> ret(val.row_num(), 1);
    for (int i = 0; i < val.row_num(); ++i)
    {
        ele_type row_max = std::numeric_limits<ele_type>::lowest();
        for (int j = 0; j < val.col_num(); ++j)
        {
            if (val(i, j) > row_max)
                row_max = val(i, j);
        }
        ret(i, 0) = row_max;
    }
    return ret;
}

template <typename val_type>
class mat_softmax_t:public mat_express_1_param_stable_t<val_type, mat_softmax_t>
{
public:
    using base_type = mat_express_1_param_stable_t<val_type, mat_softmax_t>;
    using val_storage_type = typename base_type::val_storage_type;
    // val_type 可能是 `T&`（左值操作数）或 `T&&`，取 ele_type 前必须先剥掉引用
    using ele_type = typename std::remove_cvref_t<val_type>::ele_type;
    using val_base_type = typename base_type::val_base_type;
private:
    ele_type m_sum;
    ele_type m_max;     // 用于数值稳定性的最大值
public:

    mat_softmax_t(val_storage_type val)
        : mat_express_1_param_stable_t<val_type, mat_softmax_t>(std::move(val))
    {
        // 注意：val 已经被 move 进基类的 m_val，此后不能再读 val
        //（右值操作数 move 之后 val.data() 已是 nullptr）。统计量一律从 m_val 取，
        // 这样左值（m_val 是引用）和右值（m_val 是拥有的值）两条路径都对。
        m_max = max(base_type::m_val);
        m_sum = sum(exp(base_type::m_val - m_max));           // 初始化的时候求一遍和
    }

    void reset()
    {
        m_max = max(base_type::m_val);
        m_sum = sum(exp(base_type::m_val - m_max));
    }


    auto work(val_base_type i) const
    {
        return std::exp(i - m_max) / m_sum;
    }

    static std::string type_name()
    {
        return "mat_softmax_t";
    }
};

template <typename val_type>
requires std::is_arithmetic_v<val_type>
auto softmax(val_type const& val)
{
    return static_cast<val_type>(1.);
}

template <typename val_type>
requires is_matrix<val_type>
auto softmax(val_type&& val)
{
    return mat_softmax_t<val_type&&>(std::forward<val_type>(val));
}

template <typename input_type>
auto hsoftmax(const input_type& input)
{
    using val_type = typename std::decay_t<input_type>::ele_type;
    // 求得每行的最大值
    mat_t<val_type> max_val = hmax(input);
    // 求矩阵减去每行的最大值后的指数
    mat_t<val_type> exp_val = exp(input - max_val);
    // 求得每行的指数和
    mat_t<val_type> sum_exp = hsum(exp_val);
    // 求得每行的softmax值
    mat_t<val_type> softmax_val = exp_val / sum_exp;
    return softmax_val;
}

template<typename lval_type, typename rval_type>
requires is_matrix<lval_type> && is_matrix<rval_type>       // 矩阵点乘要求两边必须是矩阵
class mat_dot_t
{
public:
    // lval_type/rval_type 可能是 `T&`（左值）或 `T&&`（右值），取 ele_type 前必须先剥掉引用
    using ele_type = std::common_type_t<
        typename std::remove_cvref_t<lval_type>::ele_type,
        typename std::remove_cvref_t<rval_type>::ele_type>;
    // 与其它表达式节点一致：左值借引用、右值按值拥有。
    // 这里原来把引用写死在成员上，绕过了存储策略，是 `auto e = a.t().dot(b);` 悬垂的来源。
    using lval_storage_type = storage_type<lval_type>;
    using rval_storage_type = storage_type<rval_type>;
private:
    lval_storage_type m_lval;
    rval_storage_type m_rval;
public:
    mat_dot_t(lval_storage_type lval, rval_storage_type rval)
        : m_lval(std::move(lval)), m_rval(std::move(rval))
    {
        if (m_lval.col_num() != m_rval.row_num())
        {
            throw std::invalid_argument("mat_dot_t: inner dimensions do not match for dot product");
        }
    }

    int row_num() const
    {
        return m_lval.row_num();
    }

    int col_num() const
    {
        return m_rval.col_num();
    }

    std::tuple<int, int> shape() const
    {
        return std::make_tuple(row_num(), col_num());
    }

    auto operator()(int i, int j) const
    {
        ele_type s = 0.;
        for (int k = 0; k < m_lval.col_num(); ++k)
            s += m_lval(i, k) * m_rval(k, j);
        return s;
    }

    // 同其它基类：`*this` 恒为左值，需按接收者值类别分派，否则 `a.t().dot(b).dot(c)` 会悬垂
    template<typename other_type>
    auto dot(other_type&& m) const &
    {
        return jasmine::dot(*this, std::forward<other_type>(m));
    }

    template<typename other_type>
    auto dot(other_type&& m) const &&
    {
        return jasmine::dot(std::move(*this), std::forward<other_type>(m));
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "mat_dot_t(" << row_num() << ", " << col_num() << ")";
        for (int i = 0; i < row_num(); ++i)
        {
            ss << "\n[ ";
            for (int j = 0; j < col_num(); ++j)
            {
                ss << (*this)(i, j) << " ";
            }
            ss << " ]";
        }
        return ss.str();
    }

    operator mat_t<ele_type>() const
    {
        return clone();
    }

    mat_t<ele_type> clone() const
    {
        mat_t<ele_type> m(row_num(), col_num());
        if (detail::try_fast_gemm(m_lval, m_rval, m))
            return m;
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(i, j) = (*this)(i, j);
            }
        }
        return m;
    }
};

template<typename lval_type, typename rval_type>
requires is_matrix<lval_type> && is_matrix<rval_type>
auto dot(lval_type&& lval, rval_type&& rval)
{
    return mat_dot_t<lval_type&&, rval_type&&>(
        std::forward<lval_type>(lval), std::forward<rval_type>(rval));
}

// 在这里实现mat_t::dot成员函数
template <typename val_type>
requires std::is_arithmetic_v<val_type>
template<typename other_type>
requires is_matrix<other_type>
auto mat_t<val_type>::dot(other_type&& m) const &
{
    return jasmine::dot(*this, std::forward<other_type>(m));
}

template <typename val_type>
requires std::is_arithmetic_v<val_type>
template<typename other_type>
requires is_matrix<other_type>
auto mat_t<val_type>::dot(other_type&& m) const &&
{
    // 临时矩阵必须被拥有：std::move(*this) 使其以右值身份进入存储策略，
    // 于是节点内保存的是一份真实的矩阵拷贝，而不是对已销毁临时量的引用
    return jasmine::dot(std::move(*this), std::forward<other_type>(m));
}

template <typename val_type>
template<typename other_type>
requires is_matrix<other_type>
auto mat_view_t<val_type>::dot(other_type&& m) const &
{
    return jasmine::dot(*this, std::forward<other_type>(m));
}

template <typename val_type>
template<typename other_type>
requires is_matrix<other_type>
auto mat_view_t<val_type>::dot(other_type&& m) const &&
{
    return jasmine::dot(std::move(*this), std::forward<other_type>(m));
}

template<typename val_type>
requires is_matrix<val_type>
auto sqrt(val_type const& val)
{
    val_type result(val.row_num(), val.col_num());
    for (int i = 0; i < val.row_num(); ++i)
        for (int j = 0; j < val.col_num(); ++j)
            result(i, j) = std::sqrt(val(i, j));
    return result;
}


} // namespace jasmine
#endif
