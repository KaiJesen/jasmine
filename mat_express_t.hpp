#ifndef __MAT_EXPRESS_T_HPP__
#define __MAT_EXPRESS_T_HPP__
#include <cassert>
#include <math.h>
#include <sstream>
#include <tuple>
#include <string>
#include <vector>
#include <numeric>

#include "mat_concepts.hpp"
#include "mat_view_t.hpp"

template<typename lval_type, typename rval_type>
requires is_matrix<lval_type> && is_matrix<rval_type>
auto dot(lval_type const& lval, rval_type const& rval);

template<typename T, bool is_trivial>
struct storage_selector;
template<typename T>
struct storage_selector<T, true>
{
    using type = mat_t<T>;
};

template<typename T>
struct storage_selector<T, false>
{
    using type = T const &;
};

template<typename T>
using storage_type = typename storage_selector<T, std::is_trivial_v<T>>::type;

template <typename lval_type, typename rval_type, template<typename,typename> class tpl>
class mat_express_2_param_stable_t
{
public:
    using lval_storage_type = storage_type<lval_type>;
    using rval_storage_type = storage_type<rval_type>;
    using lval_base_type = typename std::decay_t<lval_storage_type>::return_type;
    using rval_base_type = typename std::decay_t<rval_storage_type>::return_type;
    using derived_type = tpl<lval_type, rval_type>;
    using return_type = std::common_type_t<lval_base_type, rval_base_type>;

    mat_express_2_param_stable_t(lval_type const& left, rval_type const& right)
        : m_left(left), m_right(right)
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

    template<typename other_type>
    auto dot(const other_type& m) const
    {
        return ::dot(*reinterpret_cast<derived_type const*>(this), m);
    }

    // 计算所有元素的值，并赋值给一个新的mat_t对象并返回
    mat_t<return_type> clone() const
    {
        mat_t<return_type> m(row_num(), col_num());
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(i, j) = (*this)(i, j);
            }
        }
        return m;
    }

    operator mat_t<return_type>() const
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
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using return_type = std::common_type_t<lval_base_type, rval_base_type>;
    mat_greater_t(lval_type const& left, rval_type const& right)
        : base_type(left, right)
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
auto operator>(lval_type const& left, rval_type const& right)
{
    return mat_greater_t<lval_type, rval_type>(left, right);
}

template <typename lval_type, typename rval_type>
class mat_less_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_less_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_less_t>;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using return_type = std::common_type_t<lval_base_type, rval_base_type>;

    mat_less_t(lval_type const& left, rval_type const& right)
        : base_type(left, right)
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
auto operator<(lval_type const& left, rval_type const& right)
{
    return mat_less_t<lval_type, rval_type>(left, right);
}

template <typename lval_type, typename rval_type>
class mat_add_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_add_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_add_t>;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using return_type = std::common_type_t<lval_base_type, rval_base_type>;


    mat_add_t(lval_type const& left, rval_type const& right)
        : base_type(left, right)
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
auto operator+(lval_type const& left, rval_type const& right)
{
    return mat_add_t<lval_type, rval_type>(left, right);
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
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using return_type = std::common_type_t<lval_base_type, rval_base_type>;

    mat_sub_t(lval_type const& left, rval_type const& right)
        : base_type(left, right)
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
auto operator-(lval_type const& left, rval_type const& right)
{
    return mat_sub_t<lval_type, rval_type>(left, right);
}

template <typename lval_type, typename rval_type>
class mat_mul_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_mul_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_mul_t>;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using return_type = std::common_type_t<lval_base_type, rval_base_type>;

    mat_mul_t(lval_type const& left, rval_type const& right)
        : base_type(left, right)
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
auto operator*(lval_type const& left, rval_type const& right)
{
    return mat_mul_t<lval_type, rval_type>(left, right);
}

template <typename lval_type, typename rval_type>
class mat_div_t:public mat_express_2_param_stable_t<lval_type, rval_type, mat_div_t>
{
public:
    using lreturn_type = lval_type;
    using rreturn_type = rval_type;

    using base_type = mat_express_2_param_stable_t<lval_type, rval_type, mat_div_t>;
    using lval_base_type = typename base_type::lval_base_type;
    using rval_base_type = typename base_type::rval_base_type;
    using return_type = std::common_type_t<lval_base_type, rval_base_type>;


    mat_div_t(lval_type const& left, rval_type const& right)
        : base_type(left, right)
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
auto operator/(lval_type const& left, rval_type const& right)
{
    return mat_div_t<lval_type, rval_type>(left, right);
}

template<typename val_type, template<typename> class tpl>
class mat_express_1_param_stable_t
{
public:
    using derived_type = tpl<val_type>;
    using val_storage_type = storage_type<val_type>;
    using val_base_type = typename std::decay_t<val_storage_type>::return_type;
    using return_type = val_base_type;

    mat_express_1_param_stable_t(val_type const& val)
        : m_val(val)
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

    template<typename other_type>
    auto dot(const other_type& m) const
    {
        return ::dot(*reinterpret_cast<derived_type const*>(this), m);
    }

    mat_t<return_type> clone() const
    {
        mat_t<return_type> m(row_num(), col_num());
        for (int i = 0; i < row_num(); ++i)
        {
            for (int j = 0; j < col_num(); ++j)
            {
                m(i, j) = (*this)(i, j);
            }
        }
        return m;
    }

    operator mat_t<return_type>() const
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
    using return_type = typename val_type::return_type;
    using val_base_type = typename base_type::val_base_type;

    mat_exp_t(val_type const& val)
        : mat_express_1_param_stable_t<val_type, mat_exp_t>(val)
    {
    }

    auto work(val_base_type i) const
    {
        return exp(i);
    }

    static std::string type_name()
    {
        return "mat_exp_t";
    }
};


template <typename val_type>
requires is_matrix<val_type>
auto exp(val_type const& val)
{
    return mat_exp_t<val_type>(val);
}

template <typename val_type>
requires std::is_arithmetic_v<val_type>
val_type sigmoid(val_type const& val)
{
    return 1.0 / (1.0 + exp(-val));
}


template <typename val_type>
requires is_matrix<val_type>
class mat_sigmoid_t:public mat_express_1_param_stable_t<val_type, mat_sigmoid_t>
{
public:
    using base_type = mat_express_1_param_stable_t<val_type, mat_sigmoid_t>;
    using val_base_type = typename base_type::val_base_type;
    using return_type = typename val_type::return_type;

    mat_sigmoid_t(val_type const& val)
        : mat_express_1_param_stable_t<val_type, mat_sigmoid_t>(val)
    {
    }

    auto work(val_base_type i) const
    {
        return 1.0 / (1.0 + exp(-i));
    }

    static std::string type_name()
    {
        return "mat_sigmoid_t";
    }
};

template <typename val_type>
requires is_matrix<val_type>
auto sigmoid(val_type const& val)
{
    return mat_sigmoid_t<val_type>(val);
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
    using return_type = typename val_type::return_type;
    return_type s = 0.;
    for (int i = 0; i < val.row_num(); ++i)
        for (int j = 0; j < val.col_num(); ++j)
            s += val(i, j);
    return s;
}

template <typename val_type>
requires is_matrix<val_type>
auto vsum(val_type const& val)           // 每一列的和，返回一个1行col_num列的矩阵
{
    using return_type = typename val_type::return_type;
    mat_t<return_type> result(1, val.col_num());
    for (int j = 0; j < val.col_num(); ++j)
    {
        return_type s = 0.;
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
    using return_type = typename val_type::return_type;
    mat_t<return_type> result(val.row_num(), 1);
    for (int i = 0; i < val.row_num(); ++i)
    {
        return_type s = 0.;
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
    using val_type = typename std::decay_t<input_type>::return_type;
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
    using return_type = typename input_type::return_type;
    return_type ret = std::numeric_limits<return_type>::lowest();
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
    using return_type = typename input_type::return_type;
    mat_t<return_type> ret(val.row_num(), 1);
    for (int i = 0; i < val.row_num(); ++i)
    {
        return_type row_max = std::numeric_limits<return_type>::lowest();
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
    using return_type = typename val_type::return_type;
    using val_base_type = typename base_type::val_base_type;
private:
    return_type m_sum;
    return_type m_max;     // 用于数值稳定性的最大值
public:

    mat_softmax_t(val_type const& val)
        : mat_express_1_param_stable_t<val_type, mat_softmax_t>(val)
    {
        m_max = max(val);
        m_sum = sum(exp(val - m_max));           // 初始化的时候求一遍和
    }

    void reset()
    {
        m_max = max(base_type::m_val);
        m_sum = sum(exp(base_type::m_val - m_max));
    }


    auto work(val_base_type i) const
    {
        return exp(i - m_max) / m_sum;
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
auto softmax(val_type const& val)
{
    return mat_softmax_t<val_type>(val);
}

template <typename input_type>
auto hsoftmax(const input_type& input)
{
    using val_type = typename std::decay_t<input_type>::return_type;
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
    using return_type = std::common_type_t<typename lval_type::return_type, typename rval_type::return_type>;
private:
    lval_type const& m_lval;
    rval_type const& m_rval;
public:
    mat_dot_t(lval_type const& lval, rval_type const& rval)
        : m_lval(lval), m_rval(rval)
    {
        if (lval.col_num() != rval.row_num())
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
        return_type s = 0.;
        for (int k = 0; k < m_lval.col_num(); ++k)
            s += m_lval(i, k) * m_rval(k, j);
        return s;
    }

    template<typename other_type>
    auto dot(const other_type& m) const
    {
        return ::dot(*this, m);
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

    operator mat_t<return_type>() const
    {
        return clone();
    }

    mat_t<return_type> clone() const
    {
        mat_t<return_type> m(row_num(), col_num());
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
auto dot(lval_type const& lval, rval_type const& rval)
{
    return mat_dot_t<lval_type, rval_type>(lval, rval);
}

// 在这里实现mat_t::dot成员函数
template <typename val_type>
requires std::is_arithmetic_v<val_type>
template<typename other_type>
requires is_matrix<other_type>
auto mat_t<val_type>::dot(const other_type& m) const
{
    return ::dot(*this, m);
}

template <typename val_type>
template<typename other_type>
requires is_matrix<other_type>
auto mat_view_t<val_type>::dot(const other_type& m) const
{
    return ::dot(*this, m);
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

void test_mat_express_t()
{ 
    mat_t m1{3, 3, {1,2,3,
                   4,5,6,
                   7,8,9}};
    mat_t m2{3, 3, {9,8,7,
                   6,5,4,
                   3,2,1}};
    auto m3 = ((m1 + m2 - m1) * m2 / m1).dot(m2).dot(m2);
    std::cout << m3 << std::endl;

    mat_t<double> m4{2, 4, {2, 2, 2, 2,
                            3, 3, 3, 3}};
    std::cout << hsoftmax(m4) << std::endl;
}

#endif
