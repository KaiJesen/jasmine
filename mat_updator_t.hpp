#ifndef __MAT_UPDATOR_T_HPP__
#define __MAT_UPDATOR_T_HPP__

#include "mat_t.hpp"
#include "mat_express_t.hpp"
#include "mat_net_t.hpp"
#include "mat_init_t.hpp"

template <typename val_type>
class sgd_t
{
private:
    val_type m_learning_rate;
public:
    sgd_t(val_type learning_rate = 0.001)
        : m_learning_rate(learning_rate)
    { 
    }

    void set(val_type learning_rate = 0.001)
    {
        m_learning_rate = learning_rate;
    }

    void set_lr(val_type lr)
    {
        m_learning_rate = lr;
    }

    void update(const mat_t<val_type>& grad, mat_t<val_type>& mat)
    { 
        mat = mat - m_learning_rate * grad;
    }

    void step()
    {
        // sgd不需要额外的step操作
    }
};


template <typename val_type>
class adam_t
{
private:
    mat_t<val_type> m_m;    // 一阶矩估计
    mat_t<val_type> m_v;    // 二阶矩估计
    val_type m_beta1;
    val_type m_beta1_t;
    val_type m_beta2;
    val_type m_beta2_t;
    val_type m_learning_rate;
    val_type m_epsilon;
    int m_t;                    // 时间步长

public:
    adam_t(val_type learning_rate = 0.001,
         val_type beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : m_beta1(beta1), m_beta1_t(1), m_beta2(beta2), m_beta2_t(1),
          m_learning_rate(learning_rate), m_epsilon(epsilon), m_t(0)
    { 
    }

    void set(val_type learning_rate = 0.001,
             val_type beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        m_beta1 = beta1;
        m_beta1_t = 1;
        m_beta2 = beta2;
        m_beta2_t = 1;
        m_learning_rate = learning_rate;
        m_epsilon = epsilon;
        m_t = 0;
    }

    void set_lr(val_type lr)
    {
        m_learning_rate = lr;
    }

    void update(const mat_t<val_type>& grad, mat_t<val_type>& mat)
    { 
        m_t++;
        if (m_m.valid() == false)
        {
            m_m = mat_t<val_type>(grad.row_num(), grad.col_num());
            m_v = mat_t<val_type>(grad.row_num(), grad.col_num());
        }
        m_m = m_beta1 * m_m + (1 - m_beta1) * grad;
        m_v = m_beta2 * m_v + (1 - m_beta2) * grad * grad;
        m_beta1_t *= m_beta1;
        m_beta2_t *= m_beta2;

        mat_t<val_type> m_hat = m_m / (1 - m_beta1_t);
        mat_t<val_type> v_hat = m_v / (1 - m_beta2_t);

        mat = mat - m_learning_rate * m_hat / (sqrt(v_hat) + m_epsilon);
    }

    void step()
    {
        // nadam不需要额外的step操作
    }

};

template <typename val_type>
class nadam_t
{
private:
    mat_t<val_type> m_m;    // 一阶矩估计
    mat_t<val_type> m_v;    // 二阶矩估计
    val_type m_beta1;
    val_type m_beta1_t;
    val_type m_beta2;
    val_type m_beta2_t;
    val_type m_learning_rate;
    val_type m_epsilon;
    int m_t;                    // 时间步长

public:
    nadam_t(val_type learning_rate = 0.002,
          val_type beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : m_beta1(beta1), m_beta1_t(1), m_beta2(beta2), m_beta2_t(1),
          m_learning_rate(learning_rate), m_epsilon(epsilon), m_t(0)
    { 
    }

    void set(val_type learning_rate = 0.002,
             val_type beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        m_beta1 = beta1;
        m_beta2 = beta2;
        m_learning_rate = learning_rate;
        m_epsilon = epsilon;
        m_t = 0;
    }

    void set_lr(val_type lr)
    {
        m_learning_rate = lr;
    }

    void update(const mat_t<val_type>& grad, mat_t<val_type>& mat)
    { 
        m_t++;
        if (m_m.valid() == false)
        {
            m_m = mat_t<val_type>(grad.row_num(), grad.col_num());
            m_v = mat_t<val_type>(grad.row_num(), grad.col_num());
        }
        m_m = m_beta1 * m_m + (1 - m_beta1) * grad;
        m_v = m_beta2 * m_v + (1 - m_beta2) * grad * grad;

        m_beta1_t *= m_beta1;
        m_beta2_t *= m_beta2;

        mat_t<val_type> m_hat = m_m / (1 - m_beta1_t);
        mat_t<val_type> v_hat = m_v / (1 - m_beta2_t);

        auto mtv_n = (m_learning_rate * (m_beta1 * m_hat / (1 - m_beta1_t * m_beta1) + (1 - m_beta1) / (1 - m_beta1_t) * grad)).clone();
        mat = mat - mtv_n / (sqrt(v_hat) + m_epsilon);
    }

    void step()
    {
        // nadam不需要额外的step操作
    }

};

template<typename val_type, template<typename> class updator_type>
class cache_updator_t
{
private:
    updator_type<val_type> m_updator;
    mat_t<val_type> m_cache;   // 用于保存累加的梯度
    mat_t<val_type>* m_mat;
    val_type m_count;                // 用于记录累加的梯度数量
public:
    cache_updator_t(val_type learning_rate = 0.001)
        : m_updator(learning_rate), m_cache(), m_mat(nullptr), m_count(0)
    {
    }

    void update(const mat_t<val_type>& grad, mat_t<val_type>& mat)
    {
        if (m_mat == nullptr)
        {
            m_cache.reshape(grad.row_num(), grad.col_num());
            m_mat = &mat;
        }
        m_cache = m_cache * (val_type(1.0) - val_type(1.0) / val_type(m_count + 1)) + grad / val_type(m_count + 1);   // 累加梯度
        m_count++;
    }

    void step()
    {
        if (m_mat != nullptr)
        {
            m_updator.update(m_cache, *m_mat);   // 使用累加的梯度更新参数
            m_cache = 0.0;  // 清空缓存
            m_count = 0;    // 重置计数器
        }
    }

    void set_lr(val_type lr)
    {
        m_updator.set_lr(lr);
    }

    template<typename... upr_arg_types>
    void set(val_type learning_rate, upr_arg_types&&... args)
    {
        m_updator.set(learning_rate, std::forward<upr_arg_types>(args)...);
    }

};

#endif
