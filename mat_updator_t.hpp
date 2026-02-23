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

        auto mtv_n = m_learning_rate * (m_beta1 * m_hat / (1 - m_beta1_t * m_beta1) + (1 - m_beta1) / (1 - m_beta1_t) * grad);
        mat = mat - mtv_n / (sqrt(v_hat) + m_epsilon);
    }
};

#endif
