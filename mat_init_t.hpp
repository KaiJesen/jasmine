#ifndef __MAT_INIT_T_HPP__
#define __MAT_INIT_T_HPP__
#include <random>

#include "mat_concepts.hpp"
#include "mat_t.hpp"

static std::default_random_engine g_random_engine;

template <typename init_type>
struct init_t
{
    template <typename mat_type>
    requires is_matrix<mat_type>
    static void cal(mat_type& m)
    { 
    }
};

template <>
struct init_t<class zeros_t>
{ 
    template <typename mat_type>
    requires is_matrix<mat_type>
    static void cal(mat_type& m)
    {
        for (int i = 0; i < m.row_num(); ++i)
        {
            for (int j = 0; j < m.col_num(); ++j)
            {
                m(i, j) = 0;
            }
        }
    }
};

template <>
struct init_t<class xavier_gaussian_t>
{
    template <typename mat_type>
    requires is_matrix<mat_type>
    static void cal(mat_type& m)
    { 
        double stddev = std::sqrt(2.0 / (m.row_num() + m.col_num()));
        std::normal_distribution<double> distribution(0.0, stddev);
        for (int i = 0; i < m.row_num(); ++i)
        {
            for (int j = 0; j < m.col_num(); ++j)
            {
                m(i, j) = distribution(g_random_engine);
            }
        }
    }
};

template <>
struct init_t<class xavier_uniform_t>
{
    template <typename mat_type>
    requires is_matrix<mat_type>
    static void cal(mat_type& m)
    {
        double stddev = std::sqrt(6.0 / (m.row_num() + m.col_num()));
        std::uniform_real_distribution<double> distribution(-stddev, stddev);
        for (int i = 0; i < m.row_num(); ++i)
        {
            for (int j = 0; j < m.col_num(); ++j)
            {
                m(i, j) = distribution(g_random_engine);
            }
        }
    }
};

template<>
struct init_t<class he_gaussian_t>
{
    template <typename mat_type>
    requires is_matrix<mat_type>
    static void cal(mat_type& m)
    {
        double stddev = std::sqrt(2.0 / m.col_num());
        std::normal_distribution<double> distribution(0.0, stddev);
        for (int i = 0; i < m.row_num(); ++i)
        {
            for (int j = 0; j < m.col_num(); ++j)
            {
                m(i, j) = distribution(g_random_engine);
            }
        }
    }
};

template<>
struct init_t<class he_uniform_t>
{
    template <typename mat_type>
    requires is_matrix<mat_type>
    static void cal(mat_type& m)
    {
        double stddev = std::sqrt(6.0 / m.col_num());
        std::uniform_real_distribution<double> distribution(-stddev, stddev);
        for (int i = 0; i < m.row_num(); ++i)
        {
            for (int j = 0; j < m.col_num(); ++j)
            {
                m(i, j) = distribution(g_random_engine);
            }
        }
    }
};

template <typename init_type, typename mat_type, typename...args_types>
requires is_matrix<mat_type>
void init_matrix(mat_type& m, args_types&&... args)
{
    init_t<init_type>::cal(m, std::forward<args_types>(args)...);
}


#endif
