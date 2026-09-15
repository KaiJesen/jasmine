#ifndef __JAS_KV_CACHE_T_HPP__
#define __JAS_KV_CACHE_T_HPP__

#include <algorithm>
#include <stdexcept>

#include "jas_mat_concepts.hpp"
#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"

namespace jasmine {

/**
 * 单头 KV cache：存投影后、RoPE 后的 K/V（列 = 时间步）。
 * 仅推理 / forward_one 使用；训练整段 forward 不走这里。
 *
 * dynamic: 容量不够时扩容拷贝
 * static_fixed: 预 reserve，超出抛异常
 */
enum class kv_cache_mode
{
    dynamic,
    static_fixed
};

template <typename val_type>
class kv_cache_t
{
private:
    mat_t<val_type> m_k;
    mat_t<val_type> m_v;
    int m_d = 0;
    int m_len = 0;
    int m_cap = 0;
    kv_cache_mode m_mode = kv_cache_mode::dynamic;

    static constexpr int EXPAND = 64;

    void grow_to(int need_cap)
    {
        if (need_cap <= m_cap)
            return;
        if (m_mode == kv_cache_mode::static_fixed)
        {
            throw std::runtime_error(
                "KV cache static capacity exceeded; call reserve() with a larger max_seq");
        }
        const int new_cap = std::max(need_cap, m_cap + EXPAND);
        mat_t<val_type> nk(m_d, new_cap);
        mat_t<val_type> nv(m_d, new_cap);
        nk = val_type(0);
        nv = val_type(0);
        for (int j = 0; j < m_len; ++j)
        {
            for (int i = 0; i < m_d; ++i)
            {
                nk(i, j) = m_k(i, j);
                nv(i, j) = m_v(i, j);
            }
        }
        m_k = std::move(nk);
        m_v = std::move(nv);
        m_cap = new_cap;
    }

public:
    kv_cache_t() = default;

    void set_mode(kv_cache_mode mode)
    {
        m_mode = mode;
    }

    kv_cache_mode mode() const { return m_mode; }

    int dim() const { return m_d; }
    int length() const { return m_len; }
    int capacity() const { return m_cap; }

    /** 预分配 [d × max_seq]；可在 static / dynamic 下调用 */
    void reserve(int d, int max_seq)
    {
        if (d <= 0 || max_seq <= 0)
            throw std::invalid_argument("kv_cache reserve sizes must be positive");
        m_d = d;
        m_cap = max_seq;
        m_len = 0;
        m_k = mat_t<val_type>(d, max_seq);
        m_v = mat_t<val_type>(d, max_seq);
        m_k = val_type(0);
        m_v = val_type(0);
    }

    void clear()
    {
        m_len = 0;
    }

    /**
     * 追加一段 K/V（d × n_new）。调用方保证已做 RoPE（K）且 d 与 cache 一致。
     */
    template<typename K, typename V>
    requires is_matrix<K> && is_matrix<V>
    void append(const K& k, const V& v)
    {
        if (k.row_num() != v.row_num() || k.col_num() != v.col_num())
            throw std::runtime_error("kv_cache append: K/V shape mismatch");
        if (m_d == 0)
        {
            // 首次动态追加：按需开一块
            reserve(k.row_num(), std::max(k.col_num(), EXPAND));
            if (m_mode == kv_cache_mode::static_fixed)
            {
                // reserve 已设好；下面走正常路径
            }
        }
        if (k.row_num() != m_d)
            throw std::runtime_error("kv_cache append: dim mismatch");

        const int n_new = k.col_num();
        grow_to(m_len + n_new);
        for (int j = 0; j < n_new; ++j)
        {
            for (int i = 0; i < m_d; ++i)
            {
                m_k(i, m_len + j) = k(i, j);
                m_v(i, m_len + j) = v(i, j);
            }
        }
        m_len += n_new;
    }

    mat_view_t<mat_t<val_type>> keys()
    {
        if (m_len == 0)
            throw std::runtime_error("kv_cache keys: empty");
        return m_k.view(0, 0, m_d, m_len);
    }

    mat_view_t<mat_t<val_type>> values()
    {
        if (m_len == 0)
            throw std::runtime_error("kv_cache values: empty");
        return m_v.view(0, 0, m_d, m_len);
    }

    mat_view_t<const mat_t<val_type>> keys() const
    {
        if (m_len == 0)
            throw std::runtime_error("kv_cache keys: empty");
        return m_k.view(0, 0, m_d, m_len);
    }

    mat_view_t<const mat_t<val_type>> values() const
    {
        if (m_len == 0)
            throw std::runtime_error("kv_cache values: empty");
        return m_v.view(0, 0, m_d, m_len);
    }
};

} // namespace jasmine
#endif
