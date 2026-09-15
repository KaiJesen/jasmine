#ifndef _JAS_MAT_STORAGE_HPP_
#define _JAS_MAT_STORAGE_HPP_

#include "jas_mat_t.hpp"

namespace jasmine {
namespace detail {

/** 反向需持久输入时：表达式物化一次；mat rvalue 移动；mat lvalue 拷贝 */
template<typename val_type, typename Src>
void store_for_backward(mat_t<val_type>& dst, Src&& src)
{
    using Decay = std::decay_t<Src>;
    if constexpr (std::is_same_v<Decay, mat_t<val_type>>)
    {
        if constexpr (std::is_rvalue_reference_v<Src&&>)
            dst = std::move(src);
        else
            dst = src;
    }
    else
    {
        dst = std::forward<Src>(src);
    }
}

} // namespace detail
} // namespace jasmine

#endif
