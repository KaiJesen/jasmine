#ifndef __JAS_CUDA_NET_HPP__
#define __JAS_CUDA_NET_HPP__

/**
 * 设备端网络层：`jas_net_t.hpp` 那些层的 GPU 对应物，带 forward + backward。
 *
 * ## 为什么要有这一层
 *
 * 到 RoPE 为止，设备端攒下的是**算子**（GEMM、归约、softmax、RoPE）。但「在 GPU 上
 * 训练一个模型」需要的是**层** —— 层才有「前向留下什么、反向怎么算、参数怎么更新」
 * 这套结构。主机端把这套结构写在 `weight_net_t` / `layer_norm_net_t` 里，
 * 这里逐个对应，公式**逐字对齐**主机实现，所以同一条数据流喂给两边，
 * 梯度会逐元素相同（测例就是干这个的）。
 *
 * ## 与主机端协议的对应关系
 *
 * 主机端的网是**鸭子类型**（C++20 concept + `std::apply` 折叠），没有基类。
 * 设备端保持同样的风格，成员名一致，于是熟悉主机端的人不用学新东西：
 *
 *   | 主机                       | 设备                             |
 *   |----------------------------|----------------------------------|
 *   | `forward(x) -> mat_t`      | `forward(x) -> dev_matrix_t`     |
 *   | `backward(delta) -> mat_t` | `backward(delta) -> dev_matrix_t`|
 *   | `step()`                   | `step()`                         |
 *   | `set_lr(v)`                | `set_lr(v)`                      |
 *   | `set_updator(std::forward<arg_types>(args)...)`     | `set_updator(std::forward<arg_types>(args)...)`           |
 *   | `init_weight<init_t>()`    | `init_weight<init_t>()`          |
 *   | `net_type()`               | `net_type()`                     |
 *   | `reinit({in,out})`         | `reinit({in,out})`               |
 *
 * `reinit(std::vector<int>)` 特意保留同名同参：主机端的 `is_reinitable_net`
 * concept 正是靠它判定的，设备端若要复用那些 `complex_net_builder` 之类的
 * 编译期装置，就必须满足同一个签名。
 *
 * ## 三条贯穿所有层的约定
 *
 * 1. **梯度是返回值，参数更新在 backward 里发生。** 主机端就是这么做的
 *    （`weight_net_t::backward` 里 `m_weight_updator.update(...)` 之后才 return），
 *    设备端照搬。`step()` 只为梯度累积器（`dev_cache_updator_t`）而存在。
 *
 * 2. **顺序不能动。** 主机端 `weight_net_t::backward` 先用**更新前**的权重算输入梯度
 *    （`ret = m_weight.t().dot(delta)` 是立即物化的），然后才更新权重。
 *    设备端必须同序，否则同一串梯度喂两边会得到不同的数。
 *
 * 3. **输入要缓存。** 反向需要 `m_input`，与主机端 `store_for_backward` 同义。
 *    设备端是显式物化：`detail::materialize_input(input, m_input)`。
 *    代价是每个线性层每次前向多一次全矩阵拷贝 —— 与主机端的行为一致，
 *    不是设备端特有的开销，但确实值得将来优化（见 CUDA.md 第 13 节）。
 *
 * ## 关于 `mat_t` 与 `dev_matrix_t` 参与表达式的不对称
 *
 * 主机端 `mat_t` 既能当拥有者、又能直接进表达式。设备端不行：`dev_matrix_t`
 * 刻意**没有** `operator()` 也不是 `JAS_HD`（它只是个「缓冲区 + 尺寸」的拥有者），
 * 所以它进不了表达式，必须先取 `leaf()` / `const_leaf()`。
 * 本文件里 `materialize_input` 和 `forward` 的各种重载就是在吸收这个不对称，
 * 让调用方既能传 `dev_matrix_t`、也能传叶子和表达式。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_updator.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_mat_t.hpp"

namespace jasmine {
namespace cuda {

namespace detail {

/** 「拥有显存、需要先取叶子才能进表达式」的设备矩阵（判据是有 `const_leaf()`）。 */
template <typename X, typename = void>
struct is_dev_matrix_owner : std::false_type {};

template <typename X>
struct is_dev_matrix_owner<X, std::void_t<decltype(std::declval<const X&>().const_leaf())>>
    : std::true_type {};

template <typename X>
inline constexpr bool is_dev_matrix_owner_v = is_dev_matrix_owner<X>::value;

/**
 * 把任意设备矩阵物化成一个**拥有者**矩阵。
 *
 * 两类输入走两条完全不同的路：
 *   - 拥有者（`dev_matrix_t`）：不能进表达式，但是一块连续显存，直接设备间整块拷贝。
 *     它**必须**拷贝而不是留个叶子 —— 我们要的是一个活过本次调用的副本
 *     （调用方的缓冲区随时可能被复用或释放），这与主机端 `store_for_backward`
 *     对左值做拷贝是同一个理由。
 *   - 叶子 / 表达式：塞进融合 kernel 求值一遍。叶子走这条路也不吃亏 ——
 *     `eval_fused` 本来就会处理转置与 leading dim，比手写拷贝更稳妥。
 */
template <typename X>
void materialize_into(const X& src, dev_matrix_t<typename X::ele_type>& dst)
{
    using T = typename X::ele_type;

    dst.allocate(src.row_num(), src.col_num());
    if (dst.row_num() == 0 || dst.col_num() == 0)
        return;

    if constexpr (is_dev_matrix_owner_v<X>)
    {
        JAS_CUDA_CHECK(cudaMemcpy(dst.buffer().data(), src.const_leaf().m_data,
                                  dst.buffer().size() * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else
    {
        eval_fused(src, dst.buffer());
    }
}

/**
 * 「输入物化」的统一入口 —— 比 `materialize_into` 多认一种东西：**主机 `mat_t`**。
 *
 * 设备端层的输入主力当然是设备矩阵，但有两类调用方手里只有主机矩阵：
 *
 *   - token id：本来就产自主机上的 tokenizer；
 *   - 测试：参照数据一律先在主机上生成，再送进设备路径算。
 *
 * 让这两类调用点直接传主机矩阵，比逼它们各自 upload 一次更不容易出错 ——
 * 上传只是一次 H2D 拷贝，和后面前向/反向的算力开销不在一个量级。
 * 主机矩阵的 `data()` 是稠密行主序，与 `dev_matrix_t` 的布局一致。
 */
template <typename X, typename T>
void materialize_input(const X& src, dev_matrix_t<T>& dst)
{
    if constexpr (std::is_same_v<std::remove_cvref_t<X>, mat_t<T>>)
    {
        dst.allocate(src.row_num(), src.col_num());
        if (dst.row_num() == 0 || dst.col_num() == 0)
            return;
        dst.upload(src);
    }
    else
    {
        materialize_into(src, dst);
    }
}

/** 上传一段主机标量成列向量。 */
template <typename T>
dev_colvec_t<T> colvec_from_host(const std::vector<T>& host){
    dev_colvec_t<T> v(static_cast<int>(host.size()));
    if (!host.empty())
        v.upload(host.data(), host.size());
    return v;
}

/** 上传一个主机 (d × 1) 矩阵成列向量（gamma / beta / bias 都长这样）。 */
template <typename T>
dev_colvec_t<T> colvec_from_mat(const mat_t<T>& host, const char* who)
{
    if (host.col_num() != 1)
        throw std::invalid_argument(std::string(who) + ": 形状应为 (d × 1)");
    std::vector<T> h(static_cast<std::size_t>(host.row_num()));
    for (int i = 0; i < host.row_num(); ++i)
        h[static_cast<std::size_t>(i)] = host(i, 0);
    return colvec_from_host(h);
}

/**
 * 用主机端那套初始化策略随机初始化一个设备矩阵。
 *
 * 直接在主机上跑 `init_matrix<init_type>` 再上传，于是**同一套策略、同一个全局引擎**
 * 得到同一批数 —— 设备端与主机端的初始化因此可以逐元素对拍，而不是各随机各的。
 * 代价是初始化不是「设备内生」的（`g_random_engine` 是主机全局引擎），
 * 这一点在 CUDA.md 里记为已知限制。
 */
template <typename init_type, typename T>
void init_device_matrix(dev_matrix_t<T>& m, int rows, int cols)
{
    mat_t<T> host(rows, cols);
    init_matrix<init_type>(host);
    m.allocate(rows, cols);
    m.upload(host);
}

} // namespace detail

// ---------------------------------------------------------------------------
// 线性层：weight_net_t 的设备对应物
// ---------------------------------------------------------------------------

/**
 * 全连接层 `y = W·x + b`。
 *
 * 权重 (out × in)、偏置 (out × 1)，与主机 `weight_net_t` 完全同形 —— 包括
 * 「偏置是列向量、沿列广播」这一点（主机 `m_bias` 就是 [out, 1]）。
 *
 * 反向的三个量（对应主机 `weight_net_t::backward` 的三行）：
 *
 *   dW  = delta · xᵀ        // (out×T)·(T×in) = (out×in)
 *   db  = row_sum(delta)    // 沿序列维（列）累加 → (out×1)，即主机的 hsum
 *   dx  = Wᵀ · delta        // (in×out)·(out×T) = (in×T)
 *
 * 注意 `dx` 用的是**更新前**的 W（顺序见文件头约定 2）。
 */
template <typename T, template <typename> class updator_type>
class dev_linear_t
{
public:
    using ele_type = T;

    dev_linear_t() = default;
    dev_linear_t(int input_size, int output_size) { set_param(input_size, output_size); }

    /** 分配权重与偏置（不初始化数值，见 `init_weight` / `upload_*`）。 */
    void set_param(int input_size, int output_size)
    {
        if (input_size <= 0 || output_size <= 0)
            throw std::invalid_argument("dev_linear_t: 尺寸必须为正");
        m_in = input_size;
        m_out = output_size;
        m_weight.allocate(m_out, m_in);
        m_bias = detail::colvec_from_host(
            std::vector<T>(static_cast<std::size_t>(m_out), T(0)));
        m_input.allocate(m_in, 0);
    }

    /** 与主机 `reinit({in, out})` 同义（主机在本层读的是 container[1], container[0]）。 */
    void reinit(const std::vector<int>& container)
    {
        if (container.size() < 2)
            throw std::invalid_argument("dev_linear_t::reinit: 需要 {in, out}");
        set_param(container[0], container[1]);
    }

    /** 与主机 `weight_net_t::init_weight` 逐字对应：同一套策略、同一个引擎。 */
    template <typename init_type>
    void init_weight()
    {
        detail::init_device_matrix<init_type, T>(m_weight, m_out, m_in);

        mat_t<T> host_bias(m_out, 1);
        init_matrix<init_type>(host_bias);
        m_bias = detail::colvec_from_mat(host_bias, "dev_linear_t::init_weight");
    }

    /** LLaMA 这类要求零偏置的模型在加载后调用（主机端是 `zero_all_biases`）。 */
    void zero_bias()
    {
        m_bias = detail::colvec_from_host(std::vector<T>(static_cast<std::size_t>(m_out), T(0)));
    }

    void upload_weight(const mat_t<T>& host)
    {
        if (host.row_num() != m_out || host.col_num() != m_in)
            throw std::invalid_argument("dev_linear_t::upload_weight: 形状应为 (out × in)");
        m_weight.allocate(m_out, m_in);
        m_weight.upload(host);
    }

    void upload_bias(const mat_t<T>& host)
    {
        if (host.row_num() != m_out || host.col_num() != 1)
            throw std::invalid_argument("dev_linear_t::upload_bias: 形状应为 (out × 1)");
        m_bias = detail::colvec_from_mat(host, "dev_linear_t::upload_bias");
    }

    /** 回读参数，便于和主机对拍。 */
    mat_t<T> weight_to_host() const { return m_weight.download(); }
    mat_t<T> bias_to_host() const { return m_bias.download(); }
    dev_matrix_t<T>& cached_input() { return m_input; }

    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        if (input.row_num() != m_in)
            throw std::invalid_argument("dev_linear_t::forward: 输入行数 "
                                        + std::to_string(input.row_num()) + " 与 in "
                                        + std::to_string(m_in) + " 不一致");

        detail::materialize_input(input, m_input);

        dev_matrix_t<T> out = matmul(m_weight.const_leaf(), m_input.const_leaf());

        // 偏置是 (out × 1) 的列向量，按列广播 —— 这里在 out 自己的缓冲区上原地加。
        // 安全的原因：融合 kernel 一个线程只读写自己那个元素，且 allocate 在尺寸不变时
        // 是空操作，不会把正在读的缓冲区换掉。
        eval_fused(out.leaf() + m_bias.leaf(), out.buffer());
        return out;
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        if (delta.row_num() != m_out)
            throw std::invalid_argument("dev_linear_t::backward: delta 行数 "
                                        + std::to_string(delta.row_num()) + " 与 out "
                                        + std::to_string(m_out) + " 不一致");

        // 1. 权重梯度 (out × in)
        dev_matrix_t<T> delta_weight = matmul(delta.const_leaf(), m_input.const_leaf().t());

        // 2. 偏置梯度 (out × 1)：沿序列维（列）累加
        dev_colvec_t<T> delta_bias = row_sum(delta.const_leaf());

        // 3. 输入梯度 —— 必须在更新权重之前算（约定 2）
        dev_matrix_t<T> dx = matmul(m_weight.const_leaf().t(), delta.const_leaf());

        m_weight_updator.update(delta_weight.const_leaf(), m_weight.buffer());
        m_bias_updator.update(delta_bias.flat_leaf(), m_bias.buffer());

        return dx;
    }

    void step()
    {
        m_weight_updator.step();
        m_bias_updator.step();
    }

    void set_lr(T lr)
    {
        m_weight_updator.set_lr(lr);
        m_bias_updator.set_lr(lr);
    }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_weight_updator.set(std::forward<arg_types>(args)...);
        m_bias_updator.set(std::forward<arg_types>(args)...);
    }

    int input_size() const { return m_in; }
    int output_size() const { return m_out; }
    dev_matrix_t<T>& weight() { return m_weight; }
    dev_colvec_t<T>& bias() { return m_bias; }

    std::string net_type(int indent = 0) const
    {
        return std::string(static_cast<std::size_t>(indent) * 4, ' ') + "dev_linear_t(in:"
               + std::to_string(m_in) + ", out:" + std::to_string(m_out) + ")";
    }

private:
    dev_matrix_t<T> m_weight;  // (out × in)
    dev_colvec_t<T> m_bias;    // (out × 1)
    dev_matrix_t<T> m_input;   // (in × T)，前向缓存
    updator_type<T> m_weight_updator;
    updator_type<T> m_bias_updator;
    int m_in = 0;
    int m_out = 0;
};

// ---------------------------------------------------------------------------
// LayerNorm / RMSNorm
// ---------------------------------------------------------------------------

/**
 * 按列标准化，对应主机 `layer_norm_net_t`。
 *
 * 前向调用 `cuda::layer_norm(...)` 并传 cache —— 于是 `hx`（不含 gamma 的归一化值）
 * 与 `std` 被留下来，反向不用重算，也不用多存一份输入。这正是主机端
 * `m_hx` / `m_std` 两个成员的作用。
 *
 * 反向调用 `cuda::layer_norm_backward(...)`，公式与主机逐字一致（含「除数是 **行数**
 * 而不是列数」这个容易看错的地方 —— 它来自前向 `vmean` 的口径）。
 */
template <typename T, template <typename> class updator_type>
class dev_layer_norm_t
{
public:
    using ele_type = T;

    static constexpr T kDefaultEps = static_cast<T>(1e-5);

    dev_layer_norm_t() = default;

    /** 显式分配 gamma/beta（gamma=1, beta=0）。权重加载前必须先调用。 */
    void set_param(int d_model, T eps = kDefaultEps)
    {
        if (d_model <= 0)
            throw std::invalid_argument("dev_layer_norm_t::set_param: d_model 必须为正");
        m_eps = eps;
        m_dim = d_model;
        m_gama = detail::colvec_from_host(std::vector<T>(static_cast<std::size_t>(d_model), T(1)));
        m_beta = detail::colvec_from_host(std::vector<T>(static_cast<std::size_t>(d_model), T(0)));
    }

    void upload_gama(const mat_t<T>& host)
    {
        if (host.row_num() != m_dim)
            throw std::invalid_argument("dev_layer_norm_t::upload_gama: 行数应为 d_model");
        m_gama = detail::colvec_from_mat(host, "dev_layer_norm_t::upload_gama");
    }

    void upload_beta(const mat_t<T>& host)
    {
        if (host.row_num() != m_dim)
            throw std::invalid_argument("dev_layer_norm_t::upload_beta: 行数应为 d_model");
        m_beta = detail::colvec_from_mat(host, "dev_layer_norm_t::upload_beta");
    }

    mat_t<T> gama_to_host() const { return m_gama.download(); }
    mat_t<T> beta_to_host() const { return m_beta.download(); }
    dev_matrix_t<T>& cached_hx() { return m_cache.hx; }
    dev_rowvec_t<T>& cached_std() { return m_cache.std; }

    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        if (m_dim <= 0)
            throw std::runtime_error("dev_layer_norm_t: 请先 set_param(d_model)");
        if (input.row_num() != m_dim)
            throw std::invalid_argument("dev_layer_norm_t::forward: 输入行数与 d_model 不一致");
        return layer_norm(input, m_gama, m_beta, m_eps, &m_cache);
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        auto d_gama = dev_colvec_t<T>(m_dim);
        auto d_beta = dev_colvec_t<T>(m_dim);

        dev_matrix_t<T> dx =
            layer_norm_backward(delta.const_leaf(), m_cache.hx, m_cache.std, m_gama, &d_gama,
                                &d_beta);

        m_gama_updator.update(d_gama.flat_leaf(), m_gama.buffer());
        m_beta_updator.update(d_beta.flat_leaf(), m_beta.buffer());
        return dx;
    }

    void step()
    {
        m_gama_updator.step();
        m_beta_updator.step();
    }

    void set_lr(T lr)
    {
        m_gama_updator.set_lr(lr);
        m_beta_updator.set_lr(lr);
    }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_gama_updator.set(std::forward<arg_types>(args)...);
        m_beta_updator.set(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        // 与主机一致：仿射参数是恒等初始化，随机初始化没有意义
    }

    std::string net_type(int indent = 0) const
    {
        return std::string(static_cast<std::size_t>(indent) * 4, ' ')
               + "dev_layer_norm_t(d_model:" + std::to_string(m_dim) + ")";
    }

private:
    layer_norm_cache_t<T> m_cache;
    dev_colvec_t<T> m_gama;
    dev_colvec_t<T> m_beta;
    updator_type<T> m_gama_updator;
    updator_type<T> m_beta_updator;
    T m_eps = kDefaultEps;
    int m_dim = 0;
};

/**
 * RMSNorm，对应主机 `rms_norm_net_t`。比 LayerNorm 少一个 beta、少一次行归约。
 *
 * 反向上也精确地「少一项」（没有 `vsum(dx_norm)` 那一项），差别就一处 —— 见
 * `cuda::rms_norm_backward` 的注释。
 */
template <typename T, template <typename> class updator_type>
class dev_rms_norm_t
{
public:
    using ele_type = T;

    static constexpr T kDefaultEps = static_cast<T>(1e-5);

    dev_rms_norm_t() = default;

    void set_param(int d_model, T eps = kDefaultEps)
    {
        if (d_model <= 0)
            throw std::invalid_argument("dev_rms_norm_t::set_param: d_model 必须为正");
        m_eps = eps;
        m_dim = d_model;
        m_gama = detail::colvec_from_host(std::vector<T>(static_cast<std::size_t>(d_model), T(1)));
    }

    void upload_gama(const mat_t<T>& host)
    {
        if (host.row_num() != m_dim)
            throw std::invalid_argument("dev_rms_norm_t::upload_gama: 行数应为 d_model");
        m_gama = detail::colvec_from_mat(host, "dev_rms_norm_t::upload_gama");
    }

    mat_t<T> gama_to_host() const { return m_gama.download(); }
    dev_matrix_t<T>& cached_hx() { return m_cache.hx; }
    dev_rowvec_t<T>& cached_rms() { return m_cache.rms; }

    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        if (m_dim <= 0)
            throw std::runtime_error("dev_rms_norm_t: 请先 set_param(d_model)");
        if (input.row_num() != m_dim)
            throw std::invalid_argument("dev_rms_norm_t::forward: 输入行数与 d_model 不一致");
        return rms_norm(input, m_gama, m_eps, &m_cache);
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        auto d_gama = dev_colvec_t<T>(m_dim);
        dev_matrix_t<T> dx =
            rms_norm_backward(delta.const_leaf(), m_cache.hx, m_cache.rms, m_gama, &d_gama);
        m_gama_updator.update(d_gama.flat_leaf(), m_gama.buffer());
        return dx;
    }

    void step() { m_gama_updator.step(); }
    void set_lr(T lr) { m_gama_updator.set_lr(lr); }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_gama_updator.set(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
    }

    std::string net_type(int indent = 0) const
    {
        return std::string(static_cast<std::size_t>(indent) * 4, ' ')
               + "dev_rms_norm_t(d_model:" + std::to_string(m_dim) + ")";
    }

private:
    rms_norm_cache_t<T> m_cache;
    dev_colvec_t<T> m_gama;
    updator_type<T> m_gama_updator;
    T m_eps = kDefaultEps;
    int m_dim = 0;
};

// ---------------------------------------------------------------------------
// 激活函数
// ---------------------------------------------------------------------------

/**
 * SiLU（Swish）：`x · sigmoid(x)`，对应主机 `silu_net_t`。
 *
 * `sigmoid` 已经是**设备可求值**的表达式节点（主机端那个 `mat_sigmoid_t` 内部就用了
 * `detail::device_exp`），所以前向可以一次融合。反向需要 sigmoid 参与三项，
 * 先把 s 物化一遍再用，免得同一个 exp 在融合树里被求值三次。
 *
 * 主机端反向是**重算** sigmoid 而不是缓存（`silu_net_t::backward` 里现算），
 * 设备端同理 —— 这里物化的只是同一次反向内部的复用。
 */
template <typename T>
class dev_silu_t
{
public:
    using ele_type = T;

    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        detail::materialize_input(input, m_input);
        dev_matrix_t<T> out(m_input.row_num(), m_input.col_num());
        if (out.row_num() == 0 || out.col_num() == 0)
            return out;
        eval_fused(m_input.leaf() * sigmoid(m_input.leaf()), out.buffer());
        return out;
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        const int rows = m_input.row_num();
        const int cols = m_input.col_num();
        dev_matrix_t<T> out(rows, cols);
        if (rows == 0 || cols == 0)
            return out;

        dev_matrix_t<T> s(rows, cols);
        eval_fused(sigmoid(m_input.leaf()), s.buffer());

        // 主机：delta * (s + m_input*s*(1-s))
        eval_fused(delta.const_leaf() * (s.leaf() + m_input.leaf() * s.leaf() * (T(1) - s.leaf())),
                   out.buffer());
        return out;
    }

    void step() {}

    /**
     * 无参数的层也要有这两个接口：`dev_chain_t` / `dev_gated_t` 这类容器会对
     * **每一个**成员统一调 `set_lr` / `set_updator`（主机端的 `silu_net_t`
     * 也是空实现），缺一个就会在容器实例化时报「没有成员」。
     */
    void set_lr(T) {}

    template <typename... arg_types>
    void set_updator(arg_types&&...)
    {
    }

    template <typename init_type>
    void init_weight()
    {
    }

    dev_matrix_t<T>& cached_input() { return m_input; }

    std::string net_type(int indent = 0) const
    {
        return std::string(static_cast<std::size_t>(indent) * 4, ' ') + "dev_silu_t";
    }

private:
    dev_matrix_t<T> m_input;
};

/**
 * 残差连接：`out = net(x) + x`，对应主机 `residual_net_t`。
 *
 * 主机前向会把输入**拷贝**一份作为 skip（`mat_t<val_type> skip(input)`），
 * 设备端也物化一次 —— 表达式只求值一遍，且反向要用它。
 */
template <typename base_net_type>
class dev_residual_t
{
public:
    using ele_type = typename base_net_type::ele_type;
    using T = ele_type;

    dev_residual_t() = default;
    explicit dev_residual_t(base_net_type net) : m_net(std::move(net)) {}

    base_net_type& net() { return m_net; }
    const base_net_type& net() const { return m_net; }
    dev_matrix_t<T>& cached_skip() { return m_skip; }

    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        detail::materialize_input(input, m_skip);
        dev_matrix_t<T> inner = m_net.forward(m_skip.const_leaf());
        dev_matrix_t<T> out(m_skip.row_num(), m_skip.col_num());
        if (out.row_num() == 0 || out.col_num() == 0)
            return out;
        eval_fused(inner.leaf() + m_skip.leaf(), out.buffer());
        return out;
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        dev_matrix_t<T> inner = m_net.backward(delta);
        dev_matrix_t<T> out(inner.row_num(), inner.col_num());
        if (out.row_num() == 0 || out.col_num() == 0)
            return out;
        // 残差把梯度**原样**送回去一路：∂(net(x)+x)/∂x 里那个恒等项
        eval_fused(inner.leaf() + delta.const_leaf(), out.buffer());
        return out;
    }

    void step() { m_net.step(); }
    void set_lr(T lr) { m_net.set_lr(lr); }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_net.set_updator(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        m_net.template init_weight<init_type>();
    }

    std::string net_type(int indent = 0) const
    {
        const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');
        return pad + "dev_residual_t(\n" + m_net.net_type(indent + 1) + "\n" + pad + ")";
    }

private:
    base_net_type m_net{};
    dev_matrix_t<T> m_skip;
};

/**
 * SwiGLU 门控：`out = gate(x) ⊙ up(x)`，对应主机 `gated_net_t`。
 *
 * 反向要把上游梯度**按元素**分给两条分支（两支共享同一个输入，所以梯度相加）：
 *
 *   ∂L/∂gate = delta ⊙ up_out
 *   ∂L/∂up   = delta ⊙ gate_out
 *
 * 所以两条分支的输出必须都被留下来 —— 这正是主机 `m_gate_out` / `m_up_out` 的用途，
 * 也是 LLaMA 的 FFN 里最大的两份激活缓存。
 */
template <typename gate_net_type, typename up_net_type>
class dev_gated_t
{
public:
    using ele_type = typename gate_net_type::ele_type;
    using T = ele_type;

    dev_gated_t() = default;

    gate_net_type& gate() { return m_gate; }
    up_net_type& up() { return m_up; }

    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        // 两支都要用同一个 x，先物化一次（同主机 gated_net_t 的注释）
        detail::materialize_input(input, m_x);
        m_gate_out = m_gate.forward(m_x.const_leaf());
        m_up_out = m_up.forward(m_x.const_leaf());

        dev_matrix_t<T> out(m_gate_out.row_num(), m_gate_out.col_num());
        if (out.row_num() == 0 || out.col_num() == 0)
            return out;
        eval_fused(m_gate_out.leaf() * m_up_out.leaf(), out.buffer());
        return out;
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        dev_matrix_t<T> d_gate(m_gate_out.row_num(), m_gate_out.col_num());
        dev_matrix_t<T> d_up(m_up_out.row_num(), m_up_out.col_num());

        if (d_gate.row_num() > 0 && d_gate.col_num() > 0)
        {
            eval_fused(delta.const_leaf() * m_up_out.leaf(), d_gate.buffer());
            eval_fused(delta.const_leaf() * m_gate_out.leaf(), d_up.buffer());
        }

        dev_matrix_t<T> a = m_gate.backward(d_gate);
        dev_matrix_t<T> b = m_up.backward(d_up);
        if (a.row_num() == 0 || a.col_num() == 0)
            return b;
        if (b.row_num() == 0 || b.col_num() == 0)
            return a;

        dev_matrix_t<T> out(a.row_num(), a.col_num());
        eval_fused(a.leaf() + b.leaf(), out.buffer());
        return out;
    }

    void step()
    {
        m_gate.step();
        m_up.step();
    }

    void set_lr(T lr)
    {
        m_gate.set_lr(lr);
        m_up.set_lr(lr);
    }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_gate.set_updator(std::forward<arg_types>(args)...);
        m_up.set_updator(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        m_gate.template init_weight<init_type>();
        m_up.template init_weight<init_type>();
    }

    std::string net_type(int indent = 0) const
    {
        const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');
        return pad + "dev_gated_t(\n" + m_gate.net_type(indent + 1) + "\n"
               + m_up.net_type(indent + 1) + "\n" + pad + ")";
    }

private:
    gate_net_type m_gate{};
    up_net_type m_up{};
    dev_matrix_t<T> m_x;         // (d_model × T)
    dev_matrix_t<T> m_gate_out;  // (d_ff × T)
    dev_matrix_t<T> m_up_out;    // (d_ff × T)
};

/**
 * 两级串联：`out = second(first(x))`，对应主机 `complex_net_t` 那种嵌套。
 *
 * 存在的理由很具体：SwiGLU 的 gate 分支是 `silu(linear(x))` —— 主机端用
 * `gated_ffn_branches_t<..., silu_net_t>` 表达，设备端需要一个「线性层后面接一个
 * 激活层」的容器才能塞进 `dev_gated_t`。写死在 llama 里也行，但串联本身是个
 * 通用形状（GELU / ReLU 变体同理），放在这里更省重复。
 *
 * 反向按**相反顺序**走：先激活层的反向、再线性层。中间的 `h` 必须活过第二次调用 ——
 * `first.forward(x)` 的返回值是 `dev_matrix_t`，`second.forward` 内部会先物化再算，
 * 所以这里没有悬空引用的问题。
 */
template <typename first_net_type, typename second_net_type>
class dev_chain_t
{
public:
    using ele_type = typename first_net_type::ele_type;
    using T = ele_type;

    dev_chain_t() = default;

    first_net_type& first() { return m_first; }
    const first_net_type& first() const { return m_first; }
    second_net_type& second() { return m_second; }
    const second_net_type& second() const { return m_second; }

    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        dev_matrix_t<T> h = m_first.forward(input);
        return m_second.forward(h.const_leaf());
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        dev_matrix_t<T> d = m_second.backward(delta);
        return m_first.backward(d);
    }

    void step()
    {
        m_first.step();
        m_second.step();
    }

    void set_lr(T lr)
    {
        m_first.set_lr(lr);
        m_second.set_lr(lr);
    }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_first.set_updator(std::forward<arg_types>(args)...);
        m_second.set_updator(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        m_first.template init_weight<init_type>();
        m_second.template init_weight<init_type>();
    }

    std::string net_type(int indent = 0) const
    {
        const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');
        return pad + "dev_chain_t(\n" + m_first.net_type(indent + 1) + "\n"
               + m_second.net_type(indent + 1) + "\n" + pad + ")";
    }

private:
    first_net_type m_first{};
    second_net_type m_second{};
};

// ---------------------------------------------------------------------------
// 损失
// ---------------------------------------------------------------------------

/**
 * MSE，对应主机 `mse_loss_t`。**缩放因子必须与主机完全一致**，否则训练动力学不同：
 *
 *   loss:     mean((y - target)²)
 *   backward: y - target          ← 注意**没有** 2/N 因子，这是主机端的既有语义
 *
 * `mat_loss_t::forward` 是透传（返回输入本身），这里同样透传 —— 但设备端要返回
 * 一个拥有者矩阵，所以会拷一份（每步一次，可以接受）。
 */
template <typename T>
class dev_mse_loss_t
{
public:
    using ele_type = T;

    template <typename Src>
    dev_matrix_t<T> forward(const Src& input)
    {
        detail::materialize_input(input, m_input);
        dev_matrix_t<T> out(m_input.row_num(), m_input.col_num());
        if (out.row_num() == 0 || out.col_num() == 0)
            return out;
        eval_fused(m_input.leaf(), out.buffer());
        return out;
    }

    dev_matrix_t<T> backward(const dev_matrix_t<T>& target) const
    {
        dev_matrix_t<T> out(m_input.row_num(), m_input.col_num());
        if (out.row_num() == 0 || out.col_num() == 0)
            return out;
        eval_fused(m_input.const_leaf() - target.const_leaf(), out.buffer());
        return out;
    }

    /** 与主机 `mse_loss_t::loss` 同值：`mean((y - target)²)`。这一步会同步到主机。 */
    T loss(const dev_matrix_t<T>& target) const
    {
        const int n = m_input.row_num() * m_input.col_num();
        if (n == 0)
            return T(0);
        // 主机是 mean(pow(input - target, 2.0)) —— 差值的**平方**，不是差值本身。
        // 少了这个平方，损失会因正负相消而偏小（而且仍有下降趋势，很难从曲线看出来）。
        auto diff = m_input.const_leaf() - target.const_leaf();
        const T s = sum_all(diff * diff);
        return s / static_cast<T>(n);
    }

    void step() {}

    template <typename init_type>
    void init_weight()
    {
    }

    dev_matrix_t<T>& cached_input() { return m_input; }

    std::string net_type(int indent = 0) const
    {
        return std::string(static_cast<std::size_t>(indent) * 4, ' ') + "dev_mse_loss_t";
    }

private:
    dev_matrix_t<T> m_input;
};

} // namespace cuda
} // namespace jasmine

#endif
