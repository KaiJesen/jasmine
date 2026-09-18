#ifndef __JAS_RBM_T_HPP__
#define __JAS_RBM_T_HPP__

#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_mat_storage.hpp"
#include "jas_loss_t.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"

namespace jasmine {

/**
 * RBM（受限玻尔兹曼机，Bernoulli-Bernoulli）。
 *
 *     能量    E(v,h) = -bᵀv - cᵀh - vᵀWᵀh          W: [n_hidden, n_visible]
 *     条件概率 P(h=1|v) = σ(W v + c)               P(v=1|h) = σ(Wᵀ h + b)
 *     自由能   F(v) = -bᵀv - Σ_j log(1 + exp(c_j + (Wv)_j))
 *
 * **本类同时扮演两种角色，这是理解它的关键**：
 *
 * 1. 作为**层**（静态堆叠用）：`forward` 返回隐层概率 σ(Wv+c)，`backward` 按
 *    「线性层 + sigmoid」求导并把梯度交给 updator。DBN 的监督微调走的就是这条路：
 *    RBM 堆起来 = 一个 sigmoid 前馈网络，反向传播照常。
 * 2. 作为**RBM 自己**：`contrastive_divergence(v0, k)` 做 CD-k 无监督训练（逐层贪心预训练用）。
 *    它把「负的 CD 增量」当作梯度交给同一套 updator，因此照样能和 `cache_updator_t`
 *    组合成 mini-batch（逐样本累计 → step() 落地），与有监督路径共用 lr/正则设置。
 *
 * 尺寸约定：一次处理**一个样本一列**（与 conv/pool 一致），批处理走 updator 的梯度累加。
 */
template <typename input_type, template <typename> class updator_type>
class rbm_net_t
{
public:
    using val_type = typename input_type::ele_type;

private:
    mat_t<val_type> m_weight;       // [n_hidden, n_visible]
    mat_t<val_type> m_vis_bias;     // [n_visible, 1]
    mat_t<val_type> m_hid_bias;     // [n_hidden, 1]
    updator_type<val_type> m_weight_updator;
    updator_type<val_type> m_vis_updator;
    updator_type<val_type> m_hid_updator;

    mat_t<val_type> m_input;        // forward 缓存（v）
    mat_t<val_type> m_hidden;       // forward 缓存（h 的概率）

    /** 梯度形状必须和目标参数一致（否则 updator 会静默 resize 参数） */
    static void check_grad_shape(mat_t<val_type> const& grad, mat_t<val_type> const& param,
                                 char const* what)
    {
        if (grad.row_num() != param.row_num() || grad.col_num() != param.col_num())
            throw std::runtime_error(std::string("rbm_net_t: gradient shape mismatch for ") + what
                + " (grad " + std::to_string(grad.row_num()) + "x" + std::to_string(grad.col_num())
                + " vs param " + std::to_string(param.row_num()) + "x"
                + std::to_string(param.col_num()) + ")");
    }

    /** 按概率二值采样（Bernoulli） */
    mat_t<val_type> bernoulli(mat_t<val_type> const& probs)
    {
        mat_t<val_type> out(probs.row_num(), probs.col_num());
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int i = 0; i < probs.row_num(); ++i)
            for (int j = 0; j < probs.col_num(); ++j)
                out(i, j) = (uni(g_random_engine) < static_cast<double>(probs(i, j)))
                    ? val_type(1) : val_type(0);
        return out;
    }

public:
    rbm_net_t() = default;

    rbm_net_t(int const& n_visible, int const& n_hidden)
    {
        set_param(n_visible, n_hidden);
    }

    /**
     * 配置尺寸并分配 W / b / c（都置 0，之后用 init_weight<init_type>() 填充）。
     * 与 layer_norm / conv 一样给的是 set_param；同时也提供 reinit({n_visible, n_hidden})，
     * 因为 RBM 带权重，应该参与 complex_net_t::reinit 的容器协议（每个 RBM 消费一对数）。
     */
    void set_param(int const& n_visible, int const& n_hidden)
    {
        if (n_visible < 1 || n_hidden < 1)
            throw std::invalid_argument("rbm_net_t: n_visible/n_hidden must be >= 1");
        m_n_visible = n_visible;
        m_n_hidden = n_hidden;
        m_weight = mat_t<val_type>(n_hidden, n_visible);
        m_vis_bias = mat_t<val_type>(n_visible, 1);
        m_hid_bias = mat_t<val_type>(n_hidden, 1);
        m_weight = val_type(0);
        m_vis_bias = val_type(0);
        m_hid_bias = val_type(0);
    }

    /** 容器协议：{n_visible, n_hidden}（与 weight_net_t 的 {in, out} 同形） */
    void reinit(std::vector<int> const& container)
    {
        if (container.size() < 2)
            throw std::invalid_argument("rbm_net_t::reinit needs {n_visible, n_hidden}");
        set_param(container[0], container[1]);
    }

    int n_visible() const { return m_n_visible; }
    int n_hidden() const { return m_n_hidden; }
    bool valid() const { return m_weight.valid(); }

    /** 权重 [n_hidden, n_visible]；供权重加载器写入 */
    mat_t<val_type>& weight() { return m_weight; }
    mat_t<val_type> const& weight() const { return m_weight; }
    /** 可见层偏置 [n_visible, 1] */
    mat_t<val_type>& visible_bias() { return m_vis_bias; }
    mat_t<val_type> const& visible_bias() const { return m_vis_bias; }
    /** 隐层偏置 [n_hidden, 1] */
    mat_t<val_type>& hidden_bias() { return m_hid_bias; }
    mat_t<val_type> const& hidden_bias() const { return m_hid_bias; }

    // ---------------------------------------------------------------- 概率与采样

    /** P(h=1|v) = σ(Wv + c)，形状 [n_hidden, T] */
    template <typename Src>
    mat_t<val_type> hidden_prob(Src&& v) const
    {
        return mat_t<val_type>(sigmoid(m_weight.dot(v) + m_hid_bias));
    }

    /**
     * P(v=1|h) = σ(Wᵀh + b)，形状 [n_visible, T]。
     *
     * 这里用显式三重循环而不是 `m_weight.t().dot(h)`：`mat_t::t()` 目前只有非 const 版本，
     * 而本函数应当是 const 的（不改状态）；自己累加既保持 const，也避免每次调用构造视图。
     */
    template <typename Src>
    mat_t<val_type> visible_prob(Src&& h) const
    {
        mat_t<val_type> const sh = mat_t<val_type>(std::forward<Src>(h));
        if (sh.row_num() != m_n_hidden)
            throw std::invalid_argument("rbm_net_t::visible_prob: h rows must equal n_hidden");
        mat_t<val_type> out(m_n_visible, sh.col_num());
        for (int i = 0; i < m_n_visible; ++i)
        {
            for (int t = 0; t < sh.col_num(); ++t)
            {
                val_type a = m_vis_bias(i, 0);
                for (int j = 0; j < m_n_hidden; ++j)
                    a += m_weight(j, i) * sh(j, t);
                out(i, t) = val_type(1) / (val_type(1) + std::exp(-a));
            }
        }
        return out;
    }

    /** 按概率采样隐层（Bernoulli） */
    template <typename Src>
    mat_t<val_type> sample_hidden(Src&& v)
    {
        return bernoulli(hidden_prob(std::forward<Src>(v)));
    }

    /** 按概率采样可见层 */
    template <typename Src>
    mat_t<val_type> sample_visible(Src&& h)
    {
        return bernoulli(visible_prob(std::forward<Src>(h)));
    }

    /** 一次重构（v → h 的概率 → v 的概率），用于观察重建质量 */
    template <typename Src>
    mat_t<val_type> reconstruct(Src&& v) const
    {
        return visible_prob(hidden_prob(std::forward<Src>(v)));
    }

    /** 自由能 F(v) = -bᵀv - Σ log(1+exp(c + Wv))，形状 [1, T]（T 列各自一个值） */
    template <typename Src>
    mat_t<val_type> free_energy(Src&& v) const
    {
        mat_t<val_type> const sv = mat_t<val_type>(std::forward<Src>(v));
        const int T = sv.col_num();
        mat_t<val_type> out(1, T);
        for (int t = 0; t < T; ++t)
        {
            val_type f = val_type(0);
            for (int i = 0; i < m_n_visible; ++i) f -= m_vis_bias(i, 0) * sv(i, t);
            for (int j = 0; j < m_n_hidden; ++j)
            {
                val_type a = m_hid_bias(j, 0);
                for (int i = 0; i < m_n_visible; ++i) a += m_weight(j, i) * sv(i, t);
                f -= std::log1p(std::exp(a));            // log(1+e^a)，用 log1p 保精度
            }
            out(0, t) = f;
        }
        return out;
    }

    // ---------------------------------------------------------------- 层语义（静态堆叠 / 监督微调）

    /** 作为层的前向：返回隐层概率（确定性），缓存 v 与 h 供反向 */
    template <typename Src>
    mat_t<val_type> forward(Src&& v)
    {
        detail::store_for_backward(m_input, std::forward<Src>(v));
        if (m_input.row_num() != m_n_visible)
            throw std::invalid_argument("rbm_net_t::forward: input rows must equal n_visible");
        m_hidden = hidden_prob(m_input);
        return m_hidden;
    }

    /** 无状态层：单列输入与整段 forward 相同 */
    template <typename Src>
    mat_t<val_type> forward_one(Src&& v)
    {
        return forward(std::forward<Src>(v));
    }

    /**
     * 作为层反向（监督微调）：把 RBM 当成「线性层 + sigmoid」求导
     *     g   = delta ⊙ h ⊙ (1-h)
     *     ∂L/∂W = g · vᵀ        ∂L/∂c = Σ_列 g        ∂L/∂v = Wᵀ g
     * W 与 c 走 updator 更新（与 weight_net_t 的语义一致），返回输入的梯度 ∂L/∂v。
     *
     * **可见偏置 b 在这里不更新**：层的正向 h = σ(Wv + c) 根本不经过 b，所以 ∂L/∂b = 0；
     * b 只属于生成方向 P(v|h)，由 `contrastive_divergence` 负责训练。把它留在这里不更新，
     * 也是 DBN 判别式微调的标准做法（微调阶段 RBM 就是一个 sigmoid 层）。
     */
    template <typename other_type>
    mat_t<val_type> backward(const other_type& delta)
    {
        if (delta.row_num() != m_n_hidden || delta.col_num() != m_hidden.col_num())
            throw std::runtime_error("rbm_net_t::backward: delta shape mismatch");

        mat_t<val_type> g(m_n_hidden, m_hidden.col_num());
        for (int j = 0; j < m_n_hidden; ++j)
            for (int t = 0; t < m_hidden.col_num(); ++t)
            {
                const val_type h = m_hidden(j, t);
                g(j, t) = delta(j, t) * h * (val_type(1) - h);
            }

        mat_t<val_type> const delta_weight = g.dot(m_input.t());
        mat_t<val_type> const delta_hid = hsum(g);
        mat_t<val_type> const delta_vis = m_weight.t().dot(g);

        check_grad_shape(delta_weight, m_weight, "weight");
        check_grad_shape(delta_hid, m_hid_bias, "hidden_bias");

        m_weight_updator.update(delta_weight, m_weight);
        m_hid_updator.update(delta_hid, m_hid_bias);
        // 注意：不更新可见偏置 b（∂L/∂b = 0，b 只服务于 P(v|h)，见上面的说明）
        return delta_vis;
    }

    // ---------------------------------------------------------------- CD-k（无监督预训练）

    /**
     * 对比散度 CD-k：用 v0 做一步正相位、k 步 Gibbs 采样做负相位，然后把
     *     -(v0 h0ᵀ - vk hkᵀ),  -(v0 - vk),  -(h0 - hk)
     * 当作梯度交给 updator（负号是因为 updator 做 p ← p - lr·g）。
     *
     * `sample = false` 时全程用概率（mean-field / 确定性 CD），便于测试与复现；
     * 默认 `sample = true` 走标准 CD（Bernoulli 采样）。
     *
     * 返回本次的重建误差 mean|v0 - vk|（1×1 矩阵），便于监控预训练是否在工作。
     */
    template <typename Src>
    mat_t<val_type> contrastive_divergence(Src&& v, int const& k = 1, bool const& sample = true)
    {
        mat_t<val_type> v0 = mat_t<val_type>(std::forward<Src>(v));
        if (v0.row_num() != m_n_visible)
            throw std::invalid_argument("rbm_net_t::contrastive_divergence: v rows must equal n_visible");
        if (k < 1) throw std::invalid_argument("rbm_net_t::contrastive_divergence: k must be >= 1");

        mat_t<val_type> h0 = hidden_prob(v0);                // 正相位
        mat_t<val_type> vk = v0;
        mat_t<val_type> hk = h0;
        for (int step = 0; step < k; ++step)
        {
            mat_t<val_type> const vp = visible_prob(hk);
            vk = sample ? bernoulli(vp) : vp;
            mat_t<val_type> const hp = hidden_prob(vk);
            hk = sample ? bernoulli(hp) : hp;
        }

        // updator 做的是 p ← p - lr·g，所以这里传「负的 CD 增量」。
        // 注意方向：W 是 [n_hidden, n_visible]，CD 的规则是
        //     ΔW_CD = η (v0 h0ᵀ - vk hkᵀ)   （形状 [n_visible, n_hidden]）
        // 所以给 updator 的「负增量」必须写成 hk·vkᵀ - h0·v0ᵀ，正好是 [n_hidden, n_visible]。
        // 表达式模板没有一元负号，用「被减数写在前面」表达负号。
        mat_t<val_type> const delta_weight = hk.dot(vk.t()) - h0.dot(v0.t());
        mat_t<val_type> const delta_vis = hsum(vk) - hsum(v0);
        mat_t<val_type> const delta_hid = hsum(hk) - hsum(h0);

        // 形状护栏：updator 的 update 只按 grad 的形状工作，形状不符会静默把参数 resize 掉，
        // 这类错误在训练里很难看出来，所以在交给它之前先挡住。
        check_grad_shape(delta_weight, m_weight, "weight");
        check_grad_shape(delta_vis, m_vis_bias, "visible_bias");
        check_grad_shape(delta_hid, m_hid_bias, "hidden_bias");

        m_weight_updator.update(delta_weight, m_weight);
        m_vis_updator.update(delta_vis, m_vis_bias);
        m_hid_updator.update(delta_hid, m_hid_bias);

        mat_t<val_type> err(1, 1);
        val_type e = val_type(0);
        for (int i = 0; i < m_n_visible; ++i)
            for (int t = 0; t < v0.col_num(); ++t)
                e += std::abs(v0(i, t) - vk(i, t));
        err(0, 0) = e / static_cast<val_type>(v0.row_num() * v0.col_num());
        return err;
    }

    /** 用 W 的量级做一个温和的默认初始化（RBM 常用小随机数） */
    template <typename init_type>
    void init_weight()
    {
        init_matrix<init_type>(m_weight);
        m_vis_bias = val_type(0);
        m_hid_bias = val_type(0);
    }

    template <typename... upr_arg_types>
    void set_updator(upr_arg_types&&... args)
    {
        m_weight_updator.set(std::forward<upr_arg_types>(args)...);
        m_vis_updator.set(std::forward<upr_arg_types>(args)...);
        m_hid_updator.set(std::forward<upr_arg_types>(args)...);
    }

    void set_lr(val_type lr)
    {
        m_weight_updator.set_lr(lr);
        m_vis_updator.set_lr(lr);
        m_hid_updator.set_lr(lr);
    }

    void step()
    {
        m_weight_updator.step();
        m_vis_updator.step();
        m_hid_updator.step();
    }

    std::string net_type(int const& indent = 0) const
    {
        std::stringstream ss;
        ss << print_indent(indent) << "rbm_net_t:(visible:" << m_n_visible
           << ", hidden:" << m_n_hidden << ")";
        return ss.str();
    }

private:
    int m_n_visible = 0;
    int m_n_hidden = 0;
};

// ---------------------------------------------------------------------------------------------
// DBN：把 N 个 RBM 用静态层堆叠接起来，最后接分类头 + CE
// ---------------------------------------------------------------------------------------------

/** 把同一个「可更新层」重复压进 builder N 次的辅助模板（builder 本身是变参 pack） */
template <typename builder, int n, template <typename, template <typename> class> class net_tpl,
          template <typename> class updator_tpl>
struct push_back_n_updatable
{
    using next_builder = typename builder::template push_back_updatable<net_tpl, updator_tpl>;
    using type = typename push_back_n_updatable<next_builder, n - 1, net_tpl, updator_tpl>::type;
};

template <typename builder, template <typename, template <typename> class> class net_tpl,
          template <typename> class updator_tpl>
struct push_back_n_updatable<builder, 0, net_tpl, updator_tpl>
{
    using type = builder;
};

/**
 * DBN 链：`n_rbm` 个 RBM → 分类头（weight_net）→ CE。
 *
 * - 无监督预训练：`dbn_pretrain<net_type>(net, data, n_rbm, cd_k, epochs)`，逐层贪心（第 i 层吃
 *   第 i-1 层的隐层概率），每层用 CD-k 更新，并且**顺序与堆叠顺序一致**；
 * - 监督微调：整个链直接 `forward` / `backward` / `step`（RBM 在链里就是「线性+sigmoid」层）；
 * - 容器协议：`net.reinit({n_visible, n_h1, n_h2, ..., n_class})` —— 每个 RBM 消费一对数，
 *   分类头消费最后一对（与 complex_net_t::reinit 的规则一致）。
 */
template <int n_rbm, template <typename> class updator_tpl>
using dbn_net_t = typename push_back_n_updatable<
        complex_net_builder_t<double>, n_rbm, rbm_net_t, updator_tpl>
    ::type
    ::template push_back_updatable<weight_net_t, updator_tpl>
    ::template push_back_staticnet<ce_loss_t>
    ::type;

/** 取矩阵的某一列（[n,1]），预训练时逐样本喂给 RBM */
template <typename val_type>
mat_t<val_type> column_as_vector(mat_t<val_type> const& m, int const& col)
{
    mat_t<val_type> out(m.row_num(), 1);
    for (int i = 0; i < m.row_num(); ++i) out(i, 0) = m(i, col);
    return out;
}

/** 逐层贪心预训练的实现：<当前层号, 层数> */
template <int I, int N, typename net_type, typename val_type>
void dbn_pretrain_impl(net_type& net, mat_t<val_type> const& input, int const& cd_k,
                       int const& epochs, val_type& last_recon)
{
    if constexpr (I < N)
    {
        auto& rbm = net.template get<I>();
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (int b = 0; b < input.col_num(); ++b)
            {
                const mat_t<val_type> err = rbm.contrastive_divergence(column_as_vector(input, b), cd_k);
                last_recon = err(0, 0);
            }
            rbm.step();                 // 梯度累加器落地（mini-batch = 整个输入矩阵）
        }
        // 本层的隐层概率作为下一层的输入
        const mat_t<val_type> hidden = rbm.forward(input);
        dbn_pretrain_impl<I + 1, N>(net, hidden, cd_k, epochs, last_recon);
    }
}

/**
 * 逐层贪心预训练 DBN 的所有 RBM（不动分类头）。
 * 返回最后一层最后一次 CD 的重建误差，便于观察是否在收敛。
 */
template <int N, typename net_type, typename val_type>
val_type dbn_pretrain(net_type& net, mat_t<val_type> const& data, int const& cd_k = 1,
                      int const& epochs = 1)
{
    val_type recon = val_type(0);
    dbn_pretrain_impl<0, N>(net, data, cd_k, epochs, recon);
    return recon;
}

} // namespace jasmine
#endif
