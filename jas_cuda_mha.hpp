#ifndef __JAS_CUDA_MHA_HPP__
#define __JAS_CUDA_MHA_HPP__

/**
 * 设备端多头注意力：`jas_mha_t.hpp` 的 GPU 对应物，带 forward + backward。
 *
 * ## 结构照搬主机
 *
 * 主机端把注意力拆成两层，这里一一对应：
 *
 *   - `mat_head_gen_t` —— **单头**核：已经切好的 Q/K/V（d_head × seq）进来，
 *     做 RoPE(Q/K)、打分、因果掩码、softmax、加权求和。不含任何投影权重。
 *   - `mat_mha_t`      —— **多头**外壳：Q/K/V/O 四个全维投影 + 按头切分 + 拼接。
 *     GQA 体现在「K/V 只投影出 n_kv_heads 个头」以及「多个 Q 头共享一个 KV 头」。
 *
 * 设备端保持同样的分层，因为这套切分不是历史包袱：单头核完全不知道 GQA 的存在，
 * 而多头外壳负责的正是 GQA 唯一需要小心的地方 —— 反向时**共享同一 KV 头的多个
 * Q 头，其 K/V 梯度必须累加**（主机端 `add_rows` 那一步）。设备端同序同语义。
 *
 * ## 与主机端的接口差别
 *
 *   - 主机 `forward(q,k,v)` 里 Q/K 的 RoPE 起点固定在 0（训练路径）。
 *     设备端 `forward_at` 多两个 `q_pos` / `k_pos` 参数（默认 0，与主机同），
 *     并且**反向记住前向用过的起点**。主机 `mat_head_gen_t::backward` 把列下标直接
 *     当绝对位置（隐含起点 0）—— 训练路径恰好就是起点 0，所以两者在实际用到的
 *     路径上逐位一致；设备端顺手把起点非 0 的情形也做对了。
 *
 *   - 缩放 `1/sqrt(d_head)` 用**逐元素除法**而不是折进 GEMM 的 `alpha`。
 *     `x * (1/s)` 与 `x / s` 不是同一个浮点数，折进 alpha 会与主机端产生
 *     ulp 级差异（`attend_cached` 就是为了省一趟缩放才折 alpha 的，那是另一回事：
 *     它只做推理、不对拍主机）。这里的首要目标是**与主机逐元素对齐**，
 *     多一趟小矩阵的读写换一个可复现的对拍基准，划算。
 *
 *   - `forward_one`（decode）把 K 送进 cache 之前要**逐 KV 头**旋转，而不是整块旋转：
 *     完整 K 投影有 `n_kv_heads × d_head` 行，旋转表却只有 `d_head` 维，
 *     整块旋转会把**相邻两个头的前半行配成一对**。单 KV 头时两者恰好等价，
 *     所以这个 bug 只在 GQA/MQA 上现形。训练路径没有这个问题：
 *     那里的旋转发生在单头核内部，天然是按头做的。
 *
 * ## 内存与 launch
 *
 * 反向是逐头串行做 GEMM 的（每个头 3~4 次）。主机端有 OpenMP 并行处理各头，
 * 设备端这里先把 `cuBLAS` 的 stream 语义与正确性钉住，并行留到性能那一轮 ——
 * 与 `atten_cached` 的选择一致（见 CUDA.md 第 9.5 节的说明）。
 *
 * 本头依赖 CUDA 运行时，只在 CUDA 构建里用。
 */

#include <cstddef>
#include <limits>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_compat.hpp"
#include "jas_cuda_fused.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_attention.hpp"
#include "jas_cuda_kv_cache.hpp"
#include "jas_cuda_leaf.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_net.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_rope.hpp"
#include "jas_cuda_updator.hpp"
#include "jas_mat_express_t.hpp"

namespace jasmine {
namespace cuda {

namespace detail {

/**
 * 因果掩码：把 (rows × cols) 中**列 j > 行 i + offset** 的元素就地置成 `-inf`。
 *
 * 语义与主机端两处掩码完全一致，只是把 `offset` 显式化：
 *   - 训练 / 预填（`mat_head_gen_t::forward`）：`offset = 0`，合法 ⟺ `j <= i`；
 *   - 增量解码（`attend_cached`）：`offset = pos`（本步首列的绝对位置），
 *     合法 ⟺ `j <= pos + i`，因为第 i 行的绝对位置就是 `pos + i`。
 *
 * 做成**结构化 kernel** 而不是物化一张 (q_len × k_len) 的掩码矩阵：掩码矩阵会在
 * 每个头上各存一份，`num_heads × T²` 的开销在 TinyLlama 那种规模（32 头 × 2048²）
 * 是若干 GB —— 为了一个二值约束去花这个内存，明显不值。
 */
template <typename T>
__global__ void causal_mask_inplace_kernel(T* __restrict__ data, int rows, int cols, int ld,
                                          int offset)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols)
        return;
    const int i = idx / cols;
    const int j = idx - i * cols;
    if (j > i + offset)
        data[static_cast<std::ptrdiff_t>(i) * ld + j] = -std::numeric_limits<T>::infinity();
}

/**
 * 反向的对应物：把被掩码位置的**梯度**清零。
 *
 * 与主机 `mat_head_gen_t::backward` 里那段 `delta_qt_k(i, j) = 0` 严格对应。
 * 前向用 `-inf`、反向用 `0` 是同一件事的两面：被屏蔽的位置对 loss 没有贡献，
 * 所以它既不该产生输出，也不该收到梯度。
 */
template <typename T>
__global__ void zero_future_inplace_kernel(T* __restrict__ data, int rows, int cols, int ld,
                                           int offset)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols)
        return;
    const int i = idx / cols;
    const int j = idx - i * cols;
    if (j > i + offset)
        data[static_cast<std::ptrdiff_t>(i) * ld + j] = T(0);
}

template <typename T, bool Zero>
void mask_inplace(dev_matrix_t<T>& m, int offset)
{
    if (m.row_num() <= 0 || m.col_num() <= 0 || m.buffer().empty())
        return;
    const int rows = m.row_num();
    const int cols = m.col_num();
    const int total = rows * cols;
    constexpr int kThreads = 256;
    const int blocks = (total + kThreads - 1) / kThreads;
    if constexpr (Zero)
        zero_future_inplace_kernel<T><<<blocks, kThreads>>>(m.buffer().data(), rows, cols, cols,
                                                            offset);
    else
        causal_mask_inplace_kernel<T><<<blocks, kThreads>>>(m.buffer().data(), rows, cols, cols,
                                                            offset);
    JAS_CUDA_CHECK(cudaGetLastError());
}

/** 就地 `m /= s`。逐元素除法 —— 见文件头「为什么不折进 GEMM 的 alpha」。 */
template <typename T>
void divide_inplace(dev_matrix_t<T>& m, T s)
{
    if (m.row_num() <= 0 || m.col_num() <= 0)
        return;
    eval_fused(m.leaf() / s, m.buffer());
}

/**
 * 把 src 整块放到 dst 的第 `[row0, row0 + src.row_num())` 行。
 *
 * `Accumulate == false` 是 assign（覆盖），`true` 是 `+=`。
 * 两者对应主机端 `vsplit` 出来的视图上做的 `assign` 与 `add_rows`：
 * 拼多头输出、把 delta_concat 切回各头用 assign；**GQA 下多个 Q 头往同一个
 * KV 头累加 dk/dv 用 `+=`**，少一个加号就会静默只剩最后一个头的梯度。
 *
 * 要求 dst 是连续的 `dev_matrix_t`（ld == col_num），它本来就是。
 */
template <typename T, bool Accumulate>
__global__ void place_rows_kernel(dev_mat_t<T> src, T* __restrict__ dst, int dst_ld, int row0)
{
    const int cols = src.col_num();
    const int total = src.row_num() * cols;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int i = idx / cols;
    const int j = idx - i * cols;
    T* at = dst + static_cast<std::ptrdiff_t>(row0 + i) * dst_ld + j;
    if constexpr (Accumulate)
        *at += src(i, j);
    else
        *at = src(i, j);
}

template <bool Accumulate, typename T>
void place_rows(const dev_mat_t<T>& src, dev_matrix_t<T>& dst, int row0)
{
    if (src.row_num() <= 0 || src.col_num() <= 0)
        return;
    if (src.col_num() != dst.col_num() || row0 + src.row_num() > dst.row_num())
        throw std::invalid_argument("place_rows: 形状不匹配");

    const int total = src.row_num() * src.col_num();
    constexpr int kThreads = 256;
    const int blocks = (total + kThreads - 1) / kThreads;
    place_rows_kernel<T, Accumulate><<<blocks, kThreads>>>(src, dst.buffer().data(), dst.col_num(),
                                                           row0);
    JAS_CUDA_CHECK(cudaGetLastError());
}

/** 面向「按值返回的 bwd_pack_t」的按行拼接（避免多一次拷贝）。 */
template <typename T>
void place_rows_owned(const dev_matrix_t<T>& src, dev_matrix_t<T>& dst, int row0)
{
    place_rows<false>(src.const_leaf(), dst, row0);
}

} // namespace detail

// ---------------------------------------------------------------------------
// 单头注意力核：mat_head_gen_t 的设备对应物
// ---------------------------------------------------------------------------

/**
 * 输入已是切好的 Q/K/V（d_head × seq），本类**不含任何投影权重**。
 *
 * ## 概率矩阵为什么一直是"必须留下"的
 *
 * 前向本来要留下三样东西给反向：`m_q`、`m_k`（已旋转）、`m_v`（未旋转），
 * 以及 `m_weights` —— softmax 的概率矩阵。前两个是输入本身，第三个是**输出**。
 *
 * 留着它的理由是 softmax 反向的公式 `dx = w ⊙ (dy − row_sum(w ⊙ dy))` 要用 `w`，
 * 而一个通用 softmax 层确实重算不出来：它的分母是「先减最大值再求指数和」得到的
 * 统计量，前向算完就丢了，重算等于把前向整个再走一遍。
 *
 * 但**注意力场合不一样**：这里的 `w` 有闭式表达
 *
 *   w_ij = exp(s_ij − L_i)，  s_ij = (Qᵀ·K)_ij / √d，  L_i = rowmax(s_i) + log(rowsum exp(s_i − rowmax))
 *
 * `s` 由 Q/K 重算（它们本来就留着），而分母那一整套统计量**可以压成一个数 L**
 * （logsumexp）。所以注意力既能重算 `w`，又只需要每行留一个数 —— 这正是
 * `jas_cuda_attention.hpp` 的全部立足点。
 *
 * ## 两条引擎
 *
 * 于是本类有两个前向/反向实现，`set_fused_attention(true)` 切到融合那条：
 *
 * | | 前向为反向留下的东西 | 反向的 softmax 环节 |
 * | --- | --- | --- |
 * | 非融合（默认） | `m_weights`：`q_len × k_len` 个数 | 直接读它 |
 * | 融合 | `m_lse`：`q_len` 个数 | 从 `m_lse` 重算 |
 *
 * 默认仍然是**非融合**：融合是加速路径，不是替代品，而且它只在 `q_len × k_len`
 * 真的成规模时才划算（decode 时 `q_len == 1`，概率矩阵是 `1 × len`，融合只会白绕一圈）。
 * 两条都留着还有一个测试上的理由：融合路径不能因为「只在特定规模下被选中」而腐烂，
 * 测试必须能把同一份输入两条都跑一遍再对拍（`softmax_rows` 的三趟回退路径吃过这个亏，
 * 见 CUDA.md 5.6）。
 *
 * `m_engine` 记住前向**实际**用了哪条：反向必须与前向配对，中途换开关会让缓存与
 * 引擎对不上，所以这里直接抛异常而不是静默出错。
 */
template <typename T>
class dev_head_gen_t
{
public:
    using ele_type = T;

    struct bwd_pack_t
    {
        dev_matrix_t<T> delta_q;
        dev_matrix_t<T> delta_k;
        dev_matrix_t<T> delta_v;
    };

    dev_head_gen_t() = default;
    explicit dev_head_gen_t(int d_head, bool mask = false) { set_param(d_head, mask); }

    void set_param(int d_head, bool mask = false)
    {
        if (d_head <= 0)
            throw std::invalid_argument("dev_head_gen_t: d_head 必须为正");
        m_d_head = d_head;
        m_mask = mask;
    }

    /**
     * 绑 RoPE。**不拥有**，必须活得比本对象久（实际由模型持有，见 dev_mha_t）。
     * 传 `nullptr` 表示不做 RoPE —— 绝对位置模型（GPT-2）走这条路。
     */
    void set_rope(dev_rope_t<T>* rope)
    {
        if (rope != nullptr && rope->dim() != m_d_head)
            throw std::invalid_argument("dev_head_gen_t::set_rope: RoPE 维度 "
                                        + std::to_string(rope->dim()) + " 与 d_head "
                                        + std::to_string(m_d_head) + " 不一致");
        m_rope = rope;
    }

    dev_rope_t<T>* rope() const { return m_rope; }
    int d_head() const { return m_d_head; }
    bool masked() const { return m_mask; }

    /**
     * softmax 概率矩阵 (q_len × k_len)，反向要用；也供测试检查。
     *
     * **融合引擎下它是空的** —— 那份矩阵压根不存在。这不是"忘了填"，而是融合的
     * 全部收益所在，所以测试反过来断言它为空（见 `test_cuda_attention.cu`）。
     * 需要概率数值时用 `logsumexp()` 重算（`probabilities_row()` 就是干这个的）。
     */
    const dev_matrix_t<T>& weights() const { return m_weights; }

    /** 融合引擎前向留下的 `L = logsumexp`（q_len × 1）；非融合引擎下为空。 */
    const dev_matrix_t<T>& logsumexp() const { return m_lse; }

    /** 是否启用融合引擎（开关，前向后由 `m_engine` 记录实际用了哪条）。 */
    bool fused_attention() const { return m_use_fused; }

    void set_fused_attention(bool on) { m_use_fused = on; }

    /** 前向**实际**走的引擎，供测试断言「开关确实生效了」。 */
    bool fused_attention_used() const { return m_engine == engine::fused; }

    /**
     * 从融合引擎留下的 `L` 重算第 `row` 行的概率（`len` 个数，主机侧）。
     *
     * 反向内部是逐块重算的，这个函数是**给测试开的一扇窗**：它让「重算的概率」
     * 与「非融合路径存下来的概率矩阵」能直接逐元素对拍 —— 否则融合路径的正确性
     * 就只能靠最终梯度间接推断了。
     */
    mat_t<T> probabilities_row(int row) const
    {
        if (m_engine != engine::fused)
            throw std::runtime_error("probabilities_row: 前向没有走融合引擎，没有 logsumexp 可用");
        if (row < 0 || row >= m_lse.row_num())
            throw std::out_of_range("probabilities_row: 行号越界");

        // 这是个给测试看的窗口，所以走主机：`dev_mat_t::operator()` 在主机侧是**解引用设备指针**，
        // 不能拿来当面算。也正因如此它按「调用者只需一行」设计，别在大矩阵上循环调。
        const mat_t<T> q = m_q.download();
        const mat_t<T> k = m_k.download();
        const T lse = m_lse.download()(row, 0);
        mat_t<T> p(1, k.col_num());
        const T scale = ::jasmine::detail::device_sqrt(static_cast<T>(m_d_head));
        for (int j = 0; j < p.col_num(); ++j)
        {
            T s = T(0);
            for (int f = 0; f < m_d_head; ++f)
                s += q(f, row) * k(f, j);
            s /= scale;
            // 与融合前向同一套掩码口径（前向只在 q_len > 1 时才掩）
            const bool visible = !m_mask || m_q.col_num() == 1 || j <= row + m_mask_offset;
            p(0, j) = (visible && lse != -std::numeric_limits<T>::infinity())
                          ? ::jasmine::detail::device_exp(s - lse)
                          : T(0);
        }
        return p;
    }

    template <typename Q, typename K, typename V>
    dev_matrix_t<T> forward(const Q& q, const K& k, const V& v)
    {
        return forward_at(q, k, v, 0, 0);
    }

    /**
     * Q 列 j 用绝对位置 `q_pos + j`、K 列 j 用 `k_pos + j` 做 RoPE（与主机同名方法同义）。
     * 掩码用相对口径（`j <= i`），与主机 `mat_head_gen_t::forward` 一致。
     */
    template <typename Q, typename K, typename V>
    dev_matrix_t<T> forward_at(const Q& q, const K& k, const V& v, int q_pos, int k_pos)
    {
        check_qkv(q, k, v, "dev_head_gen_t::forward_at");
        detail::materialize_input(q, m_q);
        detail::materialize_input(k, m_k);
        detail::materialize_input(v, m_v);

        if (m_rope != nullptr)
        {
            // 原地旋转：别名安全（一个线程只碰自己那一对元素），且省一次分配
            if (m_q.col_num() > 0)
                m_rope->forward_inplace(m_q, q_pos);
            if (m_k.col_num() > 0)
                m_rope->forward_inplace(m_k, k_pos);
        }
        // 反向必须用**前向同一批角度**，所以把起点记下来
        m_q_pos = q_pos;
        m_k_pos = k_pos;

        // 掩码口径由实现决定（`q_len == 1` 时不存在未来，不掩），反向要复用同一个口径
        m_mask_offset = 0;
        const int q_len = m_q.col_num();
        if (m_use_fused)
        {
            m_engine = engine::fused;
            m_weights = dev_matrix_t<T>();  // 这份矩阵在融合引擎下根本不存在，清掉免得读到上一轮残留
            attention_cache_t<T> cache =
                fused_attention_forward<T>(m_q.const_leaf(), m_k.const_leaf(), m_v.const_leaf(),
                                           m_mask && q_len > 1, m_mask_offset);
            m_lse = std::move(cache.logsumexp);
            // 反向里的 `D_i = Σ_f dO(f,i)·O(f,i)` 要用前向输出，所以这里留一份。
            // 这是融合路径**唯一**额外留的东西，量级是 `d_head × q_len` ——
            // 与非融合路径留的 `q_len × len` 相比，在长上下文下差着 `len / d_head` 倍。
            detail::materialize_into(cache.out, m_fused_out);
            return std::move(cache.out);
        }
        m_engine = engine::plain;
        m_lse = dev_matrix_t<T>();
        return attend_impl(m_q.const_leaf(), m_k.const_leaf(), m_v.const_leaf(), m_mask_offset);
    }

    /**
     * 对 cache 里已有的 K/V 做 attention（**不 append**，K 已旋转过）。
     *
     * `pos` 是本步首列 Q 的绝对位置：既用来给 Q 做 RoPE，也用来定因果掩码
     * （第 i 行的绝对位置 = `pos + i`）。与主机 `mat_head_gen_t::attend_cached` 同义。
     */
    template <typename Q>
    dev_matrix_t<T> attend_cached(const Q& q, const dev_mat_t<T>& k_cached,
                                  const dev_mat_t<T>& v_cached, int pos)
    {
        using detail::materialize_input;
        dev_matrix_t<T> q_new;
        materialize_input(q, q_new);
        if (m_rope != nullptr && q_new.col_num() > 0)
            m_rope->forward_inplace(q_new, pos);

        if (q_new.row_num() != m_d_head)
            throw std::invalid_argument("dev_head_gen_t::attend_cached: Q 行数 "
                                        + std::to_string(q_new.row_num()) + " 与 d_head "
                                        + std::to_string(m_d_head) + " 不一致");
        if (k_cached.row_num() != m_d_head || v_cached.row_num() != m_d_head)
            throw std::invalid_argument("dev_head_gen_t::attend_cached: cache 的 head_dim 不匹配");
        if (k_cached.col_num() != v_cached.col_num())
            throw std::invalid_argument("dev_head_gen_t::attend_cached: K/V 长度不一致");

        m_q_pos = pos;
        m_mask_offset = pos;
        // decode 一律走非融合路径：此时概率矩阵只有 `1 × len`，融合省下的内存可以忽略，
        // 反而多绕一圈「重算概率」的账。融合的价值只在 `q_len × len` 成规模时才存在。
        m_engine = engine::plain;
        return attend_impl(q_new.const_leaf(), k_cached, v_cached, pos);
    }

    /**
     * 反向。`delta` 形状 (d_head × q_len)，与主机 `mat_head_gen_t::backward` 同形。
     *
     * 主机那四行在这里一一对应：
     *   delta_v        = delta · w
     *   delta_w        = deltaᵀ · v
     *   delta_scores   = softmax_backward(w, delta_w)   ← 用前向留下的概率矩阵
     *   （掩码置零，与 -inf 对应）
     *   delta_q        = k · delta_scoresᵀ / sqrt(d)
     *   delta_k        = q · delta_scores  / sqrt(d)
     *   （RoPE 反向：转置旋转回原坐标系）
     */
    template <typename Delta>
    bwd_pack_t backward(const Delta& delta)
    {
        const dev_mat_t<T> dl = detail::read_leaf_of<T>(delta);

        // 先判"前向跑过没有"再判形状：没跑过前向时形状一定不匹配，此时"形状不对"是个
        // 误导人的报错（真正的原因是压根没有缓存可配对）。
        if (m_engine == engine::none)
            throw std::runtime_error("dev_head_gen_t::backward: 前向还没跑过，"
                                     "拿不到与前向配对的缓存");
        check_delta(dl);

        if (m_engine == engine::fused)
            return backward_fused(dl);

        dev_matrix_t<T> delta_v = matmul(dl, m_weights.const_leaf());
        dev_matrix_t<T> delta_w = matmul(dl.t(), m_v.const_leaf());
        dev_matrix_t<T> delta_scores = softmax_backward(m_weights.const_leaf(), delta_w.const_leaf());

        if (m_mask)
            detail::mask_inplace<T, true>(delta_scores, 0);

        dev_matrix_t<T> delta_q = matmul(m_k.const_leaf(), delta_scores.const_leaf().t());
        dev_matrix_t<T> delta_k = matmul(m_q.const_leaf(), delta_scores.const_leaf());

        const T scale = ::jasmine::detail::device_sqrt(static_cast<T>(m_d_head));
        detail::divide_inplace(delta_q, scale);
        detail::divide_inplace(delta_k, scale);

        if (m_rope != nullptr)
        {
            // 用前向的起点做转置旋转 —— 与前向同一批角度
            if (delta_q.col_num() > 0)
                m_rope->backward_inplace(delta_q, m_q_pos);
            if (delta_k.col_num() > 0)
                m_rope->backward_inplace(delta_k, m_k_pos);
        }
        return bwd_pack_t{std::move(delta_q), std::move(delta_k), std::move(delta_v)};
    }

    std::string net_type(int indent = 0) const
    {
        return std::string(static_cast<std::size_t>(indent) * 4, ' ') + "dev_head_gen_t(d_head:"
               + std::to_string(m_d_head) + (m_mask ? ",mask" : "")
               + (m_use_fused ? ",fused" : "") + ")";
    }

private:
    /**
     * 融合引擎的反向：块内从 `L = logsumexp` 重算概率，不读概率矩阵。
     *
     * 掩码与缩放都在 kernel 里（前者按 `m_mask_offset`，后者按 `√d`），所以这里
     * 只剩 RoPE 反向这一步 —— 与非融合路径用的是同一批角度。
     */
    bwd_pack_t backward_fused(const dev_mat_t<T>& dl)
    {
        const bool causal = m_mask && m_q.col_num() > 1;
        attention_cache_t<T> cache;
        detail::materialize_into(m_fused_out, cache.out);
        detail::materialize_into(m_lse, cache.logsumexp);

        attention_grad_t<T> g = fused_attention_backward<T>(
            cache, m_q.const_leaf(), m_k.const_leaf(), m_v.const_leaf(), dl, causal, m_mask_offset);

        if (m_rope != nullptr)
        {
            if (g.d_q.col_num() > 0)
                m_rope->backward_inplace(g.d_q, m_q_pos);
            if (g.d_k.col_num() > 0)
                m_rope->backward_inplace(g.d_k, m_k_pos);
        }
        return bwd_pack_t{std::move(g.d_q), std::move(g.d_k), std::move(g.d_v)};
    }

    /** 打分 → 缩放 → 掩码 → softmax → 加权求和。前向、cache 前向共用（非融合引擎）。 */
    dev_matrix_t<T> attend_impl(const dev_mat_t<T>& q, const dev_mat_t<T>& k, const dev_mat_t<T>& v,
                                int mask_offset)
    {
        // scores = Qᵀ·K，(q_len × k_len)；只有 Q·K 需要跨行求和，所以必须走 GEMM
        dev_matrix_t<T> scores = matmul(q.t(), k);
        detail::divide_inplace(scores, ::jasmine::detail::device_sqrt(static_cast<T>(m_d_head)));
        // q_len == 1 时不存在「未来」，掩码是恒等操作（与主机同一判断）
        if (m_mask && q.col_num() > 1)
            detail::mask_inplace<T, false>(scores, mask_offset);

        m_weights = softmax_rows(scores.const_leaf());
        return matmul(v, m_weights.const_leaf().t());
    }

    template <typename Q, typename K, typename V>
    void check_qkv(const Q& q, const K& k, const V& v, const char* who) const
    {
        if (q.row_num() != m_d_head || k.row_num() != m_d_head || v.row_num() != m_d_head)
            throw std::invalid_argument(std::string(who) + ": Q/K/V 的行数都必须是 d_head = "
                                        + std::to_string(m_d_head));
        if (q.col_num() != k.col_num() || k.col_num() != v.col_num())
            throw std::invalid_argument(std::string(who) + ": Q/K/V 的序列长度必须一致");
    }

    void check_delta(const dev_mat_t<T>& delta) const
    {
        if (delta.row_num() != m_d_head)
            throw std::invalid_argument("dev_head_gen_t::backward: delta 行数应为 d_head");
        // 两条引擎的「前向跑过没有」判据不同：非融合看概率矩阵，融合看 logsumexp
        const int q_len_fwd = (m_engine == engine::fused) ? m_lse.row_num() : m_weights.row_num();
        if (q_len_fwd != delta.col_num())
            throw std::invalid_argument("dev_head_gen_t::backward: delta 列数与前向 Q 长度不一致"
                                        "（forward 必须先被调用）");
    }

    /** 前向**实际**走的引擎。默认 `none`：还没跑过前向，反向就无从配对。 */
    enum class engine
    {
        none,
        plain,
        fused,
    };

    dev_matrix_t<T> m_q, m_k, m_v;  // 前向缓存（Q/K 已旋转，V 不旋转 —— 与主机一致）
    dev_matrix_t<T> m_weights;      // 非融合引擎：softmax 概率矩阵 (q_len × k_len)
    dev_matrix_t<T> m_fused_out;    // 融合引擎：前向输出 (d_head × q_len)，反向算 D_i 要用
    dev_matrix_t<T> m_lse;          // 融合引擎：logsumexp (q_len × 1)
    dev_rope_t<T>* m_rope = nullptr;  // 非拥有
    int m_d_head = 0;
    int m_q_pos = 0;      // 前向用过的 RoPE 起点，反向照用
    int m_k_pos = 0;
    int m_mask_offset = 0;  // 前向用过的掩码偏移，反向照用
    bool m_mask = false;
    bool m_use_fused = false;         // 开关：下一次前向走哪条
    engine m_engine = engine::none;   // 事实：上一次前向走了哪条
};

// ---------------------------------------------------------------------------
// 多头注意力：mat_mha_t 的设备对应物
// ---------------------------------------------------------------------------

/**
 * Q/K/V/O 四个全维投影 + 按头切分 + 单头核 + 拼接。GQA 由 `n_kv_heads` 参数化。
 *
 * 与主机 `mat_mha_t` 的两条结构性差异，都是「把注释变成代码」：
 *
 *  1. **切头靠 `row_slice` 零拷贝薄壳**，不搬数据。主机端 `vsplit` 也是视图，
 *     但设备端多一层好处：切片直接就是 `dev_mat_t`，可以当表达式或 GEMM 操作数。
 *  2. **KV cache 用 `dev_kv_caches_t`**（多 KV 头容器），写入入口只有 `append_all`。
 *     主机端那句「GQA 下每个 KV 头只能 append 一次」的注释，在这里是不需要注释的
 *     结构性质：容器压根没有「按单个 KV 头 append」的接口。
 *
 * 反向里最要紧的一行是 `place_rows<true>`（累加）：主机端为它写过一段注释 ——
 * 共享同一 KV 头的多个 Q 头如果各自 `assign`，梯度只会剩下最后一个头的。
 */
template <typename T, template <typename> class updator_type>
class dev_mha_t
{
public:
    using ele_type = T;
    using head_type = dev_head_gen_t<T>;
    using proj_type = dev_linear_t<T, updator_type>;

    dev_mha_t() = default;

    /**
     * `n_kv_heads == 0` → 经典 MHA（等于 Q 头数）；`1` → MQA；中间值 → GQA。
     * 约束与主机一致：`d_model % num_heads == 0`、`num_heads % n_kv_heads == 0`。
     */
    void set_param(int num_heads, int d_model, bool mask = false, int n_kv_heads = 0,
                   int max_seq = 0)
    {
        if (num_heads <= 0 || d_model <= 0)
            throw std::invalid_argument("dev_mha_t::set_param: 尺寸必须为正");
        if (d_model % num_heads != 0)
            throw std::invalid_argument("dev_mha_t::set_param: d_model 必须能被 num_heads 整除");
        if (n_kv_heads <= 0)
            n_kv_heads = num_heads;
        if (num_heads % n_kv_heads != 0)
            throw std::invalid_argument("dev_mha_t::set_param: num_heads 必须能被 n_kv_heads 整除");

        m_num_heads = num_heads;
        m_num_kv_heads = n_kv_heads;
        m_group_size = num_heads / n_kv_heads;
        m_d_model = d_model;
        m_d_head = d_model / num_heads;
        m_d_kv = m_num_kv_heads * m_d_head;
        m_mask = mask;

        m_q_net.set_param(d_model, d_model);
        // GQA 的关键差异：K/V 只投影出 n_kv_heads 个头，不是 d_model
        m_k_net.set_param(d_model, m_d_kv);
        m_v_net.set_param(d_model, m_d_kv);
        m_output_proj.set_param(d_model, d_model);

        m_heads.clear();
        m_heads.resize(static_cast<std::size_t>(num_heads));
        for (auto& h : m_heads)
            h.set_param(m_d_head, mask);

        m_caches.configure(num_heads, n_kv_heads, m_d_head, max_seq);
        m_q_full.allocate(d_model, 0);
        m_k_full.allocate(m_d_kv, 0);
        m_v_full.allocate(m_d_kv, 0);
    }

    /** 绑 RoPE（不拥有）。必须在 `set_param` 之后调 —— 它会同步给所有头。 */
    void set_rope(dev_rope_t<T>* rope)
    {
        if (rope != nullptr && rope->dim() != m_d_head)
            throw std::invalid_argument("dev_mha_t::set_rope: RoPE 维度与 d_head 不一致");
        m_rope = rope;
        for (auto& h : m_heads)
            h.set_rope(rope);
    }

    dev_rope_t<T>* rope() const { return m_rope; }

    /**
     * 整层切换注意力引擎（转发给每个 Q 头）。
     *
     * 只影响**训练/预填**那条路（`forward` → 各头的 `forward_at`）；decode 走
     * `forward_one` → `attend_cached`，那里概率矩阵只有 `1 × len`，融合没有收益，
     * 所以固定走非融合路径（理由见 `dev_head_gen_t` 的说明）。
     */
    void set_fused_attention(bool on)
    {
        m_use_fused = on;
        for (auto& h : m_heads)
            h.set_fused_attention(on);
    }

    bool fused_attention() const { return m_use_fused; }

    /** 本层 Q 头是否都走了融合引擎（前向后才为真），供测试断言开关确实生效。 */
    bool fused_attention_used() const
    {
        if (m_heads.empty())
            return false;
        for (const auto& h : m_heads)
            if (!h.fused_attention_used())
                return false;
        return true;
    }

    /**
     * 非融合引擎在这个形状下要留住多少字节的概率矩阵（融合收益的量化口径）。
     *
     * `q_len × len × sizeof(T)`，且**每个 Q 头一份** —— 所以传的是 `num_heads` 而不是单头。
     *
     * 这里**没有 `d_head`**，别把它乘进来：概率矩阵的形状就是 `(q_len × len)`，
     * 每头一份。乘上 `d_head` 会把收益夸大成 64 倍（`d_head = 64` 时），
     * 而这种数字一旦被写进文档就会被人当真（`weights_bytes` 的第一版就是这么错的，
     * 是 `MemorySavingScalesWithContext` 里"比值必须恰好等于 `len`"这条断言把它抓出来的）。
     *
     * 单头版本见 `attention_cache_t<T>::weights_bytes(q_len, len)`。
     */
    static std::size_t weights_bytes(int q_len, int len, int num_heads)
    {
        return static_cast<std::size_t>(q_len) * static_cast<std::size_t>(len)
               * static_cast<std::size_t>(num_heads) * sizeof(T);
    }

    // ---- KV cache（只给 decode 用；训练路径不碰它）----

    void reserve_kv_cache(int max_seq) { m_caches.reserve_all(max_seq); }
    void clear_kv_cache() { m_caches.clear_all(); }
    int kv_cache_length() const { return m_caches.length(); }
    dev_kv_caches_t<T>& caches() { return m_caches; }
    const dev_kv_caches_t<T>& caches() const { return m_caches; }

    /**
     * 训练 / 预填：整段序列一次前向，因果掩码由本层按 `j <= i` 施加。
     * 语义与主机 `mat_mha_t::forward` 完全一致（含 GQA 下 Q 头共享 KV 头）。
     */
    template <typename X>
    dev_matrix_t<T> forward(const X& input)
    {
        if (input.row_num() != m_d_model)
            throw std::invalid_argument("dev_mha_t::forward: 输入行数 "
                                        + std::to_string(input.row_num()) + " 与 d_model "
                                        + std::to_string(m_d_model) + " 不一致");

        m_q_full = m_q_net.forward(input);
        m_k_full = m_k_net.forward(input);
        m_v_full = m_v_net.forward(input);

        return assemble_heads();
    }

    /**
     * 增量推理：只投影本步 token，追加 KV cache，再 attend 历史。
     *
     * 与主机 `mat_mha_t::forward_one` 同序、同契约：**K 在进 cache 之前做 RoPE**
     * （cache 的 append 只收已旋转的 K），V 不旋转。
     */
    template <typename X>
    dev_matrix_t<T> forward_one(const X& input)
    {
        if (input.row_num() != m_d_model)
            throw std::invalid_argument("dev_mha_t::forward_one: 输入行数 "
                                        + std::to_string(input.row_num()) + " 与 d_model "
                                        + std::to_string(m_d_model) + " 不一致");

        const int pos = m_caches.length();
        m_q_full = m_q_net.forward(input);
        m_k_full = m_k_net.forward(input);
        m_v_full = m_v_net.forward(input);

        // K 在进 cache 之前做 RoPE，而且要**逐 KV 头**做：整块 K 有
        // n_kv_heads × d_head 行，而旋转表只有 d_head 维 —— 整块旋转会把
        // 相邻两个头的前半行当成一对去配（只在 GQA/MQA 的多 KV 头上出错，
        // 单头时恰好等价，所以这种 bug 很能藏）。
        // 训练路径没这个问题：那里的旋转发生在单头核内部。
        if (m_rope != nullptr && m_k_full.col_num() > 0)
        {
            dev_matrix_t<T> k_rot(m_d_kv, m_k_full.col_num());
            for (int g = 0; g < m_num_kv_heads; ++g)
                m_rope->rotate_into(row_slice(m_k_full.const_leaf(), g * m_d_head, m_d_head),
                                    row_slice(k_rot.leaf(), g * m_d_head, m_d_head), pos);
            m_caches.append_all(k_rot.const_leaf(), m_v_full.const_leaf());
        }
        else
        {
            m_caches.append_all(m_k_full.const_leaf(), m_v_full.const_leaf());
        }

        const int seq = m_q_full.col_num();
        dev_matrix_t<T> head_concat(m_num_heads * m_d_head, seq);
        for (int i = 0; i < m_num_heads; ++i)
        {
            const dev_kv_cache_t<T>& c = m_caches.cache_of_kv_head(kv_head_of(i));
            dev_matrix_t<T> out = m_heads[static_cast<std::size_t>(i)].attend_cached(
                row_slice(m_q_full.const_leaf(), i * m_d_head, m_d_head), c.keys(), c.values(), pos);
            detail::place_rows_owned(out, head_concat, i * m_d_head);
        }
        return m_output_proj.forward(head_concat.const_leaf());
    }

    /** 反向。`delta` 形状 (d_model × T)，对应主机 `mat_mha_t::backward`。 */
    dev_matrix_t<T> backward(const dev_matrix_t<T>& delta)
    {
        if (delta.row_num() != m_d_model)
            throw std::invalid_argument("dev_mha_t::backward: delta 行数应为 d_model");

        // 1. 输出投影的反向 —— 先把 delta 切成各头的行段
        dev_matrix_t<T> delta_concat = m_output_proj.backward(delta);

        const int seq = delta.col_num();
        dev_matrix_t<T> delta_q(m_d_model, seq);
        dev_matrix_t<T> delta_k(m_d_kv, seq);
        dev_matrix_t<T> delta_v(m_d_kv, seq);
        delta_q.buffer().zero();
        delta_k.buffer().zero();
        delta_v.buffer().zero();

        // 2. 逐头反向，按 Q→KV 头映射归并
        for (int i = 0; i < m_num_heads; ++i)
        {
            const int kv = kv_head_of(i);
            const auto d_i = row_slice(delta_concat.const_leaf(), i * m_d_head, m_d_head);
            typename head_type::bwd_pack_t g = m_heads[static_cast<std::size_t>(i)].backward(d_i);

            // Q 头独占自己的行段 → assign
            detail::place_rows<false>(g.delta_q.const_leaf(), delta_q, i * m_d_head);
            // GQA：多个 Q 头共享一个 KV 头 → **必须累加**（少一个加号就只剩最后一个头）
            detail::place_rows<true>(g.delta_k.const_leaf(), delta_k, kv * m_d_head);
            detail::place_rows<true>(g.delta_v.const_leaf(), delta_v, kv * m_d_head);
        }

        // 3. 三个投影层各自回传，再相加（三个层互不相干，顺序不影响结果）
        dev_matrix_t<T> dq = m_q_net.backward(delta_q);
        dev_matrix_t<T> dk = m_k_net.backward(delta_k);
        dev_matrix_t<T> dv = m_v_net.backward(delta_v);

        dev_matrix_t<T> out(dq.row_num(), dq.col_num());
        if (out.row_num() > 0 && out.col_num() > 0)
            eval_fused(dq.leaf() + dk.leaf() + dv.leaf(), out.buffer());
        return out;
    }

    // ---- 参数访问：权重加载器按主机同名访问器逐个搬运 ----

    proj_type& q_proj() { return m_q_net; }
    const proj_type& q_proj() const { return m_q_net; }
    proj_type& k_proj() { return m_k_net; }
    const proj_type& k_proj() const { return m_k_net; }
    proj_type& v_proj() { return m_v_net; }
    const proj_type& v_proj() const { return m_v_net; }
    proj_type& out_proj() { return m_output_proj; }
    const proj_type& out_proj() const { return m_output_proj; }
    head_type& head(int i) { return m_heads.at(static_cast<std::size_t>(i)); }
    const head_type& head(int i) const { return m_heads.at(static_cast<std::size_t>(i)); }

    int num_heads() const { return m_num_heads; }
    int num_kv_heads() const { return m_num_kv_heads; }
    int group_size() const { return m_group_size; }
    int d_head() const { return m_d_head; }
    int d_kv() const { return m_d_kv; }
    int d_model() const { return m_d_model; }

    /** Q 头 `i` 该读哪个 KV 头（连续 `group_size` 个 Q 头共享一个）。 */
    int kv_head_of(int q_head) const
    {
        if (q_head < 0 || q_head >= m_num_heads)
            throw std::out_of_range("dev_mha_t::kv_head_of: Q 头下标越界");
        return q_head / m_group_size;
    }

    void step()
    {
        m_q_net.step();
        m_k_net.step();
        m_v_net.step();
        m_output_proj.step();
    }

    void set_lr(T lr)
    {
        m_q_net.set_lr(lr);
        m_k_net.set_lr(lr);
        m_v_net.set_lr(lr);
        m_output_proj.set_lr(lr);
    }

    template <typename... arg_types>
    void set_updator(arg_types&&... args)
    {
        m_q_net.set_updator(std::forward<arg_types>(args)...);
        m_k_net.set_updator(std::forward<arg_types>(args)...);
        m_v_net.set_updator(std::forward<arg_types>(args)...);
        m_output_proj.set_updator(std::forward<arg_types>(args)...);
    }

    template <typename init_type>
    void init_weight()
    {
        m_q_net.template init_weight<init_type>();
        m_k_net.template init_weight<init_type>();
        m_v_net.template init_weight<init_type>();
        m_output_proj.template init_weight<init_type>();
    }

    /** LLaMA 系没有线性层偏置；加载权重后调用（对应主机 `zero_all_biases`）。 */
    void zero_biases()
    {
        m_q_net.zero_bias();
        m_k_net.zero_bias();
        m_v_net.zero_bias();
        m_output_proj.zero_bias();
    }

    std::string net_type(int indent = 0) const
    {
        const std::string tag = (m_num_kv_heads == 1) ? "MQA"
                                : (m_num_kv_heads != m_num_heads) ? "GQA"
                                                                  : "MHA";
        return std::string(static_cast<std::size_t>(indent) * 4, ' ') + "dev_mha_t(" + tag
               + ",q_heads:" + std::to_string(m_num_heads)
               + ",kv_heads:" + std::to_string(m_num_kv_heads)
               + ",d_model:" + std::to_string(m_d_model) + ")";
    }

private:
    /**
     * 训练路径的「逐头 attend 并拼接」。
     *
     * RoPE 起点是 (0, 0)：与主机 `mat_mha_t::forward` 一致。cross-attn 那种
     * 「Q 用解码绝对位置、K 从 0」的组合由单头核的 `forward_at` 支持，等
     * cross-attn 外壳上设备时再加参数（主机也是分成 `forward` / `forward_at` 两个入口的）。
     */
    dev_matrix_t<T> assemble_heads()
    {
        const int seq = m_q_full.col_num();
        dev_matrix_t<T> head_concat(m_num_heads * m_d_head, seq);
        for (int i = 0; i < m_num_heads; ++i)
        {
            const int kv = kv_head_of(i);
            dev_matrix_t<T> out = m_heads[static_cast<std::size_t>(i)].forward_at(
                row_slice(m_q_full.const_leaf(), i * m_d_head, m_d_head),
                row_slice(m_k_full.const_leaf(), kv * m_d_head, m_d_head),
                row_slice(m_v_full.const_leaf(), kv * m_d_head, m_d_head), 0, 0);
            detail::place_rows_owned(out, head_concat, i * m_d_head);
        }
        return m_output_proj.forward(head_concat.const_leaf());
    }

    proj_type m_q_net, m_k_net, m_v_net, m_output_proj;
    std::vector<head_type> m_heads;
    dev_kv_caches_t<T> m_caches;
    dev_matrix_t<T> m_q_full, m_k_full, m_v_full;  // 前向缓存（按 KV 头切分用）
    dev_rope_t<T>* m_rope = nullptr;               // 非拥有
    int m_num_heads = 0;
    int m_num_kv_heads = 0;
    int m_group_size = 1;
    int m_d_model = 0;
    int m_d_head = 0;
    int m_d_kv = 0;
    bool m_mask = false;
    bool m_use_fused = false;
};

} // namespace cuda
} // namespace jasmine

#endif
