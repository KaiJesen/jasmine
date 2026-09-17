#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "jas_RoPE_t.hpp"
#include "jas_cuda_buffer.hpp"
#include "jas_cuda_embedding.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_mha.hpp"
#include "jas_cuda_net.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_rope.hpp"
#include "jas_embedding_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "jas_mha_t.hpp"
#include "jas_updator_t.hpp"

/**
 * 设备端注意力与 embedding 的测试：与主机 `mat_mha_t` / `embedding_net_t` 逐步对拍，
 * 外加一条不依赖主机实现的**有限差分**裁判。
 *
 * 散热约束同其它 CUDA 测试：本机是无风扇的 Tesla P4，矩阵刻意取小、
 * 每个用例前后测温、超 80℃ 就 skip。详见 CUDA.md。
 *
 * ## 这两组用例守的是什么
 *
 * `dev_mha_t` 里唯一「不能靠肉眼看出来」的地方是 GQA 的梯度归并：共享同一个 KV 头
 * 的多个 Q 头，其 dk/dv 必须**累加**。写成 assign 的话形状、维度、前向全对，
 * 只有梯度少了一部分 —— 而少掉的部分会让训练「仍然在下降，只是下降得慢」。
 * 所以对拍特意把输入梯度（`dq/dk/dv` 经过三个投影层合起来的结果）也钉住，
 * 而不只是比前向输出。
 */

using namespace jasmine;

namespace
{

constexpr int kHotCelsius = 80;

int gpu_temperature_c()
{
    // Thermal control is only needed on the fanless P4 development card.
    // A800 and other well-cooled sm_80+ targets should not emit temperature spam.
    if (jasmine::cuda::device_info().compute_capability() != 61)
        return -1;

    FILE* pipe =
        ::popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null", "r");
    if (pipe == nullptr)
        return -1;

    char buf[64] = {};
    char* got = std::fgets(buf, sizeof(buf), pipe);
    ::pclose(pipe);
    if (got == nullptr)
        return -1;

    return std::atoi(buf);
}

/** 确定性地填一个矩阵；行列量级不同，能暴露广播/转置搞反的错误。 */
mat_t<double> make_host(int rows, int cols, double base)
{
    mat_t<double> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + 0.01 * ((i * 7 + j * 13) % 41) - 0.02 * i + 0.03 * j;
    return m;
}

void expect_close(const mat_t<double>& got, const mat_t<double>& want, double rel_tol,
                  const char* what)
{
    ASSERT_EQ(got.row_num(), want.row_num()) << what;
    ASSERT_EQ(got.col_num(), want.col_num()) << what;
    for (int i = 0; i < got.row_num(); ++i)
        for (int j = 0; j < got.col_num(); ++j)
        {
            const double scale = std::max(1.0, std::abs(want(i, j)));
            ASSERT_NEAR(got(i, j), want(i, j), rel_tol * scale)
                << what << " 在 (" << i << "," << j << ") 不一致（" << got(i, j) << " vs "
                << want(i, j) << "）";
        }
}

/** 主机的单头注意力核（RoPE 从注册中心取，按 d_head + 配对约定共享）。 */
using host_head_t = mat_head_gen_t<mat_t<double>, sgd_t>;

/** 主机的多头注意力外壳。 */
using host_mha_t = mat_mha_t<mat_t<double>, sgd_t>;


/**
 * 把主机的 Q/K/V/O 四个投影搬到设备端。
 *
 * 只搬权重与偏置，不搬 RoPE —— RoPE 的表是算出来的常量，两边各自生成即可
 * （表本身的正确性由 `test_cuda_rope.cu` 单独钉住）。
 */
void upload_mha(host_mha_t& host, cuda::dev_mha_t<double, cuda::dev_sgd_t>& dev)
{
    dev.q_proj().upload_weight(host.q_proj().weight());
    dev.q_proj().upload_bias(host.q_proj().bias());
    dev.k_proj().upload_weight(host.k_proj().weight());
    dev.k_proj().upload_bias(host.k_proj().bias());
    dev.v_proj().upload_weight(host.v_proj().weight());
    dev.v_proj().upload_bias(host.v_proj().bias());
    dev.out_proj().upload_weight(host.out_proj().weight());
    dev.out_proj().upload_bias(host.out_proj().bias());
}

/** 同一次反向里给设备端与主机端准备的随机（确定性）输入与上游梯度。 */
struct mha_case_t
{
    mat_t<double> x;
    mat_t<double> upstream;
};

mha_case_t make_case(int d_model, int seq, int seed)
{
    mha_case_t c;
    c.x = make_host(d_model, seq, 0.1);
    c.upstream = make_host(d_model, seq, 0.05 + 0.001 * seed);
    return c;
}

} // namespace

class CudaModelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const int t = gpu_temperature_c();
        if (t >= kHotCelsius)
            GTEST_SKIP() << "GPU 结温 " << t << "℃ 已达上限 " << kHotCelsius << "℃，跳过";
        if (t >= 0)
            std::fprintf(stderr, "[测温] 测试开始前 GPU %d℃\n", t);
    }

    void TearDown() override
    {
        const int t = gpu_temperature_c();
        if (t >= 0)
            std::fprintf(stderr, "[测温] 测试结束后 GPU %d℃\n", t);
        cuda::sync();
    }
};

// ===========================================================================
// 单头核：mat_head_gen_t 的对应物
// ===========================================================================

TEST_F(CudaModelTest, HeadForwardMatchesHost)
{
    constexpr int d_head = 8;
    constexpr int seq = 6;
    constexpr int max_seq = 16;

    const mat_t<double> qh = make_host(d_head, seq, 0.2);
    const mat_t<double> kh = make_host(d_head, seq, -0.3);
    const mat_t<double> vh = make_host(d_head, seq, 0.4);

    auto& registry = rope_registry_t<double>::instance();

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        for (bool mask : {false, true})
        {
            host_head_t host(d_head, mask, seq);
            host.set_rope(registry.get(d_head, max_seq, layout));

            cuda::dev_head_gen_t<double> dev(d_head, mask);
            cuda::dev_rope_t<double> dev_rope(d_head, layout);
            dev_rope.reserve(max_seq);
            dev.set_rope(&dev_rope);

            const mat_t<double> want = host.forward(qh, kh, vh);
            const mat_t<double> got = dev.forward(qh, kh, vh).download();

            expect_close(got, want, 1e-12,
                         (std::string("单头前向 layout=")
                          + (layout == rope_pair_layout::half_split ? "half_split" : "interleaved")
                          + (mask ? " mask" : " nomask"))
                             .c_str());
        }
    }
}

TEST_F(CudaModelTest, HeadBackwardMatchesHost)
{
    constexpr int d_head = 8;
    constexpr int seq = 5;
    constexpr int max_seq = 16;

    const mat_t<double> qh = make_host(d_head, seq, 0.2);
    const mat_t<double> kh = make_host(d_head, seq, -0.3);
    const mat_t<double> vh = make_host(d_head, seq, 0.4);
    mat_t<double> delta = make_host(d_head, seq, 0.15);

    auto& registry = rope_registry_t<double>::instance();

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        for (bool mask : {false, true})
        {
            host_head_t host(d_head, mask, seq);
            host.set_rope(registry.get(d_head, max_seq, layout));

            cuda::dev_head_gen_t<double> dev(d_head, mask);
            cuda::dev_rope_t<double> dev_rope(d_head, layout);
            dev_rope.reserve(max_seq);
            dev.set_rope(&dev_rope);

            host.forward(qh, kh, vh);
            dev.forward(qh, kh, vh);

            cuda::dev_matrix_t<double> d(d_head, seq, delta);
            const auto want = host.backward(delta.view());
            const auto got = dev.backward(d);

            const std::string tag =
                std::string(layout == rope_pair_layout::half_split ? "half_split" : "interleaved")
                + (mask ? " mask" : " nomask");
            expect_close(got.delta_q.download(), want.delta_q, 1e-11, ("单头 dQ " + tag).c_str());
            expect_close(got.delta_k.download(), want.delta_k, 1e-11, ("单头 dK " + tag).c_str());
            expect_close(got.delta_v.download(), want.delta_v, 1e-11, ("单头 dV " + tag).c_str());
        }
    }
}

TEST_F(CudaModelTest, HeadAttendCachedMatchesHost)
{
    constexpr int d_head = 8;
    constexpr int ctx = 7;
    constexpr int max_seq = 16;

    const mat_t<double> qh = make_host(d_head, 2, 0.25);
    const mat_t<double> kh = make_host(d_head, ctx, -0.2);
    const mat_t<double> vh = make_host(d_head, ctx, 0.35);

    host_head_t host(d_head, true, ctx);
    host.set_rope(rope_registry_t<double>::instance().get(
        d_head, max_seq, rope_pair_layout::half_split));

    cuda::dev_head_gen_t<double> dev(d_head, true);
    cuda::dev_rope_t<double> dev_rope(d_head, rope_pair_layout::half_split);
    dev_rope.reserve(max_seq);
    dev.set_rope(&dev_rope);

    // 主机侧：把 K 先旋转再放进 cache（decode 的契约），Q 由 attend_cached 自己转
    kv_cache_t<double> host_cache;
    host_cache.reserve(d_head, max_seq);
    host_cache.append(host.rope()->forward_at(kh, 0), vh);

    // 设备侧：同一份「已旋转的 K」——直接把主机的 cache 内容搬上来，避免两边各自旋转
    // 带来的差异（RoPE 表本身的正确性由 test_cuda_rope.cu 单独钉住）
    cuda::dev_matrix_t<double> k_dev(d_head, ctx, host_cache.keys().clone());
    cuda::dev_matrix_t<double> v_dev(d_head, ctx, host_cache.values().clone());

    cuda::dev_kv_cache_t<double> dev_cache;
    dev_cache.reserve(d_head, max_seq);
    dev_cache.append(k_dev.const_leaf(), v_dev.const_leaf());

    const int pos = ctx;
    const mat_t<double> want = host.attend_cached(qh, pos, host_cache);
    const mat_t<double> got = dev.attend_cached(qh, dev_cache.keys(), dev_cache.values(), pos).download();
    expect_close(got, want, 1e-12, "attend_cached");
}

// ===========================================================================
// 多头外壳：mat_mha_t 的对应物
// ===========================================================================

/** 前向对拍：经典 MHA 与 GQA 各一遍（GQA 才是多 KV 头路径真正被走到的用例）。 */
TEST_F(CudaModelTest, MhaForwardMatchesHostAcrossConfigs)
{
    struct cfg_t
    {
        int heads;
        int kv_heads;
        const char* tag;
    };
    const cfg_t configs[] = {{4, 4, "MHA"}, {4, 2, "GQA"}, {4, 1, "MQA"}};

    constexpr int d_model = 16;
    constexpr int seq = 6;

    for (const auto& c : configs)
    {
        host_mha_t host(c.heads, d_model, /*mask=*/true, seq, c.kv_heads);
        host.set_rope_pair_layout(rope_pair_layout::half_split);

        cuda::dev_mha_t<double, cuda::dev_sgd_t> dev;
        dev.set_param(c.heads, d_model, /*mask=*/true, c.kv_heads);
        cuda::dev_rope_t<double> dev_rope(d_model / c.heads, rope_pair_layout::half_split);
        dev_rope.reserve(seq);
        dev.set_rope(&dev_rope);
        upload_mha(host, dev);

        const mha_case_t tc = make_case(d_model, seq, 1);
        const mat_t<double> want = host.forward(tc.x);
        const mat_t<double> got = dev.forward(tc.x).download();
        expect_close(got, want, 1e-12, (std::string("MHA 前向 ") + c.tag).c_str());
    }
}

/**
 * 反向对拍。学习率置 0 ⇒ 参数不动，于是输入梯度可以直接逐元素比 ——
 * 这一条同时覆盖了「各头 dQ 归位」「GQA 下 dK/dV 累加」「RoPE 反向」「掩码置零」
 * 这四件事：任何一处错，合出来的输入梯度就不对。
 */
TEST_F(CudaModelTest, MhaBackwardMatchesHost)
{
    struct cfg_t
    {
        int heads;
        int kv_heads;
        const char* tag;
    };
    const cfg_t configs[] = {{4, 4, "MHA"}, {4, 2, "GQA"}, {4, 1, "MQA"}};

    constexpr int d_model = 16;
    constexpr int seq = 5;

    for (const auto& c : configs)
    {
        host_mha_t host(c.heads, d_model, /*mask=*/true, seq, c.kv_heads);
        host.set_rope_pair_layout(rope_pair_layout::half_split);
        host.set_lr(0.0);  // 冻结参数：只比输入梯度，避开更新器语义的干扰

        cuda::dev_mha_t<double, cuda::dev_sgd_t> dev;
        dev.set_param(c.heads, d_model, /*mask=*/true, c.kv_heads);
        cuda::dev_rope_t<double> dev_rope(d_model / c.heads, rope_pair_layout::half_split);
        dev_rope.reserve(seq);
        dev.set_rope(&dev_rope);
        dev.set_lr(0.0);
        upload_mha(host, dev);

        const mha_case_t tc = make_case(d_model, seq, 2);
        host.forward(tc.x);
        dev.forward(tc.x);

        cuda::dev_matrix_t<double> up(d_model, seq, tc.upstream);
        const mat_t<double> want = host.backward(tc.upstream);
        const mat_t<double> got = dev.backward(up).download();
        expect_close(got, want, 1e-11, (std::string("MHA 输入梯度 ") + c.tag).c_str());
    }
}

/**
 * 参数梯度也要比 —— 用的是「初始参数 − 更新后参数，再除以学习率」这个反推法
 * （sgd 下是精确值），顺带把**更新语义本身**一起验证了。
 *
 * 用 GQA 配置：Q/K/V 三个投影的输出宽度不同（d_model vs d_kv），
 * 搬运权重时搞错一个维度，这里就会露出来。
 */
TEST_F(CudaModelTest, MhaParameterGradientsMatchHost)
{
    constexpr int heads = 4;
    constexpr int kv_heads = 2;
    constexpr int d_model = 16;
    constexpr int seq = 5;
    constexpr double lr = 0.05;

    host_mha_t host(heads, d_model, /*mask=*/true, seq, kv_heads);
    host.set_rope_pair_layout(rope_pair_layout::half_split);
    host.set_lr(lr);

    cuda::dev_mha_t<double, cuda::dev_sgd_t> dev;
    dev.set_param(heads, d_model, /*mask=*/true, kv_heads);
    cuda::dev_rope_t<double> dev_rope(d_model / heads, rope_pair_layout::half_split);
    dev_rope.reserve(seq);
    dev.set_rope(&dev_rope);
    dev.set_lr(lr);
    upload_mha(host, dev);

    const mha_case_t tc = make_case(d_model, seq, 3);
    host.forward(tc.x);
    dev.forward(tc.x);

    const mat_t<double> w_q_before = host.q_proj().weight();
    const mat_t<double> w_k_before = host.k_proj().weight();
    const mat_t<double> w_v_before = host.v_proj().weight();
    const mat_t<double> w_o_before = host.out_proj().weight();
    const mat_t<double> b_q_before = host.q_proj().bias();

    cuda::dev_matrix_t<double> up(d_model, seq, tc.upstream);
    host.backward(tc.upstream);
    dev.backward(up);

    // (init − updated) / lr == 梯度
    auto grad_of = [&](const mat_t<double>& before, const mat_t<double>& after) {
        mat_t<double> g(before.row_num(), before.col_num());
        for (int i = 0; i < g.row_num(); ++i)
            for (int j = 0; j < g.col_num(); ++j)
                g(i, j) = (before(i, j) - after(i, j)) / lr;
        return g;
    };

    expect_close(grad_of(w_q_before, dev.q_proj().weight_to_host()),
                 grad_of(w_q_before, host.q_proj().weight()), 1e-10, "W_Q 梯度");
    expect_close(grad_of(w_k_before, dev.k_proj().weight_to_host()),
                 grad_of(w_k_before, host.k_proj().weight()), 1e-10, "W_K 梯度（GQA 维度）");
    expect_close(grad_of(w_v_before, dev.v_proj().weight_to_host()),
                 grad_of(w_v_before, host.v_proj().weight()), 1e-10, "W_V 梯度（GQA 维度）");
    expect_close(grad_of(w_o_before, dev.out_proj().weight_to_host()),
                 grad_of(w_o_before, host.out_proj().weight()), 1e-10, "W_O 梯度");
    // 偏置梯度是 row_sum(delta)：形状是 (out × 1)，与权重的 (out × in) 不同，
    // 各自走一条搬运路径（dev_colvec_t vs dev_matrix_t），所以单独比一次
    expect_close(grad_of(b_q_before, dev.q_proj().bias_to_host()),
                 grad_of(b_q_before, host.q_proj().bias()), 1e-10, "b_Q 梯度（row_sum）");
}

/**
 * 有限差分：与主机实现无关的第二裁判。
 *
 * `L = Σ out ⊙ upstream`，`∂L/∂x` 就是反向该返回的东西（Jacobian-transpose-vector 积）。
 * 这条完全从「前向是个求值函数」出发，抓的是公式本身写错。
 *
 * 用最小的非平凡配置（2 头 / 1 个 KV 头 / 掩码开 / 4 列），把差分次数压到 32×2 次前向 ——
 * P4 上散热是硬约束，能省一趟是一趟。
 */
TEST_F(CudaModelTest, MhaBackwardPassesFiniteDifference)
{
    constexpr int heads = 2;
    constexpr int kv_heads = 1;
    constexpr int d_model = 8;
    constexpr int seq = 4;
    constexpr double eps = 1e-6;

    cuda::dev_mha_t<double, cuda::dev_sgd_t> dev;
    dev.set_param(heads, d_model, /*mask=*/true, kv_heads);
    cuda::dev_rope_t<double> dev_rope(d_model / heads, rope_pair_layout::half_split);
    dev_rope.reserve(seq);
    dev.set_rope(&dev_rope);

    const mat_t<double> x0 = make_host(d_model, seq, 0.12);
    const mat_t<double> upstream = make_host(d_model, seq, 0.08);

    cuda::dev_matrix_t<double> up(d_model, seq, upstream);
    auto objective = [&](const mat_t<double>& x) {
        cuda::dev_matrix_t<double> dx(d_model, seq, x);
        cuda::dev_matrix_t<double> out = dev.forward(dx.const_leaf());
        return cuda::sum_all(out.leaf() * up.const_leaf());
    };

    mat_t<double> fd(d_model, seq);
    mat_t<double> probe = x0;
    for (int i = 0; i < d_model; ++i)
        for (int j = 0; j < seq; ++j)
        {
            const double origin = x0(i, j);
            probe(i, j) = origin + eps;
            const double fp = objective(probe);
            probe(i, j) = origin - eps;
            const double fm = objective(probe);
            probe(i, j) = origin;
            fd(i, j) = (fp - fm) / (2.0 * eps);
        }

    // 解析反向放在差分之后：backward 会就地更新参数，差分必须在一套固定参数上完成
    cuda::dev_matrix_t<double> xd(d_model, seq, x0);
    dev.forward(xd.const_leaf());
    const mat_t<double> analytic = dev.backward(up).download();

    expect_close(analytic, fd, 1e-5, "MHA 输入梯度 vs 有限差分");
}

/** decode 路径：prefill 多列 + 逐 token 续写，与主机 `forward_one` 对拍。 */
TEST_F(CudaModelTest, MhaForwardOneMatchesHostWithKvCache)
{
    constexpr int heads = 4;
    constexpr int kv_heads = 2;
    constexpr int d_model = 16;
    constexpr int prompt = 3;
    constexpr int extra = 2;

    host_mha_t host(heads, d_model, /*mask=*/true, 1, kv_heads);
    host.set_rope_pair_layout(rope_pair_layout::half_split);
    host.reserve_kv_cache(32);

    cuda::dev_mha_t<double, cuda::dev_sgd_t> dev;
    dev.set_param(heads, d_model, /*mask=*/true, kv_heads);
    cuda::dev_rope_t<double> dev_rope(d_model / heads, rope_pair_layout::half_split);
    dev_rope.reserve(32);
    dev.set_rope(&dev_rope);
    upload_mha(host, dev);

    // prefill：一次喂多列（主机 mat_head_gen_t 会按绝对位置自行补因果掩码）
    const mat_t<double> xp = make_host(d_model, prompt, 0.18);
    expect_close(dev.forward_one(xp).download(), host.forward_one(xp), 1e-12, "prefill 输出");

    // 逐 token 续写
    for (int t = 0; t < extra; ++t)
    {
        const mat_t<double> x1 = make_host(d_model, 1, 0.2 + 0.03 * t);
        expect_close(dev.forward_one(x1).download(), host.forward_one(x1), 1e-12,
                     ("decode 第 " + std::to_string(t) + " 步").c_str());
    }
    ASSERT_EQ(dev.kv_cache_length(), host.kv_cache_length());
}

// ===========================================================================
// embedding
// ===========================================================================

TEST_F(CudaModelTest, EmbeddingForwardAndBackwardMatchHost)
{
    constexpr int vocab = 9;
    constexpr int d_model = 6;
    constexpr int seq = 5;
    constexpr double lr = 0.1;

    // id 里**故意重复**：反向的散射累加（同一个 token 出现多次）正是最容易写错的地方
    mat_t<double> ids(1, seq);
    const int ids_data[seq] = {3, 0, 3, 7, 3};
    for (int t = 0; t < seq; ++t)
        ids(0, t) = ids_data[t];

    embedding_net_t<mat_t<double>, sgd_t> host(vocab, d_model);
    host.weight() = make_host(d_model, vocab, 0.05);
    host.set_lr(lr);

    cuda::dev_embedding_t<double, cuda::dev_sgd_t> dev;
    dev.set_param(vocab, d_model);
    dev.upload_weight(host.weight());
    dev.set_lr(lr);

    const mat_t<double> want_fwd = host.forward(ids);
    const mat_t<double> got_fwd = dev.forward(ids).download();
    expect_close(got_fwd, want_fwd, 1e-14, "embedding 前向（gather）");

    const mat_t<double> upstream = make_host(d_model, seq, 0.07);
    const mat_t<double> w_before = host.weight();

    host.backward(upstream);
    cuda::dev_matrix_t<double> up(d_model, seq, upstream);
    dev.backward(up);

    auto grad_of = [&](const mat_t<double>& after) {
        mat_t<double> g(d_model, vocab);
        for (int i = 0; i < d_model; ++i)
            for (int j = 0; j < vocab; ++j)
                g(i, j) = (w_before(i, j) - after(i, j)) / lr;
        return g;
    };
    expect_close(grad_of(dev.weight_to_host()), grad_of(host.weight()), 1e-12,
                 "embedding 权重梯度（含重复 id 的散射累加）");

    // 没出现过的 token 列梯度必须恰好是 0（散射不该碰它们）
    const mat_t<double> g = grad_of(dev.weight_to_host());
    for (int i = 0; i < d_model; ++i)
        for (int j = 0; j < vocab; ++j)
        {
            const bool touched = (j == 0 || j == 3 || j == 7);
            if (!touched)
                ASSERT_EQ(g(i, j), 0.0) << "未出现的 token 列 " << j << " 梯度应为 0";
        }
}

TEST_F(CudaModelTest, EmbeddingRejectsOutOfRangeId)
{
    cuda::dev_embedding_t<double, cuda::dev_sgd_t> dev;
    dev.set_param(5, 4);

    mat_t<double> ids(1, 3);
    ids(0, 0) = 1;
    ids(0, 1) = 5;  // 越界（合法范围 0..4）
    ids(0, 2) = 2;

    EXPECT_THROW(dev.forward(ids), std::out_of_range);
}
