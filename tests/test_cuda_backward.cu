#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "jas_RoPE_t.hpp"
#include "jas_cuda_buffer.hpp"
#include "jas_cuda_gemm.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_net.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_cuda_rope.hpp"
#include "jas_cuda_updator.hpp"
#include "jas_loss_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_t.hpp"
#include "jas_net_t.hpp"
#include "jas_silu_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

/**
 * 设备端反向传播的测试：更新器 / 各层梯度 / RoPE 与 softmax 的反向。
 *
 * 散热约束同其它 CUDA 测试：本机是无风扇的 Tesla P4，矩阵都很小、
 * 每个用例前后测温、超 80℃ 就 skip。详见 CUDA.md。
 *
 * ## 为什么每类梯度都要两套验证
 *
 * 只跟主机端对拍是不够的：**主机实现自己也可能是错的**，而"两份实现犯了同一个错误"
 * 恰恰是最难发现的情况（本项目历史上就出现过 `append_cols` 的自增列偏移错误，
 * 主机/设备两边一起错，靠对拍看不出来）。
 *
 * 所以每个梯度都跑两条独立的裁判：
 *
 *   1. **对主机解析解**：公式逐字对齐（能抓住实现与主机不一致，包括符号/转置/广播方向）。
 *   2. **有限差分**：`(L(x+ε) − L(x−ε)) / 2ε` 与解析梯度比 —— 这条**不依赖任何一份解析
 *      反向的正确性**，纯从"前向是求值函数"这个定义出发。它抓的是公式本身写错。
 *
 * 有限差分刻意用**设备前向**来算 `L`，这样两边算术一致，不会被 GEMM 求和顺序差异淹没
 * （否则容差要放到 1e-5 量级，反而漏掉真实错误）。
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

    FILE* pipe = ::popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null", "r");
    if (pipe == nullptr)
        return -1;

    char buf[64] = {};
    char* got = std::fgets(buf, sizeof(buf), pipe);
    ::pclose(pipe);
    if (got == nullptr)
        return -1;

    return std::atoi(buf);
}

/** 确定性地填一个矩阵；各行/各列量级不同，避免掩盖广播方向搞反的错误。 */
template <typename T>
mat_t<T> make_host(int rows, int cols, T base)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = base + T(0.01) * ((i * 7 + j * 13) % 41) - T(0.02) * i + T(0.03) * j;
    return m;
}

/** 含负值的填充（softmax / SiLU 需要跨越 0 的输入才有意义）。 */
template <typename T>
mat_t<T> make_signed_host(int rows, int cols, T amp)
{
    mat_t<T> m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = amp * static_cast<T>((i * 5 + j * 11) % 23 - 11);
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

} // namespace

class CudaBackwardTest : public ::testing::Test
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
// 有限差分：与解析反向无关的第二裁判
// ===========================================================================

/**
 * 目标函数 `L = Σ out ⊙ upstream`。
 *
 * 为什么要这个形状：对 `L` 求 `∂L/∂x` 得到的正是反向该返回的东西 ——
 * 因为 `∂L/∂x = Jᵀ·upstream`，其中 `J = ∂out/∂x`。于是反向接口里那个 `delta`
 * 参数在数值上就等于 `upstream`，两者可直接对接。
 */
template <typename Layer>
double objective(Layer& layer, const mat_t<double>& host_x, const cuda::dev_matrix_t<double>& upstream)
{
    cuda::dev_matrix_t<double> x(host_x.row_num(), host_x.col_num(), host_x);
    cuda::dev_matrix_t<double> out = layer.forward(x.leaf());
    return cuda::sum_all(out.leaf() * upstream.const_leaf());
}

/** 中心差分估计 `∂L/∂x`。只调用 forward，**不碰 backward**（否则参数会被更新掉）。 */
template <typename Layer>
mat_t<double> finite_difference(Layer& layer, const mat_t<double>& x,
                                const cuda::dev_matrix_t<double>& upstream, double eps)
{
    mat_t<double> fd(x.row_num(), x.col_num());
    mat_t<double> probe = x;

    for (int i = 0; i < x.row_num(); ++i)
    {
        for (int j = 0; j < x.col_num(); ++j)
        {
            const double origin = x(i, j);

            probe(i, j) = origin + eps;
            const double fp = objective(layer, probe, upstream);

            probe(i, j) = origin - eps;
            const double fm = objective(layer, probe, upstream);

            probe(i, j) = origin;
            fd(i, j) = (fp - fm) / (2.0 * eps);
        }
    }
    return fd;
}

/**
 * 一条用例同时跑两种裁判：先做有限差分（只走 forward），再做一次解析反向。
 *
 * 顺序很关键 —— 放在后面做解析反向，是因为 `backward` 会就地更新参数，
 * 而有限差分必须在一套固定的参数上完成。
 */
template <typename Layer>
void check_gradients(Layer& layer, const mat_t<double>& x, const mat_t<double>& upstream_host,
                     double fd_tol, double eps = 1e-6)
{
    cuda::dev_matrix_t<double> upstream(upstream_host.row_num(), upstream_host.col_num(),
                                        upstream_host);

    mat_t<double> fd = finite_difference(layer, x, upstream, eps);

    cuda::dev_matrix_t<double> dev_x(x.row_num(), x.col_num(), x);
    layer.forward(dev_x.leaf());
    mat_t<double> analytic = layer.backward(upstream).download();

    expect_close(analytic, fd, fd_tol, "解析反向 vs 有限差分");
}

// ===========================================================================
// 更新器
// ===========================================================================

TEST_F(CudaBackwardTest, SgdMatchesHost)
{
    const int rows = 5, cols = 7;
    mat_t<double> param0 = make_host<double>(rows, cols, 0.3);

    mat_t<double> host_param = param0;
    sgd_t<double> host_upd(0.1);

    cuda::dev_matrix_t<double> dev_param(rows, cols, param0);
    cuda::dev_sgd_t<double> dev_upd(0.1);

    for (int step = 0; step < 6; ++step)
    {
        mat_t<double> grad = make_host<double>(rows, cols, 0.05 + 0.1 * step);
        cuda::dev_matrix_t<double> dev_grad(rows, cols, grad);

        host_upd.update(grad, host_param);
        dev_upd.update(dev_grad.const_leaf(), dev_param.buffer());

        expect_close(dev_param.download(), host_param, 1e-12, "sgd 参数");
    }
}

TEST_F(CudaBackwardTest, AdamMatchesHostOverManySteps)
{
    const int rows = 6, cols = 4;
    mat_t<double> param0 = make_host<double>(rows, cols, 0.5);

    mat_t<double> host_param = param0;
    adam_t<double> host_upd(0.01, 0.9, 0.999, 1e-8);

    cuda::dev_matrix_t<double> dev_param(rows, cols, param0);
    cuda::dev_adam_t<double> dev_upd(0.01, 0.9, 0.999, 1e-8);

    // 多跑几步：偏差修正（beta^t）只有在若干步之后才会显示出累积误差
    for (int step = 0; step < 12; ++step)
    {
        mat_t<double> grad = make_signed_host<double>(rows, cols, 0.03 + 0.02 * step);
        cuda::dev_matrix_t<double> dev_grad(rows, cols, grad);

        host_upd.update(grad, host_param);
        dev_upd.update(dev_grad.const_leaf(), dev_param.buffer());

        expect_close(dev_param.download(), host_param, 1e-10, "adam 参数");
    }
}

TEST_F(CudaBackwardTest, NadamMatchesHostOverManySteps)
{
    const int rows = 6, cols = 4;
    mat_t<double> param0 = make_host<double>(rows, cols, 0.5);

    mat_t<double> host_param = param0;
    nadam_t<double> host_upd(0.02, 0.9, 0.999, 1e-8);

    cuda::dev_matrix_t<double> dev_param(rows, cols, param0);
    cuda::dev_nadam_t<double> dev_upd(0.02, 0.9, 0.999, 1e-8);

    for (int step = 0; step < 12; ++step)
    {
        mat_t<double> grad = make_signed_host<double>(rows, cols, 0.03 + 0.02 * step);
        cuda::dev_matrix_t<double> dev_grad(rows, cols, grad);

        host_upd.update(grad, host_param);
        dev_upd.update(dev_grad.const_leaf(), dev_param.buffer());

        expect_close(dev_param.download(), host_param, 1e-10, "nadam 参数");
    }
}

TEST_F(CudaBackwardTest, GradientAccumulatorMatchesHost)
{
    const int rows = 4, cols = 3;
    mat_t<double> param0 = make_host<double>(rows, cols, 0.2);

    mat_t<double> host_param = param0;
    cache_updator_t<double, sgd_t> host_upd(0.1);

    cuda::dev_matrix_t<double> dev_param(rows, cols, param0);
    cuda::dev_cache_updator_t<double, cuda::dev_sgd_t> dev_upd(0.1);

    // 累积三份梯度后 step 一次，等价于「3 个 micro-batch 的平均」——与主机同式
    for (int micro = 0; micro < 3; ++micro)
    {
        mat_t<double> grad = make_host<double>(rows, cols, 0.07 + 0.05 * micro);
        cuda::dev_matrix_t<double> dev_grad(rows, cols, grad);

        host_upd.update(grad, host_param);
        dev_upd.update(dev_grad.const_leaf(), dev_param.buffer());
    }

    ASSERT_EQ(dev_upd.count(), 3);
    host_upd.step();
    dev_upd.step(dev_param.buffer());

    expect_close(dev_param.download(), host_param, 1e-12, "累积后的参数");
}

// ===========================================================================
// 逐元素层：SiLU
// ===========================================================================

TEST_F(CudaBackwardTest, SiluGradientMatchesFiniteDifference)
{
    cuda::dev_silu_t<double> layer;
    check_gradients(layer, make_signed_host<double>(5, 6, 0.7), make_host<double>(5, 6, 0.4),
                    1e-5);
}

TEST_F(CudaBackwardTest, SiluMatchesHostNet)
{
    mat_t<double> x = make_signed_host<double>(5, 6, 0.9);
    mat_t<double> upstream = make_host<double>(5, 6, 0.3);

    silu_net_t<mat_t<double>> host_net;
    host_net.forward(x);
    mat_t<double> host_dx = host_net.backward(upstream);

    cuda::dev_silu_t<double> dev_net;
    cuda::dev_matrix_t<double> dev_x(5, 6, x);
    cuda::dev_matrix_t<double> dev_up(5, 6, upstream);
    dev_net.forward(dev_x.leaf());
    mat_t<double> dev_dx = dev_net.backward(dev_up).download();

    expect_close(dev_dx, host_dx, 1e-12, "silu 输入梯度");
}

// ===========================================================================
// 归一化层
// ===========================================================================

TEST_F(CudaBackwardTest, LayerNormGradientMatchesFiniteDifference)
{
    cuda::dev_layer_norm_t<double, cuda::dev_sgd_t> layer;
    layer.set_param(6);
    check_gradients(layer, make_host<double>(6, 5, 0.8), make_signed_host<double>(6, 5, 0.5), 1e-5);
}

TEST_F(CudaBackwardTest, LayerNormMatchesHostNet)
{
    const int d = 6, t = 5;
    mat_t<double> x = make_host<double>(d, t, 0.8);
    mat_t<double> upstream = make_signed_host<double>(d, t, 0.5);
    const double lr = 0.05;

    layer_norm_net_t<mat_t<double>, sgd_t> host_net;
    host_net.set_param(d);
    host_net.set_lr(lr);

    cuda::dev_layer_norm_t<double, cuda::dev_sgd_t> dev_net;
    dev_net.set_param(d);
    dev_net.set_lr(lr);

    // 前向也要对拍：反向对了不能说明前向对了（可能两边都错在同一个地方）
    mat_t<double> host_y = host_net.forward(x);
    cuda::dev_matrix_t<double> dev_x(d, t, x);
    cuda::dev_matrix_t<double> dev_y = dev_net.forward(dev_x.leaf());
    expect_close(dev_y.download(), host_y, 1e-12, "layernorm 前向");

    mat_t<double> host_gama_before = host_net.gama();

    mat_t<double> host_dx = host_net.backward(upstream);
    cuda::dev_matrix_t<double> dev_up(d, t, upstream);
    mat_t<double> dev_dx = dev_net.backward(dev_up).download();

    expect_close(dev_dx, host_dx, 1e-11, "layernorm 输入梯度");

    // gamma / beta 的梯度没有直接返回值，用「参数差值 / 学习率」反推（sgd 下精确）
    mat_t<double> host_dgama = (host_gama_before - host_net.gama()) / lr;
    mat_t<double> dev_dgama = (host_gama_before - dev_net.gama_to_host()) / lr;
    expect_close(dev_dgama, host_dgama, 1e-11, "layernorm gamma 梯度");
}

TEST_F(CudaBackwardTest, RmsNormGradientMatchesFiniteDifference)
{
    cuda::dev_rms_norm_t<double, cuda::dev_sgd_t> layer;
    layer.set_param(6);
    check_gradients(layer, make_host<double>(6, 5, 0.9), make_signed_host<double>(6, 5, 0.6), 1e-5);
}

TEST_F(CudaBackwardTest, RmsNormMatchesHostNet)
{
    const int d = 6, t = 5;
    mat_t<double> x = make_host<double>(d, t, 0.9);
    mat_t<double> upstream = make_signed_host<double>(d, t, 0.6);
    const double lr = 0.05;

    rms_norm_net_t<mat_t<double>, sgd_t> host_net;
    host_net.set_param(d);
    host_net.set_lr(lr);

    cuda::dev_rms_norm_t<double, cuda::dev_sgd_t> dev_net;
    dev_net.set_param(d);
    dev_net.set_lr(lr);

    mat_t<double> host_y = host_net.forward(x);
    cuda::dev_matrix_t<double> dev_x(d, t, x);
    cuda::dev_matrix_t<double> dev_y = dev_net.forward(dev_x.leaf());
    expect_close(dev_y.download(), host_y, 1e-12, "rmsnorm 前向");

    // gamma 给一个非 1 的值，否则「乘 gamma」这条路径退化成恒等、测不出错误
    mat_t<double> gamma(d, 1);
    for (int i = 0; i < d; ++i)
        gamma(i, 0) = 0.5 + 0.1 * i;
    host_net.gama() = gamma;
    dev_net.upload_gama(gamma);

    mat_t<double> host_dx = host_net.backward(upstream);
    cuda::dev_matrix_t<double> dev_up(d, t, upstream);
    mat_t<double> dev_dx = dev_net.backward(dev_up).download();

    expect_close(dev_dx, host_dx, 1e-11, "rmsnorm 输入梯度");

    mat_t<double> host_dgama = (gamma - host_net.gama()) / lr;
    mat_t<double> dev_dgama = (gamma - dev_net.gama_to_host()) / lr;
    expect_close(dev_dgama, host_dgama, 1e-11, "rmsnorm gamma 梯度");
}

// ===========================================================================
// softmax 反向
// ===========================================================================

TEST_F(CudaBackwardTest, SoftmaxGradientMatchesFiniteDifference)
{
    // 用 softmax_rows 前向 + softmax_backward；这里手工串起来
    mat_t<double> x = make_signed_host<double>(4, 6, 0.8);
    mat_t<double> upstream = make_host<double>(4, 6, 0.5);

    auto forward_of = [](const mat_t<double>& host_x) {
        cuda::dev_matrix_t<double> d(host_x.row_num(), host_x.col_num(), host_x);
        return cuda::softmax_rows(d.const_leaf());
    };

    // 中心差分：L = Σ w ⊙ upstream
    cuda::dev_matrix_t<double> up(4, 6, upstream);
    const double eps = 1e-6;
    mat_t<double> fd(4, 6);
    mat_t<double> probe = x;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            const double origin = x(i, j);
            probe(i, j) = origin + eps;
            double fp = cuda::sum_all(forward_of(probe).leaf() * up.const_leaf());
            probe(i, j) = origin - eps;
            double fm = cuda::sum_all(forward_of(probe).leaf() * up.const_leaf());
            probe(i, j) = origin;
            fd(i, j) = (fp - fm) / (2 * eps);
        }
    }

    cuda::dev_matrix_t<double> w = forward_of(x);
    mat_t<double> analytic = cuda::softmax_backward(w.const_leaf(), up.const_leaf()).download();

    expect_close(analytic, fd, 1e-5, "softmax 反向 vs 有限差分");
}

TEST_F(CudaBackwardTest, SoftmaxMatchesHostSoftmaxNet)
{
    mat_t<double> x = make_signed_host<double>(4, 6, 0.8);
    mat_t<double> upstream = make_host<double>(4, 6, 0.5);

    hsoftmax_net_t<mat_t<double>> host_net;
    host_net.forward(x);
    mat_t<double> host_dx = host_net.backward(upstream);

    cuda::dev_matrix_t<double> dev_x(4, 6, x);
    cuda::dev_matrix_t<double> w = cuda::softmax_rows(dev_x.const_leaf());
    cuda::dev_matrix_t<double> dev_up(4, 6, upstream);
    mat_t<double> dev_dx = cuda::softmax_backward(w.const_leaf(), dev_up.const_leaf()).download();

    expect_close(dev_dx, host_dx, 1e-12, "softmax 输入梯度");
}

// ===========================================================================
// RoPE 反向
// ===========================================================================

TEST_F(CudaBackwardTest, RopeMatchesHostNetBothLayouts)
{
    const int d = 6, t = 4;
    mat_t<double> x = make_signed_host<double>(d, t, 0.7);
    mat_t<double> upstream = make_host<double>(d, t, 0.4);

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        RoPE_net_t<mat_t<double>> host_net(d);
        host_net.set_pair_layout(layout);

        cuda::dev_rope_t<double> dev_net(d, layout);

        // 前向先对拍
        mat_t<double> host_y = host_net.forward_at(x, 0);
        cuda::dev_matrix_t<double> dev_x(d, t, x);
        mat_t<double> dev_y = dev_net.forward_at(dev_x.const_leaf(), 0).download();
        expect_close(dev_y, host_y, 1e-12,
                     layout == rope_pair_layout::half_split ? "rope 前向(half_split)"
                                                            : "rope 前向(interleaved)");

        // 反向对拍（主机 backward 隐含 start_pos == 0）
        mat_t<double> host_dx = host_net.backward(upstream);
        cuda::dev_matrix_t<double> dev_up(d, t, upstream);
        mat_t<double> dev_dx = dev_net.backward(dev_up.const_leaf()).download();
        expect_close(dev_dx, host_dx, 1e-12,
                     layout == rope_pair_layout::half_split ? "rope 反向(half_split)"
                                                            : "rope 反向(interleaved)");
    }
}

TEST_F(CudaBackwardTest, RopeBackwardIsOrthogonalTranspose)
{
    // 旋转是正交变换：forward 之后 backward 应当把数据原样转回来。
    // 这条不依赖主机实现，且能抓住「反向用了错的角度」这类错误（比如漏了 start_pos）。
    const int d = 8, t = 5;
    mat_t<double> x = make_signed_host<double>(d, t, 0.9);

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        for (int start_pos : {0, 3})
        {
            cuda::dev_rope_t<double> dev_net(d, layout);
            cuda::dev_matrix_t<double> dev_x(d, t, x);

            cuda::dev_matrix_t<double> rotated = dev_net.forward_at(dev_x.const_leaf(), start_pos);
            cuda::dev_matrix_t<double> back = dev_net.backward_at(rotated.const_leaf(), start_pos);

            expect_close(back.download(), x, 1e-11, "rope 转置回旋");
        }
    }
}

TEST_F(CudaBackwardTest, RopeGradientMatchesFiniteDifference)
{
    const int d = 6, t = 3;
    mat_t<double> x = make_signed_host<double>(d, t, 0.7);
    mat_t<double> upstream = make_host<double>(d, t, 0.4);

    for (auto layout : {rope_pair_layout::interleaved, rope_pair_layout::half_split})
    {
        cuda::dev_rope_t<double> layer(d, layout);
        cuda::dev_matrix_t<double> up(d, t, upstream);

        const double eps = 1e-6;
        mat_t<double> fd(d, t);
        mat_t<double> probe = x;
        for (int i = 0; i < d; ++i)
        {
            for (int j = 0; j < t; ++j)
            {
                const double origin = x(i, j);

                probe(i, j) = origin + eps;
                cuda::dev_matrix_t<double> xp(d, t, probe);
                const double fp = cuda::sum_all(layer.forward_at(xp.const_leaf(), 0).leaf()
                                                * up.const_leaf());

                probe(i, j) = origin - eps;
                cuda::dev_matrix_t<double> xm(d, t, probe);
                const double fm = cuda::sum_all(layer.forward_at(xm.const_leaf(), 0).leaf()
                                                * up.const_leaf());

                probe(i, j) = origin;
                fd(i, j) = (fp - fm) / (2 * eps);
            }
        }

        cuda::dev_matrix_t<double> dev_up(d, t, upstream);
        mat_t<double> analytic = layer.backward(dev_up.const_leaf()).download();
        expect_close(analytic, fd, 1e-5, "rope 反向 vs 有限差分");
    }
}

// ===========================================================================
// 线性层
// ===========================================================================

TEST_F(CudaBackwardTest, LinearGradientMatchesFiniteDifference)
{
    const int in = 5, out = 4, t = 3;

    cuda::dev_linear_t<double, cuda::dev_sgd_t> layer;
    layer.set_param(in, out);
    layer.upload_weight(make_host<double>(out, in, 0.3));
    layer.upload_bias(make_host<double>(out, 1, 0.1));

    layer.set_lr(0.0);  // 差分阶段参数必须固定（backward 里会更新参数）
    check_gradients(layer, make_host<double>(in, t, 0.6), make_signed_host<double>(out, t, 0.5), 1e-5);
}

TEST_F(CudaBackwardTest, LinearMatchesHostWeightNet)
{
    const int in = 5, out = 4, t = 3;
    const double lr = 0.05;

    mat_t<double> w = make_host<double>(out, in, 0.3);
    mat_t<double> b = make_host<double>(out, 1, 0.1);
    mat_t<double> x = make_host<double>(in, t, 0.6);
    mat_t<double> upstream = make_signed_host<double>(out, t, 0.5);

    weight_net_t<mat_t<double>, sgd_t> host_net(in, out);
    host_net.weight() = w;
    host_net.bias() = b;
    host_net.set_lr(lr);

    cuda::dev_linear_t<double, cuda::dev_sgd_t> dev_net(in, out);
    dev_net.upload_weight(w);
    dev_net.upload_bias(b);
    dev_net.set_lr(lr);

    mat_t<double> host_y = host_net.forward(x);
    cuda::dev_matrix_t<double> dev_x(in, t, x);
    cuda::dev_matrix_t<double> dev_y = dev_net.forward(dev_x.leaf());
    expect_close(dev_y.download(), host_y, 1e-12, "linear 前向");

    mat_t<double> host_dx = host_net.backward(upstream);
    cuda::dev_matrix_t<double> dev_up(out, t, upstream);
    mat_t<double> dev_dx = dev_net.backward(dev_up).download();
    expect_close(dev_dx, host_dx, 1e-11, "linear 输入梯度");

    // 权重/偏置梯度用「初始参数 − 更新后参数，再除以学习率」反推
    mat_t<double> host_dw = (w - host_net.weight()) / lr;
    mat_t<double> dev_dw = (w - dev_net.weight_to_host()) / lr;
    expect_close(dev_dw, host_dw, 1e-11, "linear 权重梯度");

    mat_t<double> host_db = (b - host_net.bias()) / lr;
    mat_t<double> dev_db = (b - dev_net.bias_to_host()) / lr;
    expect_close(dev_db, host_db, 1e-11, "linear 偏置梯度");
}

// ===========================================================================
// 端到端：一个真的在设备上训练的栈
// ===========================================================================

namespace
{

/** 一个「LayerNorm → Linear → SiLU → Linear → MSE」的小栈，主机与设备各一份。 */
struct StackSpec
{
    int d = 6;
    int hidden = 8;
    int t = 4;
    double lr = 0.02;

    mat_t<double> w1, b1, w2, b2, x, target;

    StackSpec()
    {
        w1 = make_host<double>(hidden, d, 0.2);
        b1 = make_host<double>(hidden, 1, 0.05);
        w2 = make_host<double>(d, hidden, 0.1);
        b2 = make_host<double>(d, 1, 0.02);
        x = make_signed_host<double>(d, t, 0.6);
        target = make_signed_host<double>(d, t, 0.2);
    }
};

struct HostStack
{
    layer_norm_net_t<mat_t<double>, sgd_t> ln;
    weight_net_t<mat_t<double>, sgd_t> l1;
    silu_net_t<mat_t<double>> act;
    weight_net_t<mat_t<double>, sgd_t> l2;
    mse_loss_t<mat_t<double>> loss;
    int d = 0;

    void configure(const StackSpec& s)
    {
        d = s.d;
        ln.set_param(s.d);
        l1.reinit({s.d, s.hidden});
        l2.reinit({s.hidden, s.d});
        l1.weight() = s.w1;
        l1.bias() = s.b1;
        l2.weight() = s.w2;
        l2.bias() = s.b2;
        l1.set_lr(s.lr);
        l2.set_lr(s.lr);
        ln.set_lr(s.lr);
    }

    double train_step(const StackSpec& s)
    {
        mat_t<double> h = ln.forward(s.x);
        mat_t<double> a = l1.forward(h);
        mat_t<double> g = act.forward(a);
        mat_t<double> pred = l2.forward(g);
        loss.forward(pred);

        const double value = loss.loss(s.target);

        mat_t<double> delta = loss.backward(s.target);
        delta = l2.backward(delta);
        delta = act.backward(delta);
        delta = l1.backward(delta);
        ln.backward(delta);

        ln.step();
        l1.step();
        l2.step();
        return value;
    }
};

struct DeviceStack
{
    cuda::dev_layer_norm_t<double, cuda::dev_sgd_t> ln;
    cuda::dev_linear_t<double, cuda::dev_sgd_t> l1;
    cuda::dev_silu_t<double> act;
    cuda::dev_linear_t<double, cuda::dev_sgd_t> l2;
    cuda::dev_mse_loss_t<double> loss;

    void configure(const StackSpec& s)
    {
        ln.set_param(s.d);
        l1.set_param(s.d, s.hidden);
        l2.set_param(s.hidden, s.d);
        l1.upload_weight(s.w1);
        l1.upload_bias(s.b1);
        l2.upload_weight(s.w2);
        l2.upload_bias(s.b2);
        l1.set_lr(s.lr);
        l2.set_lr(s.lr);
        ln.set_lr(s.lr);
    }

    double train_step(const StackSpec& s)
    {
        cuda::dev_matrix_t<double> x(s.d, s.t, s.x);
        cuda::dev_matrix_t<double> target(s.d, s.t, s.target);

        cuda::dev_matrix_t<double> h = ln.forward(x.const_leaf());
        cuda::dev_matrix_t<double> a = l1.forward(h);
        cuda::dev_matrix_t<double> g = act.forward(a);
        cuda::dev_matrix_t<double> pred = l2.forward(g);
        loss.forward(pred);

        const double value = loss.loss(target);

        cuda::dev_matrix_t<double> delta = loss.backward(target);
        delta = l2.backward(delta);
        delta = act.backward(delta);
        delta = l1.backward(delta);
        ln.backward(delta);

        ln.step();
        l1.step();
        l2.step();
        return value;
    }
};

} // namespace

TEST_F(CudaBackwardTest, TrainableStackMatchesHostStepByStep)
{
    StackSpec spec;

    HostStack host;
    host.configure(spec);

    DeviceStack dev;
    dev.configure(spec);

    // 每一步都比较损失值。容差刻意分两段：
    //   - 第 0 步两边走的是**同一批参数**，只有 GEMM 求和顺序不同 → 可以卡很紧
    //     （这一步卡紧才有意义：它证明梯度算对了）
    //   - 之后每步的差异会被训练这个反馈过程放大，容差必须放宽，否则测的是混沌而不是正确性
    for (int step = 0; step < 6; ++step)
    {
        const double host_loss = host.train_step(spec);
        const double dev_loss = dev.train_step(spec);

        const double tol = (step == 0) ? 1e-12 : 1e-7;
        const double scale = std::max(1.0, std::abs(host_loss));
        EXPECT_NEAR(dev_loss, host_loss, tol * scale)
            << "第 " << step << " 步损失不一致（host " << host_loss << " vs dev " << dev_loss << "）";

        // 参数也要跟着收敛到同一处，否则说明某一条路径的更新语义不同
        expect_close(dev.l1.weight_to_host(), host.l1.weight(), (step == 0) ? 1e-11 : 1e-6,
                     "训练后 l1 权重");
    }
}

TEST_F(CudaBackwardTest, TrainableStackLossDecreases)
{
    // 与主机无关的自检：这套梯度方向**真的**能把损失降下去。
    // 上面那条对拍能证明「两边一致」，这条能证明「一致的方向是对的」。
    StackSpec spec;
    DeviceStack dev;
    dev.configure(spec);

    const double first = dev.train_step(spec);
    double last = first;
    for (int step = 1; step < 60; ++step)
        last = dev.train_step(spec);

    EXPECT_LT(last, first) << "训练 60 步后损失没有下降（" << first << " -> " << last << "）";
}
