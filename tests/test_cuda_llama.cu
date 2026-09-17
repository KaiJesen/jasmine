#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "jas_cuda_buffer.hpp"
#include "jas_cuda_llama.hpp"
#include "jas_cuda_matrix.hpp"
#include "jas_cuda_net.hpp"
#include "jas_cuda_reduce.hpp"
#include "jas_llama_t.hpp"
#include "jas_mat_express_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_mat_t.hpp"
#include "jas_updator_t.hpp"

/**
 * 设备端 LLaMA 模型的测试：参数搬运 → 前向 / 增量解码 → 反向。
 *
 * 散热约束同其它 CUDA 测试：模型刻意取极小（2 层、d_model 8、词表 7）、
 * 每个用例前后测温、超 80℃ 就 skip。详见 CUDA.md。
 *
 * ## 三层验收，一层比一层难伪造
 *
 *  1. **参数搬运后的前向对拍**：与主机 `llama_model_t` 逐元素比 `forward_stages`
 *     与 logits。搬错一个矩阵、少搬一层，这里立刻红。
 *  2. **增量解码对拍**：prefill + 逐 token 续写与主机 `forward_one` 比 ——
 *     这条覆盖的是 KV cache 与 RoPE 位置的配合（位置来自 cache 长度）。
 *  3. **反向的有限差分**：主机 `llama_model_t` 是 inference-only，**没有 backward
 *     可以对照**。所以整模型的反向用有限差分来验收：把某个具体参数矩阵的每个元素
 *     各扰动 ±ε、重新前向算一次 `L`，与「(初始参数 − 更新后参数) / 学习率」反推出的
 *     解析梯度比。这条从「前向是个求值函数」出发，能独立抓出整栈组合里的错误。
 *
 * 第 3 条对**每一个参数矩阵**都做差分（2 层 × 9 个矩阵 + `wte` + `ln_f` + `lm_head`）。
 * 只钉头、中、尾三处是不够的：整栈反向里任何一处「漏了 / 符号反了 / GQA 少累加
 * 一个共享 KV 头」都只影响它自己那一段，三处之外的错误可以安然通过。差分本来就便宜
 * （每个元素两次前向），全量覆盖换来的确定性远比省下的两秒值钱。
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

/**
 * 极小的 LLaMA 配置。
 *
 * `kv_heads < heads`（GQA）是**必须**的：MHA 下「多个头共享一个 KV 头」这条路径
 * 根本不会被执行，而它恰恰是设备端反向里唯一需要累加的地方。
 */
struct cfg_t
{
    int layers = 2;
    int heads = 2;
    int kv_heads = 1;
    int d_model = 8;
    int d_ff = 16;
    int vocab = 7;
    int n_pos = 32;
};

using host_model_t = llama_model_t<mat_t<double>, sgd_t>;
using dev_model_t = cuda::dev_llama_t<double, cuda::dev_sgd_t>;

/** 主机端随机初始化 → 搬到设备端。权重来源只有主机一处，两边必然一致。 */
void build_models(const cfg_t& c, host_model_t& host, dev_model_t& dev)
{
    host.set_param(c.layers, c.heads, c.d_model, c.d_ff, c.vocab, c.n_pos, c.kv_heads);
    host.init_weight<xavier_gaussian_t>();

    dev.set_param(c.layers, c.heads, c.d_model, c.d_ff, c.vocab, c.n_pos, c.kv_heads);
    dev.upload_from(host);
}

mat_t<double> make_ids(int seq, int vocab, unsigned seed)
{
    mat_t<double> ids(1, seq);
    unsigned s = seed;
    for (int t = 0; t < seq; ++t)
    {
        s = s * 1103515245u + 12345u;
        ids(0, t) = static_cast<double>((s >> 16) % static_cast<unsigned>(vocab));
    }
    return ids;
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

/** `L = Σ logits ⊙ upstream` —— 对 `L` 求参数梯度，正好是 `backward` 里累积的那些量。 */
double device_objective(dev_model_t& model, const mat_t<double>& ids_host,
                        const cuda::dev_matrix_t<double>& upstream)
{
    cuda::dev_matrix_t<double> ids(1, ids_host.col_num(), ids_host);
    cuda::dev_matrix_t<double> logits = model.forward(ids.const_leaf());
    return cuda::sum_all(logits.leaf() * upstream.const_leaf());
}

/**
 * 对某个参数矩阵做中心差分。
 *
 * 参数通过一对 `dump` / `put` 访问：这样同一个 helper 能覆盖三种存放方式
 * （权重是 `dev_matrix_t`、归一化的 gamma 是设备列向量），代价只是两个 lambda。
 */
struct param_probe_t
{
    std::string what;
    std::function<mat_t<double>(dev_model_t&)> dump;
    std::function<void(dev_model_t&, const mat_t<double>&)> put;
    // 下面两项由用例填：扰动前的基线快照，以及差分算出来的数值梯度
    mat_t<double> base;
    mat_t<double> fd;
};

/** 权重类（`dev_matrix_t`）：download / upload 直接对上。 */
template <typename Getter>
param_probe_t mat_probe(std::string what, Getter get)
{
    param_probe_t p;
    p.what = std::move(what);
    p.dump = [get](dev_model_t& m) { return get(m).download(); };
    p.put = [get](dev_model_t& m, const mat_t<double>& v) { get(m).upload(v); };
    return p;
}

/** 归一化的 gamma：设备端存成列向量，走各自的 `gama_to_host` / `upload_gama`。 */
template <typename Getter>
param_probe_t gama_probe(std::string what, Getter get)
{
    param_probe_t p;
    p.what = std::move(what);
    p.dump = [get](dev_model_t& m) { return get(m).gama_to_host(); };
    p.put = [get](dev_model_t& m, const mat_t<double>& v) { get(m).upload_gama(v); };
    return p;
}

mat_t<double> finite_difference_param(dev_model_t& model, const param_probe_t& p,
                                      const mat_t<double>& base, const mat_t<double>& ids,
                                      const cuda::dev_matrix_t<double>& upstream, double eps)
{
    mat_t<double> fd(base.row_num(), base.col_num());
    mat_t<double> probe = base;

    for (int i = 0; i < base.row_num(); ++i)
        for (int j = 0; j < base.col_num(); ++j)
        {
            const double origin = base(i, j);

            probe(i, j) = origin + eps;
            p.put(model, probe);
            const double lp = device_objective(model, ids, upstream);

            probe(i, j) = origin - eps;
            p.put(model, probe);
            const double lm = device_objective(model, ids, upstream);

            probe(i, j) = origin;
            fd(i, j) = (lp - lm) / (2.0 * eps);
        }

    p.put(model, base);  // 还原（后面解析反向必须在一套固定参数上做）
    return fd;
}

/** 同一次反向里给两边准备的确定性上游梯度（logits 的形状：vocab × T）。 */
mat_t<double> make_upstream(int vocab, int seq)
{
    mat_t<double> up(vocab, seq);
    for (int i = 0; i < vocab; ++i)
        for (int j = 0; j < seq; ++j)
            up(i, j) = 0.05 + 0.01 * ((i * 3 + j * 5) % 17) - 0.02 * i;
    return up;
}

/** 「初始参数 − 更新后参数，再除以学习率」＝ sgd 下的精确梯度。 */
mat_t<double> grad_of(const mat_t<double>& before, const mat_t<double>& after, double lr)
{
    mat_t<double> g(before.row_num(), before.col_num());
    for (int i = 0; i < before.row_num(); ++i)
        for (int j = 0; j < before.col_num(); ++j)
            g(i, j) = (before(i, j) - after(i, j)) / lr;
    return g;
}

} // namespace

class CudaLlamaTest : public ::testing::Test
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
// 参数搬运 + 前向
// ===========================================================================

TEST_F(CudaLlamaTest, UploadedWeightsReproduceHostForwardStageByStage)
{
    cfg_t c;
    c.d_model = 16;
    c.heads = 4;
    c.kv_heads = 2;
    c.d_ff = 24;

    host_model_t host;
    dev_model_t dev;
    build_models(c, host, dev);

    const mat_t<double> ids = make_ids(5, c.vocab, 7);

    // 逐层对拍比 logits 更有用：RMSNorm 与 lm_head 都是仿射的，会把偏差掩盖或放大
    const std::vector<mat_t<double>> want = host.forward_stages(ids);
    const std::vector<cuda::dev_matrix_t<double>> got = dev.forward_stages(ids);

    ASSERT_EQ(got.size(), want.size());
    ASSERT_EQ(got.size(), static_cast<std::size_t>(c.layers + 1));
    for (std::size_t k = 0; k < got.size(); ++k)
        expect_close(got[k].download(), want[k], 1e-11,
                     ("逐层输出 stage " + std::to_string(k)).c_str());

    expect_close(dev.forward(ids).download(), host.forward(ids), 1e-11, "logits");
}

// ===========================================================================
// 增量解码
// ===========================================================================

TEST_F(CudaLlamaTest, IncrementalDecodeMatchesHost)
{
    cfg_t c;
    c.d_model = 16;
    c.heads = 4;
    c.kv_heads = 2;
    c.d_ff = 24;

    host_model_t host;
    dev_model_t dev;
    build_models(c, host, dev);

    const mat_t<double> prompt = make_ids(4, c.vocab, 11);

    // prefill 多列：位置与因果掩码都按绝对位置算
    expect_close(dev.prefill(prompt).download(), host.prefill(prompt), 1e-11, "prefill logits");
    ASSERT_EQ(dev.kv_cache_length(), host.kv_cache_length());

    // 逐 token 续写：位置来自 cache 长度（没有位置参数）
    for (int t = 0; t < 3; ++t)
    {
        const mat_t<double> step = make_ids(1, c.vocab, 100u + static_cast<unsigned>(t));
        expect_close(dev.forward_one(step).download(), host.forward_one(step), 1e-11,
                     ("decode 第 " + std::to_string(t) + " 步").c_str());
    }
    ASSERT_EQ(dev.kv_cache_length(), host.kv_cache_length());
}

/**
 * 「整段前向」与「逐 token 前向」必须给同一个答案 —— 这是 decode 正确性的
 * 内在一致性检查，**不依赖主机实现**（主机的 `forward_one` 与 `forward` 是两条代码路径，
 * 一起错的可能存在）。
 */
TEST_F(CudaLlamaTest, DeviceDecodeAgreesWithDeviceFullForward)
{
    cfg_t c;
    c.d_model = 16;
    c.heads = 4;
    c.kv_heads = 2;
    c.d_ff = 24;

    host_model_t host;
    dev_model_t dev;
    build_models(c, host, dev);

    const mat_t<double> ids = make_ids(4, c.vocab, 23);

    const mat_t<double> full = dev.forward(ids).download();

    dev.clear_kv_cache();
    mat_t<double> step_logits;
    for (int t = 0; t < ids.col_num(); ++t)
    {
        mat_t<double> one(1, 1);
        one(0, 0) = ids(0, t);
        step_logits = dev.forward_one(one).download();
    }

    mat_t<double> last(full.row_num(), 1);
    for (int i = 0; i < full.row_num(); ++i)
        last(i, 0) = full(i, ids.col_num() - 1);
    expect_close(step_logits, last, 1e-11, "逐 token 最后一个位置 vs 整段前向");
}

// ===========================================================================
// 反向：主机没有 backward 可对照，所以用有限差分
// ===========================================================================

TEST_F(CudaLlamaTest, BackwardMatchesFiniteDifference)
{
    cfg_t c;  // 2 层 / 2 Q 头 / 1 KV 头 / d_model 8 / T=3：差分次数压到两千次以内
    host_model_t host;
    dev_model_t dev;
    build_models(c, host, dev);

    constexpr int seq = 3;
    constexpr double eps = 1e-6;
    constexpr double lr = 0.01;

    const mat_t<double> ids = make_ids(seq, c.vocab, 31);
    const mat_t<double> upstream = make_upstream(c.vocab, seq);
    cuda::dev_matrix_t<double> up(c.vocab, seq, upstream);

    dev.set_lr(lr);

    // ---- 探针清单：**每一个参数矩阵**都在里面 ----
    //
    // 只钉头、中、尾三处（头 = wte、中 = 某一层的 W_Q、尾 = lm_head）过不了这一关：
    // 整栈反向里任何一处「漏了 / 符号反了 / 少累加一个共享 KV 头」都只影响它自己
    // 那一段，三处之外的错误可以安然通过。差分本来就便宜，索性全量。
    std::vector<param_probe_t> probes;
    probes.push_back(mat_probe("wte 权重（散射累加，含重复 id）",
                               [](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                   return m.wte().weight();
                               }));
    for (int layer = 0; layer < c.layers; ++layer)
    {
        const std::string p = "block" + std::to_string(layer) + ".";
        probes.push_back(gama_probe(p + "ln_1.gamma",
                                    [layer](dev_model_t& m) -> cuda::dev_rms_norm_t<double, cuda::dev_sgd_t>& {
                                        return m.block(layer).ln_1();
                                    }));
        probes.push_back(mat_probe(p + "attn.W_Q",
                                   [layer](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                       return m.block(layer).attn().q_proj().weight();
                                   }));
        probes.push_back(mat_probe(p + "attn.W_K（GQA：2 个 Q 头共享）",
                                   [layer](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                       return m.block(layer).attn().k_proj().weight();
                                   }));
        probes.push_back(mat_probe(p + "attn.W_V（GQA：2 个 Q 头共享）",
                                   [layer](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                       return m.block(layer).attn().v_proj().weight();
                                   }));
        probes.push_back(mat_probe(p + "attn.W_O",
                                   [layer](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                       return m.block(layer).attn().out_proj().weight();
                                   }));
        probes.push_back(gama_probe(p + "ln_2.gamma",
                                    [layer](dev_model_t& m) -> cuda::dev_rms_norm_t<double, cuda::dev_sgd_t>& {
                                        return m.block(layer).ln_2();
                                    }));
        probes.push_back(mat_probe(p + "mlp.gate（linear→SiLU）",
                                   [layer](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                       return m.block(layer).mlp_gate().first().weight();
                                   }));
        probes.push_back(mat_probe(p + "mlp.up",
                                   [layer](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                       return m.block(layer).mlp_up().weight();
                                   }));
        probes.push_back(mat_probe(p + "mlp.down",
                                   [layer](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                       return m.block(layer).mlp_down().weight();
                                   }));
    }
    probes.push_back(gama_probe("ln_f.gamma",
                                [](dev_model_t& m) -> cuda::dev_rms_norm_t<double, cuda::dev_sgd_t>& {
                                    return m.ln_f();
                                }));
    probes.push_back(mat_probe("lm_head 权重",
                               [](dev_model_t& m) -> cuda::dev_matrix_t<double>& {
                                   return m.lm_head().weight();
                               }));

    // ---- 1. 基线 + 差分（差分只走前向，每条探针自己还原）----
    for (param_probe_t& p : probes)
        p.base = p.dump(dev);
    for (param_probe_t& p : probes)
        p.fd = finite_difference_param(dev, p, p.base, ids, up, eps);

    // ---- 2. 解析反向（会就地更新所有参数）----
    cuda::dev_matrix_t<double> ids_dev(1, seq, ids);
    dev.forward(ids_dev.const_leaf());
    dev.backward(up);

    // ---- 3. 逐组比 ----
    for (param_probe_t& p : probes)
        expect_close(grad_of(p.base, p.dump(dev), lr), p.fd, 1e-5, p.what.c_str());
}

/**
 * 训练确实在收敛：把同一批数据反复喂给设备端模型，损失应当明显下降。
 *
 * 这条**不是**正确性证明（9.4 那节记过一个「损失在降但公式是错的」的真实事故），
 * 它是给上面那条差分用例做「方向确认」：差分证明了梯度与数值梯度一致，
 * 这条证明沿着那个梯度走确实能优化目标 —— 两条合起来才排除了「梯度对但接线错
 * （比如反向用错了缓存）」这类只在多步迭代里才显形的问题。
 *
 * 顺带钉住一个容易误判的现象：**朴素 SGD 的步长有硬上限**（`lr < 2/λ_max`），
 * 超一点就是单调发散。所以「损失炸了」首先要怀疑步长，而不是先怀疑梯度 ——
 * 分清楚这两件事，靠的就是上面那条差分用例。
 */
TEST_F(CudaLlamaTest, OverfitTinyBatchDrivesLossDown)
{
    cfg_t c;
    c.layers = 1;
    c.heads = 2;
    c.kv_heads = 1;
    c.d_model = 8;
    c.d_ff = 16;
    c.vocab = 6;

    host_model_t host;
    dev_model_t dev;
    build_models(c, host, dev);

    constexpr int seq = 3;
    const mat_t<double> ids = make_ids(seq, c.vocab, 5);
    const mat_t<double> target = make_upstream(c.vocab, seq);

    cuda::dev_matrix_t<double> ids_dev(1, seq, ids);
    cuda::dev_matrix_t<double> target_dev(c.vocab, seq, target);
    cuda::dev_mse_loss_t<double> loss_net;

    // 步长取 0.05 而不是更大：本用例跑的是**朴素 SGD**，而 SGD 只保证在
    // `lr < 2/λ_max`（Hessian 最大特征值）内收敛。这批参数的 λ_max 不大，
    // 实测 lr=0.2 时损失单调**增大**（60 步后 1e76）—— 那是步长越界，不是梯度错。
    // 「梯度对不对」由上一条差分用例负责（它用 lr=0.01，一次更新即可判定），
    // 本条只负责「沿着那个梯度走确实能优化目标」。
    constexpr double lr = 0.05;
    constexpr int steps = 60;
    dev.set_lr(lr);

    // 损失的**上游梯度**是 `y − target`（主机 `mse_loss_t::backward` 的语义，
    // 没有 2/N 因子）。这里直接用设备端损失层算，避免手写出第二个版本。
    auto step_loss = [&]() {
        cuda::dev_matrix_t<double> logits = dev.forward(ids_dev.const_leaf());
        loss_net.forward(logits.const_leaf());
        return loss_net.loss(target_dev);
    };

    const double first = step_loss();
    std::vector<double> traj{first};
    for (int step = 0; step < steps; ++step)
    {
        cuda::dev_matrix_t<double> logits = dev.forward(ids_dev.const_leaf());
        loss_net.forward(logits.const_leaf());
        dev.backward(loss_net.backward(target_dev));
        dev.step();
        if ((step + 1) % 10 == 0 || step + 1 == steps)
            traj.push_back(step_loss());
    }
    const double last = traj.back();

    // 打印整条轨迹而不是首尾两点：发散、抖动、平台期在轨迹里一眼可辨，
    // 只给首尾则「下降得慢」和「先降后炸」看起来一样。
    std::fprintf(stderr, "[收敛] lr=%g %d 步:", lr, steps);
    for (double v : traj)
        std::fprintf(stderr, " %.4g", v);
    std::fprintf(stderr, "\n");
    EXPECT_LT(last, first * 0.2) << "损失没有明显下降（" << first << " → " << last << "）";
}
