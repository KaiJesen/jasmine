#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_gpt2_t.hpp"
#include "jas_gelu_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_weight_io.hpp"
#include "test_helpers.hpp"

using namespace jasmine;
template <typename val_type>
using gpt2_upr_tpl = cache_updator_t<val_type, nadam_t>;

using gpt2_type = gpt2_model_t<mat_t<double>, gpt2_upr_tpl>;

namespace
{

double MaxAbsDiff(const mat_t<double>& a, const mat_t<double>& b)
{
    double m = 0.0;
    for (int i = 0; i < a.row_num(); ++i)
        for (int j = 0; j < a.col_num(); ++j)
            m = std::max(m, std::abs(a(i, j) - b(i, j)));
    return m;
}

/** 小配置随机权重模型：d_model=8, heads=2 -> d_head=4 */
gpt2_type make_small_gpt2(int layers = 2, int heads = 2, int d_model = 8,
                          int d_ff = 32, int vocab = 11, int n_pos = 16)
{
    gpt2_type model(layers, heads, d_model, d_ff, vocab, n_pos);
    model.init_weight<xavier_gaussian_t>();
    return model;
}

/**
 * 找一个可用的 GPT-2 权重文件（含 HF 黄金值）。
 * 顺序：环境变量 JASMINE_GPT2_WEIGHTS -> ctest 工作目录下若干默认名字。
 * 找不到就 GTEST_SKIP，避免 CI 依赖 300MB 大文件。
 */
std::string find_gpt2_weights()
{
    std::vector<std::string> candidates;
    if (const char* env = std::getenv("JASMINE_GPT2_WEIGHTS"))
        candidates.emplace_back(env);
    for (const char* name : {"distilgpt2_weights.bin", "gpt2_weights.bin"})
    {
        candidates.emplace_back(name);
        candidates.emplace_back(std::string("../") + name);
        candidates.emplace_back(std::string("build/") + name);
    }
    for (const auto& c : candidates)
        if (!c.empty() && std::filesystem::exists(c))
            return c;
    return {};
}

/** 需要导出权重（含黄金值）的测试基类 */
class Gpt2AlignmentTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_path = find_gpt2_weights();
        if (m_path.empty())
        {
            GTEST_SKIP() << "no GPT-2 weight file found; run "
                            "`python tools/export_gpt2.py --model distilgpt2 "
                            "--out build/distilgpt2_weights.bin --golden-prompt \"Hello, my dog is cute\"` "
                            "or set JASMINE_GPT2_WEIGHTS";
        }
        ASSERT_NO_THROW(m_wf.load(m_path)) << "failed to load " << m_path;
        ASSERT_TRUE(m_wf.has("wte.weight")) << "weight file looks malformed: " << m_path;
        ASSERT_TRUE(m_wf.has("golden.logits"))
            << "weight file has no golden logits; re-export with --golden-prompt";
    }

    weight_file_t m_wf;
    std::string m_path;

    gpt2_type make_model() const
    {
        auto cfg = read_gpt2_config(m_wf);
        return gpt2_type(cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff, cfg.vocab, cfg.n_pos);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// GELU（gelu_new / tanh 近似）
// ---------------------------------------------------------------------------

TEST(Gelu, MatchesTanhFormula)
{
    const double alpha = 0.7978845608028654;
    const double beta = 0.044715;
    for (double x : {-3.0, -1.0, -0.5, 0.0, 0.25, 1.0, 2.0, 5.0})
    {
        const double expect = 0.5 * x * (1.0 + std::tanh(alpha * (x + beta * x * x * x)));
        EXPECT_NEAR(gelu_net_t<mat_t<double>>::gelu(x), expect, 1e-12) << "x=" << x;
    }
}

TEST(Gelu, KnownValues)
{
    // 与 pytorch F.gelu(approximate='tanh') 一致
    EXPECT_NEAR(gelu_net_t<mat_t<double>>::gelu(0.0), 0.0, 1e-12);
    EXPECT_NEAR(gelu_net_t<mat_t<double>>::gelu(1.0), 0.8411919906082768, 1e-12);
    EXPECT_NEAR(gelu_net_t<mat_t<double>>::gelu(-1.0), -0.15880800939172324, 1e-12);
    EXPECT_NEAR(gelu_net_t<mat_t<double>>::gelu(0.5), 0.34571400982514394, 1e-12);
    EXPECT_NEAR(gelu_net_t<mat_t<double>>::gelu(2.0), 1.954597694087775, 1e-12);
    EXPECT_NEAR(gelu_net_t<mat_t<double>>::gelu(8.0), 8.0, 1e-6);
}

TEST(Gelu, ForwardShapeAndValues)
{
    gelu_net_t<mat_t<double>> act;
    mat_t<double> in(3, 2, {-1.0, 0.5, 2.0, -0.25, 0.0, 3.0});
    auto out = act.forward(in);
    ExpectShape(out, 3, 2);
    EXPECT_NEAR(out(0, 0), gelu_net_t<mat_t<double>>::gelu(-1.0), 1e-12);
    EXPECT_NEAR(out(2, 1), gelu_net_t<mat_t<double>>::gelu(3.0), 1e-12);
}

TEST(Gelu, BackwardMatchesNumericalGradient)
{
    gelu_net_t<mat_t<double>> act;
    const double x = 0.7;
    mat_t<double> in(1, 1, {x});
    act.forward(in);
    mat_t<double> delta(1, 1, {1.0});
    const double analytic = act.backward(delta)(0, 0);

    const double h = 1e-6;
    const double num = (gelu_net_t<mat_t<double>>::gelu(x + h) -
                        gelu_net_t<mat_t<double>>::gelu(x - h)) / (2 * h);
    EXPECT_NEAR(analytic, num, 1e-6);
}

// ---------------------------------------------------------------------------
// 权重文件读写（合成小文件，不依赖 HF 导出）
// ---------------------------------------------------------------------------

namespace
{

std::string write_synthetic_weight_file()
{
    const std::string path = "test_gpt2_synthetic_weights.bin";
    std::vector<std::string> lines = {
        "JASMINE_WEIGHTS_V1",
        "f32",
        "3",
        "a.weight 2 3 0",       // 6 floats, 24 bytes
        "b.weight 1 2 24",      // 2 floats,  8 bytes
        "c.scalar 1 1 32",      // 1 float,   4 bytes
        "",
    };
    std::ofstream out(path, std::ios::binary);
    std::string header;
    for (const auto& l : lines) header += l + "\n";
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const std::vector<float> a = {1, 2, 3, 4, 5, 6};
    const std::vector<float> b = {7.5f, -8.25f};
    const std::vector<float> c = {42.5f};
    out.write(reinterpret_cast<const char*>(a.data()), sizeof(float) * a.size());
    out.write(reinterpret_cast<const char*>(b.data()), sizeof(float) * b.size());
    out.write(reinterpret_cast<const char*>(c.data()), sizeof(float) * c.size());
    return path;
}

} // namespace

TEST(Gpt2WeightIo, ReadsBackRowMajor)
{
    const std::string path = write_synthetic_weight_file();
    weight_file_t wf;
    wf.load(path);
    EXPECT_EQ(wf.size(), 3u);
    EXPECT_TRUE(wf.has("a.weight"));
    EXPECT_FALSE(wf.has("nope"));

    mat_t<double> a(2, 3);
    wf.read_into("a.weight", a);
    EXPECT_DOUBLE_EQ(a(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(a(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(a(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(a(1, 2), 6.0);

    mat_t<float> b(1, 2);
    wf.read_into("b.weight", b);
    EXPECT_FLOAT_EQ(b(0, 0), 7.5f);
    EXPECT_FLOAT_EQ(b(0, 1), -8.25f);

    EXPECT_DOUBLE_EQ(wf.read_scalar<double>("c.scalar"), 42.5);
    std::filesystem::remove(path);
}

TEST(Gpt2WeightIo, RejectsShapeMismatchAndMissing)
{
    const std::string path = write_synthetic_weight_file();
    weight_file_t wf;
    wf.load(path);

    mat_t<double> wrong(3, 2);
    EXPECT_THROW(wf.read_into("a.weight", wrong), std::runtime_error);
    mat_t<double> ok(2, 3);
    EXPECT_THROW(wf.read_into("does.not.exist", ok), std::runtime_error);
    EXPECT_THROW(wf.entry("does.not.exist"), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Gpt2WeightIo, WriterReaderRoundTrip)
{
    const std::string path = "test_gpt2_roundtrip.bin";
    mat_t<double> a(3, 2, {1.5, -2.25, 3.0, 0.125, -7.75, 4.0});
    mat_t<double> b(1, 1, {42.5});

    {
        weight_writer_t w;
        w.add("a", a);
        w.add("b", b);
        EXPECT_EQ(w.size(), 2u);
        w.write(path);
    }

    weight_file_t r;
    r.load(path);
    EXPECT_EQ(r.size(), 2u);

    mat_t<double> ra(3, 2);
    r.read_into("a", ra);
    ExpectNearMat(ra, a, 0.0);          // float32 可精确表示这些值

    EXPECT_DOUBLE_EQ(r.read_scalar<double>("b"), 42.5);
    std::filesystem::remove(path);
}

TEST(Gpt2WeightIo, WriterRejectsBadShape)
{
    weight_writer_t w;
    mat_t<double> empty;
    EXPECT_THROW(w.add("empty", empty), std::runtime_error);
}

TEST(Gpt2WeightIo, RejectsBadMagic)
{
    const std::string path = "test_gpt2_bad_magic.bin";
    std::ofstream out(path, std::ios::binary);
    const std::string junk = "NOT_A_JASMINE_FILE\nf32\n0\n\n";
    out.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    out.close();

    weight_file_t wf;
    EXPECT_THROW(wf.load(path), std::runtime_error);
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// GPT-2 结构（随机权重，验证拓扑与数据流）
// ---------------------------------------------------------------------------

TEST(Gpt2Structure, ForwardShape)
{
    auto model = make_small_gpt2();
    mat_t<double> ids(1, 4, {1.0, 2.0, 3.0, 4.0});
    auto logits = model.forward(ids);
    ExpectShape(logits, 11, 4);
    for (int i = 0; i < logits.row_num(); ++i)
        for (int j = 0; j < logits.col_num(); ++j)
            EXPECT_TRUE(std::isfinite(logits(i, j)));
}

TEST(Gpt2Structure, CausalFirstPositionStable)
{
    auto model = make_small_gpt2();
    mat_t<double> ids1(1, 1, {5.0});
    mat_t<double> ids4(1, 4, {5.0, 7.0, 2.0, 9.0});

    auto out1 = model.forward(ids1);
    auto out4 = model.forward(ids4);
    for (int i = 0; i < out1.row_num(); ++i)
        EXPECT_NEAR(out1(i, 0), out4(i, 0), 1e-9) << "row " << i;
}

TEST(Gpt2Structure, FutureTokensDoNotAffectPast)
{
    auto model = make_small_gpt2();
    mat_t<double> a(1, 3, {1.0, 2.0, 3.0});
    mat_t<double> b(1, 3, {1.0, 2.0, 10.0});   // 只改最后一位
    auto oa = model.forward(a);
    auto ob = model.forward(b);
    for (int i = 0; i < oa.row_num(); ++i)
    {
        EXPECT_NEAR(oa(i, 0), ob(i, 0), 1e-9);
        EXPECT_NEAR(oa(i, 1), ob(i, 1), 1e-9);
    }
}

TEST(Gpt2Structure, AbsolutePositionChangesOutput)
{
    auto model = make_small_gpt2();
    mat_t<double> one(1, 1, {3.0});
    mat_t<double> two(1, 2, {3.0, 3.0});

    auto out = model.forward(two);
    bool any_diff = false;
    for (int i = 0; i < out.row_num(); ++i)
        if (std::abs(out(i, 0) - out(i, 1)) > 1e-9) any_diff = true;
    EXPECT_TRUE(any_diff) << "绝对位置嵌入未生效";

    model.clear_kv_cache();
    auto at0 = model.forward_one(one, 0);
    model.clear_kv_cache();
    auto at5 = model.forward_one(one, 5);
    any_diff = false;
    for (int i = 0; i < at0.row_num(); ++i)
        if (std::abs(at0(i, 0) - at5(i, 0)) > 1e-9) any_diff = true;
    EXPECT_TRUE(any_diff) << "forward_one 未按 pos 取 wpe";
}

TEST(Gpt2Structure, ForwardOneMatchesFullForward)
{
    auto model = make_small_gpt2(2, 2, 8, 32, 11, 16);
    mat_t<double> ids(1, 5, {1.0, 4.0, 2.0, 7.0, 3.0});

    auto ref = model.forward(ids);

    model.clear_kv_cache();
    for (int t = 0; t < 5; ++t)
    {
        auto step = model.forward_one(ids.view(0, t, 1, 1).clone(), t);
        ExpectShape(step, 11, 1);
        for (int i = 0; i < step.row_num(); ++i)
            EXPECT_NEAR(step(i, 0), ref(i, t), 1e-9) << "t=" << t << " row=" << i;
    }
    EXPECT_EQ(model.kv_cache_length(), 5);
}

TEST(Gpt2Structure, PrefillMatchesForward)
{
    auto model = make_small_gpt2();
    mat_t<double> ids(1, 4, {1.0, 2.0, 3.0, 4.0});
    auto ref = model.forward(ids);

    model.reserve_kv_cache(16);
    auto last = model.prefill(ids);
    ExpectShape(last, 11, 1);
    for (int i = 0; i < last.row_num(); ++i)
        EXPECT_NEAR(last(i, 0), ref(i, 3), 1e-9) << "row=" << i;
}

TEST(Gpt2Structure, MultiTurnCacheMatchesFullForward)
{
    // 交互式对话的关键不变量：跨轮复用 KV cache（中途不 clear）得到的 logits，
    // 必须与「把整段拼起来做一次 forward」完全一致。若这里挂了，gpt2_chat 的多轮
    // 输出就是错的，而这种错误只比 logits 很难看出来（数值看着都合理）。
    auto model = make_small_gpt2(2, 2, 8, 32, 11, 16);
    mat_t<double> all(1, 9, {1.0, 4.0, 2.0, 7.0, 3.0, 5.0, 0.0, 6.0, 9.0});
    auto ref = model.forward(all);
    ExpectShape(ref, 11, 9);

    // 分三段喂（模拟三轮对话），段边界不重置 cache
    model.clear_kv_cache();
    const int chunk_sizes[] = {2, 3, 4};
    int pos = 0;
    for (int ci = 0; ci < 3; ++ci)
    {
        for (int k = 0; k < chunk_sizes[ci]; ++k, ++pos)
        {
            auto step = model.forward_one(all.view(0, pos, 1, 1).clone(), pos);
            for (int r = 0; r < step.row_num(); ++r)
                EXPECT_NEAR(step(r, 0), ref(r, pos), 1e-9)
                    << "chunk " << ci << " pos " << pos << " row " << r;
        }
    }
    EXPECT_EQ(model.kv_cache_length(), 9);

    // 重置后重新分块，结果必须一致（reset 不残留状态）
    model.clear_kv_cache();
    EXPECT_EQ(model.kv_cache_length(), 0);
    pos = 0;
    for (int ci = 0; ci < 3; ++ci)
        for (int k = 0; k < chunk_sizes[ci]; ++k, ++pos)
        {
            auto step = model.forward_one(all.view(0, pos, 1, 1).clone(), pos);
            for (int r = 0; r < step.row_num(); ++r)
                EXPECT_NEAR(step(r, 0), ref(r, pos), 1e-9);
        }
}

TEST(Gpt2Structure, ForwardStagesMatchBlockByBlock)
{
    auto model = make_small_gpt2(3, 2, 8, 32, 11, 16);
    mat_t<double> ids(1, 4, {1.0, 2.0, 3.0, 4.0});
    auto stages = model.forward_stages(ids);
    EXPECT_EQ(stages.size(), 4u);   // 嵌入 + 3 层

    // 逐层手工重算，应完全一致
    mat_t<double> h = model.embed(ids);
    EXPECT_LT(MaxAbsDiff(h, stages[0]), 1e-12);
    for (int i = 0; i < model.n_layers(); ++i)
    {
        h = model.block_forward(i, h);
        EXPECT_LT(MaxAbsDiff(h, stages[i + 1]), 1e-12) << "layer " << i;
    }
    // 末端 head 与 forward 一致
    EXPECT_LT(MaxAbsDiff(model.head(stages.back()), model.forward(ids)), 1e-12);
}

TEST(Gpt2Structure, TiedWordEmbeddings)
{
    auto model = make_small_gpt2();
    model.tie_word_embeddings();
    const auto& wte = model.wte().weight();       // [d_model, vocab]
    const auto& head = model.lm_head().weight();  // [vocab, d_model]
    ExpectShape(head, 11, 8);
    for (int v = 0; v < 11; ++v)
        for (int d = 0; d < 8; ++d)
            EXPECT_NEAR(head(v, d), wte(d, v), 1e-12);
    for (int v = 0; v < 11; ++v)
        EXPECT_NEAR(model.lm_head().bias()(v, 0), 0.0, 1e-12);
}

TEST(Gpt2Structure, LayerNormAffineLoadableBeforeForward)
{
    // set_param 必须显式分配 gamma/beta，否则加载器写入的是无效矩阵
    gpt2_type model(1, 2, 8, 32, 11, 16);
    auto& ln = model.ln_1(0);
    ASSERT_TRUE(ln.gama().valid());
    ASSERT_TRUE(ln.beta().valid());
    ExpectShape(ln.gama(), 8, 1);
    ExpectShape(ln.beta(), 8, 1);

    ln.gama() = 2.0;
    ln.beta() = 3.0;
    mat_t<double> in(8, 1, {1, 2, 3, 4, 5, 6, 7, 8});
    ln.forward(in);
    EXPECT_NEAR(ln.gama()(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(ln.beta()(0, 0), 3.0, 1e-12);
}

TEST(Gpt2Structure, RopeCanBeDisabled)
{
    // GPT-2 组装默认关闭 RoPE；开关必须真的改变输出，且能恢复
    gpt2_type model(1, 2, 8, 32, 11, 16);
    model.init_weight<xavier_gaussian_t>();
    mat_t<double> ids(1, 3, {1.0, 2.0, 3.0});

    auto default_out = model.forward(ids);

    for (int i = 0; i < model.n_layers(); ++i)
        model.attn(i).set_use_rope(true);
    auto with_rope = model.forward(ids);
    EXPECT_GT(MaxAbsDiff(default_out, with_rope), 1e-9)
        << "开启 RoPE 后输出应改变（说明默认确实是关闭的）";

    for (int i = 0; i < model.n_layers(); ++i)
        model.attn(i).set_use_rope(false);
    auto back_off = model.forward(ids);
    EXPECT_LT(MaxAbsDiff(default_out, back_off), 1e-12)
        << "关掉 RoPE 后应回到默认结果";
}

// ---------------------------------------------------------------------------
// 与 HuggingFace 黄金值对齐
// ---------------------------------------------------------------------------

TEST_F(Gpt2AlignmentTest, HiddenStatesMatchLayerByLayer)
{
    auto model = make_model();
    load_gpt2(model, m_wf);

    const auto& id_entry = m_wf.entry("golden.input_ids");
    const int T = id_entry.cols;
    mat_t<double> ids(1, T);
    m_wf.read_into("golden.input_ids", ids);

    auto stages = model.forward_stages(ids);
    ASSERT_EQ(static_cast<int>(stages.size()), model.n_layers() + 1);

    const double tol = 1e-3;
    for (int i = 0; i < static_cast<int>(stages.size()); ++i)
    {
        const std::string name = "golden.hidden." + std::to_string(i);
        if (!m_wf.has(name)) continue;
        const auto& e = m_wf.entry(name);
        mat_t<double> ref(e.rows, e.cols);
        m_wf.read_into(name, ref);
        const double diff = MaxAbsDiff(stages[i], ref);
        const std::string label = (i == 0) ? "embeddings" : ("block " + std::to_string(i - 1));
        EXPECT_LT(diff, tol) << "hidden stage " << i << " (" << label
                             << ") mismatch, max_abs_diff=" << diff;
    }
}

TEST_F(Gpt2AlignmentTest, GoldenLogitsMatch)
{
    auto model = make_model();
    load_gpt2(model, m_wf);

    const int T = m_wf.entry("golden.input_ids").cols;
    mat_t<double> ids(1, T);
    m_wf.read_into("golden.input_ids", ids);

    const auto& le = m_wf.entry("golden.logits");
    mat_t<double> ref(le.rows, le.cols);
    m_wf.read_into("golden.logits", ref);

    auto logits = model.forward(ids);
    ExpectShape(logits, ref.row_num(), ref.col_num());

    const double diff = MaxAbsDiff(logits, ref);
    EXPECT_LT(diff, 1e-3) << "logits max_abs_diff=" << diff;

    // argmax 序列完全一致（生成任务真正关心的量）
    for (int t = 0; t < T; ++t)
    {
        int j_ours = 0, j_ref = 0;
        for (int v = 1; v < ref.row_num(); ++v)
        {
            if (logits(v, t) > logits(j_ours, t)) j_ours = v;
            if (ref(v, t) > ref(j_ref, t)) j_ref = v;
        }
        EXPECT_EQ(j_ours, j_ref) << "argmax mismatch at position " << t;
    }
}

TEST_F(Gpt2AlignmentTest, KVCacheDecodeMatchesGolden)
{
    auto model = make_model();
    load_gpt2(model, m_wf);

    const int T = m_wf.entry("golden.input_ids").cols;
    mat_t<double> ids(1, T);
    m_wf.read_into("golden.input_ids", ids);

    const auto& le = m_wf.entry("golden.logits");
    mat_t<double> ref(le.rows, le.cols);
    m_wf.read_into("golden.logits", ref);

    // 用 KV cache 逐步解码，每一步都应与黄金 logits 对齐
    model.clear_kv_cache();
    for (int t = 0; t < T; ++t)
    {
        auto step = model.forward_one(ids.view(0, t, 1, 1).clone(), t);
        ExpectShape(step, ref.row_num(), 1);
        double diff = 0.0;
        for (int v = 0; v < step.row_num(); ++v)
            diff = std::max(diff, std::abs(step(v, 0) - ref(v, t)));
        EXPECT_LT(diff, 1e-3) << "step " << t << " max_abs_diff=" << diff;
    }
}
