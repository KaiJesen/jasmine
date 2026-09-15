#include <gtest/gtest.h>

#include "mat_kv_cache_t.hpp"
#include "mat_mha_t.hpp"
#include "mat_init_t.hpp"
#include "mat_transformer_kernel_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

TEST(KvCache, AppendAndViews)
{
    kv_cache_t<double> cache;
    cache.reserve(2, 4);
    EXPECT_EQ(cache.length(), 0);

    mat_t<double> k1(2, 1, {1.0, 2.0});
    mat_t<double> v1(2, 1, {3.0, 4.0});
    cache.append(k1, v1);
    EXPECT_EQ(cache.length(), 1);
    EXPECT_NEAR(cache.keys()(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(cache.values()(1, 0), 4.0, 1e-12);

    mat_t<double> k2(2, 1, {5.0, 6.0});
    mat_t<double> v2(2, 1, {7.0, 8.0});
    cache.append(k2, v2);
    EXPECT_EQ(cache.length(), 2);
    EXPECT_NEAR(cache.keys()(0, 1), 5.0, 1e-12);

    cache.clear();
    EXPECT_EQ(cache.length(), 0);
}

TEST(KvCache, StaticOverflowThrows)
{
    kv_cache_t<float> cache;
    cache.set_mode(kv_cache_mode::static_fixed);
    cache.reserve(2, 1);
    mat_t<float> k(2, 1, {1.f, 2.f});
    mat_t<float> v(2, 1, {3.f, 4.f});
    cache.append(k, v);
    EXPECT_THROW(cache.append(k, v), std::runtime_error);
}

TEST(MhaKvCache, DecodeStepMatchesFullForwardLastCol)
{
    // causal MHA：逐步 forward_one 每步输出 == 整段 forward 对应列
    constexpr int heads = 2;
    constexpr int d_model = 4; // d_head=2，RoPE 开启
    constexpr int seq = 4;

    mat_mha_t<mat_t<double>, nadam_t> mha(heads, d_model, true, seq);
    mha.init_weight<xavier_gaussian_t>();

    mat_t<double> input(d_model, seq, {
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6
    });

    auto ref = mha.forward(input);

    mha.clear_kv_cache();
    mat_t<double> step_out(d_model, seq);
    for (int t = 0; t < seq; ++t)
    {
        auto y = mha.forward_one(input.view(0, t, d_model, 1).clone());
        ExpectShape(y, d_model, 1);
        for (int i = 0; i < d_model; ++i)
            step_out(i, t) = y(i, 0);
    }
    EXPECT_EQ(mha.kv_cache_length(), seq);
    ExpectNearMat(ref, step_out, 1e-9);
}

TEST(MhaKvCache, PrefillThenDecodeMatchesFull)
{
    constexpr int heads = 2;
    constexpr int d_model = 4;
    mat_mha_t<mat_t<double>, adam_t> mha(heads, d_model, true, 8);
    mha.init_weight<xavier_gaussian_t>();

    mat_t<double> input(d_model, 3, {
        0.2, 0.4, 0.6,
        0.1, 0.3, 0.5,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    });
    auto ref = mha.forward(input);

    mha.clear_kv_cache();
    // prefill 前两列
    auto y01 = mha.forward_one(input.view(0, 0, d_model, 2).clone());
    ExpectShape(y01, d_model, 2);
    ExpectNearMat(ref.view(0, 0, d_model, 2), y01, 1e-9);

    auto y2 = mha.forward_one(input.view(0, 2, d_model, 1).clone());
    ExpectNearMat(ref.view(0, 2, d_model, 1), y2, 1e-9);
}

TEST(DecoderKvCache, DecodeStepMatchesFullForward)
{
    // 整 decoder（含 cross-attn）：encode 一次后，逐步 decode == 整段 forward
    transformer_kernel_t<double, nadam_t> tf(1, 1, 2, 4, 16, 8);
    tf.init_weight<xavier_gaussian_t>();

    mat_t<double> enc_in(4, 2, {
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
        0.7, 0.8
    });
    mat_t<double> dec_in(4, 3, {
        0.2, 0.3, 0.4,
        0.5, 0.6, 0.7,
        0.8, 0.9, 1.0,
        1.1, 1.2, 1.3
    });

    tf.encoder_forward(enc_in);
    auto ref = tf.forward(dec_in);

    tf.clear_kv_cache();  // 只清 self-attn；cross K/V 在 encoder_forward 时已准备
    mat_t<double> step_out(4, 3);
    for (int t = 0; t < 3; ++t)
    {
        auto y = tf.forward_one(dec_in.view(0, t, 4, 1).clone());
        for (int i = 0; i < 4; ++i)
            step_out(i, t) = y(i, 0);
    }
    ExpectNearMat(ref, step_out, 1e-8);
}

TEST(MhcaKvCache, PrepareThenDecodeMatchesFull)
{
    // 单独测 cross-attn：prepare_cross_kv 后逐步 forward_one == 整段 forward
    mat_t<double> enc_mem(4, 3, {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    });
    mat_t<double> enc_delta(4, 3);
    enc_delta = 0.0;

    mat_mhca_t<mat_t<double>, nadam_t> mhca(2, 4, false, 8);
    mhca.init_weight<xavier_gaussian_t>();
    mhca.set_encoder_param(enc_mem, enc_delta);

    mat_t<double> dec(4, 2, {
        0.2, 0.4,
        0.3, 0.5,
        0.6, 0.8,
        0.7, 0.9
    });

    auto ref = mhca.forward(dec);
    mhca.prepare_cross_kv();
    EXPECT_EQ(mhca.kv_cache_length(), 3);

    mat_t<double> step_out(4, 2);
    for (int t = 0; t < 2; ++t)
    {
        auto y = mhca.forward_one(dec.view(0, t, 4, 1).clone());
        for (int i = 0; i < 4; ++i)
            step_out(i, t) = y(i, 0);
    }
    // prepare 后再 decode，cache 长度仍为 encoder seq（不 append）
    EXPECT_EQ(mhca.kv_cache_length(), 3);
    ExpectNearMat(ref, step_out, 1e-9);
}
