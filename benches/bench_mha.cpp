#include <benchmark/benchmark.h>
#include "mat_mha_t.hpp"
#include "mat_init_t.hpp"


using namespace jasmine;
static void BM_MhaForward(benchmark::State& state)
{
    const int num_heads = static_cast<int>(state.range(0));
    const int d_model = static_cast<int>(state.range(1));
    const int seq_len = static_cast<int>(state.range(2));

    mat_mha_t<mat_t<float>, nadam_t> mha(num_heads, d_model, false, seq_len);
    mha.init_weight<xavier_gaussian_t>();

    mat_t<float> input(d_model, seq_len);
    input = 0.1f;

    for (auto _ : state)
    {
        auto out = mha.forward(input);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_MhaForward)
    ->Args({2, 32, 8})
    ->Args({4, 64, 16})
    ->Args({4, 64, 32})
    ->Args({8, 128, 64})
    ->Args({8, 256, 64});

static void BM_MhaDecodeStep(benchmark::State& state)
{
    const int num_heads = static_cast<int>(state.range(0));
    const int d_model = static_cast<int>(state.range(1));
    const int prefill_len = static_cast<int>(state.range(2));

    mat_mha_t<mat_t<float>, nadam_t> mha(num_heads, d_model, true, prefill_len + 64);
    mha.init_weight<xavier_gaussian_t>();

    mat_t<float> prompt(d_model, prefill_len);
    prompt = 0.1f;
    (void)mha.forward_one(prompt);

    mat_t<float> token(d_model, 1);
    token = 0.2f;

    for (auto _ : state)
    {
        auto out = mha.forward_one(token);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_MhaDecodeStep)
    ->Args({4, 64, 32})
    ->Args({4, 64, 128})
    ->Args({8, 256, 128});
