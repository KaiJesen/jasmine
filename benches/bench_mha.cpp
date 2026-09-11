#include <benchmark/benchmark.h>
#include "mat_mha_t.hpp"
#include "mat_init_t.hpp"

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
    ->Args({4, 64, 16});
