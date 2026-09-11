#include <benchmark/benchmark.h>
#include "mat_t.hpp"
#include "mat_express_t.hpp"

static void BM_MatDot(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    mat_t<float> a(n, n);
    mat_t<float> b(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            a(i, j) = static_cast<float>(i + j);
            b(i, j) = static_cast<float>(i * 0.1 + j);
        }
    }

    for (auto _ : state)
    {
        mat_t<float> c = a.dot(b).clone();
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * n * n);
}
BENCHMARK(BM_MatDot)->Arg(32)->Arg(64);

static void BM_MatElementwise(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    mat_t<float> a(n, n);
    mat_t<float> b(n, n);
    a = 1.0f;
    b = 2.0f;

    for (auto _ : state)
    {
        mat_t<float> c = (a + b * a).clone();
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_MatElementwise)->Arg(128)->Arg(256);
