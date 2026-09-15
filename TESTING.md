# Testing & Benchmarking

Library headers contain **only** library code. Checks and demos live under:

| Dir | Tool | Purpose |
|-----|------|---------|
| `tests/` | GoogleTest | Correctness / regression |
| `benches/` | Google Benchmark | Performance |
| `examples/` | plain binary | Interactive demos (`cin`) |

## Build

```bash
cmake -S . -B build -DJASMINE_USE_OPENMP=ON
cmake --build build -j
```

## Run

```bash
ctest --test-dir build --output-on-failure
./build/benches/bench_jasmine --benchmark_filter=BM_MhaForward
./build/examples/train_transformer      # MSE：连续向量 + 特征维 SOS/EOS
./build/examples/train_transformer_ce   # CE：离散 token + embedding + CE
./build/examples/train_tf_base
./build/examples/train_tf_kernel
```

Demo harnesses（非 GoogleTest）：

- `examples/transformer_mse_demo.hpp` — `mse_transformer_demo_t`
- `examples/transformer_ce_demo.hpp` — `ce_transformer_demo_t`
