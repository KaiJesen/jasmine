# Testing & Benchmarking

Library headers contain **only** library code. Checks live under:

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
./build/examples/train_transformer
./build/examples/train_tf_base
./build/examples/train_tf_kernel
```

`transformer_test.hpp` keeps the training harness helpers (`test_transformer_t`, SOS/EOS utilities), not executable `test_*` entrypoints.
