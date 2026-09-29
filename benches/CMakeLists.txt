add_executable(bench_jasmine
  bench_matmul.cpp
  bench_mha.cpp
)

target_link_libraries(bench_jasmine
  PRIVATE
    jasmine_headers
    benchmark::benchmark
    benchmark::benchmark_main
)

# Separate from bench_jasmine: needs an external tokenizer directory and has its
# own main so it can print the chunk-length report before timing.
add_executable(bench_bpe
  bench_bpe.cpp
)

target_link_libraries(bench_bpe
  PRIVATE
    jasmine_headers
    benchmark::benchmark
)
