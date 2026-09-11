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
