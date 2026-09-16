add_executable(unit_tests
  test_mha_tools.cpp
  test_causal_attention.cpp
  test_mat_basic.cpp
  test_expression_lifetime.cpp
  test_mat_init.cpp
  test_gemm.cpp
  test_rope.cpp
  test_mha.cpp
  test_kv_cache.cpp
  test_net.cpp
  test_embedding_ce.cpp
  test_transformer_kernel.cpp
  test_layer_norm.cpp
  test_gpt2_weights.cpp
  test_silu.cpp
  test_gated.cpp
  test_gqa.cpp
  test_rms_norm.cpp
  test_llama_weights.cpp
)

target_link_libraries(unit_tests
  PRIVATE
    jasmine_headers
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)

# CUDA 后端测试单独一个可执行文件：设备代码必须由 nvcc 编译，
# 和纯 CPU 的 unit_tests 混在一个目标里会把整个目标拖成 CUDA 目标。
if(JASMINE_USE_CUDA)
  add_executable(cuda_tests
    test_cuda_fused.cu
  )

  target_link_libraries(cuda_tests
    PRIVATE
      jasmine_headers
      jasmine_cuda_headers
      GTest::gtest_main
  )

  # 让 gtest 的测试发现也走 nvcc 目标所需的设置
  set_target_properties(cuda_tests PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )

  gtest_discover_tests(cuda_tests)
endif()
