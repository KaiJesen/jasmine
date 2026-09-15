add_executable(unit_tests
  test_mha_tools.cpp
  test_causal_attention.cpp
  test_mat_basic.cpp
  test_mat_init.cpp
  test_rope.cpp
  test_mha.cpp
  test_kv_cache.cpp
  test_net.cpp
  test_transformer_kernel.cpp
  test_layer_norm.cpp
)

target_link_libraries(unit_tests
  PRIVATE
    jasmine_headers
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
