add_executable(train_transformer train_transformer.cpp)
target_link_libraries(train_transformer PRIVATE jasmine_headers)

add_executable(train_transformer_ce train_transformer_ce.cpp)
target_link_libraries(train_transformer_ce PRIVATE jasmine_headers)

add_executable(train_tf_base train_tf_base.cpp)
target_link_libraries(train_tf_base PRIVATE jasmine_headers)

add_executable(train_tf_kernel train_tf_kernel.cpp)
target_link_libraries(train_tf_kernel PRIVATE jasmine_headers)

add_executable(gpt2_generate gpt2_generate.cpp)
target_link_libraries(gpt2_generate PRIVATE jasmine_headers)

add_executable(gpt2_chat gpt2_chat.cpp)
target_link_libraries(gpt2_chat PRIVATE jasmine_headers)

add_executable(llama_chat llama_chat.cpp)
target_link_libraries(llama_chat PRIVATE jasmine_headers)

# MNIST 小玩具：卷积 + 编码器的训练 / 评估 / 序列化
add_executable(mnist_conv mnist_conv.cpp)
target_link_libraries(mnist_conv PRIVATE jasmine_headers)

# MNIST + DBN（RBM 静态堆叠）：逐层 CD 预训练 + 监督微调
add_executable(mnist_dbn mnist_dbn.cpp)
target_link_libraries(mnist_dbn PRIVATE jasmine_headers)

if(JASMINE_USE_CUDA)
  add_executable(cuda_fused_demo cuda_fused_demo.cu)
  target_link_libraries(cuda_fused_demo PRIVATE jasmine_headers jasmine_cuda_headers)
  set_target_properties(cuda_fused_demo PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
endif()
