cmake_minimum_required(VERSION 3.16)
project(jasmine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(JASMINE_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(JASMINE_BUILD_BENCH "Build Google Benchmark targets" ON)
option(JASMINE_BUILD_EXAMPLES "Build example programs" ON)
option(JASMINE_USE_OPENMP "Enable OpenMP" ON)
option(JASMINE_USE_BLAS "Use BLAS (cblas_*gemm) for large matmul when available" ON)
option(JASMINE_USE_CUDA "Build the CUDA backend and its tests" OFF)
set(JASMINE_CUDA_ARCHITECTURES "61" CACHE STRING
    "CUDA 目标架构。Tesla P4 是 Pascal，即 61。注意 CUDA 13 已移除 Pascal 支持。")

add_library(jasmine_headers INTERFACE)
target_include_directories(jasmine_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# -pedantic 不能给 CUDA 编译单元：nvcc 生成的宿主代码里全是 `# line` 行指令，
# gcc 会把每一条都当成扩展用法报一遍，把真正的问题淹掉
target_compile_options(jasmine_headers INTERFACE
  $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall>
  $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wextra>
  $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pedantic>
)

if(JASMINE_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(jasmine_headers INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(jasmine_headers INTERFACE JASMINE_USE_OPENMP)
  endif()
endif()

if(JASMINE_USE_BLAS)
  # Distros often ship only the SONAME (libblas.so.3) without libblas-dev.
  find_library(JASMINE_BLAS_LIBRARY
    NAMES openblas blas
    PATHS
      /usr/lib/x86_64-linux-gnu
      /lib/x86_64-linux-gnu
      /usr/lib/x86_64-linux-gnu/blas
      /usr/lib
      /lib
  )
  if(NOT JASMINE_BLAS_LIBRARY)
    foreach(_cand
        /usr/lib/x86_64-linux-gnu/libblas.so.3
        /lib/x86_64-linux-gnu/libblas.so.3
        /usr/lib/x86_64-linux-gnu/blas/libblas.so.3
        /usr/lib/x86_64-linux-gnu/libopenblas.so.0)
      if(EXISTS "${_cand}")
        set(JASMINE_BLAS_LIBRARY "${_cand}")
        break()
      endif()
    endforeach()
  endif()
  if(JASMINE_BLAS_LIBRARY)
    message(STATUS "Jasmine BLAS: ${JASMINE_BLAS_LIBRARY}")
    target_link_libraries(jasmine_headers INTERFACE ${JASMINE_BLAS_LIBRARY})
    target_compile_definitions(jasmine_headers INTERFACE JASMINE_USE_BLAS)
  else()
    message(STATUS "Jasmine BLAS: not found; using blocked GEMM fallback")
  endif()
endif()

if(JASMINE_USE_CUDA)
  # CUDA 13 起不再支持 Pascal(sm_61)，而这台机器上的 /usr/local/cuda 恰好指向 13.x。
  # 所以这里主动挑一个还支持目标架构的 toolkit，否则 nvcc 会直接拒绝出码。
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED ENV{CUDACXX})
    foreach(_jas_cuda_hint
        /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda-12.4 /usr/local/cuda-12
        /usr/local/cuda)
      if(EXISTS "${_jas_cuda_hint}/bin/nvcc")
        # enable_language(CUDA) 认的是 CMAKE_CUDA_COMPILER，光设 CUDAToolkit_ROOT 不够
        set(CMAKE_CUDA_COMPILER "${_jas_cuda_hint}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
        set(CUDAToolkit_ROOT "${_jas_cuda_hint}" CACHE PATH "CUDA toolkit root")
        break()
      endif()
    endforeach()
  endif()

  # 必须在 enable_language(CUDA) 之前定好，CMake 会据此生成 -arch 参数
  set(CMAKE_CUDA_ARCHITECTURES "${JASMINE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
  # 头文件大量使用 C++20 概念（requires 子句），设备编译单元必须同标准
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  message(STATUS "Jasmine CUDA: ${CMAKE_CUDA_COMPILER}")
  message(STATUS "Jasmine CUDA arch: sm_${JASMINE_CUDA_ARCHITECTURES}")

  # 设备端模板（表达式树、设备叶子）都是头文件，随 CUDA 编译单元一起编。
  add_library(jasmine_cuda_headers INTERFACE)
  target_include_directories(jasmine_cuda_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(jasmine_cuda_headers INTERFACE JASMINE_USE_CUDA)
  target_link_libraries(jasmine_cuda_headers INTERFACE CUDA::cudart)
  # cuBLAS 只在用到 GEMM 的单元里需要，但量很小，统一挂上省得每个目标各配一遍
  target_link_libraries(jasmine_cuda_headers INTERFACE CUDA::cublas)

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # 主机侧模板大量依赖 C++20 概念/requires，设备侧编译单元要沿用同样的标准
    target_compile_options(jasmine_cuda_headers INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
    # jas_mat_gemm.hpp 里的主机侧 GEMM 回退带 OpenMP 指令，CUDA 编译单元也要能编过。
    # 编译期用 -Xcompiler 把 -fopenmp 交给宿主编译器；链接期不能再用 -Xcompiler=
    # （nvcc 会把整串原样透传给 g++，g++ 不认），直接挂 OpenMP 的库更稳。
    if(JASMINE_USE_OPENMP AND OpenMP_CXX_FOUND)
      target_compile_options(jasmine_cuda_headers INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
      target_link_libraries(jasmine_cuda_headers INTERFACE ${OpenMP_CXX_LIBRARIES})
    endif()
  endif()
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
if(JASMINE_BUILD_TESTS)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_subdirectory(tests)
endif()

if(JASMINE_BUILD_BENCH)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )
  FetchContent_MakeAvailable(googlebenchmark)
  add_subdirectory(benches)
endif()

if(JASMINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
