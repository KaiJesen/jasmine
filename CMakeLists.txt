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

add_library(jasmine_headers INTERFACE)
target_include_directories(jasmine_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(jasmine_headers INTERFACE -Wall -Wextra -pedantic)

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
