cmake_minimum_required(VERSION 3.16)
project(jasmine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(JASMINE_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(JASMINE_BUILD_BENCH "Build Google Benchmark targets" ON)
option(JASMINE_BUILD_EXAMPLES "Build example programs" ON)
option(JASMINE_USE_OPENMP "Enable OpenMP" ON)

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
