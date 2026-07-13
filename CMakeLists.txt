cmake_minimum_required(VERSION 3.20)
project(anchor_sampling LANGUAGES CXX)

option(ANCHOR_ENABLE_SANITIZERS "Enable ASan and UBSan in tests" OFF)

add_library(anchor_sampling STATIC cpp/src/anchor.cpp)
target_include_directories(anchor_sampling PUBLIC cpp/include)
target_compile_features(anchor_sampling PUBLIC cxx_std_20)
target_compile_options(anchor_sampling PRIVATE -Wall -Wextra -Wconversion -Wshadow)

if(ANCHOR_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(anchor_sampling PUBLIC -fsanitize=address,undefined
                                                -fno-omit-frame-pointer)
  target_link_options(anchor_sampling PUBLIC -fsanitize=address,undefined)
endif()

enable_testing()
add_executable(anchor_core_tests cpp/tests/test_core.cpp)
target_link_libraries(anchor_core_tests PRIVATE anchor_sampling)
target_compile_features(anchor_core_tests PRIVATE cxx_std_20)
target_compile_options(anchor_core_tests PRIVATE -Wall -Wextra -Wconversion -Wshadow)
add_test(NAME anchor_core_tests COMMAND anchor_core_tests)

set(ANCHOR_BENCH_SOURCES
    cpp/app/benchmark_main.cpp
    cpp/app/workload.cpp
    cpp/app/sha256.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/app/benchmark_main.cpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/app/workload.cpp")
  add_executable(anchor_bench ${ANCHOR_BENCH_SOURCES})
  target_link_libraries(anchor_bench PRIVATE anchor_sampling)
  target_include_directories(anchor_bench PRIVATE cpp/app)
  target_compile_features(anchor_bench PRIVATE cxx_std_20)
  target_compile_options(anchor_bench PRIVATE -Wall -Wextra -Wconversion -Wshadow)
endif()
