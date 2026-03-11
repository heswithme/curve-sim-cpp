cmake_minimum_required(VERSION 3.14)
project(arb_harness_modular CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.83 REQUIRED COMPONENTS json)
find_package(Threads REQUIRED)

# Common sources (will grow as we add modules)
set(COMMON_SOURCES
    src/main.cpp
    src/events.cpp
    src/cli.cpp
)

# Compiler flags
set(COMMON_FLAGS -Wall -Wextra)
set(RELEASE_FLAGS -O3 -march=native -flto -funroll-loops -DNDEBUG)

function(arb_target_defaults TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Boost_INCLUDE_DIRS}
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        ${COMMON_FLAGS}
        $<$<CONFIG:Release>:${RELEASE_FLAGS}>
    )
endfunction()

# --- arb_harness (runtime-selectable pool backend) ---
add_executable(arb_harness ${COMMON_SOURCES})
target_link_libraries(arb_harness Boost::json Threads::Threads)
arb_target_defaults(arb_harness)

# --- arb_eval_server (persistent single-template evaluator) ---
add_executable(arb_eval_server
    src/eval_server.cpp
    src/events.cpp
)
target_link_libraries(arb_eval_server Boost::json Threads::Threads)
arb_target_defaults(arb_eval_server)

# --- stableswap_math_i (C ABI for Python benchmarks) ---
add_library(stableswap_math_i SHARED src/stableswap_math_i.cpp)
arb_target_defaults(stableswap_math_i)
set_target_properties(stableswap_math_i PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# --- benchmark_harness (pool action sequences) ---
add_executable(benchmark_harness src/benchmark_harness.cpp)
target_link_libraries(benchmark_harness Boost::json Threads::Threads)
arb_target_defaults(benchmark_harness)

add_executable(benchmark_harness_i src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_i PRIVATE HARNESS_MODE_I)
target_link_libraries(benchmark_harness_i Boost::json Threads::Threads)
arb_target_defaults(benchmark_harness_i)

add_executable(benchmark_harness_d src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_d PRIVATE HARNESS_MODE_D)
target_link_libraries(benchmark_harness_d Boost::json Threads::Threads)
arb_target_defaults(benchmark_harness_d)

add_executable(benchmark_harness_f src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_f PRIVATE HARNESS_MODE_F)
target_link_libraries(benchmark_harness_f Boost::json Threads::Threads)
arb_target_defaults(benchmark_harness_f)

add_executable(benchmark_harness_ld src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_ld PRIVATE HARNESS_MODE_LD)
target_link_libraries(benchmark_harness_ld Boost::json Threads::Threads)
arb_target_defaults(benchmark_harness_ld)
