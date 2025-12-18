cmake_minimum_required(VERSION 3.14)
project(arb_harness_modular CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.83 REQUIRED COMPONENTS json)
find_package(Threads REQUIRED)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)
# Also include the original cpp/include for stableswap_math.hpp and twocrypto.hpp
include_directories(${CMAKE_SOURCE_DIR}/../cpp/include)

# Common sources (will grow as we add modules)
set(COMMON_SOURCES
    src/main.cpp
    src/events.cpp
    src/cli.cpp
)

# Compiler flags
set(COMMON_FLAGS -Wall -Wextra)
set(RELEASE_FLAGS -O3 -march=native)

# --- arb_harness (default: double) ---
add_executable(arb_harness ${COMMON_SOURCES})
target_link_libraries(arb_harness Boost::json Threads::Threads)
target_compile_options(arb_harness PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness PRIVATE ${RELEASE_FLAGS})
endif()

# --- arb_harness_f (float) ---
add_executable(arb_harness_f ${COMMON_SOURCES})
target_compile_definitions(arb_harness_f PRIVATE ARB_MODE_F)
target_link_libraries(arb_harness_f Boost::json Threads::Threads)
target_compile_options(arb_harness_f PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness_f PRIVATE ${RELEASE_FLAGS})
endif()

# --- arb_harness_ld (long double) ---
add_executable(arb_harness_ld ${COMMON_SOURCES})
target_compile_definitions(arb_harness_ld PRIVATE ARB_MODE_LD)
target_link_libraries(arb_harness_ld Boost::json Threads::Threads)
target_compile_options(arb_harness_ld PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arb_harness_ld PRIVATE ${RELEASE_FLAGS})
endif()

# --- stableswap_math_i (C ABI for Python benchmarks) ---
add_library(stableswap_math_i SHARED src/stableswap_math_i.cpp)
target_compile_options(stableswap_math_i PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(stableswap_math_i PRIVATE ${RELEASE_FLAGS})
endif()
set_target_properties(stableswap_math_i PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# --- benchmark_harness (pool action sequences) ---
add_executable(benchmark_harness src/benchmark_harness.cpp)
target_link_libraries(benchmark_harness Boost::json Threads::Threads)
target_compile_options(benchmark_harness PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness PRIVATE ${RELEASE_FLAGS})
endif()

add_executable(benchmark_harness_i src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_i PRIVATE HARNESS_MODE_I)
target_link_libraries(benchmark_harness_i Boost::json Threads::Threads)
target_compile_options(benchmark_harness_i PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_i PRIVATE ${RELEASE_FLAGS})
endif()

add_executable(benchmark_harness_d src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_d PRIVATE HARNESS_MODE_D)
target_link_libraries(benchmark_harness_d Boost::json Threads::Threads)
target_compile_options(benchmark_harness_d PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_d PRIVATE ${RELEASE_FLAGS})
endif()

add_executable(benchmark_harness_f src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_f PRIVATE HARNESS_MODE_F)
target_link_libraries(benchmark_harness_f Boost::json Threads::Threads)
target_compile_options(benchmark_harness_f PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_f PRIVATE ${RELEASE_FLAGS})
endif()

add_executable(benchmark_harness_ld src/benchmark_harness.cpp)
target_compile_definitions(benchmark_harness_ld PRIVATE HARNESS_MODE_LD)
target_link_libraries(benchmark_harness_ld Boost::json Threads::Threads)
target_compile_options(benchmark_harness_ld PRIVATE ${COMMON_FLAGS})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_harness_ld PRIVATE ${RELEASE_FLAGS})
endif()

