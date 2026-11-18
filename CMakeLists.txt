cmake_minimum_required(VERSION 3.18)

project(gpgpu_vulkan
    VERSION 0.1.0
    DESCRIPTION "High-Performance Developer-Friendly GPGPU Compute Framework on Vulkan"
    LANGUAGES C CXX
)

# ==============================================================================
# Language Standards & Build Options
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GPGPU_BUILD_SHARED     "Build shared library (.dll / .so)" ON)
option(GPGPU_BUILD_TESTS      "Build GoogleTest unit and integration test suite" ON)
option(GPGPU_BUILD_EXAMPLES   "Build example applications" ON)
option(GPGPU_BUILD_BENCHMARKS "Build Google Benchmark performance suite" ON)

# ==============================================================================
# Standalone Local Dependencies (external/)
# ==============================================================================
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake")

if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
endif()

# ==============================================================================
# Core Library Sources
# ==============================================================================
set(GPGPU_CORE_SOURCES
    src/api/data.cpp
    src/api/gpu.cpp
    src/api/c_api.cpp
    src/compiler/kernel_parser.cpp
    src/compiler/glsl_generator.cpp
    src/compiler/spirv_compiler.cpp
    src/compiler/pipeline_cache.cpp
    src/backend/vulkan/vk_device.cpp
    src/backend/vulkan/vk_memory.cpp
    src/backend/vulkan/vk_pipeline.cpp
    src/backend/vulkan/vk_execution.cpp
    src/backend/vulkan/vk_backend.cpp
)

# 1. Static Library Target (gpgpu_vulkan_static)
add_library(gpgpu_vulkan_static STATIC ${GPGPU_CORE_SOURCES})
add_library(gpgpu::static ALIAS gpgpu_vulkan_static)

target_include_directories(gpgpu_vulkan_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gpgpu_vulkan_static
    PUBLIC
        gpgpu_dep_vulkan_headers
    PRIVATE
        gpgpu_dep_volk
        gpgpu_dep_vma
        gpgpu_dep_xxhash
        gpgpu_dep_spdlog
        gpgpu_dep_compiler
)

if(UNIX AND NOT APPLE)
    target_link_libraries(gpgpu_vulkan_static PUBLIC dl pthread)
endif()

# 2. Shared Library Target (gpgpu_vulkan_shared -> gpgpu_vulkan.dll / libgpgpu_vulkan.so)
if(GPGPU_BUILD_SHARED)
    add_library(gpgpu_vulkan_shared SHARED ${GPGPU_CORE_SOURCES})
    add_library(gpgpu::shared ALIAS gpgpu_vulkan_shared)
    set_target_properties(gpgpu_vulkan_shared PROPERTIES OUTPUT_NAME "gpgpu_vulkan")
    if(WIN32)
        set_target_properties(gpgpu_vulkan_shared PROPERTIES PREFIX "")
    endif()

    target_compile_definitions(gpgpu_vulkan_shared
        PRIVATE GPGPU_VULKAN_BUILD_SHARED
        INTERFACE GPGPU_VULKAN_USE_SHARED
    )

    target_include_directories(gpgpu_vulkan_shared
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(gpgpu_vulkan_shared
        PUBLIC
            gpgpu_dep_vulkan_headers
        PRIVATE
            gpgpu_dep_volk
            gpgpu_dep_vma
            gpgpu_dep_xxhash
            gpgpu_dep_spdlog
            gpgpu_dep_compiler
    )

    if(UNIX AND NOT APPLE)
        target_link_libraries(gpgpu_vulkan_shared PUBLIC dl pthread)
    endif()
endif()

# ==============================================================================
# Unit & Integration Tests (GoogleTest)
# ==============================================================================
if(GPGPU_BUILD_TESTS)
    enable_testing()

    add_executable(gpgpu_tests
        tests/test_sanity.cpp
        tests/test_device_selection.cpp
        tests/test_memory_transfer.cpp
        tests/test_compute_raw.cpp
        tests/test_kernel_dsl.cpp
        tests/test_interpolation.cpp
        tests/test_c_api.cpp
        tests/test_jumping_glsl.cpp
        tests/test_multipass_channels.cpp
    )

    target_link_libraries(gpgpu_tests
        PRIVATE
            gpgpu_vulkan_static
            GTest::gtest_main
    )

    add_test(NAME gpgpu_sanity_tests COMMAND gpgpu_tests)
endif()

# ==============================================================================
# Example Applications (examples/)
# ==============================================================================
if(GPGPU_BUILD_EXAMPLES)
    add_executable(example_01_vector_cos examples/01_vector_cos.cpp)
    target_link_libraries(example_01_vector_cos PRIVATE gpgpu_vulkan_static)

    add_executable(example_02_matrix_filter_2d examples/02_matrix_filter_2d.cpp)
    target_link_libraries(example_02_matrix_filter_2d PRIVATE gpgpu_vulkan_static)

    add_executable(example_03_shadertoy_runner examples/03_shadertoy_runner.cpp)
    target_link_libraries(example_03_shadertoy_runner PRIVATE gpgpu_vulkan_static)

    add_executable(example_04_async_chaining examples/04_async_chaining.cpp)
    target_link_libraries(example_04_async_chaining PRIVATE gpgpu_vulkan_static)
endif()

# ==============================================================================
# Performance Benchmarks (Google Benchmark)
# ==============================================================================
if(GPGPU_BUILD_BENCHMARKS)
    add_executable(gpgpu_benchmarks benchmarks/bench_compute_throughput.cpp)
    target_link_libraries(gpgpu_benchmarks
        PRIVATE
            gpgpu_vulkan_static
            benchmark::benchmark
    )
endif()
