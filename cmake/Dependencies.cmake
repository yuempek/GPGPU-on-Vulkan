# ==============================================================================
# cmake/Dependencies.cmake
# Standalone / Local Dependency Configuration (100% Offline Portable)
# All third-party dependencies reside under ${PROJECT_SOURCE_DIR}/external/
# ==============================================================================

set(GPGPU_EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external")

include("${PROJECT_SOURCE_DIR}/cmake/toolchains/GitDownloader.cmake")

# --- Bağımlılık listesi ----------------------------------------------------
# Bicim: <klasor adi>|<git url>|<tag / branch / commit>
set(GPGPU_DEPENDENCIES
    "benchmark|https://github.com/google/benchmark.git|v1.8.4"
    "glslang|https://github.com/KhronosGroup/glslang.git|14.2.0"
    "googletest|https://github.com/google/googletest.git|v1.13.0"
    "spdlog|https://github.com/gabime/spdlog.git|v1.17.0"
    "SPIRV-Headers|https://github.com/arm/SPIRV-Headers.git|vulkan-sdk-1.3.283.0"
    "SPIRV-Tools|https://github.com/KhronosGroup/SPIRV-Tools.git|vulkan-sdk-1.3.283.0"
    "volk|https://github.com/zeux/volk.git|vulkan-sdk-1.3.283.0"
    "Vulkan-Headers|https://github.com/KhronosGroup/Vulkan-Headers.git|vulkan-sdk-1.3.283.0"
    "VulkanMemoryAllocator|https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git|v3.1.0"
    "xxHash|https://github.com/pombredanne/xxHash-3.git|7bff939"
)

# GitDownloader(<RootPath> <bagimlilik listesi>)
GitDownloader("${GPGPU_EXTERNAL_DIR}" "${GPGPU_DEPENDENCIES}")





# 1. Vulkan-Headers (Header-only, VK_NO_PROTOTYPES for dynamic volk loading)
if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/Vulkan-Headers/include/vulkan/vulkan.h")
    message(FATAL_ERROR "Missing local dependency: external/Vulkan-Headers")
endif()
add_library(gpgpu_dep_vulkan_headers INTERFACE)
target_include_directories(gpgpu_dep_vulkan_headers SYSTEM INTERFACE
    "${GPGPU_EXTERNAL_DIR}/Vulkan-Headers/include"
)
target_compile_definitions(gpgpu_dep_vulkan_headers INTERFACE VK_NO_PROTOTYPES)

# 2. volk (Meta-loader for Vulkan; implementation compiled in vk_device.cpp)
if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/volk/volk.h")
    message(FATAL_ERROR "Missing local dependency: external/volk")
endif()
add_library(gpgpu_dep_volk INTERFACE)
target_include_directories(gpgpu_dep_volk SYSTEM INTERFACE
    "${GPGPU_EXTERNAL_DIR}/volk"
)
target_link_libraries(gpgpu_dep_volk INTERFACE gpgpu_dep_vulkan_headers)

# 3. VulkanMemoryAllocator (VMA; implementation compiled in vk_memory.cpp)
if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/VulkanMemoryAllocator/include/vk_mem_alloc.h")
    message(FATAL_ERROR "Missing local dependency: external/VulkanMemoryAllocator")
endif()
add_library(gpgpu_dep_vma INTERFACE)
target_include_directories(gpgpu_dep_vma SYSTEM INTERFACE
    "${GPGPU_EXTERNAL_DIR}/VulkanMemoryAllocator/include"
)
target_link_libraries(gpgpu_dep_vma INTERFACE gpgpu_dep_vulkan_headers)

# 4. xxHash (XXH3_64bits inline header-only mode)
if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/xxHash/xxhash.h")
    message(FATAL_ERROR "Missing local dependency: external/xxHash")
endif()
add_library(gpgpu_dep_xxhash INTERFACE)
target_include_directories(gpgpu_dep_xxhash SYSTEM INTERFACE
    "${GPGPU_EXTERNAL_DIR}/xxHash"
)
target_compile_definitions(gpgpu_dep_xxhash INTERFACE XXH_INLINE_ALL)

# 5. spdlog (Header-only mode for zero link-time friction)
if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/spdlog/include/spdlog/spdlog.h")
    message(FATAL_ERROR "Missing local dependency: external/spdlog")
endif()
add_library(gpgpu_dep_spdlog INTERFACE)
target_include_directories(gpgpu_dep_spdlog SYSTEM INTERFACE
    "${GPGPU_EXTERNAL_DIR}/spdlog/include"
)
target_compile_definitions(gpgpu_dep_spdlog INTERFACE SPDLOG_HEADER_ONLY)

# 6. SPIRV-Headers & SPIRV-Tools
if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/SPIRV-Tools/CMakeLists.txt")
    message(FATAL_ERROR "Missing local dependency: external/SPIRV-Tools")
endif()
set(SPIRV-Headers_SOURCE_DIR "${GPGPU_EXTERNAL_DIR}/SPIRV-Headers" CACHE PATH "" FORCE)
set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ENABLE_SPIRV_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${GPGPU_EXTERNAL_DIR}/SPIRV-Tools" "${CMAKE_BINARY_DIR}/external/SPIRV-Tools" EXCLUDE_FROM_ALL)

# 7. glslang (In-memory runtime shader compiler)
if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/glslang/CMakeLists.txt")
    message(FATAL_ERROR "Missing local dependency: external/glslang")
endif()
set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT ON CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${GPGPU_EXTERNAL_DIR}/glslang" "${CMAKE_BINARY_DIR}/external/glslang" EXCLUDE_FROM_ALL)

# Aggregated compiler interface library
add_library(gpgpu_dep_compiler INTERFACE)
target_include_directories(gpgpu_dep_compiler SYSTEM INTERFACE
    "${GPGPU_EXTERNAL_DIR}/glslang"
    "${GPGPU_EXTERNAL_DIR}/SPIRV-Tools/include"
    "${GPGPU_EXTERNAL_DIR}/SPIRV-Headers/include"
)
target_link_libraries(gpgpu_dep_compiler INTERFACE
    glslang
    glslang-default-resource-limits
    SPIRV
    SPIRV-Tools-opt
    SPIRV-Tools
)

# 8. GoogleTest (Local standalone source under external/googletest)
if(GPGPU_BUILD_TESTS)
    if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/googletest/CMakeLists.txt")
        message(FATAL_ERROR "Missing local dependency: external/googletest")
    endif()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GPGPU_EXTERNAL_DIR}/googletest" "${CMAKE_BINARY_DIR}/external/googletest" EXCLUDE_FROM_ALL)
endif()

# 9. Google Benchmark (Local standalone source under external/benchmark)
if(GPGPU_BUILD_BENCHMARKS)
    if(NOT EXISTS "${GPGPU_EXTERNAL_DIR}/benchmark/CMakeLists.txt")
        message(FATAL_ERROR "Missing local dependency: external/benchmark")
    endif()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GPGPU_EXTERNAL_DIR}/benchmark" "${CMAKE_BINARY_DIR}/external/benchmark" EXCLUDE_FROM_ALL)
endif()
