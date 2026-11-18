cmake_minimum_required(VERSION 3.18)

# ==============================================================================
# GPGPU.VULKAN Pre-built SDK CMake Integration
#
# Usage in another project's CMakeLists.txt:
#   add_subdirectory(path/to/deploy gpgpu_sdk)
#   target_link_libraries(my_app PRIVATE gpgpu::shared)  # or gpgpu::static
# ==============================================================================

if(NOT TARGET gpgpu::shared)
    add_library(gpgpu_vulkan_shared_imported SHARED IMPORTED GLOBAL)
    add_library(gpgpu::shared ALIAS gpgpu_vulkan_shared_imported)

    set_target_properties(gpgpu_vulkan_shared_imported PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include"
        INTERFACE_COMPILE_DEFINITIONS "GPGPU_VULKAN_USE_SHARED"
        INTERFACE_COMPILE_FEATURES "cxx_std_20"
    )

    if(WIN32)
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/lib/windows-x64/gpgpu_vulkan.dll")
            set(_LIB_LOC "${CMAKE_CURRENT_LIST_DIR}/lib/windows-x64/gpgpu_vulkan.dll")
            set(_IMP_LOC "${CMAKE_CURRENT_LIST_DIR}/lib/windows-x64/libgpgpu_vulkan.dll.a")
        else()
            set(_LIB_LOC "${CMAKE_CURRENT_LIST_DIR}/lib/gpgpu_vulkan.dll")
            set(_IMP_LOC "${CMAKE_CURRENT_LIST_DIR}/lib/libgpgpu_vulkan.dll.a")
        endif()
        set_target_properties(gpgpu_vulkan_shared_imported PROPERTIES
            IMPORTED_LOCATION "${_LIB_LOC}"
            IMPORTED_IMPLIB   "${_IMP_LOC}"
        )
    else()
        set(_ARCH_SO "${CMAKE_CURRENT_LIST_DIR}/lib/linux-${CMAKE_SYSTEM_PROCESSOR}/libgpgpu_vulkan.so")
        if(EXISTS "${_ARCH_SO}")
            set(_LIB_LOC "${_ARCH_SO}")
        else()
            set(_LIB_LOC "${CMAKE_CURRENT_LIST_DIR}/lib/libgpgpu_vulkan.so")
        endif()
        set_target_properties(gpgpu_vulkan_shared_imported PROPERTIES
            IMPORTED_LOCATION "${_LIB_LOC}"
        )
    endif()
endif()

if(NOT TARGET gpgpu::static)
    add_library(gpgpu_vulkan_static_imported STATIC IMPORTED GLOBAL)
    add_library(gpgpu::static ALIAS gpgpu_vulkan_static_imported)

    set_target_properties(gpgpu_vulkan_static_imported PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include"
        INTERFACE_COMPILE_FEATURES "cxx_std_20"
        IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/libgpgpu_vulkan_static.a"
    )

    if(UNIX AND NOT APPLE)
        set_target_properties(gpgpu_vulkan_static_imported PROPERTIES
            INTERFACE_LINK_LIBRARIES "dl;pthread"
        )
    endif()
endif()
