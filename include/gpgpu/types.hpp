#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gpgpu {

/**
 * @brief Supported interpolation modes for sampling Data buffers in GLSL kernels.
 */
enum class Interpolation : uint8_t {
    None = 0,     // Direct SSBO array index access (fastest, zero interpolation overhead)
    Nearest = 1,  // Nearest-neighbor coordinate rounding
    Linear = 2,   // Bilinear (2D) / Trilinear (3D) / Linear (1D) interpolation
    Cubic = 3     // Catmull-Rom Bicubic / Cubic spline interpolation
};

/**
 * @brief Parse string representation ("linear"|"lineer"|"cubic"|"nearest"|"none") into Interpolation enum.
 */
inline Interpolation parseInterpolation(std::string_view str) noexcept {
    std::string lower;
    lower.reserve(str.size());
    for (char c : str) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "linear" || lower == "lineer" || lower == "bilinear") {
        return Interpolation::Linear;
    }
    if (lower == "cubic" || lower == "bicubic" || lower == "kubik") {
        return Interpolation::Cubic;
    }
    if (lower == "nearest" || lower == "point") {
        return Interpolation::Nearest;
    }
    return Interpolation::None;
}

/**
 * @brief Supported scalar and vector data types across CPU and GPU (std430 layout compatible).
 */
enum class DataType : uint8_t {
    Float32 = 0,
    Float64,
    Int32,
    UInt32,
    Int16,
    UInt16,
    Int8,
    UInt8,
    Vec2,
    Vec3,
    Vec4,
    IVec2,
    IVec4,
    UVec2,
    UVec4
};

/**
 * @brief Byte size of a single element of a given DataType on the host (CPU).
 */
constexpr size_t dataTypeSize(DataType dt) noexcept {
    switch (dt) {
        case DataType::Float32: return 4;
        case DataType::Float64: return 8;
        case DataType::Int32:   return 4;
        case DataType::UInt32:  return 4;
        case DataType::Int16:   return 2;
        case DataType::UInt16:  return 2;
        case DataType::Int8:    return 1;
        case DataType::UInt8:   return 1;
        case DataType::Vec2:    return 8;
        case DataType::Vec3:    return 12;
        case DataType::Vec4:    return 16;
        case DataType::IVec2:   return 8;
        case DataType::IVec4:   return 16;
        case DataType::UVec2:   return 8;
        case DataType::UVec4:   return 16;
    }
    return 4;
}

/**
 * @brief Byte stride of a single element of a given DataType in GLSL std430 storage buffers on the GPU.
 * Note: In std430 layout, vec3 arrays have a base alignment and array stride of 16 bytes (4 floats).
 */
constexpr size_t dataTypeDeviceStride(DataType dt) noexcept {
    if (dt == DataType::Vec3) {
        return 16;
    }
    return dataTypeSize(dt);
}

/**
 * @brief Corresponding GLSL 450 type keyword for a given DataType.
 */
constexpr std::string_view dataTypeToGlslString(DataType dt) noexcept {
    switch (dt) {
        case DataType::Float32: return "float";
        case DataType::Float64: return "double";
        case DataType::Int32:   return "int";
        case DataType::UInt32:  return "uint";
        case DataType::Int16:   return "int16_t";
        case DataType::UInt16:  return "uint16_t";
        case DataType::Int8:    return "int8_t";
        case DataType::UInt8:   return "uint8_t";
        case DataType::Vec2:    return "vec2";
        case DataType::Vec3:    return "vec3";
        case DataType::Vec4:    return "vec4";
        case DataType::IVec2:   return "ivec2";
        case DataType::IVec4:   return "ivec4";
        case DataType::UVec2:   return "uvec2";
        case DataType::UVec4:   return "uvec4";
    }
    return "float";
}

// Lightweight vector types for host-side convenience when not using GLM
struct alignas(8)  vec2  { float x = 0.0f, y = 0.0f; };
struct             vec3  { float x = 0.0f, y = 0.0f, z = 0.0f; };
struct alignas(16) vec4  { float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f; };
struct alignas(8)  ivec2 { int32_t x = 0, y = 0; };
struct alignas(16) ivec4 { int32_t x = 0, y = 0, z = 0, w = 0; };
struct alignas(8)  uvec2 { uint32_t x = 0, y = 0; };
struct alignas(16) uvec4 { uint32_t x = 0, y = 0, z = 0, w = 0; };

template <typename T>
struct TypeTraits;

template <> struct TypeTraits<float>    { static constexpr DataType type = DataType::Float32; };
template <> struct TypeTraits<double>   { static constexpr DataType type = DataType::Float64; };
template <> struct TypeTraits<int32_t>  { static constexpr DataType type = DataType::Int32;   };
template <> struct TypeTraits<uint32_t> { static constexpr DataType type = DataType::UInt32;  };
template <> struct TypeTraits<int16_t>  { static constexpr DataType type = DataType::Int16;   };
template <> struct TypeTraits<uint16_t> { static constexpr DataType type = DataType::UInt16;  };
template <> struct TypeTraits<int8_t>   { static constexpr DataType type = DataType::Int8;    };
template <> struct TypeTraits<uint8_t>  { static constexpr DataType type = DataType::UInt8;   };
template <> struct TypeTraits<vec2>     { static constexpr DataType type = DataType::Vec2;    };
template <> struct TypeTraits<vec3>     { static constexpr DataType type = DataType::Vec3;    };
template <> struct TypeTraits<vec4>     { static constexpr DataType type = DataType::Vec4;    };
template <> struct TypeTraits<ivec2>    { static constexpr DataType type = DataType::IVec2;   };
template <> struct TypeTraits<ivec4>    { static constexpr DataType type = DataType::IVec4;   };
template <> struct TypeTraits<uvec2>    { static constexpr DataType type = DataType::UVec2;   };
template <> struct TypeTraits<uvec4>    { static constexpr DataType type = DataType::UVec4;   };

#if defined(_WIN32)
template <> struct TypeTraits<long>          { static constexpr DataType type = DataType::Int32;   };
template <> struct TypeTraits<unsigned long> { static constexpr DataType type = DataType::UInt32;  };
#endif

/**
 * @brief 1D, 2D, or 3D logical data / dispatch dimensions.
 */
struct Dim3 {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;

    constexpr Dim3() noexcept = default;
    constexpr Dim3(uint32_t width, uint32_t height = 1, uint32_t depth = 1) noexcept
        : x(width), y(height), z(depth) {}

    constexpr size_t totalElements() const noexcept {
        return static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
    }

    constexpr int dimensions() const noexcept {
        return (z > 1) ? 3 : ((y > 1) ? 2 : 1);
    }

    constexpr bool operator==(const Dim3& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
    constexpr bool operator!=(const Dim3& other) const noexcept {
        return !(*this == other);
    }
};

/**
 * @brief Detailed diagnostic information returned by gpu.hasError().
 */
struct GPUError {
    enum class Code : uint32_t {
        None = 0,
        NoVulkanSupport,
        DeviceNotFound,
        OutOfDeviceMemory,
        OutOfHostMemory,
        KernelParseError,
        ShaderCompileError,
        SpirvValidationError,
        PipelineCreationError,
        InvalidDimensions,
        UnboundDataVariable,
        TransferError,
        DeviceLost,
        InternalError
    };

    Code code = Code::None;
    std::string message;
    std::string shaderLog;      // Compiler log mapped to user kernel line numbers
    std::string generatedGlsl;  // Full generated GLSL source for debugging

    constexpr explicit operator bool() const noexcept {
        return code != Code::None;
    }
};

/**
 * @brief Hardware profile classification for adaptive memory and workgroup tuning.
 */
enum class HardwareProfile : uint8_t {
    EmbeddedUMA = 0,  // Raspberry Pi (VideoCore VI/VII V3DV), mobile SoCs (small workgroups, Zero-Copy UMA)
    IntegratedUMA,    // Laptop iGPUs (Intel Iris/Arc, AMD APU, Apple Silicon - 256 workgroup, Zero-Copy UMA)
    DiscreteGPU,      // Dedicated PCIe GPUs (NVIDIA RTX, AMD Radeon RX - VRAM + Staging Ring-Buffer)
    SoftwareCPU       // Fallback software rasterizer/compute (llvmpipe / SwiftShader)
};

/**
 * @brief Physical GPU device metadata returned by GPU::devices() and gpu.info().
 */
struct DeviceInfo {
    uint32_t index = 0;
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    std::string name;
    std::string vendor;
    std::string driverVersion;
    std::string apiVersion;
    bool isDiscrete = false;
    bool isUnifiedMemory = false;
    HardwareProfile profile = HardwareProfile::IntegratedUMA;
    uint64_t totalMemoryBytes = 0;
};

/**
 * @brief Live memory usage statistics returned by gpu.memoryInfo().
 */
struct MemoryInfo {
    uint64_t totalDeviceMemoryBytes = 0;
    uint64_t usedDeviceMemoryBytes = 0;
    uint64_t availableDeviceMemoryBytes = 0;
    uint64_t pooledBufferBytes = 0;
    uint32_t activeAllocations = 0;
    uint32_t pooledAllocations = 0;
};

/**
 * @brief Device compute limits and feature flags returned by gpu.capabilities().
 */
struct Capabilities {
    uint32_t maxWorkGroupCount[3] = {65535, 65535, 65535};
    uint32_t maxWorkGroupSize[3] = {256, 256, 64};
    uint32_t maxWorkGroupInvocations = 256;
    uint32_t subgroupSize = 32;
    uint64_t maxStorageBufferRange = 134217728ULL;
    uint32_t maxPushConstantsSize = 128;
    float timestampPeriod = 1.0f;
    bool supportsFloat64 = false;
    bool supportsFloat16 = false;
    bool supportsInt16 = false;
    bool supportsInt8 = false;
    bool supportsTimelineSemaphore = true;
};

} // namespace gpgpu
