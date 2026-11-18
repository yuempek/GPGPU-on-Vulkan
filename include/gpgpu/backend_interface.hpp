#pragma once

#include <gpgpu/types.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gpgpu {

using BufferHandle = uint64_t;
constexpr BufferHandle kInvalidBufferHandle = 0;

using PipelineHandle = uint64_t;
constexpr PipelineHandle kInvalidPipelineHandle = 0;

/**
 * @brief Standard 64-byte Push Constant block shared across all generated compute shaders.
 * Guaranteed to fit within the Vulkan minimum maxPushConstantsSize (128 bytes) on all devices.
 */
struct alignas(16) KernelPushConstants {
    uint32_t dims[4] = {1, 1, 1, 1};             // x: width, y: height, z: depth, w: totalElements
    float    iResolution[4] = {1.0f, 1.0f, 1.0f, 0.0f}; // ShaderToy compatibility
    float    iMouse[4] = {0.0f, 0.0f, 0.0f, 0.0f};      // ShaderToy compatibility
    float    iTime = 0.0f;                       // Elapsed time in seconds
    float    iTimeDelta = 0.0f;                  // Delta time between dispatches
    int32_t  iFrame = 0;                         // Dispatch frame counter
    uint32_t flags = 0;                          // Reserved runtime flags
};
static_assert(sizeof(KernelPushConstants) == 64, "KernelPushConstants must be exactly 64 bytes");

/**
 * @brief Metadata for a single bound Data variable during kernel compilation and dispatch.
 */
struct BoundVariableDesc {
    std::string name;
    BufferHandle buffer = kInvalidBufferHandle;
    DataType dtype = DataType::Float32;
    Interpolation interpolation = Interpolation::None;
    Dim3 dims{};
    size_t sizeBytes = 0;
    bool isReadOnly = false;
    bool isWriteOnly = false;
};

/**
 * @brief Complete description of a single compute dispatch command.
 */
struct DispatchDesc {
    uint64_t pipelineHash = 0;
    std::span<const uint32_t> spirvCode;
    Dim3 logicalDims{1, 1, 1};
    Dim3 workgroupSize{256, 1, 1};
    std::span<const BoundVariableDesc> bindings;
    KernelPushConstants pushConstants{};
};

/**
 * @brief Abstract Hardware Abstraction Layer (HAL) interface for compute backends.
 * Primary implementation: VulkanBackend (Vulkan 1.2 + volk + VMA).
 * Designed to allow future CUDA, Metal, DirectX 12, or WebGPU backends without changing public API.
 */
class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    // Lifecycle & Status
    virtual bool isValid() const noexcept = 0;
    virtual std::optional<GPUError> lastError() const noexcept = 0;
    virtual void clearError() noexcept = 0;
    virtual void setError(GPUError error) noexcept = 0;

    // Device & Capability Queries
    virtual const DeviceInfo& deviceInfo() const noexcept = 0;
    virtual const Capabilities& capabilities() const noexcept = 0;
    virtual MemoryInfo memoryInfo() const noexcept = 0;

    // Buffer & Memory Management
    virtual BufferHandle createBuffer(size_t sizeBytes) = 0;
    virtual void destroyBuffer(BufferHandle handle) = 0;
    virtual bool uploadBuffer(BufferHandle dst, const void* srcHostPtr, size_t sizeBytes, size_t dstOffset = 0) = 0;
    virtual bool downloadBuffer(BufferHandle src, void* dstHostPtr, size_t sizeBytes, size_t srcOffset = 0) = 0;
    virtual size_t bufferSize(BufferHandle handle) const noexcept = 0;

    // Compute Execution & Synchronization
    virtual uint64_t dispatch(const DispatchDesc& desc, bool waitImmediately) = 0;
    virtual void waitTicket(uint64_t ticket) = 0;
    virtual void waitIdle() = 0;
    virtual bool isTicketCompleted(uint64_t ticket) const = 0;
    virtual void registerCallback(uint64_t ticket, std::function<void()> callback) = 0;
    virtual double ticketElapsedMilliseconds(uint64_t ticket) const noexcept = 0;

    // Explicit Cleanup
    virtual void dispose() = 0;
};

/**
 * @brief Factory functions for enumerating devices and instantiating the Vulkan backend.
 */
std::vector<DeviceInfo> enumerateVulkanDevices();
std::unique_ptr<IComputeBackend> createVulkanBackendAuto();
std::unique_ptr<IComputeBackend> createVulkanBackendByIndex(uint32_t deviceIndex);
std::unique_ptr<IComputeBackend> createVulkanBackendByName(std::string_view nameSubstr);

} // namespace gpgpu
