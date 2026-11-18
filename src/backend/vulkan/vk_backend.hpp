#pragma once

#include "vk_device.hpp"
#include "vk_execution.hpp"
#include "vk_memory.hpp"
#include "vk_pipeline.hpp"

#include <gpgpu/backend_interface.hpp>

#include <mutex>

namespace gpgpu::vk {

/**
 * @brief Primary Vulkan 1.2 implementation of IComputeBackend.
 */
class VulkanBackend final : public IComputeBackend {
public:
    VulkanBackend() = default;
    ~VulkanBackend() override;

    bool initAuto();
    bool initByIndex(uint32_t deviceIndex);
    bool initByName(std::string_view nameSubstr);

    // IComputeBackend overrides
    bool isValid() const noexcept override { return device_.isValid(); }
    std::optional<GPUError> lastError() const noexcept override;
    void clearError() noexcept override;
    void setError(GPUError error) noexcept override;

    const DeviceInfo& deviceInfo() const noexcept override { return device_.info(); }
    const Capabilities& capabilities() const noexcept override { return device_.capabilities(); }
    MemoryInfo memoryInfo() const noexcept override { return memory_.getMemoryInfo(); }

    BufferHandle createBuffer(size_t sizeBytes) override;
    void destroyBuffer(BufferHandle handle) override;
    bool uploadBuffer(BufferHandle dst, const void* srcHostPtr, size_t sizeBytes, size_t dstOffset = 0) override;
    bool downloadBuffer(BufferHandle src, void* dstHostPtr, size_t sizeBytes, size_t srcOffset = 0) override;
    size_t bufferSize(BufferHandle handle) const noexcept override;

    uint64_t dispatch(const DispatchDesc& desc, bool waitImmediately) override;
    void waitTicket(uint64_t ticket) override;
    void waitIdle() override;
    bool isTicketCompleted(uint64_t ticket) const override;
    void registerCallback(uint64_t ticket, std::function<void()> callback) override;
    double ticketElapsedMilliseconds(uint64_t ticket) const noexcept override;

    void dispose() override;

private:
    bool initSubsystems();

    VulkanDevice device_;
    VulkanMemoryManager memory_;
    VulkanPipelineManager pipelines_;
    VulkanExecutionEngine execution_;

    mutable std::mutex errorMutex_;
    std::optional<GPUError> lastError_;
};

} // namespace gpgpu::vk
