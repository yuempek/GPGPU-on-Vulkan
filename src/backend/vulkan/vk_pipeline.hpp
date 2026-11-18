#pragma once

#include "vk_device.hpp"
#include <gpgpu/backend_interface.hpp>
#include <gpgpu/types.hpp>

#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace gpgpu::vk {

/**
 * @brief Selects optimal workgroup dimensions (local_size_x/y/z) and calculates
 * dispatch group counts based on hardware profile and logical data dimensions.
 */
struct WorkgroupSelector {
    static Dim3 selectWorkgroupSize(const DeviceInfo& devInfo, const Capabilities& caps, Dim3 logicalDims) noexcept;
    static Dim3 calculateDispatchGroups(Dim3 logicalDims, Dim3 workgroupSize) noexcept;
};

/**
 * @brief Holds Vulkan pipeline objects for a compiled compute kernel.
 */
struct PipelineInfo {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    uint32_t numBindings = 0;
};

/**
 * @brief Manages Vulkan Descriptor Set Layouts, Pipeline Layouts, and Compute Pipelines.
 * Caches pipelines by 64-bit hash (XXH3 of SPIR-V bytecode).
 */
class VulkanPipelineManager {
public:
    VulkanPipelineManager() = default;
    ~VulkanPipelineManager();

    VulkanPipelineManager(const VulkanPipelineManager&) = delete;
    VulkanPipelineManager& operator=(const VulkanPipelineManager&) = delete;

    bool init(VulkanDevice& device);
    void shutdown();

    PipelineInfo getOrCreatePipeline(uint64_t hash,
                                     std::span<const uint32_t> spirvCode,
                                     uint32_t numBindings,
                                     GPUError* outErr = nullptr);

    void clearCache();

private:
    VulkanDevice* device_ = nullptr;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, PipelineInfo> cache_;
};

} // namespace gpgpu::vk
