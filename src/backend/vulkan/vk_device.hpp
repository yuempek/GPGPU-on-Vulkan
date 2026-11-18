#pragma once

#include <gpgpu/types.hpp>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gpgpu::vk {

/**
 * @brief Manages Vulkan Instance (via volk), Physical Device selection,
 * HardwareProfile detection (Raspberry Pi UMA vs Laptop iGPU vs Discrete GPU),
 * and Logical Device + Queue creation.
 */
class VulkanDevice {
public:
    VulkanDevice() = default;
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    bool initAuto();
    bool initByIndex(uint32_t deviceIndex);
    bool initByName(std::string_view nameSubstr);
    void shutdown();

    static std::vector<DeviceInfo> enumerateDevices(GPUError* outErr = nullptr);

    bool isValid() const noexcept { return device_ != VK_NULL_HANDLE; }
    VkInstance instance() const noexcept { return instance_; }
    VkPhysicalDevice physicalDevice() const noexcept { return physicalDevice_; }
    VkDevice device() const noexcept { return device_; }
    VkDevice logicalDevice() const noexcept { return device_; }
    const VolkDeviceTable& table() const noexcept { return deviceTable_; }

    VkQueue computeQueue() const noexcept { return computeQueue_; }
    uint32_t computeQueueFamily() const noexcept { return computeQueueFamily_; }
    VkQueue transferQueue() const noexcept { return transferQueue_; }
    uint32_t transferQueueFamily() const noexcept { return transferQueueFamily_; }

    const DeviceInfo& info() const noexcept { return deviceInfo_; }
    const Capabilities& capabilities() const noexcept { return capabilities_; }
    std::optional<GPUError> lastError() const noexcept { return lastError_; }

private:
    bool createInstance();
    bool selectPhysicalDevice(int targetIndex, std::string_view targetName);
    bool createLogicalDevice();
    static DeviceInfo buildDeviceInfo(uint32_t index, VkPhysicalDevice physDev, Capabilities* outCaps = nullptr);
    static int64_t scoreDevice(const DeviceInfo& info);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VolkDeviceTable deviceTable_{};

    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamily_ = UINT32_MAX;
    VkQueue transferQueue_ = VK_NULL_HANDLE;
    uint32_t transferQueueFamily_ = UINT32_MAX;

    DeviceInfo deviceInfo_{};
    Capabilities capabilities_{};
    std::optional<GPUError> lastError_;
};

} // namespace gpgpu::vk
