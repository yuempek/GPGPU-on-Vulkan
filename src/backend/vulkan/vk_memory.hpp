#pragma once

#include "vk_device.hpp"
#include <gpgpu/backend_interface.hpp>

#include <vk_mem_alloc.h>

#include <map>
#include <mutex>
#include <unordered_map>

namespace gpgpu::vk {

struct InternalBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo{};
    size_t sizeBytes = 0;
    size_t alignedSizeBytes = 0;
    void* mappedPtr = nullptr;
    bool isHostVisible = false;
    bool hostDirty = false;
    bool deviceDirty = false;
    uint64_t lastUsedSeq = 0;
};

/**
 * @brief VMA-backed GPU memory manager with BufferPool reuse and adaptive UMA Zero-Copy vs Staging transfers.
 */
class VulkanMemoryManager {
public:
    VulkanMemoryManager() = default;
    ~VulkanMemoryManager();

    VulkanMemoryManager(const VulkanMemoryManager&) = delete;
    VulkanMemoryManager& operator=(const VulkanMemoryManager&) = delete;

    bool init(VulkanDevice& device);
    void shutdown();

    BufferHandle createBuffer(size_t sizeBytes, GPUError* outErr = nullptr);
    void releaseBuffer(BufferHandle handle);

    bool upload(BufferHandle dst, const void* srcHostPtr, size_t sizeBytes, size_t dstOffset = 0, GPUError* outErr = nullptr);
    bool download(BufferHandle src, void* dstHostPtr, size_t sizeBytes, size_t srcOffset = 0, GPUError* outErr = nullptr);

    VkBuffer getVkBuffer(BufferHandle handle) const noexcept;
    size_t getBufferSize(BufferHandle handle) const noexcept;
    MemoryInfo getMemoryInfo() const noexcept;

private:
    static constexpr size_t alignUp(size_t val, size_t alignment) noexcept {
        return (val + alignment - 1) & ~(alignment - 1);
    }

    bool ensureStagingBuffer(InternalBuffer& stagingBuf, size_t requiredBytes, VkBufferUsageFlags usage);
    void destroyInternalBuffer(InternalBuffer& buf);
    void evictPoolLRU(uint64_t maxAllowedBytes);

    VulkanDevice* device_ = nullptr;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkCommandPool transferCmdPool_ = VK_NULL_HANDLE;

    mutable std::mutex mutex_;
    BufferHandle nextHandle_ = 1;
    uint64_t currentSeq_ = 1;
    std::unordered_map<BufferHandle, InternalBuffer> activeBuffers_;
    std::multimap<size_t, InternalBuffer> freePool_;
    uint64_t usedBytes_ = 0;
    uint64_t pooledBytes_ = 0;

    // Persistent staging buffers for discrete GPUs (zero allocations during hot-path upload/download)
    InternalBuffer stagingUploadBuffer_{};
    InternalBuffer stagingDownloadBuffer_{};
};

} // namespace gpgpu::vk
