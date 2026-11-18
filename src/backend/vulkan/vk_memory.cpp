#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include "vk_memory.hpp"

#include <cstring>

namespace gpgpu::vk {

VulkanMemoryManager::~VulkanMemoryManager() {
    shutdown();
}

bool VulkanMemoryManager::init(VulkanDevice& device) {
    device_ = &device;
    const auto& tbl = device.table();

    VmaVulkanFunctions vmaFuncs{};
    vmaFuncs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFuncs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vmaFuncs.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vmaFuncs.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vmaFuncs.vkAllocateMemory = tbl.vkAllocateMemory;
    vmaFuncs.vkFreeMemory = tbl.vkFreeMemory;
    vmaFuncs.vkMapMemory = tbl.vkMapMemory;
    vmaFuncs.vkUnmapMemory = tbl.vkUnmapMemory;
    vmaFuncs.vkFlushMappedMemoryRanges = tbl.vkFlushMappedMemoryRanges;
    vmaFuncs.vkInvalidateMappedMemoryRanges = tbl.vkInvalidateMappedMemoryRanges;
    vmaFuncs.vkBindBufferMemory = tbl.vkBindBufferMemory;
    vmaFuncs.vkBindImageMemory = tbl.vkBindImageMemory;
    vmaFuncs.vkGetBufferMemoryRequirements = tbl.vkGetBufferMemoryRequirements;
    vmaFuncs.vkGetImageMemoryRequirements = tbl.vkGetImageMemoryRequirements;
    vmaFuncs.vkCreateBuffer = tbl.vkCreateBuffer;
    vmaFuncs.vkDestroyBuffer = tbl.vkDestroyBuffer;
    vmaFuncs.vkCreateImage = tbl.vkCreateImage;
    vmaFuncs.vkDestroyImage = tbl.vkDestroyImage;
    vmaFuncs.vkCmdCopyBuffer = tbl.vkCmdCopyBuffer;
    vmaFuncs.vkGetBufferMemoryRequirements2KHR = tbl.vkGetBufferMemoryRequirements2;
    vmaFuncs.vkGetImageMemoryRequirements2KHR = tbl.vkGetImageMemoryRequirements2;
    vmaFuncs.vkBindBufferMemory2KHR = tbl.vkBindBufferMemory2;
    vmaFuncs.vkBindImageMemory2KHR = tbl.vkBindImageMemory2;
    vmaFuncs.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;

    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.physicalDevice = device.physicalDevice();
    allocInfo.device = device.device();
    allocInfo.instance = device.instance();
    allocInfo.pVulkanFunctions = &vmaFuncs;
    allocInfo.vulkanApiVersion = VK_API_VERSION_1_2;

    if (vmaCreateAllocator(&allocInfo, &allocator_) != VK_SUCCESS) {
        return false;
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.transferQueueFamily();

    if (tbl.vkCreateCommandPool(device.device(), &poolInfo, nullptr, &transferCmdPool_) != VK_SUCCESS) {
        return false;
    }

    return true;
}

void VulkanMemoryManager::destroyInternalBuffer(InternalBuffer& buf) {
    if (allocator_ != VK_NULL_HANDLE && buf.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buf.buffer, buf.allocation);
        buf.buffer = VK_NULL_HANDLE;
        buf.allocation = VK_NULL_HANDLE;
        buf.mappedPtr = nullptr;
        buf.sizeBytes = 0;
        buf.alignedSizeBytes = 0;
    }
}

void VulkanMemoryManager::evictPoolLRU(uint64_t maxAllowedBytes) {
    while (pooledBytes_ > maxAllowedBytes && !freePool_.empty()) {
        auto oldestIt = freePool_.begin();
        for (auto it = freePool_.begin(); it != freePool_.end(); ++it) {
            if (it->second.lastUsedSeq < oldestIt->second.lastUsedSeq) {
                oldestIt = it;
            }
        }
        pooledBytes_ -= oldestIt->second.alignedSizeBytes;
        destroyInternalBuffer(oldestIt->second);
        freePool_.erase(oldestIt);
    }
}

bool VulkanMemoryManager::ensureStagingBuffer(InternalBuffer& stagingBuf, size_t requiredBytes, VkBufferUsageFlags usage) {
    if (stagingBuf.buffer != VK_NULL_HANDLE && stagingBuf.alignedSizeBytes >= requiredBytes) {
        return true;
    }

    destroyInternalBuffer(stagingBuf);

    const size_t targetCapacity = alignUp(std::max<size_t>(requiredBytes, 16ULL * 1024 * 1024), 1024 * 1024);

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = targetCapacity;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vmaReq{};
    vmaReq.usage = VMA_MEMORY_USAGE_AUTO;
    vmaReq.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkResult res = vmaCreateBuffer(allocator_, &bufInfo, &vmaReq,
                                   &stagingBuf.buffer, &stagingBuf.allocation, &stagingBuf.allocInfo);
    if (res != VK_SUCCESS) {
        return false;
    }

    stagingBuf.sizeBytes = targetCapacity;
    stagingBuf.alignedSizeBytes = targetCapacity;
    stagingBuf.isHostVisible = true;
    stagingBuf.mappedPtr = stagingBuf.allocInfo.pMappedData;
    return true;
}

void VulkanMemoryManager::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (allocator_ != VK_NULL_HANDLE) {
        destroyInternalBuffer(stagingUploadBuffer_);
        destroyInternalBuffer(stagingDownloadBuffer_);

        for (auto& [handle, buf] : activeBuffers_) {
            vmaDestroyBuffer(allocator_, buf.buffer, buf.allocation);
        }
        activeBuffers_.clear();

        for (auto& [alignedSize, buf] : freePool_) {
            vmaDestroyBuffer(allocator_, buf.buffer, buf.allocation);
        }
        freePool_.clear();
    }

    if (device_ && device_->isValid() && transferCmdPool_ != VK_NULL_HANDLE) {
        device_->table().vkDestroyCommandPool(device_->device(), transferCmdPool_, nullptr);
        transferCmdPool_ = VK_NULL_HANDLE;
    }

    if (allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator_);
        allocator_ = VK_NULL_HANDLE;
    }

    usedBytes_ = 0;
    pooledBytes_ = 0;
    device_ = nullptr;
}

BufferHandle VulkanMemoryManager::createBuffer(size_t sizeBytes, GPUError* outErr) {
    if (!allocator_ || sizeBytes == 0) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::InvalidDimensions, "Cannot allocate 0-byte GPU buffer.", {}, {}};
        }
        return kInvalidBufferHandle;
    }

    const size_t alignedSize = alignUp(sizeBytes, 256);
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Check BufferPool for reusable buffer within [alignedSize, alignedSize * 1.25]
    auto it = freePool_.lower_bound(alignedSize);
    if (it != freePool_.end() && it->first <= alignedSize + (alignedSize / 4)) {
        InternalBuffer reused = it->second;
        reused.sizeBytes = sizeBytes;
        pooledBytes_ -= reused.alignedSizeBytes;
        usedBytes_ += reused.alignedSizeBytes;
        freePool_.erase(it);

        const BufferHandle handle = nextHandle_++;
        activeBuffers_.emplace(handle, reused);
        return handle;
    }

    // 2. Allocate via VMA (UMA uses Zero-Copy persistent mapping; dGPU uses DEVICE_LOCAL)
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = alignedSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vmaReq{};
    vmaReq.usage = VMA_MEMORY_USAGE_AUTO;
    if (device_->info().isUnifiedMemory) {
        vmaReq.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        vmaReq.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    InternalBuffer internal{};
    internal.sizeBytes = sizeBytes;
    internal.alignedSizeBytes = alignedSize;

    VkResult res = vmaCreateBuffer(allocator_, &bufInfo, &vmaReq,
                                   &internal.buffer, &internal.allocation, &internal.allocInfo);
    if (res != VK_SUCCESS) {
        if (outErr) {
            *outErr = GPUError{
                GPUError::Code::OutOfDeviceMemory,
                "vmaCreateBuffer failed for " + std::to_string(sizeBytes) + " bytes.",
                {}, {}
            };
        }
        return kInvalidBufferHandle;
    }

    VkMemoryPropertyFlags memFlags = 0;
    vmaGetAllocationMemoryProperties(allocator_, internal.allocation, &memFlags);
    internal.isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    internal.mappedPtr = internal.allocInfo.pMappedData;

    usedBytes_ += alignedSize;
    const BufferHandle handle = nextHandle_++;
    activeBuffers_.emplace(handle, internal);
    return handle;
}

void VulkanMemoryManager::releaseBuffer(BufferHandle handle) {
    if (handle == kInvalidBufferHandle) return;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = activeBuffers_.find(handle);
    if (it == activeBuffers_.end()) return;

    InternalBuffer buf = it->second;
    activeBuffers_.erase(it);
    usedBytes_ -= buf.alignedSizeBytes;

    // Return to BufferPool for zero-overhead reuse
    buf.lastUsedSeq = currentSeq_++;
    pooledBytes_ += buf.alignedSizeBytes;
    freePool_.emplace(buf.alignedSizeBytes, buf);

    // Evict if pooled memory exceeds 25% of total device memory (or 256 MB if memory unknown)
    uint64_t maxPooledBytes = 256ULL * 1024 * 1024;
    if (device_ && device_->info().totalMemoryBytes > 0) {
        maxPooledBytes = std::max<uint64_t>(maxPooledBytes, device_->info().totalMemoryBytes / 4);
    }
    evictPoolLRU(maxPooledBytes);
}

bool VulkanMemoryManager::upload(BufferHandle dst, const void* srcHostPtr, size_t sizeBytes,
                                 size_t dstOffset, GPUError* outErr) {
    if (!srcHostPtr || sizeBytes == 0) return false;

    InternalBuffer target{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = activeBuffers_.find(dst);
        if (it == activeBuffers_.end()) return false;
        target = it->second;
    }

    if (dstOffset + sizeBytes > target.alignedSizeBytes) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::TransferError, "Buffer upload exceeds allocated capacity.", {}, {}};
        }
        return false;
    }

    // Fast Path A: UMA / Zero-Copy persistently mapped buffer
    if (target.isHostVisible && target.mappedPtr != nullptr) {
        std::memcpy(static_cast<uint8_t*>(target.mappedPtr) + dstOffset, srcHostPtr, sizeBytes);
        vmaFlushAllocation(allocator_, target.allocation, dstOffset, sizeBytes);
        return true;
    }

    // Path B: Discrete GPU Staging Upload (reusing persistent staging buffer)
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureStagingBuffer(stagingUploadBuffer_, sizeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::OutOfHostMemory, "Failed to allocate persistent staging upload buffer.", {}, {}};
        }
        return false;
    }

    std::memcpy(stagingUploadBuffer_.mappedPtr, srcHostPtr, sizeBytes);
    vmaFlushAllocation(allocator_, stagingUploadBuffer_.allocation, 0, sizeBytes);

    const auto& tbl = device_->table();
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = transferCmdPool_;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    tbl.vkAllocateCommandBuffers(device_->device(), &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    tbl.vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = sizeBytes;
    tbl.vkCmdCopyBuffer(cmd, stagingUploadBuffer_.buffer, target.buffer, 1, &copyRegion);
    tbl.vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    tbl.vkQueueSubmit(device_->transferQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    tbl.vkQueueWaitIdle(device_->transferQueue());

    tbl.vkFreeCommandBuffers(device_->device(), transferCmdPool_, 1, &cmd);
    return true;
}

bool VulkanMemoryManager::download(BufferHandle src, void* dstHostPtr, size_t sizeBytes,
                                   size_t srcOffset, GPUError* outErr) {
    if (!dstHostPtr || sizeBytes == 0) return false;

    InternalBuffer source{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = activeBuffers_.find(src);
        if (it == activeBuffers_.end()) return false;
        source = it->second;
    }

    if (srcOffset + sizeBytes > source.alignedSizeBytes) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::TransferError, "Buffer download exceeds allocated capacity.", {}, {}};
        }
        return false;
    }

    // Fast Path A: UMA / Zero-Copy persistently mapped buffer
    if (source.isHostVisible && source.mappedPtr != nullptr) {
        vmaInvalidateAllocation(allocator_, source.allocation, srcOffset, sizeBytes);
        std::memcpy(dstHostPtr, static_cast<const uint8_t*>(source.mappedPtr) + srcOffset, sizeBytes);
        return true;
    }

    // Path B: Discrete GPU Staging Readback (reusing persistent staging buffer)
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureStagingBuffer(stagingDownloadBuffer_, sizeBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::OutOfHostMemory, "Failed to allocate persistent staging readback buffer.", {}, {}};
        }
        return false;
    }

    const auto& tbl = device_->table();
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = transferCmdPool_;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    tbl.vkAllocateCommandBuffers(device_->device(), &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    tbl.vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = 0;
    copyRegion.size = sizeBytes;
    tbl.vkCmdCopyBuffer(cmd, source.buffer, stagingDownloadBuffer_.buffer, 1, &copyRegion);
    tbl.vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    tbl.vkQueueSubmit(device_->transferQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    tbl.vkQueueWaitIdle(device_->transferQueue());

    vmaInvalidateAllocation(allocator_, stagingDownloadBuffer_.allocation, 0, sizeBytes);
    std::memcpy(dstHostPtr, stagingDownloadBuffer_.mappedPtr, sizeBytes);

    tbl.vkFreeCommandBuffers(device_->device(), transferCmdPool_, 1, &cmd);
    return true;
}

VkBuffer VulkanMemoryManager::getVkBuffer(BufferHandle handle) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = activeBuffers_.find(handle);
    return (it != activeBuffers_.end()) ? it->second.buffer : VK_NULL_HANDLE;
}

size_t VulkanMemoryManager::getBufferSize(BufferHandle handle) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = activeBuffers_.find(handle);
    return (it != activeBuffers_.end()) ? it->second.sizeBytes : 0;
}

MemoryInfo VulkanMemoryManager::getMemoryInfo() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryInfo info{};
    if (device_) {
        info.totalDeviceMemoryBytes = device_->info().totalMemoryBytes;
    }
    info.usedDeviceMemoryBytes = usedBytes_;
    info.pooledBufferBytes = pooledBytes_;
    if (info.totalDeviceMemoryBytes > usedBytes_ + pooledBytes_) {
        info.availableDeviceMemoryBytes = info.totalDeviceMemoryBytes - (usedBytes_ + pooledBytes_);
    }
    info.activeAllocations = static_cast<uint32_t>(activeBuffers_.size());
    info.pooledAllocations = static_cast<uint32_t>(freePool_.size());
    return info;
}

} // namespace gpgpu::vk
