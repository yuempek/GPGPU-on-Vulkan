#include "vk_execution.hpp"

#include <spdlog/spdlog.h>

namespace gpgpu::vk {

VulkanExecutionEngine::~VulkanExecutionEngine() {
    shutdown();
}

bool VulkanExecutionEngine::init(VulkanDevice& device,
                                 VulkanMemoryManager& memory,
                                 VulkanPipelineManager& pipelines) {
    device_ = &device;
    memory_ = &memory;
    pipelines_ = &pipelines;

    if (!device_->isValid()) {
        return false;
    }

    const auto& tbl = device_->table();
    VkDevice dev = device_->logicalDevice();

    // 1. Create Timeline Semaphore (Vulkan 1.2 core)
    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue = 0;

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &timelineInfo;

    VkResult res = tbl.vkCreateSemaphore(dev, &semInfo, nullptr, &timelineSemaphore_);
    if (res != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan timeline semaphore: {}", static_cast<int>(res));
        return false;
    }

    // 2. Initialize Command Buffer & Descriptor Pool Ring
    for (size_t i = 0; i < kRingSize; ++i) {
        VkCommandPoolCreateInfo cpInfo{};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpInfo.queueFamilyIndex = device_->computeQueueFamily();

        res = tbl.vkCreateCommandPool(dev, &cpInfo, nullptr, &ringSlots_[i].pool);
        if (res != VK_SUCCESS) {
            spdlog::error("Failed to create command pool for ring slot {}: {}", i, static_cast<int>(res));
            return false;
        }

        VkCommandBufferAllocateInfo cbInfo{};
        cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbInfo.commandPool = ringSlots_[i].pool;
        cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbInfo.commandBufferCount = 1;

        res = tbl.vkAllocateCommandBuffers(dev, &cbInfo, &ringSlots_[i].cmd);
        if (res != VK_SUCCESS) {
            spdlog::error("Failed to allocate command buffer for ring slot {}: {}", i, static_cast<int>(res));
            return false;
        }

        VkDescriptorPoolSize poolSizes[1];
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 128;

        VkDescriptorPoolCreateInfo dpInfo{};
        dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpInfo.maxSets = 16;
        dpInfo.poolSizeCount = 1;
        dpInfo.pPoolSizes = poolSizes;

        res = tbl.vkCreateDescriptorPool(dev, &dpInfo, nullptr, &ringSlots_[i].descPool);
        if (res != VK_SUCCESS) {
            spdlog::error("Failed to create descriptor pool for ring slot {}: {}", i, static_cast<int>(res));
            return false;
        }

        ringSlots_[i].inFlightTicket = 0;
    }

    // 3. Initialize Timestamp Query Pool (for hardware profiling)
    hasTimestamps_ = false;
    timestampPeriod_ = device_->capabilities().timestampPeriod;
    if (timestampPeriod_ > 0.0f) {
        VkQueryPoolCreateInfo qpInfo{};
        qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = static_cast<uint32_t>(kRingSize * 2);
        if (tbl.vkCreateQueryPool(dev, &qpInfo, nullptr, &queryPool_) == VK_SUCCESS) {
            hasTimestamps_ = true;
        }
    }

    // 4. Start Background Worker Thread for onDone Callbacks
    stopWorker_.store(false);
    workerThread_ = std::thread(&VulkanExecutionEngine::workerLoop, this);

    return true;
}

void VulkanExecutionEngine::shutdown() {
    stopWorker_.store(true);
    callbackCv_.notify_all();
    if (workerThread_.joinable()) {
        workerThread_.join();
    }

    waitIdle();

    if (device_ && device_->isValid()) {
        const auto& tbl = device_->table();
        VkDevice dev = device_->logicalDevice();

        for (auto& slot : ringSlots_) {
            if (slot.descPool != VK_NULL_HANDLE) {
                tbl.vkDestroyDescriptorPool(dev, slot.descPool, nullptr);
                slot.descPool = VK_NULL_HANDLE;
            }
            if (slot.pool != VK_NULL_HANDLE) {
                tbl.vkDestroyCommandPool(dev, slot.pool, nullptr);
                slot.pool = VK_NULL_HANDLE;
            }
            slot.cmd = VK_NULL_HANDLE;
            slot.inFlightTicket = 0;
        }

        if (timelineSemaphore_ != VK_NULL_HANDLE) {
            tbl.vkDestroySemaphore(dev, timelineSemaphore_, nullptr);
            timelineSemaphore_ = VK_NULL_HANDLE;
        }

        if (queryPool_ != VK_NULL_HANDLE) {
            tbl.vkDestroyQueryPool(dev, queryPool_, nullptr);
            queryPool_ = VK_NULL_HANDLE;
        }
    }

    device_ = nullptr;
    memory_ = nullptr;
    pipelines_ = nullptr;
}

uint64_t VulkanExecutionEngine::submitDispatch(const DispatchDesc& desc,
                                               bool waitImmediately,
                                               GPUError* outErr) {
    if (!device_ || !device_->isValid()) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::InternalError, "Device is invalid for dispatch."};
        }
        return 0;
    }

    // Logical dimensions validation
    if (desc.logicalDims.x == 0 || desc.logicalDims.y == 0 || desc.logicalDims.z == 0) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::InvalidDimensions, "Logical dispatch dimensions must be non-zero."};
        }
        return 0;
    }

    // Workgroup and Group Count calculation
    Dim3 wg = desc.workgroupSize;
    if (wg.x == 0 || wg.y == 0 || wg.z == 0) {
        wg = WorkgroupSelector::selectWorkgroupSize(device_->info(), device_->capabilities(), desc.logicalDims);
    }
    Dim3 groups = WorkgroupSelector::calculateDispatchGroups(desc.logicalDims, wg);
    const auto& caps = device_->capabilities();
    if (groups.x > caps.maxWorkGroupCount[0] ||
        groups.y > caps.maxWorkGroupCount[1] ||
        groups.z > caps.maxWorkGroupCount[2]) {
        if (outErr) {
            *outErr = GPUError{
                GPUError::Code::InvalidDimensions,
                "Calculated dispatch groups (" + std::to_string(groups.x) + ", " +
                std::to_string(groups.y) + ", " + std::to_string(groups.z) +
                ") exceed physical hardware limits."
            };
        }
        return 0;
    }

    // Pipeline acquisition
    auto pipeInfo = pipelines_->getOrCreatePipeline(desc.pipelineHash, desc.spirvCode,
                                                    static_cast<uint32_t>(desc.bindings.size()), outErr);
    if (pipeInfo.pipeline == VK_NULL_HANDLE) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(submitMutex_);
    uint64_t ticket = nextTicket_.fetch_add(1, std::memory_order_relaxed);
    size_t slotIdx = ticket % kRingSize;
    auto& slot = ringSlots_[slotIdx];

    const auto& tbl = device_->table();
    VkDevice dev = device_->logicalDevice();

    // Reclaim slot if previous ticket is still in-flight
    if (slot.inFlightTicket != 0) {
        uint64_t completed = 0;
        tbl.vkGetSemaphoreCounterValue(dev, timelineSemaphore_, &completed);
        if (completed < slot.inFlightTicket) {
            VkSemaphoreWaitInfo waitInfo{};
            waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            waitInfo.semaphoreCount = 1;
            waitInfo.pSemaphores = &timelineSemaphore_;
            waitInfo.pValues = &slot.inFlightTicket;
            tbl.vkWaitSemaphores(dev, &waitInfo, UINT64_MAX);
        }
    }

    tbl.vkResetCommandPool(dev, slot.pool, 0);
    tbl.vkResetDescriptorPool(dev, slot.descPool, 0);

    // Allocate and update descriptor set
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    if (!desc.bindings.empty()) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = slot.descPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &pipeInfo.descSetLayout;

        VkResult res = tbl.vkAllocateDescriptorSets(dev, &allocInfo, &descSet);
        if (res != VK_SUCCESS) {
            if (outErr) {
                *outErr = GPUError{GPUError::Code::InternalError, "vkAllocateDescriptorSets failed: " + std::to_string(res)};
            }
            return 0;
        }

        std::vector<VkDescriptorBufferInfo> bufferInfos(desc.bindings.size());
        std::vector<VkWriteDescriptorSet> writes(desc.bindings.size());

        for (size_t i = 0; i < desc.bindings.size(); ++i) {
            const auto& b = desc.bindings[i];
            VkBuffer vkBuf = memory_->getVkBuffer(b.buffer);
            if (vkBuf == VK_NULL_HANDLE) {
                if (outErr) {
                    *outErr = GPUError{GPUError::Code::UnboundDataVariable, "Data variable '" + b.name + "' references invalid buffer handle."};
                }
                return 0;
            }

            bufferInfos[i].buffer = vkBuf;
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = (b.sizeBytes > 0) ? b.sizeBytes : VK_WHOLE_SIZE;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descSet;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].dstArrayElement = 0;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i];
        }

        tbl.vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // Record command buffer
    slot.startTime = std::chrono::high_resolution_clock::now();
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    tbl.vkBeginCommandBuffer(slot.cmd, &beginInfo);

    if (hasTimestamps_ && queryPool_ != VK_NULL_HANDLE) {
        tbl.vkCmdResetQueryPool(slot.cmd, queryPool_, static_cast<uint32_t>(slotIdx * 2), 2);
        tbl.vkCmdWriteTimestamp(slot.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, static_cast<uint32_t>(slotIdx * 2));
    }

    // Pre-execution memory barrier
    VkMemoryBarrier preBarrier{};
    preBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    preBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    preBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    tbl.vkCmdPipelineBarrier(slot.cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &preBarrier, 0, nullptr, 0, nullptr);

    tbl.vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeInfo.pipeline);

    if (descSet != VK_NULL_HANDLE) {
        tbl.vkCmdBindDescriptorSets(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeInfo.layout, 0, 1, &descSet, 0, nullptr);
    }

    KernelPushConstants push = desc.pushConstants;
    push.dims[0] = desc.logicalDims.x;
    push.dims[1] = desc.logicalDims.y;
    push.dims[2] = desc.logicalDims.z;
    push.dims[3] = static_cast<uint32_t>(desc.logicalDims.totalElements());
    push.iResolution[0] = static_cast<float>(desc.logicalDims.x);
    push.iResolution[1] = static_cast<float>(desc.logicalDims.y);
    push.iResolution[2] = 1.0f;
    tbl.vkCmdPushConstants(slot.cmd, pipeInfo.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelPushConstants), &push);

    tbl.vkCmdDispatch(slot.cmd, groups.x, groups.y, groups.z);

    if (hasTimestamps_ && queryPool_ != VK_NULL_HANDLE) {
        tbl.vkCmdWriteTimestamp(slot.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, static_cast<uint32_t>(slotIdx * 2 + 1));
    }

    // Post-execution barrier
    VkMemoryBarrier postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    postBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    tbl.vkCmdPipelineBarrier(slot.cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &postBarrier, 0, nullptr, 0, nullptr);

    tbl.vkEndCommandBuffer(slot.cmd);

    // Submit to compute queue with timeline semaphore
    VkTimelineSemaphoreSubmitInfo timelineSubmit{};
    timelineSubmit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmit.signalSemaphoreValueCount = 1;
    timelineSubmit.pSignalSemaphoreValues = &ticket;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineSubmit;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &slot.cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &timelineSemaphore_;

    VkResult res = tbl.vkQueueSubmit(device_->computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::InternalError, "vkQueueSubmit failed with code " + std::to_string(res)};
        }
        return 0;
    }
    slot.inFlightTicket = ticket;

    if (waitImmediately) {
        waitTicket(ticket);
        auto endCpu = std::chrono::high_resolution_clock::now();
        cachedElapsedMs_[slotIdx] = std::chrono::duration<double, std::milli>(endCpu - slot.startTime).count();
    }
    return ticket;
}

double VulkanExecutionEngine::ticketElapsedMilliseconds(uint64_t ticket) const noexcept {
    if (ticket == 0 || !device_ || !device_->isValid()) return 0.0;
    size_t slotIdx = ticket % kRingSize;

    uint64_t completed = 0;
    device_->table().vkGetSemaphoreCounterValue(device_->logicalDevice(), timelineSemaphore_, &completed);
    if (completed < ticket) {
        return 0.0;
    }

    if (hasTimestamps_ && queryPool_ != VK_NULL_HANDLE) {
        uint64_t timestamps[2] = {0, 0};
        VkResult res = device_->table().vkGetQueryPoolResults(
            device_->logicalDevice(), queryPool_, static_cast<uint32_t>(slotIdx * 2), 2,
            sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (res == VK_SUCCESS && timestamps[1] >= timestamps[0]) {
            double durationNs = static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(timestampPeriod_);
            cachedElapsedMs_[slotIdx] = durationNs / 1000000.0;
            return cachedElapsedMs_[slotIdx];
        }
    }
    return cachedElapsedMs_[slotIdx];
}

void VulkanExecutionEngine::waitTicket(uint64_t ticket) {
    if (ticket == 0 || !device_ || !device_->isValid() || timelineSemaphore_ == VK_NULL_HANDLE) {
        return;
    }
    const auto& tbl = device_->table();
    VkDevice dev = device_->logicalDevice();

    uint64_t current = 0;
    tbl.vkGetSemaphoreCounterValue(dev, timelineSemaphore_, &current);
    if (current >= ticket) {
        return;
    }

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &timelineSemaphore_;
    waitInfo.pValues = &ticket;
    tbl.vkWaitSemaphores(dev, &waitInfo, UINT64_MAX);
}

void VulkanExecutionEngine::waitIdle() {
    if (device_ && device_->isValid()) {
        device_->table().vkQueueWaitIdle(device_->computeQueue());
    }
}

bool VulkanExecutionEngine::isTicketCompleted(uint64_t ticket) const {
    if (ticket == 0 || !device_ || !device_->isValid() || timelineSemaphore_ == VK_NULL_HANDLE) {
        return true;
    }
    uint64_t current = 0;
    device_->table().vkGetSemaphoreCounterValue(device_->logicalDevice(), timelineSemaphore_, &current);
    return current >= ticket;
}

void VulkanExecutionEngine::registerCallback(uint64_t ticket, std::function<void()> callback) {
    if (!callback) return;
    if (isTicketCompleted(ticket)) {
        callback();
        return;
    }
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        callbackQueue_.push_back({ticket, std::move(callback)});
    }
    callbackCv_.notify_one();
}

void VulkanExecutionEngine::workerLoop() {
    while (!stopWorker_.load(std::memory_order_relaxed)) {
        CallbackTask task;
        {
            std::unique_lock<std::mutex> lock(callbackMutex_);
            callbackCv_.wait(lock, [this]() {
                return stopWorker_.load(std::memory_order_relaxed) || !callbackQueue_.empty();
            });
            if (stopWorker_.load(std::memory_order_relaxed) && callbackQueue_.empty()) {
                break;
            }
            task = std::move(callbackQueue_.front());
            callbackQueue_.erase(callbackQueue_.begin());
        }

        if (task.ticket > 0 && device_ && device_->isValid() && timelineSemaphore_ != VK_NULL_HANDLE) {
            VkSemaphoreWaitInfo waitInfo{};
            waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            waitInfo.semaphoreCount = 1;
            waitInfo.pSemaphores = &timelineSemaphore_;
            waitInfo.pValues = &task.ticket;
            device_->table().vkWaitSemaphores(device_->logicalDevice(), &waitInfo, UINT64_MAX);
        }

        if (task.callback) {
            task.callback();
        }
    }
}

} // namespace gpgpu::vk
