#pragma once

#include "vk_device.hpp"
#include "vk_memory.hpp"
#include "vk_pipeline.hpp"
#include <gpgpu/backend_interface.hpp>
#include <gpgpu/types.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gpgpu::vk {

/**
 * @brief Manages compute command buffers, Vulkan 1.2 Timeline Semaphore synchronization,
 * descriptor allocations, and asynchronous completion callbacks.
 */
class VulkanExecutionEngine {
public:
    static constexpr size_t kRingSize = 16;

    VulkanExecutionEngine() = default;
    ~VulkanExecutionEngine();

    VulkanExecutionEngine(const VulkanExecutionEngine&) = delete;
    VulkanExecutionEngine& operator=(const VulkanExecutionEngine&) = delete;

    bool init(VulkanDevice& device, VulkanMemoryManager& memory, VulkanPipelineManager& pipelines);
    void shutdown();

    uint64_t submitDispatch(const DispatchDesc& desc, bool waitImmediately, GPUError* outErr = nullptr);
    void waitTicket(uint64_t ticket);
    void waitIdle();
    bool isTicketCompleted(uint64_t ticket) const;
    void registerCallback(uint64_t ticket, std::function<void()> callback);
    double ticketElapsedMilliseconds(uint64_t ticket) const noexcept;

private:
    struct CommandSlot {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkDescriptorPool descPool = VK_NULL_HANDLE;
        uint64_t inFlightTicket = 0;
        std::chrono::high_resolution_clock::time_point startTime{};
    };

    struct CallbackTask {
        uint64_t ticket = 0;
        std::function<void()> callback;
    };

    void workerLoop();

    VulkanDevice* device_ = nullptr;
    VulkanMemoryManager* memory_ = nullptr;
    VulkanPipelineManager* pipelines_ = nullptr;

    VkSemaphore timelineSemaphore_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    float timestampPeriod_ = 1.0f;
    bool hasTimestamps_ = false;
    std::atomic<uint64_t> nextTicket_{1};

    std::mutex submitMutex_;
    std::array<CommandSlot, kRingSize> ringSlots_{};
    mutable std::array<double, kRingSize> cachedElapsedMs_{};

    // Callback worker thread
    std::mutex callbackMutex_;
    std::condition_variable callbackCv_;
    std::vector<CallbackTask> callbackQueue_;
    std::thread workerThread_;
    std::atomic<bool> stopWorker_{false};
};

} // namespace gpgpu::vk
