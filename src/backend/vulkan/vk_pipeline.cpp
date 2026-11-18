#include "vk_pipeline.hpp"

#include <spdlog/spdlog.h>
#include <xxhash.h>

namespace gpgpu::vk {

Dim3 WorkgroupSelector::selectWorkgroupSize(const DeviceInfo& devInfo,
                                            const Capabilities& caps,
                                            Dim3 logicalDims) noexcept {
    const bool isEmbedded = (devInfo.profile == HardwareProfile::EmbeddedUMA) ||
                            (caps.maxWorkGroupInvocations < 256) ||
                            (caps.subgroupSize <= 16);

    const int dims = logicalDims.dimensions();
    if (dims == 1) {
        return isEmbedded ? Dim3{64, 1, 1} : Dim3{256, 1, 1};
    }
    if (dims == 2) {
        return isEmbedded ? Dim3{8, 8, 1} : Dim3{16, 16, 1};
    }
    return isEmbedded ? Dim3{4, 4, 4} : Dim3{8, 8, 4};
}

Dim3 WorkgroupSelector::calculateDispatchGroups(Dim3 logicalDims, Dim3 workgroupSize) noexcept {
    const uint32_t gx = (logicalDims.x + workgroupSize.x - 1) / workgroupSize.x;
    const uint32_t gy = (logicalDims.y + workgroupSize.y - 1) / workgroupSize.y;
    const uint32_t gz = (logicalDims.z + workgroupSize.z - 1) / workgroupSize.z;
    return Dim3{gx > 0 ? gx : 1, gy > 0 ? gy : 1, gz > 0 ? gz : 1};
}

VulkanPipelineManager::~VulkanPipelineManager() {
    shutdown();
}

bool VulkanPipelineManager::init(VulkanDevice& device) {
    device_ = &device;
    return true;
}

void VulkanPipelineManager::shutdown() {
    clearCache();
    device_ = nullptr;
}

void VulkanPipelineManager::clearCache() {
    if (!device_ || !device_->isValid()) {
        cache_.clear();
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& tbl = device_->table();
    VkDevice dev = device_->logicalDevice();

    for (auto& [h, info] : cache_) {
        if (info.pipeline != VK_NULL_HANDLE) {
            tbl.vkDestroyPipeline(dev, info.pipeline, nullptr);
        }
        if (info.layout != VK_NULL_HANDLE) {
            tbl.vkDestroyPipelineLayout(dev, info.layout, nullptr);
        }
        if (info.descSetLayout != VK_NULL_HANDLE) {
            tbl.vkDestroyDescriptorSetLayout(dev, info.descSetLayout, nullptr);
        }
    }
    cache_.clear();
}

PipelineInfo VulkanPipelineManager::getOrCreatePipeline(uint64_t hash,
                                                       std::span<const uint32_t> spirvCode,
                                                       uint32_t numBindings,
                                                       GPUError* outErr) {
    if (!device_ || !device_->isValid()) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::PipelineCreationError, "Device is invalid for pipeline creation."};
        }
        return {};
    }

    if (spirvCode.empty()) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::PipelineCreationError, "SPIR-V code is empty."};
        }
        return {};
    }

    if (hash == 0) {
        hash = XXH3_64bits(spirvCode.data(), spirvCode.size_bytes());
        hash ^= (static_cast<uint64_t>(numBindings) << 32);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(hash);
        if (it != cache_.end()) {
            return it->second;
        }
    }

    const auto& tbl = device_->table();
    VkDevice dev = device_->logicalDevice();

    // 1. Descriptor Set Layout
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    if (numBindings > 0) {
        std::vector<VkDescriptorSetLayoutBinding> bindings(numBindings);
        for (uint32_t i = 0; i < numBindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VkResult res = tbl.vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &descSetLayout);
        if (res != VK_SUCCESS) {
            if (outErr) {
                *outErr = GPUError{GPUError::Code::PipelineCreationError, "vkCreateDescriptorSetLayout failed with code " + std::to_string(res)};
            }
            return {};
        }
    }

    // 2. Pipeline Layout (including standard 64-byte Push Constants)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(KernelPushConstants);

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = (descSetLayout != VK_NULL_HANDLE) ? 1 : 0;
    pipeLayoutInfo.pSetLayouts = (descSetLayout != VK_NULL_HANDLE) ? &descSetLayout : nullptr;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushRange;

    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkResult res = tbl.vkCreatePipelineLayout(dev, &pipeLayoutInfo, nullptr, &pipeLayout);
    if (res != VK_SUCCESS) {
        if (descSetLayout != VK_NULL_HANDLE) {
            tbl.vkDestroyDescriptorSetLayout(dev, descSetLayout, nullptr);
        }
        if (outErr) {
            *outErr = GPUError{GPUError::Code::PipelineCreationError, "vkCreatePipelineLayout failed with code " + std::to_string(res)};
        }
        return {};
    }

    // 3. Shader Module
    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = spirvCode.size_bytes();
    smInfo.pCode = spirvCode.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    res = tbl.vkCreateShaderModule(dev, &smInfo, nullptr, &shaderModule);
    if (res != VK_SUCCESS) {
        tbl.vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
        if (descSetLayout != VK_NULL_HANDLE) {
            tbl.vkDestroyDescriptorSetLayout(dev, descSetLayout, nullptr);
        }
        if (outErr) {
            *outErr = GPUError{GPUError::Code::PipelineCreationError, "vkCreateShaderModule failed with code " + std::to_string(res)};
        }
        return {};
    }

    // 4. Compute Pipeline
    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = shaderModule;
    compInfo.stage.pName = "main";
    compInfo.layout = pipeLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    res = tbl.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &compInfo, nullptr, &pipeline);

    // Shader module can be destroyed immediately once pipeline is created
    tbl.vkDestroyShaderModule(dev, shaderModule, nullptr);

    if (res != VK_SUCCESS) {
        tbl.vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
        if (descSetLayout != VK_NULL_HANDLE) {
            tbl.vkDestroyDescriptorSetLayout(dev, descSetLayout, nullptr);
        }
        if (outErr) {
            *outErr = GPUError{GPUError::Code::PipelineCreationError, "vkCreateComputePipelines failed with code " + std::to_string(res)};
        }
        return {};
    }

    PipelineInfo info{pipeline, pipeLayout, descSetLayout, numBindings};

    std::lock_guard<std::mutex> lock(mutex_);
    cache_[hash] = info;
    return info;
}

} // namespace gpgpu::vk
