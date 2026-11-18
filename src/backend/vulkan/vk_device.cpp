#define VOLK_IMPLEMENTATION
#include "vk_device.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <mutex>
#include <sstream>

namespace gpgpu::vk {

namespace {

std::string toLower(std::string_view sv) {
    std::string out;
    out.reserve(sv.size());
    for (char c : sv) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

std::string vendorIdToName(uint32_t vendorId) {
    switch (vendorId) {
        case 0x10DE: return "NVIDIA";
        case 0x1002:
        case 0x1022: return "AMD";
        case 0x8086: return "Intel";
        case 0x13B5: return "ARM";
        case 0x14E4: return "Broadcom (Raspberry Pi)";
        case 0x5143: return "Qualcomm";
        case 0x106B: return "Apple";
        default:     return "Unknown Vendor";
    }
}

std::string formatVersion(uint32_t v) {
    std::ostringstream oss;
    oss << VK_API_VERSION_MAJOR(v) << "."
        << VK_API_VERSION_MINOR(v) << "."
        << VK_API_VERSION_PATCH(v);
    return oss.str();
}

bool ensureVolkInitialized() {
    static std::once_flag flag;
    static bool initialized = false;
    std::call_once(flag, []() {
        initialized = (volkInitialize() == VK_SUCCESS);
    });
    return initialized;
}

} // namespace

VulkanDevice::~VulkanDevice() {
    shutdown();
}

void VulkanDevice::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        if (deviceTable_.vkDeviceWaitIdle) {
            deviceTable_.vkDeviceWaitIdle(device_);
        }
        if (deviceTable_.vkDestroyDevice) {
            deviceTable_.vkDestroyDevice(device_, nullptr);
        }
        device_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    physicalDevice_ = VK_NULL_HANDLE;
    computeQueue_ = VK_NULL_HANDLE;
    transferQueue_ = VK_NULL_HANDLE;
}

bool VulkanDevice::initAuto() {
    lastError_.reset();
    if (!createInstance()) return false;
    if (!selectPhysicalDevice(-1, {})) return false;
    return createLogicalDevice();
}

bool VulkanDevice::initByIndex(uint32_t deviceIndex) {
    lastError_.reset();
    if (!createInstance()) return false;
    if (!selectPhysicalDevice(static_cast<int>(deviceIndex), {})) return false;
    return createLogicalDevice();
}

bool VulkanDevice::initByName(std::string_view nameSubstr) {
    lastError_.reset();
    if (!createInstance()) return false;
    if (!selectPhysicalDevice(-1, nameSubstr)) return false;
    return createLogicalDevice();
}

bool VulkanDevice::createInstance() {
    if (!ensureVolkInitialized()) {
        lastError_ = GPUError{
            GPUError::Code::NoVulkanSupport,
            "Vulkan runtime (vulkan-1.dll / libvulkan.so) could not be loaded via volk.",
            {}, {}
        };
        return false;
    }

    uint32_t instanceVersion = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) {
        vkEnumerateInstanceVersion(&instanceVersion);
    }
    const uint32_t targetApi = (instanceVersion >= VK_API_VERSION_1_2)
                                   ? VK_API_VERSION_1_2
                                   : VK_API_VERSION_1_1;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPGPU.VULKAN";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName = "GPGPU.VULKAN.Engine";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.apiVersion = targetApi;

    // Enumerate supported instance extensions
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    if (extCount > 0) {
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availableExts.data());
    }

    std::vector<const char*> enabledExtensions;
    VkInstanceCreateFlags instanceFlags = 0;

    for (const auto& ext : availableExts) {
        if (std::strcmp(ext.extensionName, "VK_KHR_portability_enumeration") == 0) {
            enabledExtensions.push_back("VK_KHR_portability_enumeration");
            instanceFlags |= 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        }
    }

    // Enumerate layers (check for VK_LAYER_KHRONOS_validation in debug mode)
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    if (layerCount > 0) {
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    }

    std::vector<const char*> enabledLayers;
#if !defined(NDEBUG)
    for (const auto& layer : availableLayers) {
        if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
            break;
        }
    }
#endif

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = instanceFlags;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.empty() ? nullptr : enabledExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
    createInfo.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();

    VkResult res = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (res != VK_SUCCESS || instance_ == VK_NULL_HANDLE) {
        lastError_ = GPUError{
            GPUError::Code::NoVulkanSupport,
            "vkCreateInstance failed with VkResult: " + std::to_string(static_cast<int>(res)),
            {}, {}
        };
        return false;
    }

    volkLoadInstanceOnly(instance_);
    return true;
}

DeviceInfo VulkanDevice::buildDeviceInfo(uint32_t index, VkPhysicalDevice physDev, Capabilities* outCaps) {
    VkPhysicalDeviceSubgroupProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;

    if (vkGetPhysicalDeviceProperties2) {
        vkGetPhysicalDeviceProperties2(physDev, &props2);
    } else {
        vkGetPhysicalDeviceProperties(physDev, &props2.properties);
        subgroupProps.subgroupSize = 32;
    }

    const VkPhysicalDeviceProperties& props = props2.properties;

    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    uint64_t totalDeviceLocalBytes = 0;
    uint64_t hostVisibleDeviceLocalBytes = 0;

    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceLocalBytes += memProps.memoryHeaps[i].size;
        }
    }

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memProps.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            uint32_t heapIdx = memProps.memoryTypes[i].heapIndex;
            hostVisibleDeviceLocalBytes = std::max(hostVisibleDeviceLocalBytes, memProps.memoryHeaps[heapIdx].size);
        }
    }

    DeviceInfo info{};
    info.index = index;
    info.vendorId = props.vendorID;
    info.deviceId = props.deviceID;
    info.name = props.deviceName;
    info.vendor = vendorIdToName(props.vendorID);
    info.driverVersion = formatVersion(props.driverVersion);
    info.apiVersion = formatVersion(props.apiVersion);
    info.isDiscrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    info.totalMemoryBytes = totalDeviceLocalBytes;

    const std::string lowerName = toLower(info.name);
    const bool isRPi = (lowerName.find("v3d") != std::string::npos) || (props.vendorID == 0x14E4);
    const bool isUMA = !info.isDiscrete ||
                       (totalDeviceLocalBytes > 0 && hostVisibleDeviceLocalBytes >= (totalDeviceLocalBytes * 3) / 4);
    info.isUnifiedMemory = isUMA;

    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        info.profile = HardwareProfile::SoftwareCPU;
    } else if (isRPi || props.limits.maxComputeWorkGroupInvocations < 256 || subgroupProps.subgroupSize <= 16) {
        info.profile = HardwareProfile::EmbeddedUMA;
    } else if (info.isDiscrete) {
        info.profile = HardwareProfile::DiscreteGPU;
    } else {
        info.profile = HardwareProfile::IntegratedUMA;
    }

    if (outCaps) {
        VkPhysicalDeviceFeatures feats{};
        vkGetPhysicalDeviceFeatures(physDev, &feats);

        for (int d = 0; d < 3; ++d) {
            outCaps->maxWorkGroupCount[d] = props.limits.maxComputeWorkGroupCount[d];
            outCaps->maxWorkGroupSize[d] = props.limits.maxComputeWorkGroupSize[d];
        }
        outCaps->maxWorkGroupInvocations = props.limits.maxComputeWorkGroupInvocations;
        outCaps->subgroupSize = subgroupProps.subgroupSize > 0 ? subgroupProps.subgroupSize : 32;
        outCaps->maxStorageBufferRange = props.limits.maxStorageBufferRange;
        outCaps->maxPushConstantsSize = props.limits.maxPushConstantsSize;
        outCaps->timestampPeriod = props.limits.timestampPeriod;
        outCaps->supportsFloat64 = (feats.shaderFloat64 == VK_TRUE);
        outCaps->supportsInt16 = (feats.shaderInt16 == VK_TRUE);
        outCaps->supportsTimelineSemaphore = (props.apiVersion >= VK_API_VERSION_1_2);
    }

    return info;
}

int64_t VulkanDevice::scoreDevice(const DeviceInfo& info) {
    int64_t score = 0;
    switch (info.profile) {
        case HardwareProfile::DiscreteGPU:   score += 10000; break;
        case HardwareProfile::IntegratedUMA: score += 5000;  break;
        case HardwareProfile::EmbeddedUMA:   score += 3000;  break;
        case HardwareProfile::SoftwareCPU:   score += 100;   break;
    }
    score += static_cast<int64_t>(info.totalMemoryBytes / (1024ULL * 1024ULL * 256ULL)) * 25;
    return score;
}

std::vector<DeviceInfo> VulkanDevice::enumerateDevices(GPUError* outErr) {
    VulkanDevice temp;
    if (!temp.createInstance()) {
        if (outErr && temp.lastError_) *outErr = *temp.lastError_;
        return {};
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(temp.instance_, &count, nullptr);
    if (count == 0) {
        return {};
    }

    std::vector<VkPhysicalDevice> physDevs(count);
    vkEnumeratePhysicalDevices(temp.instance_, &count, physDevs.data());

    std::vector<DeviceInfo> result;
    result.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        result.push_back(buildDeviceInfo(i, physDevs[i], nullptr));
    }
    return result;
}

bool VulkanDevice::selectPhysicalDevice(int targetIndex, std::string_view targetName) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        lastError_ = GPUError{
            GPUError::Code::DeviceNotFound,
            "No Vulkan-capable physical devices found on this system.",
            {}, {}
        };
        return false;
    }

    std::vector<VkPhysicalDevice> physDevs(count);
    vkEnumeratePhysicalDevices(instance_, &count, physDevs.data());

    int selectedIdx = -1;
    if (targetIndex >= 0) {
        if (static_cast<uint32_t>(targetIndex) >= count) {
            lastError_ = GPUError{
                GPUError::Code::DeviceNotFound,
                "Requested GPU index " + std::to_string(targetIndex) +
                    " is out of range (available devices: " + std::to_string(count) + ").",
                {}, {}
            };
            return false;
        }
        selectedIdx = targetIndex;
    } else if (!targetName.empty()) {
        const std::string needle = toLower(targetName);
        for (uint32_t i = 0; i < count; ++i) {
            DeviceInfo info = buildDeviceInfo(i, physDevs[i], nullptr);
            if (toLower(info.name).find(needle) != std::string::npos ||
                toLower(info.vendor).find(needle) != std::string::npos) {
                selectedIdx = static_cast<int>(i);
                break;
            }
        }
        if (selectedIdx < 0) {
            lastError_ = GPUError{
                GPUError::Code::DeviceNotFound,
                "No Vulkan GPU matching name substring '" + std::string(targetName) + "' was found.",
                {}, {}
            };
            return false;
        }
    } else {
        int64_t bestScore = -1;
        for (uint32_t i = 0; i < count; ++i) {
            DeviceInfo info = buildDeviceInfo(i, physDevs[i], nullptr);
            int64_t s = scoreDevice(info);
            if (s > bestScore) {
                bestScore = s;
                selectedIdx = static_cast<int>(i);
            }
        }
    }

    physicalDevice_ = physDevs[static_cast<size_t>(selectedIdx)];
    deviceInfo_ = buildDeviceInfo(static_cast<uint32_t>(selectedIdx), physicalDevice_, &capabilities_);
    return true;
}

bool VulkanDevice::createLogicalDevice() {
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qCount, qProps.data());

    // Prefer dedicated compute queue if available, otherwise any queue with VK_QUEUE_COMPUTE_BIT
    for (uint32_t i = 0; i < qCount; ++i) {
        if ((qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            computeQueueFamily_ = i;
            break;
        }
    }
    if (computeQueueFamily_ == UINT32_MAX) {
        for (uint32_t i = 0; i < qCount; ++i) {
            if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeQueueFamily_ = i;
                break;
            }
        }
    }

    if (computeQueueFamily_ == UINT32_MAX) {
        lastError_ = GPUError{
            GPUError::Code::DeviceNotFound,
            "Selected GPU '" + deviceInfo_.name + "' does not expose a VK_QUEUE_COMPUTE_BIT queue.",
            {}, {}
        };
        return false;
    }

    // Look for dedicated transfer queue family on discrete GPUs
    transferQueueFamily_ = computeQueueFamily_;
    if (deviceInfo_.isDiscrete) {
        for (uint32_t i = 0; i < qCount; ++i) {
            if ((qProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                !(qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                transferQueueFamily_ = i;
                break;
            }
        }
    }

    const float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    VkDeviceQueueCreateInfo computeQInfo{};
    computeQInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    computeQInfo.queueFamilyIndex = computeQueueFamily_;
    computeQInfo.queueCount = 1;
    computeQInfo.pQueuePriorities = &priority;
    queueCreateInfos.push_back(computeQInfo);

    if (transferQueueFamily_ != computeQueueFamily_) {
        VkDeviceQueueCreateInfo transferQInfo{};
        transferQInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transferQInfo.queueFamilyIndex = transferQueueFamily_;
        transferQInfo.queueCount = 1;
        transferQInfo.pQueuePriorities = &priority;
        queueCreateInfos.push_back(transferQInfo);
    }

    // Query supported Vulkan 1.2 and device features
    VkPhysicalDeviceVulkan12Features supportedVk12Features{};
    supportedVk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 supportedFeatures2{};
    supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures2.pNext = &supportedVk12Features;

    if (vkGetPhysicalDeviceFeatures2) {
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &supportedFeatures2);
    } else {
        vkGetPhysicalDeviceFeatures(physicalDevice_, &supportedFeatures2.features);
    }

    VkPhysicalDeviceVulkan12Features enabledVk12Features{};
    enabledVk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    if (supportedVk12Features.timelineSemaphore) {
        enabledVk12Features.timelineSemaphore = VK_TRUE;
        capabilities_.supportsTimelineSemaphore = true;
    }
    if (supportedVk12Features.shaderFloat16) {
        enabledVk12Features.shaderFloat16 = VK_TRUE;
        capabilities_.supportsFloat16 = true;
    }
    if (supportedVk12Features.storageBuffer8BitAccess) {
        enabledVk12Features.storageBuffer8BitAccess = VK_TRUE;
        capabilities_.supportsInt8 = true;
    }

    VkPhysicalDeviceFeatures enabledFeatures{};
    if (supportedFeatures2.features.shaderFloat64) {
        enabledFeatures.shaderFloat64 = VK_TRUE;
        capabilities_.supportsFloat64 = true;
    }
    if (supportedFeatures2.features.shaderInt16) {
        enabledFeatures.shaderInt16 = VK_TRUE;
        capabilities_.supportsInt16 = true;
    }

    VkDeviceCreateInfo devCreateInfo{};
    devCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCreateInfo.pNext = capabilities_.supportsTimelineSemaphore ? &enabledVk12Features : nullptr;
    devCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    devCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    devCreateInfo.pEnabledFeatures = &enabledFeatures;

    VkResult res = vkCreateDevice(physicalDevice_, &devCreateInfo, nullptr, &device_);
    if (res != VK_SUCCESS || device_ == VK_NULL_HANDLE) {
        lastError_ = GPUError{
            GPUError::Code::InternalError,
            "vkCreateDevice failed with VkResult: " + std::to_string(static_cast<int>(res)),
            {}, {}
        };
        return false;
    }

    volkLoadDeviceTable(&deviceTable_, device_);
    deviceTable_.vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);
    if (transferQueueFamily_ != computeQueueFamily_) {
        deviceTable_.vkGetDeviceQueue(device_, transferQueueFamily_, 0, &transferQueue_);
    } else {
        transferQueue_ = computeQueue_;
    }
    return true;
}

} // namespace gpgpu::vk
