#include "vk_backend.hpp"

namespace gpgpu::vk {

VulkanBackend::~VulkanBackend() {
    dispose();
}

bool VulkanBackend::initAuto() {
    if (!device_.initAuto()) {
        setError(*device_.lastError());
        return false;
    }
    return initSubsystems();
}

bool VulkanBackend::initByIndex(uint32_t deviceIndex) {
    if (!device_.initByIndex(deviceIndex)) {
        setError(*device_.lastError());
        return false;
    }
    return initSubsystems();
}

bool VulkanBackend::initByName(std::string_view nameSubstr) {
    if (!device_.initByName(nameSubstr)) {
        setError(*device_.lastError());
        return false;
    }
    return initSubsystems();
}

bool VulkanBackend::initSubsystems() {
    if (!memory_.init(device_)) {
        setError(GPUError{GPUError::Code::InternalError, "Failed to initialize VMA memory manager.", {}, {}});
        return false;
    }
    if (!pipelines_.init(device_)) {
        setError(GPUError{GPUError::Code::InternalError, "Failed to initialize pipeline manager.", {}, {}});
        return false;
    }
    if (!execution_.init(device_, memory_, pipelines_)) {
        setError(GPUError{GPUError::Code::InternalError, "Failed to initialize execution engine.", {}, {}});
        return false;
    }
    return true;
}

std::optional<GPUError> VulkanBackend::lastError() const noexcept {
    std::lock_guard<std::mutex> lock(errorMutex_);
    return lastError_;
}

void VulkanBackend::clearError() noexcept {
    std::lock_guard<std::mutex> lock(errorMutex_);
    lastError_.reset();
}

void VulkanBackend::setError(GPUError error) noexcept {
    std::lock_guard<std::mutex> lock(errorMutex_);
    lastError_ = std::move(error);
}

BufferHandle VulkanBackend::createBuffer(size_t sizeBytes) {
    GPUError err{};
    BufferHandle handle = memory_.createBuffer(sizeBytes, &err);
    if (handle == kInvalidBufferHandle && err) {
        setError(std::move(err));
    }
    return handle;
}

void VulkanBackend::destroyBuffer(BufferHandle handle) {
    memory_.releaseBuffer(handle);
}

bool VulkanBackend::uploadBuffer(BufferHandle dst, const void* srcHostPtr, size_t sizeBytes, size_t dstOffset) {
    GPUError err{};
    if (!memory_.upload(dst, srcHostPtr, sizeBytes, dstOffset, &err)) {
        if (err) setError(std::move(err));
        return false;
    }
    return true;
}

bool VulkanBackend::downloadBuffer(BufferHandle src, void* dstHostPtr, size_t sizeBytes, size_t srcOffset) {
    GPUError err{};
    if (!memory_.download(src, dstHostPtr, sizeBytes, srcOffset, &err)) {
        if (err) setError(std::move(err));
        return false;
    }
    return true;
}

size_t VulkanBackend::bufferSize(BufferHandle handle) const noexcept {
    return memory_.getBufferSize(handle);
}

uint64_t VulkanBackend::dispatch(const DispatchDesc& desc, bool waitImmediately) {
    if (!isValid()) return 0;
    GPUError err{};
    uint64_t ticket = execution_.submitDispatch(desc, waitImmediately, &err);
    if (err) {
        setError(std::move(err));
    }
    return ticket;
}

void VulkanBackend::waitTicket(uint64_t ticket) {
    execution_.waitTicket(ticket);
}

void VulkanBackend::waitIdle() {
    execution_.waitIdle();
}

bool VulkanBackend::isTicketCompleted(uint64_t ticket) const {
    return execution_.isTicketCompleted(ticket);
}

void VulkanBackend::registerCallback(uint64_t ticket, std::function<void()> callback) {
    execution_.registerCallback(ticket, std::move(callback));
}

double VulkanBackend::ticketElapsedMilliseconds(uint64_t ticket) const noexcept {
    return execution_.ticketElapsedMilliseconds(ticket);
}

void VulkanBackend::dispose() {
    execution_.shutdown();
    pipelines_.shutdown();
    memory_.shutdown();
    device_.shutdown();
}

} // namespace gpgpu::vk

namespace gpgpu {

std::vector<DeviceInfo> enumerateVulkanDevices() {
    return vk::VulkanDevice::enumerateDevices(nullptr);
}

std::unique_ptr<IComputeBackend> createVulkanBackendAuto() {
    auto backend = std::make_unique<vk::VulkanBackend>();
    backend->initAuto();
    return backend;
}

std::unique_ptr<IComputeBackend> createVulkanBackendByIndex(uint32_t deviceIndex) {
    auto backend = std::make_unique<vk::VulkanBackend>();
    backend->initByIndex(deviceIndex);
    return backend;
}

std::unique_ptr<IComputeBackend> createVulkanBackendByName(std::string_view nameSubstr) {
    auto backend = std::make_unique<vk::VulkanBackend>();
    backend->initByName(nameSubstr);
    return backend;
}

} // namespace gpgpu
