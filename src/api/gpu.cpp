#include <gpgpu/gpgpu.hpp>

#include "../backend/vulkan/vk_pipeline.hpp"
#include "../compiler/glsl_generator.hpp"
#include "../compiler/kernel_parser.hpp"
#include "../compiler/pipeline_cache.hpp"
#include "../compiler/spirv_compiler.hpp"

#include <chrono>

namespace gpgpu {

// ============================================================================
// ExecutionTicket Implementation
// ============================================================================

void ExecutionTicket::wait() {
    if (ticketId_ == 0) return;
    if (auto b = backend_.lock()) {
        b->waitTicket(ticketId_);
    }
}

bool ExecutionTicket::isCompleted() const {
    if (ticketId_ == 0) return true;
    if (auto b = backend_.lock()) {
        return b->isTicketCompleted(ticketId_);
    }
    return true;
}

ExecutionTicket& ExecutionTicket::onDone(std::function<void()> callback) {
    if (auto b = backend_.lock()) {
        b->registerCallback(ticketId_, std::move(callback));
    } else if (callback) {
        callback();
    }
    return *this;
}

double ExecutionTicket::elapsedMilliseconds() const noexcept {
    if (auto b = backend_.lock()) {
        return b->ticketElapsedMilliseconds(ticketId_);
    }
    return 0.0;
}

// ============================================================================
// GPU Implementation
// ============================================================================

struct GPU::Impl {
    std::shared_ptr<IComputeBackend> backend;
    std::vector<std::shared_ptr<detail::DataState>> variables;
    std::string rawKernelSource;
    compiler::ParsedKernel parsedKernel{};
    bool hasValidKernel = false;
    uint64_t lastTicket = 0;
    int32_t frameCounter = 0;
    int32_t manualFrame = -1;
    float manualTime = -1.0f;
    float manualTimeDelta = 0.0f;
    float lastTime = 0.0f;
    float mouse[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::unordered_map<uint64_t, std::vector<uint32_t>> spirvCache;
    bool isBatching = false;
};

GPU::GPU() : impl_(std::make_unique<Impl>()) {
    impl_->backend = std::shared_ptr<IComputeBackend>(createVulkanBackendAuto());
}

GPU::GPU(int deviceIndex) : impl_(std::make_unique<Impl>()) {
    if (deviceIndex < 0) {
        impl_->backend = std::shared_ptr<IComputeBackend>(createVulkanBackendAuto());
    } else {
        impl_->backend = std::shared_ptr<IComputeBackend>(
            createVulkanBackendByIndex(static_cast<uint32_t>(deviceIndex)));
    }
}

GPU::GPU(uint32_t deviceIndex) : impl_(std::make_unique<Impl>()) {
    impl_->backend = std::shared_ptr<IComputeBackend>(createVulkanBackendByIndex(deviceIndex));
}

GPU::GPU(std::string_view deviceNameSubstr) : impl_(std::make_unique<Impl>()) {
    impl_->backend = std::shared_ptr<IComputeBackend>(createVulkanBackendByName(deviceNameSubstr));
}

GPU::GPU(const char* deviceNameSubstr)
    : GPU(deviceNameSubstr ? std::string_view(deviceNameSubstr) : std::string_view{}) {}

GPU::~GPU() {
    dispose();
}

GPU::GPU(GPU&&) noexcept = default;
GPU& GPU::operator=(GPU&&) noexcept = default;

std::shared_ptr<detail::DataState> GPU::registerVariable(
    std::string_view varName, DataType dtype, size_t elementSize, Interpolation interp) {
    if (!impl_ || !impl_->backend) return nullptr;

    // Replace existing variable if registered with the same name
    for (auto& existing : impl_->variables) {
        if (existing && existing->name == varName) {
            existing->dtype = dtype;
            existing->elementSize = elementSize;
            existing->interpolation = interp;
            return existing;
        }
    }

    auto state = std::make_shared<detail::DataState>();
    state->backend = impl_->backend;
    state->name = std::string(varName);
    state->dtype = dtype;
    state->elementSize = elementSize;
    state->interpolation = interp;
    state->isDeferred = true;
    state->isAutoSized = true;

    impl_->variables.push_back(state);
    return state;
}

void GPU::kernel(std::string_view kernelSource) {
    if (!impl_ || !impl_->backend) return;

    impl_->backend->clearError();
    impl_->rawKernelSource = std::string(kernelSource);
    GPUError parseErr{};
    if (!compiler::KernelParser::parse(kernelSource, impl_->parsedKernel, &parseErr)) {
        impl_->hasValidKernel = false;
        impl_->backend->setError(std::move(parseErr));
        return;
    }
    impl_->hasValidKernel = true;
}

ExecutionTicket GPU::runAsync(uint32_t width, uint32_t height, uint32_t depth) {
    if (!impl_ || !impl_->backend || !impl_->backend->isValid()) {
        return {};
    }

    if (width == 0 || height == 0 || depth == 0) {
        impl_->backend->setError(GPUError{
            GPUError::Code::InvalidDimensions,
            "Dispatch dimensions (width, height, depth) must all be >= 1.",
            {}, {}
        });
        return {};
    }

    if (!impl_->hasValidKernel) {
        if (!impl_->backend->lastError()) {
            impl_->backend->setError(GPUError{
                GPUError::Code::KernelParseError,
                "Cannot call run() before defining a valid kernel via gpu.kernel(...).",
                {}, {}
            });
        }
        return {};
    }

    const Dim3 logicalDims{width, height, depth};

    // Resolve Deferred / Auto-Sized Data variables (e.g. createData("outputData"))
    std::vector<BoundVariableDesc> bindings;
    bindings.reserve(impl_->variables.size());

    for (auto& var : impl_->variables) {
        if (!var) continue;
        if (var->isDeferred || var->isAutoSized) {
            if (!var->ensureAllocated(logicalDims)) {
                return {};
            }
        }

        BoundVariableDesc b{};
        b.name = var->name;
        b.buffer = var->buffer;
        b.dtype = var->dtype;
        b.interpolation = var->interpolation;
        b.dims = var->dims;
        b.sizeBytes = var->elementCount * var->elementSize;
        bindings.push_back(std::move(b));
    }

    const Dim3 wgSize = vk::WorkgroupSelector::selectWorkgroupSize(
        impl_->backend->deviceInfo(), impl_->backend->capabilities(), logicalDims);

    const uint64_t pipelineHash = compiler::PipelineCache::computeKernelHash(
        impl_->rawKernelSource, wgSize, logicalDims.dimensions(), bindings);

    auto it = impl_->spirvCache.find(pipelineHash);
    if (it != impl_->spirvCache.end()) {
        compiler::PipelineCache::recordHit();
    } else {
        std::vector<uint32_t> spirv;
        if (compiler::PipelineCache::loadFromDisk(pipelineHash, spirv)) {
            compiler::PipelineCache::recordHit();
            impl_->spirvCache[pipelineHash] = std::move(spirv);
        } else {
            compiler::PipelineCache::recordMiss();
            std::string glsl = compiler::GLSLGenerator::generate(
                impl_->parsedKernel, bindings, logicalDims, wgSize, impl_->backend->capabilities());

            GPUError compErr{};
            if (!compiler::SpirvCompiler::compileGlslToSpirv(glsl, spirv, true, &compErr)) {
                impl_->backend->setError(std::move(compErr));
                return {};
            }
            compiler::PipelineCache::saveToDisk(pipelineHash, spirv);
            impl_->spirvCache[pipelineHash] = std::move(spirv);
        }
    }

    const auto& spirv = impl_->spirvCache[pipelineHash];

    DispatchDesc desc{};
    desc.pipelineHash = pipelineHash;
    desc.spirvCode = spirv;
    desc.logicalDims = logicalDims;
    desc.workgroupSize = wgSize;
    desc.bindings = bindings;
    desc.pushConstants.dims[0] = width;
    desc.pushConstants.dims[1] = height;
    desc.pushConstants.dims[2] = depth;
    desc.pushConstants.dims[3] = static_cast<uint32_t>(logicalDims.totalElements());
    desc.pushConstants.iResolution[0] = static_cast<float>(width);
    desc.pushConstants.iResolution[1] = static_cast<float>(height);
    desc.pushConstants.iResolution[2] = 1.0f;
    desc.pushConstants.iResolution[3] = 0.0f;
    desc.pushConstants.iFrame = (impl_->manualFrame >= 0) ? impl_->manualFrame : impl_->frameCounter++;

    const auto now = std::chrono::steady_clock::now();
    if (impl_->manualTime >= 0.0f) {
        desc.pushConstants.iTime = impl_->manualTime;
        desc.pushConstants.iTimeDelta = impl_->manualTimeDelta;
    } else {
        float t = std::chrono::duration<float>(now - impl_->startTime).count();
        desc.pushConstants.iTime = t;
        desc.pushConstants.iTimeDelta = t - impl_->lastTime;
        impl_->lastTime = t;
    }
    desc.pushConstants.iMouse[0] = impl_->mouse[0];
    desc.pushConstants.iMouse[1] = impl_->mouse[1];
    desc.pushConstants.iMouse[2] = impl_->mouse[2];
    desc.pushConstants.iMouse[3] = impl_->mouse[3];

    impl_->lastTicket = impl_->backend->dispatch(desc, false);
    return ExecutionTicket(impl_->backend, impl_->lastTicket);
}

void GPU::run(uint32_t width, uint32_t height, uint32_t depth) {
    auto ticket = runAsync(width, height, depth);
    ticket.wait();
}

void GPU::wait() {
    if (!impl_ || !impl_->backend) return;
    if (impl_->lastTicket != 0) {
        impl_->backend->waitTicket(impl_->lastTicket);
    } else {
        impl_->backend->waitIdle();
    }
}

void GPU::onDone(std::function<void()> callback) {
    if (!impl_ || !impl_->backend) {
        if (callback) callback();
        return;
    }
    impl_->backend->registerCallback(impl_->lastTicket, std::move(callback));
}

void GPU::beginBatch() {
    if (impl_) {
        impl_->isBatching = true;
    }
}

void GPU::endBatch() {
    if (!impl_) return;
    impl_->isBatching = false;
    if (impl_->lastTicket != 0) {
        impl_->backend->waitTicket(impl_->lastTicket);
    }
}

double GPU::lastKernelTimeMs() const noexcept {
    if (!impl_ || !impl_->backend || impl_->lastTicket == 0) return 0.0;
    return impl_->backend->ticketElapsedMilliseconds(impl_->lastTicket);
}

std::optional<GPUError> GPU::hasError() const noexcept {
    if (!impl_ || !impl_->backend) {
        return GPUError{GPUError::Code::InternalError, "GPU instance has been disposed.", {}, {}};
    }
    return impl_->backend->lastError();
}

void GPU::clearError() noexcept {
    if (impl_ && impl_->backend) {
        impl_->backend->clearError();
    }
}

std::vector<DeviceInfo> GPU::devices() {
    return enumerateVulkanDevices();
}

DeviceInfo GPU::info() const {
    if (!impl_ || !impl_->backend) return {};
    return impl_->backend->deviceInfo();
}

MemoryInfo GPU::memoryInfo() const {
    if (!impl_ || !impl_->backend) return {};
    return impl_->backend->memoryInfo();
}

Capabilities GPU::capabilities() const {
    if (!impl_ || !impl_->backend) return {};
    return impl_->backend->capabilities();
}

void GPU::setMouse(float x, float y, float clickX, float clickY) {
    if (!impl_) return;
    impl_->mouse[0] = x;
    impl_->mouse[1] = y;
    impl_->mouse[2] = clickX;
    impl_->mouse[3] = clickY;
}

void GPU::setTime(float timeSeconds, float timeDeltaSeconds) {
    if (!impl_) return;
    impl_->manualTime = timeSeconds;
    impl_->manualTimeDelta = timeDeltaSeconds;
}

void GPU::resetTime() {
    if (!impl_) return;
    impl_->startTime = std::chrono::steady_clock::now();
    impl_->lastTime = 0.0f;
    impl_->frameCounter = 0;
    impl_->manualTime = -1.0f;
    impl_->manualTimeDelta = 0.0f;
    impl_->manualFrame = -1;
}

void GPU::setFrame(int32_t frame) {
    if (!impl_) return;
    impl_->manualFrame = frame;
}

void GPU::dispose() {
    if (!impl_) return;
    for (auto& var : impl_->variables) {
        if (var) var->release();
    }
    impl_->variables.clear();
    if (impl_->backend) {
        impl_->backend->dispose();
    }
}

} // namespace gpgpu
