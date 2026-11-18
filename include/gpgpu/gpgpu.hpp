#pragma once

#include <gpgpu/backend_interface.hpp>
#include <gpgpu/types.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#  if defined(GPGPU_VULKAN_BUILD_SHARED)
#    define GPGPU_CPP_API __declspec(dllexport)
#  elif defined(GPGPU_VULKAN_USE_SHARED)
#    define GPGPU_CPP_API __declspec(dllimport)
#  else
#    define GPGPU_CPP_API
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define GPGPU_CPP_API __attribute__((visibility("default")))
#  else
#    define GPGPU_CPP_API
#  endif
#endif

namespace gpgpu {

namespace detail {

/**
 * @brief Internal type-erased state for a GPU Data variable.
 * Shared between the user's Data<T> handle and the owning GPU instance.
 */
struct GPGPU_CPP_API DataState {
    std::weak_ptr<IComputeBackend> backend;
    std::string name;
    DataType dtype = DataType::Float32;
    Interpolation interpolation = Interpolation::None;
    Dim3 dims{0, 0, 0};
    size_t elementCount = 0;
    size_t elementSize = sizeof(float);
    BufferHandle buffer = kInvalidBufferHandle;
    bool isDeferred = true;     // True when created via createData("out") without size/data
    bool isAutoSized = true;    // True if size is controlled by gpu.run(x, y, z)
    bool hostDirty = false;
    bool deviceDirty = false;

    ~DataState();

    bool ensureAllocated(Dim3 targetDims);
    bool upload(const void* srcPtr, size_t bytes, Dim3 newDims = {});
    bool download(void* dstPtr, size_t bytes);
    void release();
};

} // namespace detail

/**
 * @brief Asynchronous execution handle returned by gpu.runAsync().
 */
class GPGPU_CPP_API ExecutionTicket {
public:
    ExecutionTicket() = default;
    ExecutionTicket(std::weak_ptr<IComputeBackend> backend, uint64_t ticketId) noexcept
        : backend_(std::move(backend)), ticketId_(ticketId) {}

    /**
     * @brief Block calling CPU thread until this GPU dispatch finishes.
     */
    void wait();

    /**
     * @brief Non-blocking query checking if the GPU has completed this dispatch.
     */
    bool isCompleted() const;

    /**
     * @brief Register a callback invoked asynchronously as soon as this GPU dispatch finishes.
     */
    ExecutionTicket& onDone(std::function<void()> callback);

    /**
     * @brief Hardware GPU execution duration in milliseconds (requires timestamp query support).
     */
    double elapsedMilliseconds() const noexcept;

    constexpr uint64_t id() const noexcept { return ticketId_; }
    constexpr bool valid() const noexcept { return ticketId_ != 0; }

private:
    std::weak_ptr<IComputeBackend> backend_;
    uint64_t ticketId_ = 0;
};

/**
 * @brief Typed handle representing a GPU buffer/variable bound to kernels by name.
 * Lightweight reference-counted value type (can be copied freely).
 */
template <typename T = float>
class Data {
public:
    using value_type = T;

    Data() = default;
    explicit Data(std::shared_ptr<detail::DataState> state) noexcept
        : state_(std::move(state)) {}

    /**
     * @brief Upload data from a contiguous span (CPU -> GPU).
     */
    void set(std::span<const T> hostData, Dim3 dims = {}) {
        if (!state_) return;
        Dim3 targetDims = dims.totalElements() == hostData.size() && dims.totalElements() > 0
                              ? dims
                              : Dim3(static_cast<uint32_t>(hostData.size()), 1, 1);
        state_->isAutoSized = false;
        state_->upload(hostData.data(), hostData.size_bytes(), targetDims);
    }

    /**
     * @brief Upload data from a std::vector (CPU -> GPU).
     */
    void set(const std::vector<T>& hostVec, Dim3 dims = {}) {
        set(std::span<const T>(hostVec.data(), hostVec.size()), dims);
    }

    /**
     * @brief Upload data from a raw pointer and element count (CPU -> GPU).
     */
    void set(const T* ptr, size_t elementCount, Dim3 dims = {}) {
        if (!ptr || elementCount == 0) return;
        set(std::span<const T>(ptr, elementCount), dims);
    }

    /**
     * @brief Download data from GPU back to CPU DRAM as a std::vector<T>.
     * Automatically waits for any pending asynchronous compute dispatch.
     */
    std::vector<T> get() {
        if (!state_ || state_->elementCount == 0) {
            return {};
        }
        std::vector<T> result(state_->elementCount);
        state_->download(result.data(), result.size() * sizeof(T));
        return result;
    }

    /**
     * @brief Zero-allocation download directly into a user-provided span.
     */
    bool getTo(std::span<T> outBuffer) {
        if (!state_ || outBuffer.empty()) {
            return false;
        }
        const size_t bytesToCopy = std::min(outBuffer.size_bytes(), sizeBytes());
        return state_->download(outBuffer.data(), bytesToCopy);
    }

    bool get(std::span<T> outBuffer) {
        return getTo(outBuffer);
    }

    std::string_view name() const noexcept {
        return state_ ? std::string_view(state_->name) : std::string_view{};
    }

    Dim3 dimensions() const noexcept {
        return state_ ? state_->dims : Dim3{0, 0, 0};
    }

    size_t elementCount() const noexcept {
        return state_ ? state_->elementCount : 0;
    }

    size_t sizeBytes() const noexcept {
        return state_ ? (state_->elementCount * state_->elementSize) : 0;
    }

    Interpolation interpolation() const noexcept {
        return state_ ? state_->interpolation : Interpolation::None;
    }

    bool isDeferred() const noexcept {
        return state_ ? state_->isDeferred : true;
    }

    bool valid() const noexcept {
        return state_ != nullptr;
    }

    /**
     * @brief Explicitly release underlying GPU memory back to the buffer pool.
     */
    void dispose() {
        if (state_) {
            state_->release();
        }
    }

    const std::shared_ptr<detail::DataState>& internalState() const noexcept {
        return state_;
    }

private:
    std::shared_ptr<detail::DataState> state_;
};

/**
 * @brief Main Developer Facade for GPGPU.VULKAN.
 *
 * Example flow:
 * @code
 *   GPU gpu(0); // or GPU("RTX 3060")
 *   auto remoteData = gpu.createData("varNameIn", localInputData);
 *   auto remoteOutData = gpu.createData("outputData");
 *   gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }");
 *   gpu.run(800, 600);
 *   auto localOutputData = remoteOutData.get();
 *   gpu.dispose();
 * @endcode
 */
class GPGPU_CPP_API GPU {
public:
    /**
     * @brief Automatically select the highest-scoring Vulkan GPU (Discrete > Integrated > Embedded).
     */
    GPU();

    /**
     * @brief Select GPU by zero-based device index.
     */
    explicit GPU(int deviceIndex);
    explicit GPU(uint32_t deviceIndex);

    /**
     * @brief Select GPU by case-insensitive device name substring (e.g. "RTX 3060", "Intel", "V3D").
     */
    explicit GPU(std::string_view deviceNameSubstr);
    explicit GPU(const char* deviceNameSubstr);

    ~GPU();

    GPU(GPU&&) noexcept;
    GPU& operator=(GPU&&) noexcept;
    GPU(const GPU&) = delete;
    GPU& operator=(const GPU&) = delete;

    // ========================================================================
    // Data Creation Overloads
    // ========================================================================

    /**
     * @brief Create a deferred-size GPU variable (allocated automatically upon gpu.run(W, H, D)).
     */
    template <typename T = float>
    Data<T> createData(std::string_view varName, Interpolation interp = Interpolation::None) {
        auto state = registerVariable(varName, TypeTraits<T>::type, sizeof(T), interp);
        return Data<T>(std::move(state));
    }

    /**
     * @brief Create a deferred-size GPU variable with string interpolation ("linear"|"cubic"|"nearest").
     */
    template <typename T = float>
    Data<T> createData(std::string_view varName, std::string_view interpolationMethod) {
        return createData<T>(varName, parseInterpolation(interpolationMethod));
    }

    /**
     * @brief Create a pre-sized GPU variable with explicit dimensions.
     */
    template <typename T = float>
    Data<T> createData(std::string_view varName, Dim3 dims, Interpolation interp = Interpolation::None) {
        auto state = registerVariable(varName, TypeTraits<T>::type, sizeof(T), interp);
        if (state && dims.totalElements() > 0) {
            state->isAutoSized = false;
            state->ensureAllocated(dims);
        }
        return Data<T>(std::move(state));
    }

    /**
     * @brief Create a GPU variable and immediately upload initial host vector data.
     */
    template <typename T>
    Data<T> createData(std::string_view varName, const std::vector<T>& localData,
                       Interpolation interp = Interpolation::None) {
        auto data = createData<T>(varName, interp);
        data.set(localData);
        return data;
    }

    /**
     * @brief Create a GPU variable, upload initial host vector data, and set interpolation by string.
     */
    template <typename T>
    Data<T> createData(std::string_view varName, const std::vector<T>& localData,
                       std::string_view interpolationMethod) {
        return createData<T>(varName, localData, parseInterpolation(interpolationMethod));
    }

    /**
     * @brief Create a GPU variable, upload initial host vector data with explicit 2D/3D dimensions.
     */
    template <typename T>
    Data<T> createData(std::string_view varName, const std::vector<T>& localData,
                       Dim3 dims, Interpolation interp = Interpolation::None) {
        auto data = createData<T>(varName, interp);
        data.set(localData, dims);
        return data;
    }

    template <typename T>
    Data<T> createData(std::string_view varName, const std::vector<T>& localData,
                       Dim3 dims, std::string_view interpolationMethod) {
        return createData<T>(varName, localData, dims, parseInterpolation(interpolationMethod));
    }

    /**
     * @brief Create a GPU variable from a std::span.
     */
    template <typename T>
    Data<T> createData(std::string_view varName, std::span<const T> localData,
                       Interpolation interp = Interpolation::None) {
        auto data = createData<T>(varName, interp);
        data.set(localData);
        return data;
    }

    template <typename T>
    Data<T> createData(std::string_view varName, std::span<const T> localData,
                       std::string_view interpolationMethod) {
        return createData<T>(varName, localData, parseInterpolation(interpolationMethod));
    }

    template <typename T>
    Data<T> createData(std::string_view varName, std::span<const T> localData,
                       Dim3 dims, Interpolation interp = Interpolation::None) {
        auto data = createData<T>(varName, interp);
        data.set(localData, dims);
        return data;
    }

    template <typename T>
    Data<T> createData(std::string_view varName, std::span<const T> localData,
                       Dim3 dims, std::string_view interpolationMethod) {
        return createData<T>(varName, localData, dims, parseInterpolation(interpolationMethod));
    }

    template <typename T>
    Data<T> createData(std::string_view varName, const T* ptr, size_t count,
                       Interpolation interp = Interpolation::None) {
        return createData<T>(varName, std::span<const T>(ptr, count), interp);
    }

    template <typename T>
    Data<T> createData(std::string_view varName, const T* ptr, size_t count,
                       std::string_view interpolationMethod) {
        return createData<T>(varName, std::span<const T>(ptr, count), parseInterpolation(interpolationMethod));
    }

    // ========================================================================
    // Kernel Definition & Execution
    // ========================================================================

    /**
     * @brief Define or dynamically replace the active GLSL compute kernel.
     * Supports DSL ("kernel(coord){ ... }"), ShaderToy ("mainImage(...)"), or raw "#version 450" GLSL.
     */
    void kernel(std::string_view kernelSource);

    /**
     * @brief Execute the active kernel synchronously over 1D, 2D, or 3D logical dimensions.
     * Equivalent to runAsync(width, height, depth) + wait().
     */
    void run(uint32_t width, uint32_t height = 1, uint32_t depth = 1);

    /**
     * @brief Execute the active kernel asynchronously and return an ExecutionTicket.
     */
    ExecutionTicket runAsync(uint32_t width, uint32_t height = 1, uint32_t depth = 1);

    /**
     * @brief Wait for the most recently submitted asynchronous execution to complete.
     */
    void wait();

    /**
     * @brief Register a completion callback for the most recently submitted asynchronous execution.
     */
    void onDone(std::function<void()> callback);

    /**
     * @brief Begin a command batch to chain multiple kernels with minimal launch overhead.
     */
    void beginBatch();

    /**
     * @brief End and submit the active command batch.
     */
    void endBatch();

    /**
     * @brief GPU execution duration of the last completed kernel in milliseconds.
     */
    double lastKernelTimeMs() const noexcept;

    /**
     * @brief Set or override the Shadertoy iMouse uniform (x, y, clickX, clickY).
     */
    void setMouse(float x, float y, float clickX = 0.0f, float clickY = 0.0f);

    /**
     * @brief Set or override the Shadertoy iTime and iTimeDelta uniforms in seconds.
     * Pass a negative value to revert to internal automatic steady_clock timing.
     */
    void setTime(float timeSeconds, float timeDeltaSeconds = 0.0f);

    /**
     * @brief Reset internal elapsed time and frame counter to zero.
     */
    void resetTime();

    /**
     * @brief Set or override the Shadertoy iFrame uniform counter.
     * Pass a negative value to revert to internal automatic incrementing frame counter.
     */
    void setFrame(int32_t frame);

    // ========================================================================
    // Error Handling & Device Queries
    // ========================================================================

    /**
     * @brief Returns std::nullopt if no error occurred, or GPUError with detailed diagnostics.
     */
    std::optional<GPUError> hasError() const noexcept;

    /**
     * @brief Clear any recorded error state.
     */
    void clearError() noexcept;

    /**
     * @brief List all available Vulkan-capable GPUs on the system.
     */
    static std::vector<DeviceInfo> devices();

    /**
     * @brief Get metadata for the currently selected GPU.
     */
    DeviceInfo info() const;

    /**
     * @brief Get live VRAM/UMA memory allocation and buffer pool metrics.
     */
    MemoryInfo memoryInfo() const;

    /**
     * @brief Get hardware limits and supported shader capabilities for the selected GPU.
     */
    Capabilities capabilities() const;

    /**
     * @brief Release all GPU buffers, pipelines, and Vulkan device handles held by this instance.
     */
    void dispose();

    /**
     * @brief Access internal untyped variable registration (also used by C99 FFI layer).
     */
    std::shared_ptr<detail::DataState> registerVariable(
        std::string_view varName, DataType dtype, size_t elementSize, Interpolation interp);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gpgpu
