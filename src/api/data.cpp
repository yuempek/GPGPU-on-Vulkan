#include <gpgpu/gpgpu.hpp>

namespace gpgpu::detail {

DataState::~DataState() {
    release();
}

bool DataState::ensureAllocated(Dim3 targetDims) {
    auto b = backend.lock();
    if (!b || !b->isValid()) return false;

    const size_t newCount = targetDims.totalElements();
    if (newCount == 0) return false;
    const size_t requiredBytes = newCount * elementSize;

    if (buffer != kInvalidBufferHandle && elementCount == newCount && !isDeferred) {
        dims = targetDims;
        return true;
    }

    if (buffer != kInvalidBufferHandle) {
        b->destroyBuffer(buffer);
        buffer = kInvalidBufferHandle;
    }

    buffer = b->createBuffer(requiredBytes);
    if (buffer == kInvalidBufferHandle) {
        return false;
    }

    dims = targetDims;
    elementCount = newCount;
    isDeferred = false;
    return true;
}

bool DataState::upload(const void* srcPtr, size_t bytes, Dim3 newDims) {
    auto b = backend.lock();
    if (!b || !b->isValid() || !srcPtr || bytes == 0) return false;

    const size_t count = bytes / elementSize;
    Dim3 targetDims = (newDims.totalElements() == count && newDims.totalElements() > 0)
                          ? newDims
                          : Dim3(static_cast<uint32_t>(count), 1, 1);

    if (!ensureAllocated(targetDims)) {
        return false;
    }

    if (!b->uploadBuffer(buffer, srcPtr, bytes, 0)) {
        return false;
    }

    hostDirty = false;
    deviceDirty = false;
    return true;
}

bool DataState::download(void* dstPtr, size_t bytes) {
    auto b = backend.lock();
    if (!b || !b->isValid() || !dstPtr || bytes == 0 || buffer == kInvalidBufferHandle) {
        return false;
    }

    // Ensure any pending async execution is finished before reading back
    b->waitIdle();

    if (!b->downloadBuffer(buffer, dstPtr, bytes, 0)) {
        return false;
    }

    deviceDirty = false;
    return true;
}

void DataState::release() {
    if (buffer != kInvalidBufferHandle) {
        if (auto b = backend.lock()) {
            b->destroyBuffer(buffer);
        }
        buffer = kInvalidBufferHandle;
    }
    elementCount = 0;
    dims = Dim3{0, 0, 0};
    isDeferred = true;
}

} // namespace gpgpu::detail
