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
    const size_t deviceElemSize = dataTypeDeviceStride(dtype);
    const size_t requiredBytes = newCount * deviceElemSize;

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

    if (dtype == DataType::Vec3) {
        // In GLSL std430, vec3 arrays have a 16-byte (4-float) stride on the GPU,
        // while host vec3 data is tightly packed as 12 bytes (3 floats).
        std::vector<float> padded(count * 4, 0.0f);
        const float* srcFloats = static_cast<const float*>(srcPtr);
        for (size_t i = 0; i < count; ++i) {
            padded[i * 4 + 0] = srcFloats[i * 3 + 0];
            padded[i * 4 + 1] = srcFloats[i * 3 + 1];
            padded[i * 4 + 2] = srcFloats[i * 3 + 2];
            padded[i * 4 + 3] = 0.0f;
        }
        if (!b->uploadBuffer(buffer, padded.data(), padded.size() * sizeof(float), 0)) {
            return false;
        }
    } else {
        if (!b->uploadBuffer(buffer, srcPtr, bytes, 0)) {
            return false;
        }
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

    if (dtype == DataType::Vec3) {
        // Unpack 16-byte (4-float) std430 GPU vec3 elements into 12-byte (3-float) host elements
        const size_t count = bytes / elementSize;
        std::vector<float> padded(count * 4);
        if (!b->downloadBuffer(buffer, padded.data(), padded.size() * sizeof(float), 0)) {
            return false;
        }
        float* dstFloats = static_cast<float*>(dstPtr);
        for (size_t i = 0; i < count; ++i) {
            dstFloats[i * 3 + 0] = padded[i * 4 + 0];
            dstFloats[i * 3 + 1] = padded[i * 4 + 1];
            dstFloats[i * 3 + 2] = padded[i * 4 + 2];
        }
    } else {
        if (!b->downloadBuffer(buffer, dstPtr, bytes, 0)) {
            return false;
        }
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
