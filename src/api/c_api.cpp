#include <gpgpu/gpgpu.hpp>
#include <gpgpu/gpgpu_c_api.h>

#include <cstring>
#include <memory>
#include <string>

struct GPGPU_Context_T {
    std::unique_ptr<gpgpu::GPU> gpu;
    std::string cachedErrorMsg;
};

struct GPGPU_Data_T {
    std::shared_ptr<gpgpu::detail::DataState> state;
};

namespace {

gpgpu::DataType mapDType(GPGPU_DType dt) noexcept {
    switch (dt) {
        case GPGPU_DTYPE_FLOAT32: return gpgpu::DataType::Float32;
        case GPGPU_DTYPE_FLOAT64: return gpgpu::DataType::Float64;
        case GPGPU_DTYPE_INT32:   return gpgpu::DataType::Int32;
        case GPGPU_DTYPE_UINT32:  return gpgpu::DataType::UInt32;
        case GPGPU_DTYPE_INT16:   return gpgpu::DataType::Int16;
        case GPGPU_DTYPE_UINT16:  return gpgpu::DataType::UInt16;
        case GPGPU_DTYPE_INT8:    return gpgpu::DataType::Int8;
        case GPGPU_DTYPE_UINT8:   return gpgpu::DataType::UInt8;
        case GPGPU_DTYPE_VEC2:    return gpgpu::DataType::Vec2;
        case GPGPU_DTYPE_VEC3:    return gpgpu::DataType::Vec3;
        case GPGPU_DTYPE_VEC4:    return gpgpu::DataType::Vec4;
    }
    return gpgpu::DataType::Float32;
}

gpgpu::Interpolation mapInterp(GPGPU_Interp interp) noexcept {
    switch (interp) {
        case GPGPU_INTERP_NONE:    return gpgpu::Interpolation::None;
        case GPGPU_INTERP_NEAREST: return gpgpu::Interpolation::Nearest;
        case GPGPU_INTERP_LINEAR:  return gpgpu::Interpolation::Linear;
        case GPGPU_INTERP_CUBIC:   return gpgpu::Interpolation::Cubic;
    }
    return gpgpu::Interpolation::None;
}

} // namespace

extern "C" {

GPGPU_API int gpgpu_get_device_count(void) {
    try {
        return static_cast<int>(gpgpu::GPU::devices().size());
    } catch (...) {
        return 0;
    }
}

GPGPU_API int gpgpu_get_device_name(int device_index, char* out_buf, size_t max_len) {
    if (!out_buf || max_len == 0 || device_index < 0) return -1;
    try {
        auto devs = gpgpu::GPU::devices();
        if (static_cast<size_t>(device_index) >= devs.size()) return -1;
        const auto& name = devs[static_cast<size_t>(device_index)].name;
        const size_t copyLen = (name.size() < max_len - 1) ? name.size() : (max_len - 1);
        std::memcpy(out_buf, name.data(), copyLen);
        out_buf[copyLen] = '\0';
        return 0;
    } catch (...) {
        return -1;
    }
}

GPGPU_API GPGPU_Context gpgpu_create_auto(void) {
    try {
        auto ctx = new GPGPU_Context_T();
        ctx->gpu = std::make_unique<gpgpu::GPU>();
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

GPGPU_API GPGPU_Context gpgpu_create_by_index(int device_index) {
    try {
        auto ctx = new GPGPU_Context_T();
        ctx->gpu = std::make_unique<gpgpu::GPU>(device_index);
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

GPGPU_API GPGPU_Context gpgpu_create_by_name(const char* device_name_substr) {
    try {
        auto ctx = new GPGPU_Context_T();
        ctx->gpu = std::make_unique<gpgpu::GPU>(device_name_substr ? device_name_substr : "");
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

GPGPU_API void gpgpu_dispose(GPGPU_Context ctx) {
    if (!ctx || !ctx->gpu) return;
    try {
        ctx->gpu->dispose();
    } catch (...) {}
}

GPGPU_API void gpgpu_destroy(GPGPU_Context ctx) {
    if (!ctx) return;
    gpgpu_dispose(ctx);
    delete ctx;
}

GPGPU_API GPGPU_Data gpgpu_create_data(GPGPU_Context ctx,
                                       const char* var_name,
                                       const void* host_ptr,
                                       size_t elem_count,
                                       GPGPU_DType dtype,
                                       GPGPU_Interp interp) {
    if (!ctx || !ctx->gpu || !var_name) return nullptr;
    try {
        const auto cppType = mapDType(dtype);
        const size_t elemSize = gpgpu::dataTypeSize(cppType);
        auto state = ctx->gpu->registerVariable(var_name, cppType, elemSize, mapInterp(interp));
        if (!state) return nullptr;

        if (host_ptr && elem_count > 0) {
            state->isAutoSized = false;
            state->upload(host_ptr, elem_count * elemSize, gpgpu::Dim3(static_cast<uint32_t>(elem_count), 1, 1));
        }

        auto handle = new GPGPU_Data_T();
        handle->state = std::move(state);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

GPGPU_API int gpgpu_data_set(GPGPU_Data data, const void* host_ptr, size_t elem_count) {
    if (!data || !data->state || !host_ptr || elem_count == 0) return -1;
    try {
        data->state->isAutoSized = false;
        const size_t bytes = elem_count * data->state->elementSize;
        return data->state->upload(host_ptr, bytes, gpgpu::Dim3(static_cast<uint32_t>(elem_count), 1, 1)) ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

GPGPU_API size_t gpgpu_data_get_element_count(GPGPU_Data data) {
    if (!data || !data->state) return 0;
    return data->state->elementCount;
}

GPGPU_API int gpgpu_data_get(GPGPU_Data data, void* out_host_ptr, size_t max_elem_count) {
    if (!data || !data->state || !out_host_ptr || max_elem_count == 0) return -1;
    try {
        const size_t count = (max_elem_count < data->state->elementCount) ? max_elem_count : data->state->elementCount;
        return data->state->download(out_host_ptr, count * data->state->elementSize) ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

GPGPU_API void gpgpu_data_dispose(GPGPU_Data data) {
    if (!data) return;
    if (data->state) {
        data->state->release();
    }
    delete data;
}

GPGPU_API void gpgpu_data_destroy(GPGPU_Data data) {
    gpgpu_data_dispose(data);
}

GPGPU_API int gpgpu_data_get_dims(GPGPU_Data data, uint32_t* out_x, uint32_t* out_y, uint32_t* out_z) {
    if (!data || !data->state) return -1;
    if (out_x) *out_x = data->state->dims.x;
    if (out_y) *out_y = data->state->dims.y;
    if (out_z) *out_z = data->state->dims.z;
    return 0;
}

GPGPU_API int gpgpu_kernel(GPGPU_Context ctx, const char* kernel_code) {
    if (!ctx || !ctx->gpu || !kernel_code) return -1;
    try {
        ctx->gpu->kernel(kernel_code);
        return ctx->gpu->hasError() ? -1 : 0;
    } catch (...) {
        return -1;
    }
}

GPGPU_API int gpgpu_set_kernel(GPGPU_Context ctx, const char* kernel_code) {
    return gpgpu_kernel(ctx, kernel_code);
}

GPGPU_API int gpgpu_run(GPGPU_Context ctx, uint32_t width, uint32_t height, uint32_t depth) {
    if (!ctx || !ctx->gpu) return -1;
    try {
        ctx->gpu->run(width, height, depth);
        return ctx->gpu->hasError() ? -1 : 0;
    } catch (...) {
        return -1;
    }
}

GPGPU_API uint64_t gpgpu_run_async(GPGPU_Context ctx, uint32_t width, uint32_t height, uint32_t depth) {
    if (!ctx || !ctx->gpu) return 0;
    try {
        auto ticket = ctx->gpu->runAsync(width, height, depth);
        return ticket.id();
    } catch (...) {
        return 0;
    }
}

GPGPU_API void gpgpu_wait(GPGPU_Context ctx) {
    if (!ctx || !ctx->gpu) return;
    try {
        ctx->gpu->wait();
    } catch (...) {}
}

GPGPU_API void gpgpu_on_done(GPGPU_Context ctx, GPGPU_DoneCallback callback, void* user_data) {
    if (!ctx || !ctx->gpu || !callback) return;
    try {
        ctx->gpu->onDone([callback, user_data]() {
            callback(user_data);
        });
    } catch (...) {}
}

GPGPU_API const char* gpgpu_has_error(GPGPU_Context ctx) {
    if (!ctx || !ctx->gpu) return "Invalid GPGPU_Context";
    try {
        auto err = ctx->gpu->hasError();
        if (!err) return nullptr;
        ctx->cachedErrorMsg = err->message;
        return ctx->cachedErrorMsg.c_str();
    } catch (...) {
        return "Internal exception in gpgpu_has_error";
    }
}

GPGPU_API void gpgpu_clear_error(GPGPU_Context ctx) {
    if (!ctx || !ctx->gpu) return;
    ctx->gpu->clearError();
}

GPGPU_API double gpgpu_last_elapsed_ms(GPGPU_Context ctx) {
    if (!ctx || !ctx->gpu) return 0.0;
    try {
        return ctx->gpu->lastKernelTimeMs();
    } catch (...) {
        return 0.0;
    }
}

} // extern "C"
