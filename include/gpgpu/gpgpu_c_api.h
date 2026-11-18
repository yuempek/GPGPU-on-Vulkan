#ifndef GPGPU_VULKAN_C_API_H
#define GPGPU_VULKAN_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(GPGPU_VULKAN_BUILD_SHARED)
#    define GPGPU_API __declspec(dllexport)
#  elif defined(GPGPU_VULKAN_USE_SHARED)
#    define GPGPU_API __declspec(dllimport)
#  else
#    define GPGPU_API
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define GPGPU_API __attribute__((visibility("default")))
#  else
#    define GPGPU_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GPGPU_Context_T* GPGPU_Context;
typedef struct GPGPU_Data_T*    GPGPU_Data;
typedef void (*GPGPU_DoneCallback)(void* user_data);

typedef enum GPGPU_DType {
    GPGPU_DTYPE_FLOAT32 = 0,
    GPGPU_DTYPE_FLOAT64 = 1,
    GPGPU_DTYPE_INT32   = 2,
    GPGPU_DTYPE_UINT32  = 3,
    GPGPU_DTYPE_INT16   = 4,
    GPGPU_DTYPE_UINT16  = 5,
    GPGPU_DTYPE_INT8    = 6,
    GPGPU_DTYPE_UINT8   = 7,
    GPGPU_DTYPE_VEC2    = 8,
    GPGPU_DTYPE_VEC3    = 9,
    GPGPU_DTYPE_VEC4    = 10
} GPGPU_DType;

typedef enum GPGPU_Interp {
    GPGPU_INTERP_NONE    = 0,
    GPGPU_INTERP_NEAREST = 1,
    GPGPU_INTERP_LINEAR  = 2,
    GPGPU_INTERP_CUBIC   = 3
} GPGPU_Interp;

/* Device & Context Lifecycle */
GPGPU_API int           gpgpu_get_device_count(void);
GPGPU_API int           gpgpu_get_device_name(int device_index, char* out_buf, size_t max_len);
GPGPU_API GPGPU_Context gpgpu_create_auto(void);
GPGPU_API GPGPU_Context gpgpu_create_by_index(int device_index);
GPGPU_API GPGPU_Context gpgpu_create_by_name(const char* device_name_substr);
GPGPU_API void          gpgpu_dispose(GPGPU_Context ctx);
GPGPU_API void          gpgpu_destroy(GPGPU_Context ctx);

/* Data Variable Creation & Transfer */
GPGPU_API GPGPU_Data    gpgpu_create_data(GPGPU_Context ctx,
                                          const char* var_name,
                                          const void* host_ptr,
                                          size_t elem_count,
                                          GPGPU_DType dtype,
                                          GPGPU_Interp interp);
GPGPU_API int           gpgpu_data_set(GPGPU_Data data, const void* host_ptr, size_t elem_count);
GPGPU_API size_t        gpgpu_data_get_element_count(GPGPU_Data data);
GPGPU_API int           gpgpu_data_get_dims(GPGPU_Data data, uint32_t* out_x, uint32_t* out_y, uint32_t* out_z);
GPGPU_API int           gpgpu_data_get(GPGPU_Data data, void* out_host_ptr, size_t max_elem_count);
GPGPU_API void          gpgpu_data_dispose(GPGPU_Data data);
GPGPU_API void          gpgpu_data_destroy(GPGPU_Data data);

/* Kernel Compilation & Execution */
GPGPU_API int           gpgpu_kernel(GPGPU_Context ctx, const char* kernel_code);
GPGPU_API int           gpgpu_set_kernel(GPGPU_Context ctx, const char* kernel_code);
GPGPU_API int           gpgpu_run(GPGPU_Context ctx, uint32_t width, uint32_t height, uint32_t depth);
GPGPU_API uint64_t      gpgpu_run_async(GPGPU_Context ctx, uint32_t width, uint32_t height, uint32_t depth);
GPGPU_API void          gpgpu_wait(GPGPU_Context ctx);
GPGPU_API void          gpgpu_on_done(GPGPU_Context ctx, GPGPU_DoneCallback callback, void* user_data);
GPGPU_API double        gpgpu_last_elapsed_ms(GPGPU_Context ctx);

/* Error Reporting (returns NULL if no error, or pointer to error message string) */
GPGPU_API const char*   gpgpu_has_error(GPGPU_Context ctx);
GPGPU_API void          gpgpu_clear_error(GPGPU_Context ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPGPU_VULKAN_C_API_H */
