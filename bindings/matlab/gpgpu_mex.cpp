/**
 * @file gpgpu_mex.cpp
 * @brief MATLAB MEX Bridge for GPGPU.VULKAN
 *
 * Compilation command from MATLAB:
 *   mex -I../../include -L../../build/release -lgpgpu_vulkan gpgpu_mex.cpp
 *
 * Usage:
 *   gpgpu_mex('init', 0);
 *   in_h = gpgpu_mex('createData', 'varNameIn', single(rand(800, 600)));
 *   out_h = gpgpu_mex('createData', 'outputData');
 *   gpgpu_mex('kernel', 'kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }');
 *   gpgpu_mex('run', 800, 600);
 *   out = gpgpu_mex('get', out_h);
 *   gpgpu_mex('dispose');
 */

#include <gpgpu/gpgpu_c_api.h>

#include <cstring>
#include <string>
#include <vector>
#include <cstdint>

// Forward declare MATLAB MEX types and functions so this compiles even when mex.h is not present
#if defined(MATLAB_MEX_FILE) || defined(HAVE_MEX_H)
#  include "mex.h"
#else
// Standalone mock headers for non-MATLAB environments
typedef struct mxArray_tag mxArray;
typedef size_t mwSize;

extern "C" {
    void mexErrMsgIdAndTxt(const char * identifier, const char * err_msg, ...);
    void mexWarnMsgIdAndTxt(const char * identifier, const char * warn_msg, ...);
    int mxGetString(const mxArray *pm, char *str, mwSize strlen);
    char *mxArrayToString(const mxArray *array_ptr);
    bool mxIsSingle(const mxArray *pm);
    bool mxIsDouble(const mxArray *pm);
    bool mxIsChar(const mxArray *pm);
    mwSize mxGetNumberOfElements(const mxArray *pm);
    mwSize mxGetNumberOfDimensions(const mxArray *pm);
    const mwSize *mxGetDimensions(const mxArray *pm);
    float *mxGetSingles(const mxArray *pm);
    double *mxGetDoubles(const mxArray *pm);
    void *mxGetData(const mxArray *pm);
    double mxGetScalar(const mxArray *pm);
    mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims, int classid, int complexity);
    mxArray *mxCreateString(const char *str);
    mxArray *mxCreateDoubleScalar(double value);
    void mxFree(void *ptr);
}
#define mxSINGLE_CLASS 7
#define mxREAL 0
#endif

namespace {
static GPGPU_Context g_ctx = nullptr;
static std::vector<GPGPU_Data> g_dataList;
}

extern "C" void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("GPGPU:InvalidArgs", "Usage: gpgpu_mex('command', ...)");
        return;
    }

    char cmd[64] = {0};
    if (mxGetString(prhs[0], cmd, sizeof(cmd)) != 0) {
        mexErrMsgIdAndTxt("GPGPU:InvalidCommand", "First argument must be a command string.");
        return;
    }

    std::string command(cmd);

    if (command == "init") {
        if (g_ctx) {
            gpgpu_destroy(g_ctx);
            g_ctx = nullptr;
        }
        g_dataList.clear();

        if (nrhs > 1 && mxIsDouble(prhs[1])) {
            int devIdx = static_cast<int>(mxGetScalar(prhs[1]));
            g_ctx = gpgpu_create_by_index(devIdx);
        } else if (nrhs > 1 && mxIsChar(prhs[1])) {
            char* name = mxArrayToString(prhs[1]);
            g_ctx = gpgpu_create_by_name(name);
            mxFree(name);
        } else {
            g_ctx = gpgpu_create_auto();
        }

        if (!g_ctx) {
            mexErrMsgIdAndTxt("GPGPU:InitFailed", "Failed to initialize Vulkan GPU context.");
        }
    }
    else if (command == "createData") {
        if (!g_ctx) {
            mexErrMsgIdAndTxt("GPGPU:NotInitialized", "GPU context is not initialized. Call gpgpu_mex('init') first.");
        }
        if (nrhs < 2 || !mxIsChar(prhs[1])) {
            mexErrMsgIdAndTxt("GPGPU:InvalidArgs", "createData requires variable name as second argument.");
        }

        char* varName = mxArrayToString(prhs[1]);
        GPGPU_Data data = nullptr;

        GPGPU_Interp interp = GPGPU_INTERP_NONE;
        if (nrhs > 3 && mxIsChar(prhs[3])) {
            char* interpStr = mxArrayToString(prhs[3]);
            if (std::string(interpStr) == "linear" || std::string(interpStr) == "lineer") {
                interp = GPGPU_INTERP_LINEAR;
            } else if (std::string(interpStr) == "cubic") {
                interp = GPGPU_INTERP_CUBIC;
            } else if (std::string(interpStr) == "nearest") {
                interp = GPGPU_INTERP_NEAREST;
            }
            mxFree(interpStr);
        }

        if (nrhs > 2 && (mxIsSingle(prhs[2]) || mxIsDouble(prhs[2]))) {
            size_t count = mxGetNumberOfElements(prhs[2]);
            if (mxIsSingle(prhs[2])) {
                float* ptr = mxGetSingles(prhs[2]);
                data = gpgpu_create_data(g_ctx, varName, ptr, count, GPGPU_DTYPE_FLOAT32, interp);
            } else {
                double* dptr = mxGetDoubles(prhs[2]);
                std::vector<float> fbuf(count);
                for (size_t i = 0; i < count; ++i) fbuf[i] = static_cast<float>(dptr[i]);
                data = gpgpu_create_data(g_ctx, varName, fbuf.data(), count, GPGPU_DTYPE_FLOAT32, interp);
            }
        } else {
            data = gpgpu_create_data(g_ctx, varName, nullptr, 0, GPGPU_DTYPE_FLOAT32, interp);
        }
        mxFree(varName);

        if (!data) {
            const char* err = gpgpu_has_error(g_ctx);
            mexErrMsgIdAndTxt("GPGPU:CreateDataFailed", err ? err : "Failed to create Data.");
        }

        g_dataList.push_back(data);
        if (nlhs > 0) {
            // Return index into g_dataList as a 1-based handle for MATLAB
            plhs[0] = mxCreateDoubleScalar(static_cast<double>(g_dataList.size()));
        }
    }
    else if (command == "kernel") {
        if (!g_ctx || nrhs < 2 || !mxIsChar(prhs[1])) {
            mexErrMsgIdAndTxt("GPGPU:InvalidArgs", "kernel requires a shader string argument.");
        }
        char* kcode = mxArrayToString(prhs[1]);
        int res = gpgpu_set_kernel(g_ctx, kcode);
        mxFree(kcode);
        if (res != 0) {
            const char* err = gpgpu_has_error(g_ctx);
            mexErrMsgIdAndTxt("GPGPU:KernelError", err ? err : "Kernel compilation failed.");
        }
    }
    else if (command == "run") {
        if (!g_ctx || nrhs < 2) {
            mexErrMsgIdAndTxt("GPGPU:InvalidArgs", "run requires width [and optional height, depth].");
        }
        uint32_t w = static_cast<uint32_t>(mxGetScalar(prhs[1]));
        uint32_t h = (nrhs > 2) ? static_cast<uint32_t>(mxGetScalar(prhs[2])) : 1;
        uint32_t d = (nrhs > 3) ? static_cast<uint32_t>(mxGetScalar(prhs[3])) : 1;
        int res = gpgpu_run(g_ctx, w, h, d);
        if (res != 0) {
            const char* err = gpgpu_has_error(g_ctx);
            mexErrMsgIdAndTxt("GPGPU:RunError", err ? err : "Kernel execution failed.");
        }
    }
    else if (command == "get") {
        if (!g_ctx || nrhs < 2) {
            mexErrMsgIdAndTxt("GPGPU:InvalidArgs", "get requires data handle argument.");
        }
        size_t handleIdx = static_cast<size_t>(mxGetScalar(prhs[1]));
        if (handleIdx < 1 || handleIdx > g_dataList.size() || !g_dataList[handleIdx - 1]) {
            mexErrMsgIdAndTxt("GPGPU:InvalidHandle", "Invalid Data handle.");
        }
        GPGPU_Data data = g_dataList[handleIdx - 1];
        size_t count = gpgpu_data_get_element_count(data);

        uint32_t dx = 0, dy = 0, dz = 0;
        gpgpu_data_get_dims(data, &dx, &dy, &dz);

        mwSize dims[3] = { (mwSize)dx, (mwSize)(dy > 0 ? dy : 1), (mwSize)(dz > 0 ? dz : 1) };
        mwSize ndim = (dz > 1) ? 3 : ((dy > 1) ? 2 : 1);

        plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
        float* outPtr = mxGetSingles(plhs[0]);
        gpgpu_data_get(data, outPtr, count);
    }
    else if (command == "hasError") {
        const char* err = g_ctx ? gpgpu_has_error(g_ctx) : nullptr;
        if (nlhs > 0) {
            plhs[0] = err ? mxCreateString(err) : mxCreateString("");
        }
    }
    else if (command == "dispose") {
        for (auto d : g_dataList) {
            if (d) gpgpu_data_destroy(d);
        }
        g_dataList.clear();
        if (g_ctx) {
            gpgpu_destroy(g_ctx);
            g_ctx = nullptr;
        }
    }
    else {
        mexErrMsgIdAndTxt("GPGPU:UnknownCommand", "Unknown command specified.");
    }
}
