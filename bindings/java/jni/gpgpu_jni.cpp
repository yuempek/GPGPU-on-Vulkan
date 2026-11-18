/**
 * @file gpgpu_jni.cpp
 * @brief JNI Implementation connecting GPGPU.java to gpgpu_c_api.h
 */

#include <gpgpu/gpgpu_c_api.h>

#include <cstring>
#include <string>

// Include jni.h if available, or forward-declare standard JNI types
#if defined(__has_include)
#  if __has_include(<jni.h>)
#    include <jni.h>
#    define HAS_JNI_H 1
#  endif
#endif

#ifndef HAS_JNI_H
typedef unsigned char   jboolean;
typedef signed char     jbyte;
typedef unsigned short  jchar;
typedef short           jshort;
typedef int             jint;
typedef long long       jlong;
typedef float           jfloat;
typedef double          jdouble;
typedef void*           jobject;
typedef jobject         jclass;
typedef jobject         jstring;
typedef jobject         jarray;
typedef jarray          jfloatArray;
struct JNIEnv_;
typedef const struct JNINativeInterface_* JNIEnv;
#endif

#define JNI_FN(name) Java_com_gpgpu_GPGPU_##name

extern "C" {

#ifdef HAS_JNI_H

JNIEXPORT jint JNICALL JNI_FN(nativeGetDeviceCount)(JNIEnv* env, jclass) {
    return (jint)gpgpu_get_device_count();
}

JNIEXPORT jstring JNICALL JNI_FN(nativeGetDeviceName)(JNIEnv* env, jclass, jint index) {
    char buf[256] = {0};
    if (gpgpu_get_device_name(index, buf, sizeof(buf)) == 0) {
        return env->NewStringUTF(buf);
    }
    return env->NewStringUTF("");
}

JNIEXPORT jlong JNICALL JNI_FN(nativeCreateAuto)(JNIEnv* env, jclass) {
    return (jlong)(uintptr_t)gpgpu_create_auto();
}

JNIEXPORT jlong JNICALL JNI_FN(nativeCreateByIndex)(JNIEnv* env, jclass, jint index) {
    return (jlong)(uintptr_t)gpgpu_create_by_index(index);
}

JNIEXPORT jlong JNICALL JNI_FN(nativeCreateByName)(JNIEnv* env, jclass, jstring name) {
    const char* cname = env->GetStringUTFChars(name, nullptr);
    GPGPU_Context ctx = gpgpu_create_by_name(cname);
    env->ReleaseStringUTFChars(name, cname);
    return (jlong)(uintptr_t)ctx;
}

JNIEXPORT void JNICALL JNI_FN(nativeDestroy)(JNIEnv* env, jclass, jlong ctx) {
    if (ctx) gpgpu_destroy((GPGPU_Context)(uintptr_t)ctx);
}

JNIEXPORT jlong JNICALL JNI_FN(nativeCreateData)(
    JNIEnv* env, jclass, jlong ctx, jstring name, jfloatArray initialData, jint elemCount, jint dtype, jint interp) {
    const char* cname = env->GetStringUTFChars(name, nullptr);
    jfloat* elems = initialData ? env->GetFloatArrayElements(initialData, nullptr) : nullptr;

    GPGPU_Data data = gpgpu_create_data(
        (GPGPU_Context)(uintptr_t)ctx, cname, elems, (size_t)elemCount, (GPGPU_DType)dtype, (GPGPU_Interp)interp);

    if (elems) env->ReleaseFloatArrayElements(initialData, elems, JNI_ABORT);
    env->ReleaseStringUTFChars(name, cname);

    return (jlong)(uintptr_t)data;
}

JNIEXPORT jint JNICALL JNI_FN(nativeDataSet)(JNIEnv* env, jclass, jlong dataHandle, jfloatArray hostPtr, jint elemCount) {
    if (!dataHandle || !hostPtr) return -1;
    jfloat* elems = env->GetFloatArrayElements(hostPtr, nullptr);
    int res = gpgpu_data_set((GPGPU_Data)(uintptr_t)dataHandle, elems, (size_t)elemCount);
    env->ReleaseFloatArrayElements(hostPtr, elems, JNI_ABORT);
    return (jint)res;
}

JNIEXPORT jlong JNICALL JNI_FN(nativeDataGetElementCount)(JNIEnv* env, jclass, jlong dataHandle) {
    if (!dataHandle) return 0;
    return (jlong)gpgpu_data_get_element_count((GPGPU_Data)(uintptr_t)dataHandle);
}

JNIEXPORT jint JNICALL JNI_FN(nativeDataGet)(JNIEnv* env, jclass, jlong dataHandle, jfloatArray outBuf, jint maxElemCount) {
    if (!dataHandle || !outBuf) return -1;
    jfloat* elems = env->GetFloatArrayElements(outBuf, nullptr);
    int res = gpgpu_data_get((GPGPU_Data)(uintptr_t)dataHandle, elems, (size_t)maxElemCount);
    env->ReleaseFloatArrayElements(outBuf, elems, 0); // commit copy back
    return (jint)res;
}

JNIEXPORT void JNICALL JNI_FN(nativeDataDispose)(JNIEnv* env, jclass, jlong dataHandle) {
    if (dataHandle) gpgpu_data_dispose((GPGPU_Data)(uintptr_t)dataHandle);
}

JNIEXPORT jint JNICALL JNI_FN(nativeSetKernel)(JNIEnv* env, jclass, jlong ctx, jstring code) {
    if (!ctx || !code) return -1;
    const char* ccode = env->GetStringUTFChars(code, nullptr);
    int res = gpgpu_set_kernel((GPGPU_Context)(uintptr_t)ctx, ccode);
    env->ReleaseStringUTFChars(code, ccode);
    return (jint)res;
}

JNIEXPORT jint JNICALL JNI_FN(nativeRun)(JNIEnv* env, jclass, jlong ctx, jint w, jint h, jint d) {
    if (!ctx) return -1;
    return (jint)gpgpu_run((GPGPU_Context)(uintptr_t)ctx, (uint32_t)w, (uint32_t)h, (uint32_t)d);
}

JNIEXPORT jlong JNICALL JNI_FN(nativeRunAsync)(JNIEnv* env, jclass, jlong ctx, jint w, jint h, jint d) {
    if (!ctx) return 0;
    return (jlong)gpgpu_run_async((GPGPU_Context)(uintptr_t)ctx, (uint32_t)w, (uint32_t)h, (uint32_t)d);
}

JNIEXPORT void JNICALL JNI_FN(nativeWait)(JNIEnv* env, jclass, jlong ctx) {
    if (ctx) gpgpu_wait((GPGPU_Context)(uintptr_t)ctx);
}

JNIEXPORT jstring JNICALL JNI_FN(nativeHasError)(JNIEnv* env, jclass, jlong ctx) {
    if (!ctx) return env->NewStringUTF("Context disposed");
    const char* err = gpgpu_has_error((GPGPU_Context)(uintptr_t)ctx);
    return err ? env->NewStringUTF(err) : nullptr;
}

JNIEXPORT jdouble JNICALL JNI_FN(nativeLastElapsedMs)(JNIEnv* env, jclass, jlong ctx) {
    if (!ctx) return 0.0;
    return (jdouble)gpgpu_last_elapsed_ms((GPGPU_Context)(uintptr_t)ctx);
}

#endif // HAS_JNI_H

} // extern "C"
