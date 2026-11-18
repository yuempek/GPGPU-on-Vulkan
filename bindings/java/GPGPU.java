package com.gpgpu;

import java.util.Objects;

/**
 * GPGPU.VULKAN Java Binding
 * High-Performance Developer-Friendly GPU Compute Framework on Vulkan.
 *
 * Example usage:
 * <pre>
 *   try (GPGPU gpu = new GPGPU(0)) {
 *       float[] localInput = new float[]{0.0f, 0.5f, 1.0f, 1.57f};
 *       GPGPU.Data inData = gpu.createData("varNameIn", localInput);
 *       GPGPU.Data outData = gpu.createData("outputData");
 *       gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }");
 *       gpu.run(localInput.length);
 *       float[] result = outData.get();
 *   }
 * </pre>
 */
public class GPGPU implements AutoCloseable {

    static {
        try {
            System.loadLibrary("gpgpu_vulkan");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Note: Ensure libgpgpu_vulkan is in java.library.path or working directory: " + e.getMessage());
        }
    }

    private long nativeContext = 0;

    public GPGPU() {
        this.nativeContext = nativeCreateAuto();
        checkContext();
    }

    public GPGPU(int deviceIndex) {
        this.nativeContext = nativeCreateByIndex(deviceIndex);
        checkContext();
    }

    public GPGPU(String deviceName) {
        this.nativeContext = nativeCreateByName(Objects.requireNonNull(deviceName));
        checkContext();
    }

    private void checkContext() {
        if (nativeContext == 0) {
            throw new IllegalStateException("Failed to initialize GPGPU Vulkan context.");
        }
    }

    public static int getDeviceCount() {
        return nativeGetDeviceCount();
    }

    public static String getDeviceName(int index) {
        return nativeGetDeviceName(index);
    }

    public Data createData(String varName) {
        checkContext();
        long handle = nativeCreateData(nativeContext, varName, null, 0, 0, 0);
        return new Data(handle);
    }

    public Data createData(String varName, float[] initialData) {
        checkContext();
        long handle = nativeCreateData(nativeContext, varName, initialData, initialData != null ? initialData.length : 0, 0, 0);
        return new Data(handle);
    }

    public Data createData(String varName, float[] initialData, String interpolation) {
        checkContext();
        int interpCode = parseInterpolation(interpolation);
        long handle = nativeCreateData(nativeContext, varName, initialData, initialData != null ? initialData.length : 0, 0, interpCode);
        return new Data(handle);
    }

    public void kernel(String glslCode) {
        checkContext();
        int res = nativeSetKernel(nativeContext, glslCode);
        if (res != 0) {
            throw new RuntimeException("Kernel compilation error: " + hasError());
        }
    }

    public void run(int width) {
        run(width, 1, 1);
    }

    public void run(int width, int height) {
        run(width, height, 1);
    }

    public void run(int width, int height, int depth) {
        checkContext();
        int res = nativeRun(nativeContext, width, height, depth);
        if (res != 0) {
            throw new RuntimeException("Kernel execution error: " + hasError());
        }
    }

    public long runAsync(int width, int height, int depth) {
        checkContext();
        return nativeRunAsync(nativeContext, width, height, depth);
    }

    public void waitCompute() {
        if (nativeContext != 0) {
            nativeWait(nativeContext);
        }
    }

    public String hasError() {
        if (nativeContext == 0) return "Context disposed";
        return nativeHasError(nativeContext);
    }

    public double lastKernelTimeMs() {
        if (nativeContext == 0) return 0.0;
        return nativeLastElapsedMs(nativeContext);
    }

    public synchronized void dispose() {
        if (nativeContext != 0) {
            nativeDestroy(nativeContext);
            nativeContext = 0;
        }
    }

    @Override
    public void close() {
        dispose();
    }

    private static int parseInterpolation(String interp) {
        if (interp == null) return 0;
        String s = interp.toLowerCase().trim();
        switch (s) {
            case "linear":
            case "lineer":
            case "bilinear":
                return 2;
            case "cubic":
            case "bicubic":
                return 3;
            case "nearest":
            case "point":
                return 1;
            default:
                return 0;
        }
    }

    /**
     * GPU Memory Buffer Handle.
     */
    public static class Data implements AutoCloseable {
        private long nativeHandle;

        Data(long handle) {
            this.nativeHandle = handle;
        }

        public void set(float[] hostData) {
            if (nativeHandle == 0) throw new IllegalStateException("Data buffer is disposed.");
            nativeDataSet(nativeHandle, hostData, hostData.length);
        }

        public float[] get() {
            if (nativeHandle == 0) throw new IllegalStateException("Data buffer is disposed.");
            int count = (int) nativeDataGetElementCount(nativeHandle);
            if (count <= 0) return new float[0];
            float[] result = new float[count];
            nativeDataGet(nativeHandle, result, count);
            return result;
        }

        public long elementCount() {
            return nativeHandle != 0 ? nativeDataGetElementCount(nativeHandle) : 0;
        }

        public synchronized void dispose() {
            if (nativeHandle != 0) {
                nativeDataDispose(nativeHandle);
                nativeHandle = 0;
            }
        }

        @Override
        public void close() {
            dispose();
        }
    }

    // Native JNI Declarations
    private static native int nativeGetDeviceCount();
    private static native String nativeGetDeviceName(int index);
    private static native long nativeCreateAuto();
    private static native long nativeCreateByIndex(int index);
    private static native long nativeCreateByName(String name);
    private static native void nativeDestroy(long ctx);
    private static native long nativeCreateData(long ctx, String name, float[] hostPtr, int elemCount, int dtype, int interp);
    private static native int nativeDataSet(long dataHandle, float[] hostPtr, int elemCount);
    private static native long nativeDataGetElementCount(long dataHandle);
    private static native int nativeDataGet(long dataHandle, float[] outBuf, int maxElemCount);
    private static native void nativeDataDispose(long dataHandle);
    private static native int nativeSetKernel(long ctx, String code);
    private static native int nativeRun(long ctx, int w, int h, int d);
    private static native long nativeRunAsync(long ctx, int w, int h, int d);
    private static native void nativeWait(long ctx);
    private static native String nativeHasError(long ctx);
    private static native double nativeLastElapsedMs(long ctx);
}
