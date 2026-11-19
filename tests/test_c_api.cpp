#include <gpgpu/gpgpu_c_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <thread>

class CApiTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = gpgpu_create_auto();
        ASSERT_NE(ctx_, nullptr);
        ASSERT_EQ(gpgpu_has_error(ctx_), nullptr);
    }

    void TearDown() override {
        if (ctx_) {
            gpgpu_destroy(ctx_);
            ctx_ = nullptr;
        }
    }

    GPGPU_Context ctx_ = nullptr;
};

TEST_F(CApiTest, DeviceEnumeration) {
    int count = gpgpu_get_device_count();
    EXPECT_GE(count, 1);

    char nameBuf[256] = {0};
    int res = gpgpu_get_device_name(0, nameBuf, sizeof(nameBuf));
    EXPECT_EQ(res, 0);
    EXPECT_GT(std::string(nameBuf).length(), 0u);
}

TEST_F(CApiTest, VectorCosEndToEnd) {
    const size_t N = 1000;
    std::vector<float> inputData(N);
    for (size_t i = 0; i < N; ++i) {
        inputData[i] = static_cast<float>(i) * 0.01f;
    }

    GPGPU_Data inHandle = gpgpu_create_data(
        ctx_, "varNameIn", inputData.data(), N, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);
    ASSERT_NE(inHandle, nullptr);

    GPGPU_Data outHandle = gpgpu_create_data(
        ctx_, "outputData", nullptr, 0, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);
    ASSERT_NE(outHandle, nullptr);

    int kRes = gpgpu_set_kernel(ctx_, "kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }");
    EXPECT_EQ(kRes, 0);
    EXPECT_EQ(gpgpu_has_error(ctx_), nullptr);

    int rRes = gpgpu_run(ctx_, static_cast<uint32_t>(N), 1, 1);
    EXPECT_EQ(rRes, 0);
    EXPECT_EQ(gpgpu_has_error(ctx_), nullptr);

    uint32_t dx = 0, dy = 0, dz = 0;
    EXPECT_EQ(gpgpu_data_get_dims(outHandle, &dx, &dy, &dz), 0);
    EXPECT_EQ(dx, N);
    EXPECT_EQ(dy, 1u);
    EXPECT_EQ(dz, 1u);

    std::vector<float> outData(N, 0.0f);
    int gRes = gpgpu_data_get(outHandle, outData.data(), N);
    EXPECT_EQ(gRes, 0);

    for (size_t i = 0; i < N; ++i) {
        float expected = std::cos(inputData[i]);
        EXPECT_NEAR(outData[i], expected, 5e-5f) << "Mismatch at index " << i;
    }

    double elapsed = gpgpu_last_elapsed_ms(ctx_);
    EXPECT_GE(elapsed, 0.0);

    gpgpu_data_destroy(inHandle);
    gpgpu_data_destroy(outHandle);
}

TEST_F(CApiTest, AsyncExecutionAndCallback) {
    const size_t N = 2000;
    std::vector<float> input(N, 1.5707963f);

    GPGPU_Data inHandle = gpgpu_create_data(
        ctx_, "inVal", input.data(), N, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);
    GPGPU_Data outHandle = gpgpu_create_data(
        ctx_, "outVal", nullptr, 0, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);

    gpgpu_set_kernel(ctx_, "kernel(c){ outVal[c] = sin(inVal[c]); }");

    std::atomic<bool> doneFlag{false};
    auto callback = [](void* userData) {
        auto* flag = static_cast<std::atomic<bool>*>(userData);
        flag->store(true);
    };

    uint64_t ticket = gpgpu_run_async(ctx_, static_cast<uint32_t>(N), 1, 1);
    EXPECT_GT(ticket, 0u);

    gpgpu_on_done(ctx_, callback, &doneFlag);
    gpgpu_wait(ctx_);

    // Wait for callback worker thread to invoke callback
    for (int i = 0; i < 200 && !doneFlag.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(doneFlag.load());

    std::vector<float> out(N);
    gpgpu_data_get(outHandle, out.data(), N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out[i], 1.0f, 1e-4f);
    }

    gpgpu_data_dispose(inHandle);
    gpgpu_data_dispose(outHandle);
}

TEST_F(CApiTest, ErrorHandling) {
    // Malformed kernel syntax
    int kRes = gpgpu_kernel(ctx_, "kernel(coord){ this is completely broken syntax }");
    int rRes = gpgpu_run(ctx_, 100, 1, 1);
    EXPECT_NE(rRes, 0);

    const char* err = gpgpu_has_error(ctx_);
    EXPECT_NE(err, nullptr);
    EXPECT_GT(std::string(err).length(), 0u);

    gpgpu_clear_error(ctx_);
    EXPECT_EQ(gpgpu_has_error(ctx_), nullptr);
}

TEST_F(CApiTest, Vec3RgbAndGetDTypeEndToEnd) {
    const uint32_t W = 16;
    const uint32_t H = 8;
    const size_t N = static_cast<size_t>(W) * H;

    GPGPU_Data outVec3 = gpgpu_create_data(
        ctx_, "outputData", nullptr, 0, GPGPU_DTYPE_VEC3, GPGPU_INTERP_NONE);
    ASSERT_NE(outVec3, nullptr);
    EXPECT_EQ(gpgpu_data_get_dtype(outVec3), GPGPU_DTYPE_VEC3);

    int kRes = gpgpu_set_kernel(ctx_, "kernel(coord){ outputData[coord] = vec3(float(coord.x), float(coord.y), 0.75); }");
    EXPECT_EQ(kRes, 0);
    EXPECT_EQ(gpgpu_has_error(ctx_), nullptr);

    int rRes = gpgpu_run(ctx_, W, H, 1);
    EXPECT_EQ(rRes, 0);
    EXPECT_EQ(gpgpu_has_error(ctx_), nullptr);

    EXPECT_EQ(gpgpu_data_get_element_count(outVec3), N);

    std::vector<float> rgb(N * 3, 0.0f);
    int gRes = gpgpu_data_get(outVec3, rgb.data(), N);
    EXPECT_EQ(gRes, 0);

    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            size_t idx = (static_cast<size_t>(y) * W + x) * 3;
            EXPECT_FLOAT_EQ(rgb[idx + 0], static_cast<float>(x));
            EXPECT_FLOAT_EQ(rgb[idx + 1], static_cast<float>(y));
            EXPECT_FLOAT_EQ(rgb[idx + 2], 0.75f);
        }
    }

    gpgpu_data_destroy(outVec3);
}
