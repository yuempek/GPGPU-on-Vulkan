#include <gpgpu/gpgpu.hpp>
#include <gpgpu/gpgpu_c_api.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace gpgpu;

// ============================================================================
// Phase 1 Unit Tests: CPU <-> GPU Memory Transfers & BufferPool Reuse
// ============================================================================

TEST(MemoryTransferTest, SingleElementRoundTrip) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> input = {42.125f};
    auto remoteData = gpu.createData("singleVal", input);
    ASSERT_FALSE(gpu.hasError());
    EXPECT_EQ(remoteData.elementCount(), 1u);

    std::vector<float> output = remoteData.get();
    ASSERT_EQ(output.size(), 1u);
    EXPECT_FLOAT_EQ(output[0], input[0]);
}

TEST(MemoryTransferTest, OneThousandElementsRoundTrip) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    constexpr size_t count = 1'000;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<float>(i) * 1.5f - 250.0f;
    }

    auto remoteData = gpu.createData("thousandVals", input);
    ASSERT_FALSE(gpu.hasError());
    EXPECT_EQ(remoteData.elementCount(), count);

    std::vector<float> output = remoteData.get();
    ASSERT_EQ(output.size(), count);
    EXPECT_EQ(std::memcmp(output.data(), input.data(), count * sizeof(float)), 0);
}

TEST(MemoryTransferTest, OneMillionElementsFourMegabytesRoundTrip) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    constexpr size_t count = 1'000'000; // 4 MB of float data
    std::vector<float> input(count);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < count; ++i) {
        input[i] = dist(rng);
    }

    auto remoteData = gpu.createData("millionVals", input);
    ASSERT_FALSE(gpu.hasError());
    EXPECT_EQ(remoteData.elementCount(), count);
    EXPECT_EQ(remoteData.sizeBytes(), count * sizeof(float));

    std::vector<float> output = remoteData.get();
    ASSERT_EQ(output.size(), count);
    EXPECT_EQ(std::memcmp(output.data(), input.data(), count * sizeof(float)), 0);
}

TEST(MemoryTransferTest, MultipleDataTypesRoundTrip) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    // 1. int32_t
    {
        std::vector<int32_t> inInt = {-1000, 0, 42, 1000000, -999999};
        auto d = gpu.createData("intBuf", inInt);
        auto out = d.get();
        ASSERT_EQ(out.size(), inInt.size());
        EXPECT_EQ(std::memcmp(out.data(), inInt.data(), inInt.size() * sizeof(int32_t)), 0);
    }

    // 2. uint32_t
    {
        std::vector<uint32_t> inUInt = {0, 10, 50000, 4000000000U};
        auto d = gpu.createData("uintBuf", inUInt);
        auto out = d.get();
        ASSERT_EQ(out.size(), inUInt.size());
        EXPECT_EQ(std::memcmp(out.data(), inUInt.data(), inUInt.size() * sizeof(uint32_t)), 0);
    }

    // 3. int16_t
    {
        std::vector<int16_t> inShort = {-32768, -1, 0, 1, 32767};
        auto d = gpu.createData("shortBuf", inShort);
        auto out = d.get();
        ASSERT_EQ(out.size(), inShort.size());
        EXPECT_EQ(std::memcmp(out.data(), inShort.data(), inShort.size() * sizeof(int16_t)), 0);
    }

    // 4. uint8_t
    {
        std::vector<uint8_t> inByte = {0, 128, 255, 64, 32};
        auto d = gpu.createData("byteBuf", inByte);
        auto out = d.get();
        ASSERT_EQ(out.size(), inByte.size());
        EXPECT_EQ(std::memcmp(out.data(), inByte.data(), inByte.size() * sizeof(uint8_t)), 0);
    }

    // 5. vec4
    {
        std::vector<vec4> inVec = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {-0.5f, 0.5f, 10.0f, -100.0f}
        };
        auto d = gpu.createData("vecBuf", inVec);
        auto out = d.get();
        ASSERT_EQ(out.size(), inVec.size());
        EXPECT_EQ(std::memcmp(out.data(), inVec.data(), inVec.size() * sizeof(vec4)), 0);
    }
}

TEST(MemoryTransferTest, MultiDimensionalPreservation) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    // 2D Matrix (800 x 600)
    {
        const Dim3 dims2D(800, 600, 1);
        std::vector<float> mat(dims2D.totalElements(), 3.14f);
        auto d2 = gpu.createData("matrix2d", mat, dims2D);
        EXPECT_EQ(d2.dimensions(), dims2D);
        EXPECT_EQ(d2.elementCount(), dims2D.totalElements());

        auto out = d2.get();
        ASSERT_EQ(out.size(), dims2D.totalElements());
        EXPECT_FLOAT_EQ(out[0], 3.14f);
        EXPECT_FLOAT_EQ(out[dims2D.totalElements() - 1], 3.14f);
    }

    // 3D Cube (100 x 100 x 100)
    {
        const Dim3 dims3D(100, 100, 100);
        std::vector<float> cube(dims3D.totalElements(), 2.718f);
        auto d3 = gpu.createData("cube3d", cube, dims3D);
        EXPECT_EQ(d3.dimensions(), dims3D);
        EXPECT_EQ(d3.elementCount(), dims3D.totalElements());

        auto out = d3.get();
        ASSERT_EQ(out.size(), dims3D.totalElements());
        EXPECT_FLOAT_EQ(out[0], 2.718f);
        EXPECT_FLOAT_EQ(out[dims3D.totalElements() - 1], 2.718f);
    }
}

TEST(MemoryTransferTest, StabilityNoAllocationLeakOnRepeatedSetGet) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    constexpr size_t count = 50'000;
    std::vector<float> data(count, 1.0f);

    auto remote = gpu.createData("leakTest", data);
    const uint32_t baselineAllocations = gpu.memoryInfo().activeAllocations;
    EXPECT_GT(baselineAllocations, 0u);

    // 100 repeated set and get calls
    for (int i = 0; i < 100; ++i) {
        data[0] = static_cast<float>(i);
        remote.set(data);
        auto res = remote.get();
        ASSERT_EQ(res.size(), count);
        EXPECT_FLOAT_EQ(res[0], static_cast<float>(i));
    }

    // Active allocations must not grow
    EXPECT_EQ(gpu.memoryInfo().activeAllocations, baselineAllocations);
}

TEST(MemoryTransferTest, BufferPoolReuseAcrossDisposals) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    constexpr size_t count = 120'000;
    std::vector<float> data(count, 7.5f);

    auto d1 = gpu.createData("buf1", data);
    EXPECT_EQ(gpu.memoryInfo().activeAllocations, 1u);
    EXPECT_EQ(gpu.memoryInfo().pooledAllocations, 0u);

    // Release back to BufferPool
    d1.dispose();
    EXPECT_EQ(gpu.memoryInfo().activeAllocations, 0u);
    EXPECT_EQ(gpu.memoryInfo().pooledAllocations, 1u);

    // Re-create buffer with same size -> should be retrieved from pool (0 new driver allocations)
    auto d2 = gpu.createData("buf2", data);
    EXPECT_EQ(gpu.memoryInfo().activeAllocations, 1u);
    EXPECT_EQ(gpu.memoryInfo().pooledAllocations, 0u);

    auto out = d2.get();
    ASSERT_EQ(out.size(), count);
    EXPECT_FLOAT_EQ(out[0], 7.5f);
}

TEST(MemoryTransferTest, GetToUserSpanZeroHeapAllocation) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    constexpr size_t count = 256;
    std::vector<float> in(count);
    for (size_t i = 0; i < count; ++i) in[i] = static_cast<float>(i);

    auto remote = gpu.createData("spanBuf", in);

    // Preallocated output buffer
    std::vector<float> out(count, -1.0f);
    bool success = remote.getTo(std::span<float>(out.data(), out.size()));
    EXPECT_TRUE(success);
    EXPECT_EQ(std::memcmp(out.data(), in.data(), count * sizeof(float)), 0);
}

TEST(MemoryTransferTest, C99FlatApiTransfer) {
    GPGPU_Context ctx = gpgpu_create_auto();
    ASSERT_NE(ctx, nullptr);

    constexpr size_t count = 1000;
    std::vector<float> in(count);
    for (size_t i = 0; i < count; ++i) in[i] = static_cast<float>(i * 2 + 1);

    GPGPU_Data data = gpgpu_create_data(ctx, "cData", in.data(), count, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(gpgpu_data_get_element_count(data), count);

    std::vector<float> out(count, 0.0f);
    int res = gpgpu_data_get(data, out.data(), count);
    EXPECT_EQ(res, 0);
    EXPECT_EQ(std::memcmp(out.data(), in.data(), count * sizeof(float)), 0);

    // Update via gpgpu_data_set
    for (size_t i = 0; i < count; ++i) in[i] = static_cast<float>(i * 3);
    EXPECT_EQ(gpgpu_data_set(data, in.data(), count), 0);

    EXPECT_EQ(gpgpu_data_get(data, out.data(), count), 0);
    EXPECT_EQ(std::memcmp(out.data(), in.data(), count * sizeof(float)), 0);

    gpgpu_data_dispose(data);
    gpgpu_destroy(ctx);
}
