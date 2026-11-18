#include <gpgpu/gpgpu.hpp>
#include <gpgpu/gpgpu_c_api.h>

#include "../src/compiler/glsl_generator.hpp"
#include "../src/compiler/kernel_parser.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

using namespace gpgpu;

TEST(BackboneTest, EnumerateAndSelectDevice) {
    auto devs = GPU::devices();
    ASSERT_FALSE(devs.empty()) << "At least one Vulkan-capable GPU should be detected.";

    GPU gpu(0);
    EXPECT_FALSE(gpu.hasError()) << (gpu.hasError() ? gpu.hasError()->message : "");
    EXPECT_FALSE(gpu.info().name.empty());
    EXPECT_GT(gpu.capabilities().maxWorkGroupInvocations, 0u);
}

TEST(BackboneTest, InvalidDeviceSelectionRecordsErrorWithoutCrashing) {
    GPU badIndexGpu(9999);
    auto err1 = badIndexGpu.hasError();
    ASSERT_TRUE(err1.has_value());
    EXPECT_EQ(err1->code, GPUError::Code::DeviceNotFound);

    GPU badNameGpu("NonExistentQuantumGPU_9999");
    auto err2 = badNameGpu.hasError();
    ASSERT_TRUE(err2.has_value());
    EXPECT_EQ(err2->code, GPUError::Code::DeviceNotFound);
}

TEST(BackboneTest, MemoryRoundTripAndBufferPoolReuse) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    constexpr size_t kNumElements = 100'000;
    std::vector<float> input(kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
        input[i] = static_cast<float>(i) * 0.25f - 500.0f;
    }

    auto remoteData = gpu.createData("varNameIn", input);
    ASSERT_FALSE(gpu.hasError());
    EXPECT_EQ(remoteData.elementCount(), kNumElements);

    std::vector<float> output = remoteData.get();
    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(std::memcmp(output.data(), input.data(), input.size() * sizeof(float)), 0);

    // Dispose and re-allocate same size to verify BufferPool reuse
    remoteData.dispose();
    EXPECT_EQ(gpu.memoryInfo().activeAllocations, 0u);
    EXPECT_EQ(gpu.memoryInfo().pooledAllocations, 1u);

    auto reusedData = gpu.createData("varNameIn2", input);
    EXPECT_EQ(gpu.memoryInfo().activeAllocations, 1u);
    EXPECT_EQ(gpu.memoryInfo().pooledAllocations, 0u);
}

TEST(BackboneTest, DeferredAllocationAndPseudoCodeFlow) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> localInputData(800 * 600, 1.0f);
    auto remoteData = gpu.createData("varNameIn", localInputData);
    auto remoteOutData = gpu.createData("outputData");

    // Before run(), remoteOutData is in Deferred state
    EXPECT_TRUE(remoteOutData.isDeferred());
    EXPECT_EQ(remoteOutData.elementCount(), 0u);

    gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }");
    EXPECT_FALSE(gpu.hasError());

    gpu.run(800, 600);
    EXPECT_FALSE(gpu.hasError());

    // After run(800, 600), remoteOutData is automatically allocated to 800x600 floats
    EXPECT_FALSE(remoteOutData.isDeferred());
    EXPECT_EQ(remoteOutData.dimensions(), Dim3(800, 600, 1));
    EXPECT_EQ(remoteOutData.elementCount(), 800u * 600u);

    gpu.dispose();
}

TEST(BackboneTest, KernelParserAndGlslGenerator) {
    compiler::ParsedKernel parsed{};
    GPUError err{};
    ASSERT_TRUE(compiler::KernelParser::parse(
        "kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }", parsed, &err));
    EXPECT_EQ(parsed.mode, compiler::KernelMode::DSL);
    EXPECT_EQ(parsed.coordParamName, "coord");
    EXPECT_EQ(parsed.bodyCode, "outputData[coord] = cos(varNameIn[coord]);");

    std::vector<BoundVariableDesc> bindings = {
        {"varNameIn", 1, DataType::Float32, Interpolation::None, {800, 600, 1}, 800 * 600 * 4, true, false},
        {"outputData", 2, DataType::Float32, Interpolation::None, {800, 600, 1}, 800 * 600 * 4, false, true}
    };

    std::string glsl = compiler::GLSLGenerator::generate(
        parsed, bindings, {800, 600, 1}, {16, 16, 1}, Capabilities{});

    EXPECT_NE(glsl.find("#version 450"), std::string::npos);
    EXPECT_NE(glsl.find("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;"), std::string::npos);
    EXPECT_NE(glsl.find("void _gpgpu_user_kernel(ivec2 coord)"), std::string::npos);
}

TEST(BackboneTest, C99FlatApiRoundTrip) {
    ASSERT_GT(gpgpu_get_device_count(), 0);

    GPGPU_Context ctx = gpgpu_create_auto();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(gpgpu_has_error(ctx), nullptr);

    const float hostIn[4] = {1.5f, -2.5f, 3.25f, 42.0f};
    GPGPU_Data dIn = gpgpu_create_data(ctx, "inBuf", hostIn, 4, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);
    ASSERT_NE(dIn, nullptr);
    EXPECT_EQ(gpgpu_data_get_element_count(dIn), 4u);

    float hostOut[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(gpgpu_data_get(dIn, hostOut, 4), 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]);
    }

    gpgpu_data_dispose(dIn);
    gpgpu_destroy(ctx);
}
