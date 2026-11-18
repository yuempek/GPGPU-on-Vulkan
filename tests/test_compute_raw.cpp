#include <gpgpu/gpgpu.hpp>
#include <gpgpu/types.hpp>
#include "../src/compiler/spirv_compiler.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

using namespace gpgpu;

// ============================================================================
// Direct SPIR-V Compiler & Validator Tests (Adım 2.1 & 2.2)
// ============================================================================

TEST(ComputeExecutionTest, InRamGlslToSpirvCompilationAndValidation) {
    const std::string validGlsl = R"(#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uvec4 dims;
} pc;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float inData[];
};

layout(set = 0, binding = 1, std430) writeonly buffer OutputBuffer {
    float outData[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.dims.x) return;
    outData[idx] = cos(inData[idx]);
}
)";

    std::vector<uint32_t> spirv;
    GPUError err{};
    bool success = compiler::SpirvCompiler::compileGlslToSpirv(validGlsl, spirv, true, &err);

    ASSERT_TRUE(success);
    EXPECT_FALSE(err);
    EXPECT_FALSE(spirv.empty());
    // SPIR-V magic number is 0x07230203
    ASSERT_GE(spirv.size(), 5u);
    EXPECT_EQ(spirv[0], 0x07230203u);
}

TEST(ComputeExecutionTest, InvalidGlslProducesDetailedCompilerDiagnostic) {
    const std::string brokenGlsl = R"(#version 450
layout(local_size_x = 64) in;
void main() {
    this_is_an_undefined_variable = 12345;
}
)";

    std::vector<uint32_t> spirv;
    GPUError err{};
    bool success = compiler::SpirvCompiler::compileGlslToSpirv(brokenGlsl, spirv, true, &err);

    EXPECT_FALSE(success);
    EXPECT_TRUE(err);
    EXPECT_EQ(err.code, GPUError::Code::ShaderCompileError);
    EXPECT_FALSE(err.shaderLog.empty());
    EXPECT_NE(err.shaderLog.find("undefined_variable"), std::string::npos);
}

// ============================================================================
// 1D Compute: 1,000,000 Floats cos(in[i]) vs CPU std::cos (Adım 2.6)
// ============================================================================

TEST(ComputeExecutionTest, Compute1DOneMillionElementsCosAccuracy) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    const size_t N = 1'000'000;
    std::vector<float> hostInput(N);
    for (size_t i = 0; i < N; ++i) {
        hostInput[i] = static_cast<float>(i) * 0.0001f;
    }

    auto inData = gpu.createData("inData", hostInput);
    auto outData = gpu.createData("outData");

    gpu.kernel("kernel(coord){ outData[coord] = cos(inData[coord]) }");
    ASSERT_FALSE(gpu.hasError());

    gpu.run(static_cast<uint32_t>(N));
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> hostOutput(N);
    outData.get(hostOutput);

    // Verify bitwise / numerical precision within float tolerance (< 5e-5f)
    for (size_t i = 0; i < N; ++i) {
        const float expected = std::cos(hostInput[i]);
        const float diff = std::abs(hostOutput[i] - expected);
        ASSERT_LT(diff, 5e-5f) << "Mismatch at index " << i << ": expected " << expected << ", got " << hostOutput[i];
    }
}

// ============================================================================
// 2D Compute: 800 x 600 Matrix Non-Multiple Workgroup Dimension (Adım 2.6)
// ============================================================================

TEST(ComputeExecutionTest, Compute2DMatrixBoundaryAndNonMultipleWorkgroup) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    const uint32_t width = 800;
    const uint32_t height = 600; // 600 / 16 = 37.5 -> tests boundary clipping
    const size_t total = static_cast<size_t>(width) * height;

    std::vector<float> zeros(total, 0.0f);
    auto outData = gpu.createData("outData", zeros);

    gpu.kernel("kernel(coord){ outData[coord] = float(coord.x) + float(coord.y) * 10.0f; }");
    ASSERT_FALSE(gpu.hasError());

    gpu.run(width, height);
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> hostOutput(total);
    outData.get(hostOutput);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const float expected = static_cast<float>(x) + static_cast<float>(y) * 10.0f;
            const float actual = hostOutput[y * width + x];
            ASSERT_EQ(actual, expected) << "Mismatch at (" << x << ", " << y << ")";
        }
    }
}

// ============================================================================
// 3D Compute: 100 x 100 x 100 Cube (1,000,000 Elements) (Adım 2.6)
// ============================================================================

TEST(ComputeExecutionTest, Compute3DCubeOneMillionElements) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    const uint32_t sizeX = 100;
    const uint32_t sizeY = 100;
    const uint32_t sizeZ = 100;
    const size_t total = static_cast<size_t>(sizeX) * sizeY * sizeZ;

    auto outData = gpu.createData("outData");

    gpu.kernel("kernel(coord){ outData[coord] = float(coord.x) + float(coord.y) * 100.0f + float(coord.z) * 10000.0f; }");
    ASSERT_FALSE(gpu.hasError());

    gpu.run(sizeX, sizeY, sizeZ);
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> hostOutput(total);
    outData.get(hostOutput);

    // Verify corners, center and boundaries
    const uint32_t testCoords[][3] = {
        {0, 0, 0},
        {sizeX - 1, sizeY - 1, sizeZ - 1},
        {50, 50, 50},
        {12, 34, 56},
        {99, 0, 99}
    };

    for (const auto& c : testCoords) {
        const uint32_t x = c[0], y = c[1], z = c[2];
        const size_t flatIdx = (static_cast<size_t>(z) * sizeY + y) * sizeX + x;
        const float expected = static_cast<float>(x) + static_cast<float>(y) * 100.0f + static_cast<float>(z) * 10000.0f;
        EXPECT_EQ(hostOutput[flatIdx], expected) << "Mismatch at (" << x << ", " << y << ", " << z << ")";
    }
}

// ============================================================================
// Asynchronous Execution & onDone Callback (Adım 2.5)
// ============================================================================

TEST(ComputeExecutionTest, AsyncExecutionAndTimelineSemaphoreCallback) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    const size_t N = 100'000;
    std::vector<float> input(N, 2.0f);
    auto inData = gpu.createData("inData", input);
    auto outData = gpu.createData("outData");

    gpu.kernel("kernel(coord){ outData[coord] = inData[coord] * inData[coord]; }");

    std::atomic<bool> callbackFired{false};

    auto ticket = gpu.runAsync(static_cast<uint32_t>(N));
    ticket.onDone([&callbackFired]() {
        callbackFired.store(true);
    });

    ticket.wait();
    EXPECT_TRUE(ticket.isCompleted());

    // Allow worker thread to fire callback
    for (int i = 0; i < 200 && !callbackFired.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(callbackFired.load());

    std::vector<float> hostOutput(N);
    outData.get(hostOutput);
    EXPECT_EQ(hostOutput[0], 4.0f);
    EXPECT_EQ(hostOutput[N - 1], 4.0f);
}

// ============================================================================
// Dynamic Kernel Recompilation At Runtime (Faz 2/3 Feature)
// ============================================================================

TEST(ComputeExecutionTest, DynamicKernelRecompilationAtRuntime) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    const size_t N = 1000;
    std::vector<float> inVals(N, 1.0f);
    auto inData = gpu.createData("inData", inVals);
    auto outData = gpu.createData("outData");

    // First kernel: cos(x)
    gpu.kernel("kernel(coord){ outData[coord] = cos(inData[coord]); }");
    gpu.run(static_cast<uint32_t>(N));
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> out1(N);
    outData.get(out1);
    EXPECT_NEAR(out1[0], std::cos(1.0f), 1e-5f);

    // Change kernel dynamically to sin(x)
    gpu.kernel("kernel(coord){ outData[coord] = sin(inData[coord]); }");
    gpu.run(static_cast<uint32_t>(N));
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> out2(N);
    outData.get(out2);
    EXPECT_NEAR(out2[0], std::sin(1.0f), 1e-5f);
}
