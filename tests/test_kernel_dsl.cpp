#include <gpgpu/gpgpu.hpp>
#include <gpgpu/types.hpp>
#include "../src/compiler/pipeline_cache.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace gpgpu;

// ============================================================================
// DSL Syntax Tests: Semicolon-less & Multidimensional (Adım 3.5)
// ============================================================================

TEST(KernelDSLTest, SemicolonLessSingleLine1D2D3D) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    // 1D test without semicolon
    {
        std::vector<float> in1D(100, 2.0f);
        auto inData = gpu.createData("varNameIn", in1D);
        auto outData = gpu.createData("outputData");

        gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }");
        ASSERT_FALSE(gpu.hasError());

        gpu.run(100);
        ASSERT_FALSE(gpu.hasError());

        std::vector<float> res = outData.get();
        ASSERT_EQ(res.size(), 100u);
        EXPECT_NEAR(res[0], std::cos(2.0f), 5e-5f);
    }

    // 2D test without semicolon
    {
        std::vector<float> in2D(50 * 50, 1.5f);
        auto inData = gpu.createData("varNameIn", in2D);
        auto outData = gpu.createData("outputData");

        gpu.kernel("kernel(coord){ outputData[coord] = sin(varNameIn[coord]) }");
        ASSERT_FALSE(gpu.hasError());

        gpu.run(50, 50);
        ASSERT_FALSE(gpu.hasError());

        std::vector<float> res = outData.get();
        ASSERT_EQ(res.size(), 2500u);
        EXPECT_NEAR(res[0], std::sin(1.5f), 5e-5f);
    }

    // 3D test without semicolon
    {
        std::vector<float> in3D(10 * 10 * 10, 0.5f);
        auto inData = gpu.createData("varNameIn", in3D);
        auto outData = gpu.createData("outputData");

        gpu.kernel("kernel(coord){ outputData[coord] = varNameIn[coord] * 4.0 }");
        ASSERT_FALSE(gpu.hasError());

        gpu.run(10, 10, 10);
        ASSERT_FALSE(gpu.hasError());

        std::vector<float> res = outData.get();
        ASSERT_EQ(res.size(), 1000u);
        EXPECT_NEAR(res[0], 2.0f, 1e-5f);
    }
}

// ============================================================================
// Dynamic Kernel Recompilation At Runtime (cos -> sin)
// ============================================================================

TEST(KernelDSLTest, DynamicKernelSwitchingCosToSin) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> inData(100);
    for (size_t i = 0; i < 100; ++i) inData[i] = static_cast<float>(i) * 0.1f;

    auto remoteIn = gpu.createData("varNameIn", inData);
    auto remoteOut = gpu.createData("outputData");

    // First: cos
    gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }");
    gpu.run(100);
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> res1 = remoteOut.get();
    EXPECT_NEAR(res1[10], std::cos(1.0f), 5e-5f);

    // Second: switch dynamically to sin
    gpu.kernel("kernel(coord){ outputData[coord] = sin(varNameIn[coord]); }");
    gpu.run(100);
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> res2 = remoteOut.get();
    EXPECT_NEAR(res2[10], std::sin(1.0f), 5e-5f);
}

// ============================================================================
// L1 Pipeline Cache Hit Count: 1000 executions = 999 cache hits (Adım 3.5)
// ============================================================================

TEST(KernelDSLTest, CacheHitRateAcrossOneThousandExecutions) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> input(64, 1.0f);
    auto remoteIn = gpu.createData("varNameIn", input);
    auto remoteOut = gpu.createData("outputData");

    compiler::PipelineCache::resetStats();
    compiler::PipelineCache::clearDiskCache();

    gpu.kernel("kernel(coord){ outputData[coord] = varNameIn[coord] + 1.0f }");

    // Execute 1,000 times
    for (int i = 0; i < 1000; ++i) {
        gpu.run(64);
    }
    ASSERT_FALSE(gpu.hasError());

    auto stats = compiler::PipelineCache::getStats();
    // First execution is a miss (compiles), subsequent 999 are L1 cache hits
    EXPECT_EQ(stats.hits, 999u);
    EXPECT_EQ(stats.misses, 1u);
}

// ============================================================================
// ShaderToy mainImage Compatibility Test (Adım 3.5)
// ============================================================================

TEST(KernelDSLTest, ShaderToyMainImageCompatibility) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    const uint32_t width = 320;
    const uint32_t height = 240;
    const size_t total = static_cast<size_t>(width) * height;

    auto outputData = gpu.createData<vec4>("outputData");

    // Standard ShaderToy pixel shader writing UV gradient
    const std::string shaderToyCode = R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(uv.x, uv.y, 0.5, 1.0);
}
)";

    gpu.kernel(shaderToyCode);
    ASSERT_FALSE(gpu.hasError());

    gpu.run(width, height);
    ASSERT_FALSE(gpu.hasError());

    std::vector<vec4> pixels = outputData.get();
    ASSERT_EQ(pixels.size(), total);

    // Verify bottom-left pixel
    EXPECT_NEAR(pixels[0].x, (0.5f / width), 1e-3f);
    EXPECT_NEAR(pixels[0].y, (0.5f / height), 1e-3f);
    EXPECT_NEAR(pixels[0].z, 0.5f, 1e-3f);
    EXPECT_NEAR(pixels[0].w, 1.0f, 1e-3f);

    // Verify top-right pixel
    const size_t topIdx = (height - 1) * width + (width - 1);
    EXPECT_NEAR(pixels[topIdx].x, (width - 0.5f) / width, 1e-3f);
    EXPECT_NEAR(pixels[topIdx].y, (height - 0.5f) / height, 1e-3f);
    EXPECT_NEAR(pixels[topIdx].z, 0.5f, 1e-3f);
    EXPECT_NEAR(pixels[topIdx].w, 1.0f, 1e-3f);
}
