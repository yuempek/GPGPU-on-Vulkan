#include <gpgpu/gpgpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

using namespace gpgpu;

TEST(ShaderToyTest, JumpingGlslExecution) {
    std::ifstream file("ShadertoyOnVulkan/resources/jumping.glsl");
    if (!file.is_open()) {
        file.open("../../ShadertoyOnVulkan/resources/jumping.glsl");
    }
    ASSERT_TRUE(file.is_open()) << "Could not open ShadertoyOnVulkan/resources/jumping.glsl";

    std::stringstream ss;
    ss << file.rdbuf();
    std::string shaderCode = ss.str();
    ASSERT_FALSE(shaderCode.empty());

    GPU gpu;
    ASSERT_FALSE(gpu.hasError()) << gpu.hasError().value().message;

    auto outBuf = gpu.createData<vec4>("outputData");
    gpu.kernel(shaderCode);
    ASSERT_FALSE(gpu.hasError()) << gpu.hasError().value().message;

    // Run 320x240 frame
    gpu.run(320, 240);
    ASSERT_FALSE(gpu.hasError()) << gpu.hasError().value().message;

    std::vector<vec4> pixels = outBuf.get();
    ASSERT_EQ(pixels.size(), 320u * 240u);

    // Verify non-black rendering output
    vec4 center = pixels[120 * 320 + 160];
    EXPECT_GT(center.x, 0.0f);
    EXPECT_GT(center.y, 0.0f);
    EXPECT_GT(center.z, 0.0f);
    EXPECT_FLOAT_EQ(center.w, 1.0f);
}

TEST(ShaderToyTest, MouseAndTimeUniformControl) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    auto outBuf = gpu.createData<vec4>("outputData");

    // Shader that directly reflects iTime, iTimeDelta, iFrame, iMouse into fragColor
    gpu.kernel(R"(
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            fragColor = vec4(iTime, iTimeDelta, float(iFrame), iMouse.x);
        }
    )");
    ASSERT_FALSE(gpu.hasError()) << gpu.hasError().value().message;

    // Set custom uniform controls
    gpu.setTime(12.5f, 0.016f);
    gpu.setFrame(100);
    gpu.setMouse(42.0f, 84.0f, 10.0f, 20.0f);

    gpu.run(10, 10);
    ASSERT_FALSE(gpu.hasError()) << gpu.hasError().value().message;

    std::vector<vec4> pixels = outBuf.get();
    ASSERT_EQ(pixels.size(), 100u);

    EXPECT_NEAR(pixels[0].x, 12.5f, 1e-4f);
    EXPECT_NEAR(pixels[0].y, 0.016f, 1e-4f);
    EXPECT_NEAR(pixels[0].z, 100.0f, 1e-4f);
    EXPECT_NEAR(pixels[0].w, 42.0f, 1e-4f);
}
