#include <gpgpu/gpgpu.hpp>
#include <gpgpu/types.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace gpgpu;

// ============================================================================
// Interpolation Tests: Nearest, Linear, Cubic (Adım 3.3 & 3.5)
// ============================================================================

TEST(InterpolationTest, NearestNeighborSampling) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    // 1D Array [0.0, 10.0, 20.0, 30.0]
    std::vector<float> input1D = {0.0f, 10.0f, 20.0f, 30.0f};
    auto remoteIn = gpu.createData("inData", input1D, "nearest");
    auto remoteOut = gpu.createData("outData");

    // Sample at non-integer coordinates: 0.2 -> rounds to 0 (0.0), 0.8 -> rounds to 1 (10.0)
    gpu.kernel(R"(
kernel(coord) {
    if (coord == 0) outData[coord] = inData[0.2];
    else if (coord == 1) outData[coord] = inData[0.8];
    else if (coord == 2) outData[coord] = inData[1.4];
    else outData[coord] = inData[2.7];
}
)");
    ASSERT_FALSE(gpu.hasError());

    gpu.run(4);
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> result = remoteOut.get();
    ASSERT_EQ(result.size(), 4u);
    EXPECT_FLOAT_EQ(result[0], 0.0f);   // 0.2 -> 0.0
    EXPECT_FLOAT_EQ(result[1], 10.0f);  // 0.8 -> 10.0
    EXPECT_FLOAT_EQ(result[2], 10.0f);  // 1.4 -> 10.0
    EXPECT_FLOAT_EQ(result[3], 30.0f);  // 2.7 -> 30.0
}

TEST(InterpolationTest, Bilinear2DInterpolation) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    // 2x2 Grid:
    // (0,0)=0.0,  (1,0)=10.0
    // (0,1)=20.0, (1,1)=30.0
    std::vector<float> grid2x2 = {
        0.0f,  10.0f,
        20.0f, 30.0f
    };
    auto remoteIn = gpu.createData("inGrid", grid2x2, Dim3(2, 2), "linear");
    auto remoteOut = gpu.createData("outVal");

    // Sample at exact center vec2(0.5, 0.5)
    // Expected bilinear: 0.25 * (0 + 10 + 20 + 30) = 15.0
    gpu.kernel(R"(
kernel(coord) {
    outVal[coord] = inGrid[vec2(0.5, 0.5)];
}
)");
    ASSERT_FALSE(gpu.hasError());

    gpu.run(1);
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> result = remoteOut.get();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 15.0f, 1e-4f);
}

TEST(InterpolationTest, CubicCatmullRom1DInterpolation) {
    GPU gpu;
    ASSERT_FALSE(gpu.hasError());

    // 1D line: [0, 10, 20, 30]
    // For a straight line, Catmull-Rom reproduces linear interpolation exactly:
    // At x = 1.5, result should be 15.0
    std::vector<float> line = {0.0f, 10.0f, 20.0f, 30.0f};
    auto remoteIn = gpu.createData("inLine", line, Dim3(4, 1), "cubic");
    auto remoteOut = gpu.createData("outVal");

    gpu.kernel(R"(
kernel(coord) {
    outVal[coord] = inLine[1.5];
}
)");
    ASSERT_FALSE(gpu.hasError());

    gpu.run(1);
    ASSERT_FALSE(gpu.hasError());

    std::vector<float> result = remoteOut.get();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 15.0f, 1e-3f);
}
