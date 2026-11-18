/**
 * @file 02_matrix_filter_2d.cpp
 * @brief GPGPU.VULKAN Example 02: 2D Bicubic / Bilinear Matrix Filtering
 */

#include <gpgpu/gpgpu.hpp>
#include <iostream>
#include <vector>

int main() {
    using namespace gpgpu;

    std::cout << "====================================================\n";
    std::cout << " GPGPU.VULKAN - Example 02: 2D Matrix Filtering\n";
    std::cout << "====================================================\n";

    GPU gpu;
    std::cout << "Using GPU: " << gpu.info().name << "\n";

    // Create a 4x4 input grid
    const uint32_t inW = 4, inH = 4;
    std::vector<float> inputGrid = {
        0.0f, 10.0f, 20.0f, 30.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
        20.0f, 30.0f, 40.0f, 50.0f,
        30.0f, 40.0f, 50.0f, 60.0f
    };

    // Create input variable with Cubic (Catmull-Rom) interpolation enabled!
    auto srcImg = gpu.createData("srcImg", inputGrid, Dim3(inW, inH, 1), "cubic");
    auto outImg = gpu.createData("outImg");

    // Sample continuous sub-pixel coordinates from input grid:
    gpu.kernel(R"(
        kernel(coord) {
            vec2 uv = vec2(coord) * 0.5; // Upsample 2x
            outImg[coord] = srcImg[uv];
        }
    )");

    // Run over 8x8 upsampled grid
    const uint32_t outW = 8, outH = 8;
    gpu.run(outW, outH);

    if (auto err = gpu.hasError()) {
        std::cerr << "GPU Error: " << err->message << "\n";
        return 1;
    }

    auto result = outImg.get();
    std::cout << "Upsampled " << inW << "x" << inH << " -> " << outW << "x" << outH << " with Catmull-Rom Bicubic spline.\n";
    std::cout << "First row of upsampled output:\n  ";
    for (uint32_t x = 0; x < outW; ++x) {
        std::cout << result[x] << " ";
    }
    std::cout << "\n";

    gpu.dispose();
    return 0;
}
