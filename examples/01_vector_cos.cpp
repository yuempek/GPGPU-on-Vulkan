/**
 * @file 01_vector_cos.cpp
 * @brief GPGPU.VULKAN Example 01: Vector Cosine
 *
 * Demonstrates the 1:1 user pseudo-code flow:
 *   gpu = GPU(0); // GPU selector
 *   remoteData = gpu.createData("varNameIn", localInputData);
 *   remoteOutData = gpu.createData("outputData");
 *   gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }");
 *   gpu.run(800, 600);
 *   localOutputData = remoteOutData.get();
 *   gpu.dispose();
 */

#include <gpgpu/gpgpu.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    using namespace gpgpu;

    std::cout << "====================================================\n";
    std::cout << " GPGPU.VULKAN - Example 01: 2D Matrix Vector Cosine\n";
    std::cout << "====================================================\n";

    // 1. Device selection
    auto devices = GPU::devices();
    std::cout << "Found " << devices.size() << " Vulkan device(s):\n";
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << devices[i].name << "\n";
    }

    GPU gpu(0); // or GPU("RTX 3060"), or GPU() auto-select

    const uint32_t width = 800;
    const uint32_t height = 600;
    const size_t totalElements = width * height;

    // 2. Prepare local input data
    std::vector<float> localInputData(totalElements);
    for (size_t i = 0; i < totalElements; ++i) {
        localInputData[i] = static_cast<float>(i) * 0.001f;
    }

    // 3. Create GPU variables
    auto remoteData = gpu.createData("varNameIn", localInputData);
    auto remoteOutData = gpu.createData("outputData"); // Deferred size, automatically allocated upon run()

    // 4. Define GLSL DSL Kernel (semicolon-free body supported!)
    gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }");

    // 5. Execute on GPU across 2D matrix
    std::cout << "Executing kernel over " << width << " x " << height << " matrix (" << totalElements << " elements)...\n";
    gpu.run(width, height);

    if (auto err = gpu.hasError()) {
        std::cerr << "GPU Error: " << err->message << "\n";
        return 1;
    }

    // 6. Download results back to CPU DRAM
    std::vector<float> localOutputData = remoteOutData.get();

    // 7. Verify first few results
    std::cout << "Execution complete! Last GPU kernel time: " << gpu.lastKernelTimeMs() << " ms\n";
    std::cout << "Verifying results:\n";
    for (size_t i = 0; i < 5; ++i) {
        float cpuExpected = std::cos(localInputData[i]);
        std::cout << "  [" << i << "] cos(" << localInputData[i] << ") = "
                  << localOutputData[i] << " (CPU: " << cpuExpected << ")\n";
    }

    // 8. Explicit cleanup
    gpu.dispose();
    std::cout << "GPU disposed cleanly. Success!\n";
    return 0;
}
