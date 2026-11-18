/**
 * @file 03_shadertoy_runner.cpp
 * @brief GPGPU.VULKAN Example 03: ShaderToy Mode Execution & PPM Exporter
 */

#include <gpgpu/gpgpu.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    using namespace gpgpu;

    std::cout << "====================================================\n";
    std::cout << " GPGPU.VULKAN - Example 03: ShaderToy Runner\n";
    std::cout << "====================================================\n";

    GPU gpu;
    std::cout << "Using GPU: " << gpu.info().name << "\n";

    const uint32_t width = 640;
    const uint32_t height = 480;

    // Output buffer for RGBA pixels
    auto pixels = gpu.createData<vec4>("fragColor");

    // ShaderToy code: plasma with color palette and iResolution
    gpu.kernel(R"(
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord.xy / iResolution.xy;
            vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
            fragColor = vec4(col, 1.0);
        }
    )");

    std::cout << "Rendering ShaderToy frame (" << width << " x " << height << ")...\n";
    gpu.run(width, height);

    if (auto err = gpu.hasError()) {
        std::cerr << "GPU Error: " << err->message << "\n";
        return 1;
    }

    std::vector<vec4> outColors = pixels.get();
    std::cout << "Frame rendered in " << gpu.lastKernelTimeMs() << " ms.\n";

    // Export to PPM image
    const char* filename = "shadertoy_output.ppm";
    std::ofstream ppm(filename, std::ios::binary);
    if (ppm.is_open()) {
        ppm << "P6\n" << width << " " << height << "\n255\n";
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                const auto& c = outColors[y * width + x];
                uint8_t r = static_cast<uint8_t>(std::clamp(c.x * 255.0f, 0.0f, 255.0f));
                uint8_t g = static_cast<uint8_t>(std::clamp(c.y * 255.0f, 0.0f, 255.0f));
                uint8_t b = static_cast<uint8_t>(std::clamp(c.z * 255.0f, 0.0f, 255.0f));
                ppm.put(static_cast<char>(r));
                ppm.put(static_cast<char>(g));
                ppm.put(static_cast<char>(b));
            }
        }
        std::cout << "Saved rendered image to '" << filename << "'.\n";
    }

    gpu.dispose();
    return 0;
}
