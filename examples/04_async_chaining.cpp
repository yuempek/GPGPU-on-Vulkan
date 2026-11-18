/**
 * @file 04_async_chaining.cpp
 * @brief GPGPU.VULKAN Example 04: Asynchronous Kernel Chaining & onDone Callbacks
 */

#include <gpgpu/gpgpu.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <atomic>
#include <chrono>
#include <thread>

int main() {
    using namespace gpgpu;

    std::cout << "====================================================\n";
    std::cout << " GPGPU.VULKAN - Example 04: Async Kernel Chaining\n";
    std::cout << "====================================================\n";

    GPU gpu;
    std::cout << "Selected GPU: " << gpu.info().name << "\n";

    const size_t N = 1000000;
    std::vector<float> initial(N, 1.0f);

    // Create variables
    auto dataA = gpu.createData("bufA", initial);
    auto dataB = gpu.createData("bufB");
    auto dataC = gpu.createData("bufC");

    std::cout << "Step 1: Dispatch Kernel 1 (bufB = bufA * 2.0)...\n";
    gpu.kernel("kernel(coord){ bufB[coord] = bufA[coord] * 2.0; }");
    auto ticket1 = gpu.runAsync(static_cast<uint32_t>(N));

    std::cout << "Step 2: Chain Kernel 2 (bufC = sqrt(bufB) + 1.0) without CPU readback...\n";
    // Vulkan memory barrier automatically ensures Kernel 1 writes are visible to Kernel 2!
    gpu.kernel("kernel(coord){ bufC[coord] = sqrt(bufB[coord]) + 1.0; }");

    std::atomic<bool> chainFinished{false};
    auto ticket2 = gpu.runAsync(static_cast<uint32_t>(N));
    ticket2.onDone([&chainFinished]() {
        std::cout << "--> Asynchronous Callback Fired: Kernel Chain Complete!\n";
        chainFinished.store(true);
    });

    std::cout << "Waiting for compute completion on CPU thread...\n";
    gpu.wait();

    // Verify callback
    while (!chainFinished.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::vector<float> result = dataC.get();
    std::cout << "Result sample: initial=" << initial[0]
              << " -> step1=" << initial[0] * 2.0f
              << " -> step2=" << result[0] << " (expected: " << std::sqrt(2.0f) + 1.0f << ")\n";

    gpu.dispose();
    std::cout << "Finished successfully!\n";
    return 0;
}
