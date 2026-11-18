/**
 * @file bench_compute_throughput.cpp
 * @brief Google Benchmark performance suite for GPGPU.VULKAN
 */

#include <gpgpu/gpgpu.hpp>
#include <benchmark/benchmark.h>

#include <vector>
#include <cmath>

static void BM_HotLoopKernelDispatch(benchmark::State& state) {
    using namespace gpgpu;
    GPU gpu;
    auto inData = gpu.createData("inVal", std::vector<float>(1024, 1.0f));
    auto outData = gpu.createData("outVal");
    gpu.kernel("kernel(coord){ outVal[coord] = inVal[coord] + 1.0; }");

    // Warm-up to populate L1 RAM cache
    gpu.run(1024);

    for (auto _ : state) {
        gpu.run(1024);
    }

    state.SetItemsProcessed(state.iterations() * 1024);
}
BENCHMARK(BM_HotLoopKernelDispatch);

static void BM_VectorTrig_1M(benchmark::State& state) {
    using namespace gpgpu;
    GPU gpu;
    const size_t N = 1000000;
    std::vector<float> src(N);
    for (size_t i = 0; i < N; ++i) src[i] = static_cast<float>(i) * 0.001f;

    auto inData = gpu.createData("inVal", src);
    auto outData = gpu.createData("outVal");
    gpu.kernel("kernel(coord){ outVal[coord] = cos(inVal[coord]) + sin(inVal[coord]); }");

    // Warmup
    gpu.run(static_cast<uint32_t>(N));

    for (auto _ : state) {
        gpu.run(static_cast<uint32_t>(N));
    }

    state.SetBytesProcessed(state.iterations() * N * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_VectorTrig_1M);

static void BM_BicubicResampling_1080p(benchmark::State& state) {
    using namespace gpgpu;
    GPU gpu;
    const uint32_t srcW = 960, srcH = 540;
    const uint32_t dstW = 1920, dstH = 1080;

    std::vector<float> inputImg(srcW * srcH, 128.0f);
    auto inData = gpu.createData("srcImg", inputImg, Dim3(srcW, srcH, 1), "cubic");
    auto outData = gpu.createData("outImg");

    gpu.kernel(R"(
        kernel(coord) {
            vec2 uv = vec2(coord) * 0.5;
            outImg[coord] = srcImg[uv];
        }
    )");

    gpu.run(dstW, dstH);

    for (auto _ : state) {
        gpu.run(dstW, dstH);
    }

    state.SetItemsProcessed(state.iterations() * dstW * dstH);
}
BENCHMARK(BM_BicubicResampling_1080p);

BENCHMARK_MAIN();
