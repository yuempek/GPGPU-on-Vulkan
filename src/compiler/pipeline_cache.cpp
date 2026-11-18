#include "pipeline_cache.hpp"

#include <xxhash.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace gpgpu::compiler {

namespace {

constexpr uint64_t kCacheVersion = 0x2026111002ULL;

std::atomic<uint64_t> s_cacheHits{0};
std::atomic<uint64_t> s_cacheMisses{0};

} // namespace

uint64_t PipelineCache::computeKernelHash(std::string_view kernelSource,
                                          Dim3 workgroupSize,
                                          int dispatchDimensions,
                                          std::span<const BoundVariableDesc> bindings) noexcept {
    XXH3_state_t state;
    XXH3_64bits_reset(&state);

    XXH3_64bits_update(&state, &kCacheVersion, sizeof(kCacheVersion));
    XXH3_64bits_update(&state, kernelSource.data(), kernelSource.size());
    XXH3_64bits_update(&state, &workgroupSize, sizeof(workgroupSize));
    XXH3_64bits_update(&state, &dispatchDimensions, sizeof(dispatchDimensions));

    for (const auto& b : bindings) {
        XXH3_64bits_update(&state, b.name.data(), b.name.size());
        XXH3_64bits_update(&state, &b.dtype, sizeof(b.dtype));
        XXH3_64bits_update(&state, &b.interpolation, sizeof(b.interpolation));
        XXH3_64bits_update(&state, &b.dims, sizeof(b.dims));
    }

    return XXH3_64bits_digest(&state);
}

void PipelineCache::clearDiskCache() {
    try {
        std::filesystem::path dir = getDiskCacheDirectory();
        if (std::filesystem::exists(dir)) {
            std::filesystem::remove_all(dir);
            std::filesystem::create_directories(dir);
        }
    } catch (...) {}
}

std::string PipelineCache::getDiskCacheDirectory() {
    std::filesystem::path p;
#if defined(_WIN32)
    const char* localApp = std::getenv("LOCALAPPDATA");
    if (localApp) {
        p = std::filesystem::path(localApp) / "gpgpu_vulkan" / "cache";
    } else {
        const char* userProfile = std::getenv("USERPROFILE");
        p = userProfile ? (std::filesystem::path(userProfile) / ".cache" / "gpgpu_vulkan")
                        : (std::filesystem::current_path() / ".cache");
    }
#else
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg) {
        p = std::filesystem::path(xdg) / "gpgpu_vulkan";
    } else {
        const char* home = std::getenv("HOME");
        p = home ? (std::filesystem::path(home) / ".cache" / "gpgpu_vulkan")
                 : (std::filesystem::current_path() / ".cache");
    }
#endif
    return p.string();
}

bool PipelineCache::loadFromDisk(uint64_t hash, std::vector<uint32_t>& outSpirv) {
    try {
        std::filesystem::path dir = getDiskCacheDirectory();
        std::ostringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(16) << hash << ".spv";
        std::filesystem::path file = dir / ss.str();

        if (!std::filesystem::exists(file)) {
            return false;
        }

        std::ifstream is(file, std::ios::binary | std::ios::ate);
        if (!is.is_open()) {
            return false;
        }

        const auto size = is.tellg();
        if (size <= 0 || (size % sizeof(uint32_t)) != 0) {
            return false;
        }

        is.seekg(0, std::ios::beg);
        outSpirv.resize(static_cast<size_t>(size) / sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(outSpirv.data()), size);
        return is.good();
    } catch (...) {
        return false;
    }
}

bool PipelineCache::saveToDisk(uint64_t hash, std::span<const uint32_t> spirv) {
    if (spirv.empty()) return false;
    try {
        std::filesystem::path dir = getDiskCacheDirectory();
        std::filesystem::create_directories(dir);

        std::ostringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(16) << hash << ".spv";
        std::filesystem::path file = dir / ss.str();

        std::ofstream os(file, std::ios::binary | std::ios::trunc);
        if (!os.is_open()) {
            return false;
        }

        os.write(reinterpret_cast<const char*>(spirv.data()), spirv.size_bytes());
        return os.good();
    } catch (...) {
        return false;
    }
}

CacheStats PipelineCache::getStats() noexcept {
    return CacheStats{
        s_cacheHits.load(std::memory_order_relaxed),
        s_cacheMisses.load(std::memory_order_relaxed)
    };
}

void PipelineCache::resetStats() noexcept {
    s_cacheHits.store(0, std::memory_order_relaxed);
    s_cacheMisses.store(0, std::memory_order_relaxed);
}

void PipelineCache::recordHit() noexcept {
    s_cacheHits.fetch_add(1, std::memory_order_relaxed);
}

void PipelineCache::recordMiss() noexcept {
    s_cacheMisses.fetch_add(1, std::memory_order_relaxed);
}

} // namespace gpgpu::compiler
