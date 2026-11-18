#pragma once

#include <gpgpu/backend_interface.hpp>
#include <gpgpu/types.hpp>

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gpgpu::compiler {

struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
};

class PipelineCache {
public:
    static uint64_t computeKernelHash(std::string_view kernelSource,
                                      Dim3 workgroupSize,
                                      int dispatchDimensions,
                                      std::span<const BoundVariableDesc> bindings) noexcept;

    static bool loadFromDisk(uint64_t hash, std::vector<uint32_t>& outSpirv);
    static bool saveToDisk(uint64_t hash, std::span<const uint32_t> spirv);
    static void clearDiskCache();

    static CacheStats getStats() noexcept;
    static void resetStats() noexcept;
    static void recordHit() noexcept;
    static void recordMiss() noexcept;

    static std::string getDiskCacheDirectory();
};

} // namespace gpgpu::compiler
