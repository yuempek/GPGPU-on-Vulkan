#pragma once

#include <gpgpu/types.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace gpgpu::compiler {

class SpirvCompiler {
public:
    static bool compileGlslToSpirv(std::string_view glslSource,
                                   std::vector<uint32_t>& outSpirv,
                                   bool optimize = true,
                                   GPUError* outErr = nullptr);
};

} // namespace gpgpu::compiler
