#pragma once

#include <gpgpu/types.hpp>

#include <string>
#include <string_view>

namespace gpgpu::compiler {

enum class KernelMode : uint8_t {
    DSL = 0,       // kernel(coord) { outputData[coord] = cos(varNameIn[coord]); }
    ShaderToy = 1, // void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }
    RawGLSL = 2    // Full "#version 450 ..." compute shader source
};

struct ParsedKernel {
    KernelMode mode = KernelMode::DSL;
    std::string coordParamName = "coord";
    std::string preambleFunctions;
    std::string bodyCode;
    std::string rawSource;
};

class KernelParser {
public:
    static bool parse(std::string_view source, ParsedKernel& outParsed, GPUError* outErr = nullptr);
};

} // namespace gpgpu::compiler
