#pragma once

#include "kernel_parser.hpp"
#include <gpgpu/backend_interface.hpp>

#include <span>
#include <string>

namespace gpgpu::compiler {

class GLSLGenerator {
public:
    static std::string generate(const ParsedKernel& parsed,
                                std::span<const BoundVariableDesc> bindings,
                                Dim3 logicalDims,
                                Dim3 workgroupSize,
                                const Capabilities& caps);
};

} // namespace gpgpu::compiler
