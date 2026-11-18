#include "spirv_compiler.hpp"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include <cstdlib>
#include <mutex>
#include <string>

namespace gpgpu::compiler {

namespace {

std::once_flag s_glslangInitOnce;

void ensureGlslangInitialized() {
    std::call_once(s_glslangInitOnce, []() {
        glslang::InitializeProcess();
        std::atexit([]() {
            glslang::FinalizeProcess();
        });
    });
}

} // namespace

bool SpirvCompiler::compileGlslToSpirv(std::string_view glslSource,
                                       std::vector<uint32_t>& outSpirv,
                                       bool optimize,
                                       GPUError* outErr) {
    if (glslSource.empty()) {
        if (outErr) {
            *outErr = GPUError{
                GPUError::Code::ShaderCompileError,
                "Empty GLSL source provided.",
                {},
                {}
            };
        }
        return false;
    }

    ensureGlslangInitialized();

    glslang::TShader shader(EShLangCompute);
    const char* srcPtr = glslSource.data();
    const int srcLen = static_cast<int>(glslSource.size());
    shader.setStringsWithLengths(&srcPtr, &srcLen, 1);

    // Target Vulkan 1.2 with SPIR-V 1.5
    shader.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);

    const TBuiltInResource* resources = GetDefaultResources();
    const auto messages = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

    if (!shader.parse(resources, 100, false, messages)) {
        if (outErr) {
            std::string log = shader.getInfoLog();
            std::string debugLog = shader.getInfoDebugLog();
            if (!debugLog.empty()) {
                log += "\n" + debugLog;
            }
            *outErr = GPUError{
                GPUError::Code::ShaderCompileError,
                "GLSL compilation failed:\n" + log,
                std::move(log),
                std::string(glslSource)
            };
        }
        return false;
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        if (outErr) {
            std::string log = program.getInfoLog();
            *outErr = GPUError{
                GPUError::Code::ShaderCompileError,
                "GLSL linking failed:\n" + log,
                std::move(log),
                std::string(glslSource)
            };
        }
        return false;
    }

    glslang::TIntermediate* intermediate = program.getIntermediate(EShLangCompute);
    if (!intermediate) {
        if (outErr) {
            *outErr = GPUError{
                GPUError::Code::ShaderCompileError,
                "Failed to obtain glslang intermediate AST for compute shader.",
                {},
                std::string(glslSource)
            };
        }
        return false;
    }

    glslang::SpvOptions spvOptions;
    spvOptions.generateDebugInfo = false;
    spvOptions.disableOptimizer = true; // Use SPIRV-Tools for passes
    spv::SpvBuildLogger logger;
    outSpirv.clear();
    glslang::GlslangToSpv(*intermediate, outSpirv, &logger, &spvOptions);

    if (outSpirv.empty()) {
        if (outErr) {
            *outErr = GPUError{
                GPUError::Code::ShaderCompileError,
                "SPIR-V generation failed: " + logger.getAllMessages(),
                logger.getAllMessages(),
                std::string(glslSource)
            };
        }
        return false;
    }

    // SPIR-V Validation
    {
        spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
        std::string valErrors;
        validator.SetMessageConsumer([&valErrors](spv_message_level_t level, const char*, const spv_position_t& pos, const char* msg) {
            if (level == SPV_MSG_ERROR || level == SPV_MSG_INTERNAL_ERROR) {
                valErrors += "Line " + std::to_string(pos.line) + ":" + std::to_string(pos.column) + " - " + msg + "\n";
            }
        });

        if (!validator.Validate(outSpirv)) {
            if (outErr) {
                *outErr = GPUError{
                    GPUError::Code::SpirvValidationError,
                    "SPIR-V validation failed:\n" + valErrors,
                    std::move(valErrors),
                    std::string(glslSource)
                };
            }
            return false;
        }
    }

    // SPIR-V Optimization passes
    if (optimize) {
        spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_2);
        optimizer.RegisterPerformancePasses();
        std::vector<uint32_t> optimized;
        if (optimizer.Run(outSpirv.data(), outSpirv.size(), &optimized)) {
            outSpirv = std::move(optimized);
        }
    }

    return true;
}

} // namespace gpgpu::compiler
