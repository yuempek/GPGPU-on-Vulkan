#include "glsl_generator.hpp"

#include <regex>
#include <sstream>

namespace gpgpu::compiler {

namespace {

std::string generateInterpolationCode(std::span<const BoundVariableDesc> bindings) {
    std::ostringstream ss;
    bool needsCubicHelper = false;

    for (const auto& b : bindings) {
        if (b.interpolation == Interpolation::Cubic) {
            needsCubicHelper = true;
            break;
        }
    }

    if (needsCubicHelper) {
        ss << "vec4 _gpgpu_cubic_weights(float t) {\n"
           << "    float t2 = t * t;\n"
           << "    float t3 = t2 * t;\n"
           << "    return 0.5 * vec4(\n"
           << "        -t3 + 2.0 * t2 - t,\n"
           << "         3.0 * t3 - 5.0 * t2 + 2.0,\n"
           << "        -3.0 * t3 + 4.0 * t2 + t,\n"
           << "         t3 - t2\n"
           << "    );\n"
           << "}\n\n";
    }

    for (const auto& b : bindings) {
        if (b.interpolation == Interpolation::None) continue;

        const std::string_view typeStr = dataTypeToGlslString(b.dtype);
        const uint32_t dimX = (b.dims.x > 0) ? b.dims.x : 1;
        const uint32_t dimY = (b.dims.y > 0) ? b.dims.y : 1;
        const uint32_t dimZ = (b.dims.z > 0) ? b.dims.z : 1;

        ss << "const uvec3 _dims_" << b.name << " = uvec3("
           << dimX << ", " << dimY << ", " << dimZ << ");\n";

        ss << "uint _idx_" << b.name << "(int x) { return uint(clamp(x, 0, int(_dims_" << b.name << ".x) - 1)); }\n"
           << "uint _idx_" << b.name << "(ivec2 c) {\n"
           << "    uint x = uint(clamp(c.x, 0, int(_dims_" << b.name << ".x) - 1));\n"
           << "    uint y = uint(clamp(c.y, 0, int(_dims_" << b.name << ".y) - 1));\n"
           << "    return y * _dims_" << b.name << ".x + x;\n"
           << "}\n"
           << "uint _idx_" << b.name << "(ivec3 c) {\n"
           << "    uint x = uint(clamp(c.x, 0, int(_dims_" << b.name << ".x) - 1));\n"
           << "    uint y = uint(clamp(c.y, 0, int(_dims_" << b.name << ".y) - 1));\n"
           << "    uint z = uint(clamp(c.z, 0, int(_dims_" << b.name << ".z) - 1));\n"
           << "    return (z * _dims_" << b.name << ".y + y) * _dims_" << b.name << ".x + x;\n"
           << "}\n\n";

        if (b.interpolation == Interpolation::Nearest) {
            // 1D Nearest
            ss << typeStr << " _gpgpu_sample_" << b.name << "(float x) {\n"
               << "    int ix = int(round(x));\n"
               << "    return " << b.name << "[_idx_" << b.name << "(ix)];\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(int x) { return " << b.name << "[_idx_" << b.name << "(x)]; }\n"
               // 2D Nearest
               << typeStr << " _gpgpu_sample_" << b.name << "(vec2 c) {\n"
               << "    ivec2 ic = ivec2(round(c));\n"
               << "    return " << b.name << "[_idx_" << b.name << "(ic)];\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(ivec2 c) { return " << b.name << "[_idx_" << b.name << "(c)]; }\n"
               // 3D Nearest
               << typeStr << " _gpgpu_sample_" << b.name << "(vec3 c) {\n"
               << "    ivec3 ic = ivec3(round(c));\n"
               << "    return " << b.name << "[_idx_" << b.name << "(ic)];\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(ivec3 c) { return " << b.name << "[_idx_" << b.name << "(c)]; }\n\n";
        } else if (b.interpolation == Interpolation::Linear) {
            // 1D Linear
            ss << typeStr << " _gpgpu_sample_" << b.name << "(float x) {\n"
               << "    float cx = clamp(x, 0.0, float(_dims_" << b.name << ".x - 1u));\n"
               << "    int x0 = int(floor(cx));\n"
               << "    int x1 = min(x0 + 1, int(_dims_" << b.name << ".x - 1u));\n"
               << "    float fx = fract(cx);\n"
               << "    return mix(" << b.name << "[_idx_" << b.name << "(x0)], " << b.name << "[_idx_" << b.name << "(x1)], fx);\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(int x) { return _gpgpu_sample_" << b.name << "(float(x)); }\n"
               // 2D Bilinear
               << typeStr << " _gpgpu_sample_" << b.name << "(vec2 c) {\n"
               << "    vec2 cc = clamp(c, vec2(0.0), vec2(_dims_" << b.name << ".xy - uvec2(1u)));\n"
               << "    int x0 = int(floor(cc.x));\n"
               << "    int x1 = min(x0 + 1, int(_dims_" << b.name << ".x - 1u));\n"
               << "    int y0 = int(floor(cc.y));\n"
               << "    int y1 = min(y0 + 1, int(_dims_" << b.name << ".y - 1u));\n"
               << "    float fx = fract(cc.x);\n"
               << "    float fy = fract(cc.y);\n"
               << "    " << typeStr << " v00 = " << b.name << "[_idx_" << b.name << "(ivec2(x0, y0))];\n"
               << "    " << typeStr << " v10 = " << b.name << "[_idx_" << b.name << "(ivec2(x1, y0))];\n"
               << "    " << typeStr << " v01 = " << b.name << "[_idx_" << b.name << "(ivec2(x0, y1))];\n"
               << "    " << typeStr << " v11 = " << b.name << "[_idx_" << b.name << "(ivec2(x1, y1))];\n"
               << "    return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(ivec2 c) { return _gpgpu_sample_" << b.name << "(vec2(c)); }\n"
               // 3D Trilinear
               << typeStr << " _gpgpu_sample_" << b.name << "(vec3 c) {\n"
               << "    vec3 cc = clamp(c, vec3(0.0), vec3(_dims_" << b.name << ".xyz - uvec3(1u)));\n"
               << "    int x0 = int(floor(cc.x)); int x1 = min(x0 + 1, int(_dims_" << b.name << ".x - 1u));\n"
               << "    int y0 = int(floor(cc.y)); int y1 = min(y0 + 1, int(_dims_" << b.name << ".y - 1u));\n"
               << "    int z0 = int(floor(cc.z)); int z1 = min(z0 + 1, int(_dims_" << b.name << ".z - 1u));\n"
               << "    float fx = fract(cc.x); float fy = fract(cc.y); float fz = fract(cc.z);\n"
               << "    " << typeStr << " v000 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y0, z0))];\n"
               << "    " << typeStr << " v100 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y0, z0))];\n"
               << "    " << typeStr << " v010 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y1, z0))];\n"
               << "    " << typeStr << " v110 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y1, z0))];\n"
               << "    " << typeStr << " v001 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y0, z1))];\n"
               << "    " << typeStr << " v101 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y0, z1))];\n"
               << "    " << typeStr << " v011 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y1, z1))];\n"
               << "    " << typeStr << " v111 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y1, z1))];\n"
               << "    " << typeStr << " m0 = mix(mix(v000, v100, fx), mix(v010, v110, fx), fy);\n"
               << "    " << typeStr << " m1 = mix(mix(v001, v101, fx), mix(v011, v111, fx), fy);\n"
               << "    return mix(m0, m1, fz);\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(ivec3 c) { return _gpgpu_sample_" << b.name << "(vec3(c)); }\n\n";
        } else if (b.interpolation == Interpolation::Cubic) {
            // 1D Catmull-Rom
            ss << typeStr << " _gpgpu_sample_" << b.name << "(float x) {\n"
               << "    float cx = clamp(x, 0.0, float(_dims_" << b.name << ".x - 1u));\n"
               << "    int ix = int(floor(cx));\n"
               << "    float fx = fract(cx);\n"
               << "    vec4 wx = _gpgpu_cubic_weights(fx);\n"
               << "    int p0 = max(ix - 1, 0);\n"
               << "    int p1 = ix;\n"
               << "    int p2 = min(ix + 1, int(_dims_" << b.name << ".x - 1u));\n"
               << "    int p3 = min(ix + 2, int(_dims_" << b.name << ".x - 1u));\n"
               << "    return " << b.name << "[_idx_" << b.name << "(p0)] * wx.x +\n"
               << "           " << b.name << "[_idx_" << b.name << "(p1)] * wx.y +\n"
               << "           " << b.name << "[_idx_" << b.name << "(p2)] * wx.z +\n"
               << "           " << b.name << "[_idx_" << b.name << "(p3)] * wx.w;\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(int x) { return _gpgpu_sample_" << b.name << "(float(x)); }\n"
               // 2D Bicubic Catmull-Rom
               << typeStr << " _gpgpu_sample_" << b.name << "(vec2 c) {\n"
               << "    vec2 cc = clamp(c, vec2(0.0), vec2(_dims_" << b.name << ".xy - uvec2(1u)));\n"
               << "    int ix = int(floor(cc.x));\n"
               << "    int iy = int(floor(cc.y));\n"
               << "    float fx = fract(cc.x);\n"
               << "    float fy = fract(cc.y);\n"
               << "    vec4 wx = _gpgpu_cubic_weights(fx);\n"
               << "    vec4 wy = _gpgpu_cubic_weights(fy);\n"
               << "    int x0 = max(ix - 1, 0), x1 = ix, x2 = min(ix + 1, int(_dims_" << b.name << ".x - 1u)), x3 = min(ix + 2, int(_dims_" << b.name << ".x - 1u));\n"
               << "    int y0 = max(iy - 1, 0), y1 = iy, y2 = min(iy + 1, int(_dims_" << b.name << ".y - 1u)), y3 = min(iy + 2, int(_dims_" << b.name << ".y - 1u));\n"
               << "    " << typeStr << " col0 = " << b.name << "[_idx_" << b.name << "(ivec2(x0, y0))]*wx.x + " << b.name << "[_idx_" << b.name << "(ivec2(x1, y0))]*wx.y + " << b.name << "[_idx_" << b.name << "(ivec2(x2, y0))]*wx.z + " << b.name << "[_idx_" << b.name << "(ivec2(x3, y0))]*wx.w;\n"
               << "    " << typeStr << " col1 = " << b.name << "[_idx_" << b.name << "(ivec2(x0, y1))]*wx.x + " << b.name << "[_idx_" << b.name << "(ivec2(x1, y1))]*wx.y + " << b.name << "[_idx_" << b.name << "(ivec2(x2, y1))]*wx.z + " << b.name << "[_idx_" << b.name << "(ivec2(x3, y1))]*wx.w;\n"
               << "    " << typeStr << " col2 = " << b.name << "[_idx_" << b.name << "(ivec2(x0, y2))]*wx.x + " << b.name << "[_idx_" << b.name << "(ivec2(x1, y2))]*wx.y + " << b.name << "[_idx_" << b.name << "(ivec2(x2, y2))]*wx.z + " << b.name << "[_idx_" << b.name << "(ivec2(x3, y2))]*wx.w;\n"
               << "    " << typeStr << " col3 = " << b.name << "[_idx_" << b.name << "(ivec2(x0, y3))]*wx.x + " << b.name << "[_idx_" << b.name << "(ivec2(x1, y3))]*wx.y + " << b.name << "[_idx_" << b.name << "(ivec2(x2, y3))]*wx.z + " << b.name << "[_idx_" << b.name << "(ivec2(x3, y3))]*wx.w;\n"
               << "    return col0 * wy.x + col1 * wy.y + col2 * wy.z + col3 * wy.w;\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(ivec2 c) { return _gpgpu_sample_" << b.name << "(vec2(c)); }\n"
               // 3D Trilinear fallback
               << typeStr << " _gpgpu_sample_" << b.name << "(vec3 c) {\n"
               << "    vec3 cc = clamp(c, vec3(0.0), vec3(_dims_" << b.name << ".xyz - uvec3(1u)));\n"
               << "    int x0 = int(floor(cc.x)); int x1 = min(x0 + 1, int(_dims_" << b.name << ".x - 1u));\n"
               << "    int y0 = int(floor(cc.y)); int y1 = min(y0 + 1, int(_dims_" << b.name << ".y - 1u));\n"
               << "    int z0 = int(floor(cc.z)); int z1 = min(z0 + 1, int(_dims_" << b.name << ".z - 1u));\n"
               << "    float fx = fract(cc.x); float fy = fract(cc.y); float fz = fract(cc.z);\n"
               << "    " << typeStr << " v000 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y0, z0))];\n"
               << "    " << typeStr << " v100 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y0, z0))];\n"
               << "    " << typeStr << " v010 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y1, z0))];\n"
               << "    " << typeStr << " v110 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y1, z0))];\n"
               << "    " << typeStr << " v001 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y0, z1))];\n"
               << "    " << typeStr << " v101 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y0, z1))];\n"
               << "    " << typeStr << " v011 = " << b.name << "[_idx_" << b.name << "(ivec3(x0, y1, z1))];\n"
               << "    " << typeStr << " v111 = " << b.name << "[_idx_" << b.name << "(ivec3(x1, y1, z1))];\n"
               << "    " << typeStr << " m0 = mix(mix(v000, v100, fx), mix(v010, v110, fx), fy);\n"
               << "    " << typeStr << " m1 = mix(mix(v001, v101, fx), mix(v011, v111, fx), fy);\n"
               << "    return mix(m0, m1, fz);\n"
               << "}\n"
               << typeStr << " _gpgpu_sample_" << b.name << "(ivec3 c) { return _gpgpu_sample_" << b.name << "(vec3(c)); }\n\n";
        }
    }

    return ss.str();
}

} // namespace

std::string GLSLGenerator::generate(const ParsedKernel& parsed,
                                    std::span<const BoundVariableDesc> bindings,
                                    Dim3 logicalDims,
                                    Dim3 workgroupSize,
                                    const Capabilities& /*caps*/) {
    if (parsed.mode == KernelMode::RawGLSL) {
        return parsed.bodyCode;
    }

    const int dims = logicalDims.dimensions();

    // 1. ShaderToy Mode
    if (parsed.mode == KernelMode::ShaderToy) {
        std::ostringstream st;
        st << "#version 450\n";
        st << "layout(local_size_x = " << workgroupSize.x
           << ", local_size_y = " << workgroupSize.y
           << ", local_size_z = 1) in;\n\n";

        st << "layout(push_constant) uniform _GPGPUPushConstants {\n"
           << "    uvec4 _u_dims;\n"
           << "    vec4  iResolution;\n"
           << "    vec4  iMouse;\n"
           << "    float iTime;\n"
           << "    float iTimeDelta;\n"
           << "    int   iFrame;\n"
           << "    uint  _u_flags;\n"
           << "} _pc;\n\n";

        st << "#define iResolution _pc.iResolution\n"
           << "#define iTime _pc.iTime\n"
           << "#define iTimeDelta _pc.iTimeDelta\n"
           << "#define iFrame _pc.iFrame\n"
           << "#define iMouse _pc.iMouse\n\n";

        for (size_t i = 0; i < bindings.size(); ++i) {
            const auto& b = bindings[i];
            st << "layout(set = 0, binding = " << i << ", std430) buffer _Buf_" << b.name << " {\n"
               << "    " << dataTypeToGlslString(b.dtype) << " " << b.name << "[];\n"
               << "};\n";
        }
        st << "\n";

        std::string outVarName = bindings.empty() ? "" : bindings[0].name;
        DataType outDtype = bindings.empty() ? DataType::Vec4 : bindings[0].dtype;
        for (const auto& b : bindings) {
            if (b.name == "outputData" || b.name == "fragColor" || b.name == "outData" || b.name == "output") {
                outVarName = b.name;
                outDtype = b.dtype;
                break;
            }
        }

        st << "// Shadertoy compatibility keyword remap\n"
           << "#define buffer _usr_buf_\n"
           << "#define sample _usr_sample_\n\n";

        st << parsed.bodyCode << "\n\n";

        st << "void main() {\n"
           << "    uvec3 _gid = gl_GlobalInvocationID;\n"
           << "    if (_gid.x >= _pc._u_dims.x || _gid.y >= _pc._u_dims.y) return;\n"
           << "    vec2 fragCoord = vec2(float(_gid.x) + 0.5, float(_gid.y) + 0.5);\n"
           << "    vec4 _st_fragColor = vec4(0.0);\n"
           << "    mainImage(_st_fragColor, fragCoord);\n";

        if (!outVarName.empty()) {
            st << "    uint _flat_idx = _gid.y * _pc._u_dims.x + _gid.x;\n";
            if (outDtype == DataType::Vec4) {
                st << "    " << outVarName << "[_flat_idx] = _st_fragColor;\n";
            } else if (outDtype == DataType::Vec3) {
                st << "    " << outVarName << "[_flat_idx] = _st_fragColor.rgb;\n";
            } else if (outDtype == DataType::Vec2) {
                st << "    " << outVarName << "[_flat_idx] = _st_fragColor.rg;\n";
            } else if (outDtype == DataType::Float32) {
                st << "    " << outVarName << "[_flat_idx] = _st_fragColor.r;\n";
            } else {
                st << "    " << outVarName << "[_flat_idx] = " << dataTypeToGlslString(outDtype) << "(_st_fragColor.r);\n";
            }
        }
        st << "}\n";
        return st.str();
    }

    // 2. Standard DSL Mode
    std::ostringstream glsl;
    glsl << "#version 450\n";

    for (const auto& b : bindings) {
        if (b.dtype == DataType::Float64) {
            glsl << "#extension GL_ARB_gpu_shader_fp64 : enable\n";
            break;
        }
    }
    for (const auto& b : bindings) {
        if (b.dtype == DataType::Int16 || b.dtype == DataType::UInt16) {
            glsl << "#extension GL_EXT_shader_16bit_storage : enable\n";
            break;
        }
    }

    glsl << "layout(local_size_x = " << workgroupSize.x
         << ", local_size_y = " << workgroupSize.y
         << ", local_size_z = " << workgroupSize.z << ") in;\n\n";

    glsl << "layout(push_constant) uniform _GPGPUPushConstants {\n"
         << "    uvec4 _u_dims;\n"
         << "    vec4  iResolution;\n"
         << "    vec4  iMouse;\n"
         << "    float iTime;\n"
         << "    float iTimeDelta;\n"
         << "    int   iFrame;\n"
         << "    uint  _u_flags;\n"
         << "} _pc;\n\n";

    glsl << "#define iResolution _pc.iResolution\n"
         << "#define iTime _pc.iTime\n"
         << "#define iTimeDelta _pc.iTimeDelta\n"
         << "#define iFrame _pc.iFrame\n"
         << "#define iMouse _pc.iMouse\n\n";

    for (size_t i = 0; i < bindings.size(); ++i) {
        const auto& b = bindings[i];
        glsl << "layout(set = 0, binding = " << i << ", std430) buffer _Buf_" << b.name << " {\n"
             << "    " << dataTypeToGlslString(b.dtype) << " " << b.name << "[];\n"
             << "};\n";
    }

    glsl << "\nuint _gpgpu_idx(int x)   { return uint(clamp(x, 0, int(_pc._u_dims.x) - 1)); }\n"
         << "uint _gpgpu_idx(uint x)  { return min(x, _pc._u_dims.x - 1u); }\n"
         << "uint _gpgpu_idx(float x) { return uint(clamp(int(x), 0, int(_pc._u_dims.x) - 1)); }\n"
         << "uint _gpgpu_idx(ivec2 c) {\n"
         << "    uint cx = uint(clamp(c.x, 0, int(_pc._u_dims.x) - 1));\n"
         << "    uint cy = uint(clamp(c.y, 0, int(_pc._u_dims.y) - 1));\n"
         << "    return cy * _pc._u_dims.x + cx;\n"
         << "}\n"
         << "uint _gpgpu_idx(uvec2 c) {\n"
         << "    uint cx = min(c.x, _pc._u_dims.x - 1u);\n"
         << "    uint cy = min(c.y, _pc._u_dims.y - 1u);\n"
         << "    return cy * _pc._u_dims.x + cx;\n"
         << "}\n"
         << "uint _gpgpu_idx(vec2 c) {\n"
         << "    return _gpgpu_idx(ivec2(c));\n"
         << "}\n"
         << "uint _gpgpu_idx(ivec3 c) {\n"
         << "    uint cx = uint(clamp(c.x, 0, int(_pc._u_dims.x) - 1));\n"
         << "    uint cy = uint(clamp(c.y, 0, int(_pc._u_dims.y) - 1));\n"
         << "    uint cz = uint(clamp(c.z, 0, int(_pc._u_dims.z) - 1));\n"
         << "    return (cz * _pc._u_dims.y + cy) * _pc._u_dims.x + cx;\n"
         << "}\n"
         << "uint _gpgpu_idx(uvec3 c) {\n"
         << "    uint cx = min(c.x, _pc._u_dims.x - 1u);\n"
         << "    uint cy = min(c.y, _pc._u_dims.y - 1u);\n"
         << "    uint cz = min(c.z, _pc._u_dims.z - 1u);\n"
         << "    return (cz * _pc._u_dims.y + cy) * _pc._u_dims.x + cx;\n"
         << "}\n"
         << "uint _gpgpu_idx(vec3 c) {\n"
         << "    return _gpgpu_idx(ivec3(c));\n"
         << "}\n\n";

    // Interpolation sampling helpers
    glsl << generateInterpolationCode(bindings);

    if (!parsed.preambleFunctions.empty()) {
        glsl << parsed.preambleFunctions << "\n\n";
    }

    const char* coordType = (dims == 1) ? "int" : ((dims == 2) ? "ivec2" : "ivec3");
    const char* gidArg = (dims == 1) ? "int(_gid.x)" : ((dims == 2) ? "ivec2(_gid.xy)" : "ivec3(_gid.xyz)");

    std::string transformedBody = parsed.bodyCode;
    for (const auto& b : bindings) {
        if (b.interpolation != Interpolation::None) {
            std::regex lhsReg("\\b(" + b.name + ")\\s*\\[\\s*([^;\\]]+)\\s*\\]\\s*(=|\\+=|-=|\\*=|\\/=)");
            transformedBody = std::regex_replace(transformedBody, lhsReg, "$1[_gpgpu_idx($2)] $3");

            std::regex rhsReg("\\b" + b.name + "\\s*\\[\\s*([^;\\]]+)\\s*\\]");
            transformedBody = std::regex_replace(transformedBody, rhsReg, "_gpgpu_sample_" + b.name + "($1)");
        } else {
            std::regex reg("\\b" + b.name + "\\s*\\[\\s*([^;\\]]+)\\s*\\]");
            transformedBody = std::regex_replace(transformedBody, reg, b.name + "[_gpgpu_idx($1)]");
        }
    }
    if (!transformedBody.empty() && transformedBody.back() != ';') {
        transformedBody += ';';
    }

    glsl << "#line 1\n";
    glsl << "void _gpgpu_user_kernel(" << coordType << " " << parsed.coordParamName << ") {\n"
         << "    " << transformedBody << "\n"
         << "}\n\n";

    glsl << "void main() {\n"
         << "    uvec3 _gid = gl_GlobalInvocationID;\n"
         << "    if (_gid.x >= _pc._u_dims.x || _gid.y >= _pc._u_dims.y || _gid.z >= _pc._u_dims.z) return;\n"
         << "    _gpgpu_user_kernel(" << gidArg << ");\n"
         << "}\n";

    return glsl.str();
}

} // namespace gpgpu::compiler
