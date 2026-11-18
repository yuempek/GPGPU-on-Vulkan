#include <gtest/gtest.h>
#include <gpgpu/gpgpu.hpp>
#include <vector>
#include <cmath>

class MultiPassChannelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        gpu = std::make_unique<gpgpu::GPU>();
        ASSERT_FALSE(gpu->hasError().has_value());
    }

    void TearDown() override {
        if (gpu) {
            gpu->dispose();
        }
    }

    std::unique_ptr<gpgpu::GPU> gpu;
};

// 1. Test 2D Texture Channel Sampling via SSBO binding
TEST_F(MultiPassChannelsTest, Channel2DSamplingAndTexelFetch) {
    const uint32_t texW = 4;
    const uint32_t texH = 4;
    std::vector<gpgpu::vec4> textureData(texW * texH);
    for (uint32_t y = 0; y < texH; ++y) {
        for (uint32_t x = 0; x < texW; ++x) {
            textureData[y * texW + x] = gpgpu::vec4{
                static_cast<float>(x) / (texW - 1),
                static_cast<float>(y) / (texH - 1),
                0.5f,
                1.0f
            };
        }
    }

    auto ch0 = gpu->createData<gpgpu::vec4>("_Buf_iCh0", textureData);
    auto ch1 = gpu->createData<gpgpu::vec4>("_Buf_iCh1", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch2 = gpu->createData<gpgpu::vec4>("_Buf_iCh2", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch3 = gpu->createData<gpgpu::vec4>("_Buf_iCh3", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto outBuf = gpu->createData<gpgpu::vec4>("_Buf_outputData");

    const std::string glsl = R"(#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform _GPGPUPushConstants {
    uvec4 _u_dims;
    vec4  iResolution;
    vec4  iMouse;
    float iTime;
    float iTimeDelta;
    int   iFrame;
    uint  _u_flags;
} _pc;

#define iResolution _pc.iResolution
#define iTime _pc.iTime
#define iTimeDelta _pc.iTimeDelta
#define iFrame _pc.iFrame
#define iMouse _pc.iMouse

const vec4 iChannelResolution[4] = vec4[4](
    vec4(4.0, 4.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0)
);

layout(set = 0, binding = 0, std430) readonly buffer _Buf_iCh0 { vec4 _iChannel0_data[]; };
layout(set = 0, binding = 1, std430) readonly buffer _Buf_iCh1 { vec4 _iChannel1_data[]; };
layout(set = 0, binding = 2, std430) readonly buffer _Buf_iCh2 { vec4 _iChannel2_data[]; };
layout(set = 0, binding = 3, std430) readonly buffer _Buf_iCh3 { vec4 _iChannel3_data[]; };
layout(set = 0, binding = 4, std430) writeonly buffer _Buf_outputData { vec4 outputData[]; };

struct _Channel0_t { int id; } iChannel0;

vec4 texelFetch(_Channel0_t ch, ivec2 coord, int lod) {
    uvec2 res = uvec2(iChannelResolution[0].xy);
    return _iChannel0_data[uint(coord.y) * res.x + uint(coord.x)];
}

void main() {
    uvec3 gid = gl_GlobalInvocationID;
    if (gid.x >= _pc._u_dims.x || gid.y >= _pc._u_dims.y) return;
    vec4 val = texelFetch(iChannel0, ivec2(gid.x, gid.y), 0);
    outputData[gid.y * _pc._u_dims.x + gid.x] = val;
}
)";

    gpu->kernel(glsl);
    ASSERT_FALSE(gpu->hasError().has_value());

    gpu->run(texW, texH);
    ASSERT_FALSE(gpu->hasError().has_value());

    std::vector<gpgpu::vec4> result(texW * texH);
    outBuf.getTo(result);

    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i].x, textureData[i].x, 1e-4f);
        EXPECT_NEAR(result[i].y, textureData[i].y, 1e-4f);
        EXPECT_NEAR(result[i].z, textureData[i].z, 1e-4f);
        EXPECT_NEAR(result[i].w, textureData[i].w, 1e-4f);
    }
}

// 2. Test Temporal Feedback (Ping-Pong Buffering across multiple frames)
TEST_F(MultiPassChannelsTest, PingPongTemporalFeedback) {
    const uint32_t width = 8;
    const uint32_t height = 8;
    const size_t count = width * height;

    std::vector<gpgpu::vec4> pingBuffer(count, gpgpu::vec4{1.0f, 0.0f, 0.0f, 1.0f});
    std::vector<gpgpu::vec4> pongBuffer(count, gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f});

    auto ch0 = gpu->createData<gpgpu::vec4>("_Buf_iCh0", pingBuffer);
    auto ch1 = gpu->createData<gpgpu::vec4>("_Buf_iCh1", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch2 = gpu->createData<gpgpu::vec4>("_Buf_iCh2", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch3 = gpu->createData<gpgpu::vec4>("_Buf_iCh3", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto outBuf = gpu->createData<gpgpu::vec4>("_Buf_outputData");

    const std::string stepShader = R"(#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform _GPGPUPushConstants {
    uvec4 _u_dims;
    vec4  iResolution;
    vec4  iMouse;
    float iTime;
    float iTimeDelta;
    int   iFrame;
    uint  _u_flags;
} _pc;

layout(set = 0, binding = 0, std430) readonly buffer _Buf_iCh0 { vec4 _iChannel0_data[]; };
layout(set = 0, binding = 1, std430) readonly buffer _Buf_iCh1 { vec4 _iChannel1_data[]; };
layout(set = 0, binding = 2, std430) readonly buffer _Buf_iCh2 { vec4 _iChannel2_data[]; };
layout(set = 0, binding = 3, std430) readonly buffer _Buf_iCh3 { vec4 _iChannel3_data[]; };
layout(set = 0, binding = 4, std430) writeonly buffer _Buf_outputData { vec4 outputData[]; };

void main() {
    uvec3 gid = gl_GlobalInvocationID;
    if (gid.x >= _pc._u_dims.x || gid.y >= _pc._u_dims.y) return;
    uint idx = gid.y * _pc._u_dims.x + gid.x;
    vec4 prev = _iChannel0_data[idx];
    // Add 0.5 per frame to green channel
    outputData[idx] = vec4(prev.x, prev.y + 0.5, prev.z, 1.0);
}
)";

    gpu->kernel(stepShader);
    ASSERT_FALSE(gpu->hasError().has_value());

    // Frame 1
    gpu->run(width, height);
    ASSERT_FALSE(gpu->hasError().has_value());
    outBuf.getTo(pongBuffer);

    EXPECT_NEAR(pongBuffer[0].y, 0.5f, 1e-4f);

    // Frame 2: feed pongBuffer into iChannel0
    ch0.set(pongBuffer);
    gpu->run(width, height);
    ASSERT_FALSE(gpu->hasError().has_value());
    outBuf.getTo(pingBuffer);

    EXPECT_NEAR(pingBuffer[0].y, 1.0f, 1e-4f);

    // Frame 3: feed pingBuffer back into iChannel0
    ch0.set(pingBuffer);
    gpu->run(width, height);
    ASSERT_FALSE(gpu->hasError().has_value());
    outBuf.getTo(pongBuffer);

    EXPECT_NEAR(pongBuffer[0].y, 1.5f, 1e-4f);
}

// 3. Test Keyboard and Audio Mock Buffer Sampling
TEST_F(MultiPassChannelsTest, KeyboardAndAudioSampling) {
    // 256 x 3 keyboard buffer
    std::vector<gpgpu::vec4> kbData(256 * 3, gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f});
    // Key 'A' = 65, set pressed (row 0) and toggle (row 2)
    kbData[0 * 256 + 65] = gpgpu::vec4{1.0f, 0.0f, 0.0f, 0.0f};
    kbData[2 * 256 + 65] = gpgpu::vec4{1.0f, 0.0f, 0.0f, 0.0f};

    // 512 x 2 audio buffer
    std::vector<gpgpu::vec4> audioData(512 * 2, gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f});
    // Frequency bin 10 has magnitude 0.75
    audioData[0 * 512 + 10] = gpgpu::vec4{0.75f, 0.75f, 0.75f, 1.0f};

    auto ch0 = gpu->createData<gpgpu::vec4>("_Buf_iCh0", kbData);
    auto ch1 = gpu->createData<gpgpu::vec4>("_Buf_iCh1", audioData);
    auto ch2 = gpu->createData<gpgpu::vec4>("_Buf_iCh2", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch3 = gpu->createData<gpgpu::vec4>("_Buf_iCh3", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto outBuf = gpu->createData<gpgpu::vec4>("_Buf_outputData");

    const std::string testShader = R"(#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform _GPGPUPushConstants {
    uvec4 _u_dims;
    vec4  iResolution;
    vec4  iMouse;
    float iTime;
    float iTimeDelta;
    int   iFrame;
    uint  _u_flags;
} _pc;

layout(set = 0, binding = 0, std430) readonly buffer _Buf_iCh0 { vec4 _iChannel0_data[]; };
layout(set = 0, binding = 1, std430) readonly buffer _Buf_iCh1 { vec4 _iChannel1_data[]; };
layout(set = 0, binding = 2, std430) readonly buffer _Buf_iCh2 { vec4 _iChannel2_data[]; };
layout(set = 0, binding = 3, std430) readonly buffer _Buf_iCh3 { vec4 _iChannel3_data[]; };
layout(set = 0, binding = 4, std430) writeonly buffer _Buf_outputData { vec4 outputData[]; };

void main() {
    uvec3 gid = gl_GlobalInvocationID;
    if (gid.x >= _pc._u_dims.x || gid.y >= _pc._u_dims.y) return;
    
    // Read key 'A' state from row 0
    float isKeyADown = _iChannel0_data[0 * 256 + 65].x;
    // Read audio bin 10 from row 0
    float audioFreq = _iChannel1_data[0 * 512 + 10].x;

    outputData[gid.y * _pc._u_dims.x + gid.x] = vec4(isKeyADown, audioFreq, 0.0, 1.0);
}
)";

    gpu->kernel(testShader);
    ASSERT_FALSE(gpu->hasError().has_value());

    gpu->run(1, 1);
    ASSERT_FALSE(gpu->hasError().has_value());

    std::vector<gpgpu::vec4> outResult(1);
    outBuf.getTo(outResult);

    EXPECT_NEAR(outResult[0].x, 1.0f, 1e-4f);
    EXPECT_NEAR(outResult[0].y, 0.75f, 1e-4f);
}

// 4. Test Dynamic Syntax Error Recovery: Broken GLSL -> Error Reported -> Fixed GLSL -> Recovers Immediately
TEST_F(MultiPassChannelsTest, DynamicSyntaxErrorRecovery) {
    auto outBuf = gpu->createData<gpgpu::vec4>("outputData");

    const std::string validShader1 = R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
)";

    // Broken shader (missing semicolon / deleted character)
    const std::string brokenShader = R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(0.0, 1.0, 0.0, 1.0) // syntax error missing semicolon!
}
)";

    const std::string validShader2 = R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(0.0, 0.0, 1.0, 1.0);
}
)";

    // Step 1: Valid initial run
    gpu->kernel(validShader1);
    ASSERT_FALSE(gpu->hasError().has_value());
    gpu->run(4, 4);
    ASSERT_FALSE(gpu->hasError().has_value());
    std::vector<gpgpu::vec4> res1(16);
    outBuf.getTo(res1);
    EXPECT_NEAR(res1[0].x, 1.0f, 1e-4f);

    // Step 2: Broken syntax shader
    gpu->kernel(brokenShader);
    gpu->run(4, 4);
    // Must record error without crash
    EXPECT_TRUE(gpu->hasError().has_value());

    // Step 3: Fixed shader
    gpu->kernel(validShader2);
    // Must have cleared error and accepted new kernel
    ASSERT_FALSE(gpu->hasError().has_value());
    gpu->run(4, 4);
    ASSERT_FALSE(gpu->hasError().has_value());
    std::vector<gpgpu::vec4> res2(16);
    outBuf.getTo(res2);
    EXPECT_NEAR(res2[0].z, 1.0f, 1e-4f);
}

// 5. Test User Snippet with 'buffer' variable name (GLSL keyword compatibility)
TEST_F(MultiPassChannelsTest, BufferKeywordCompatibility) {
    auto outBuf = gpu->createData<gpgpu::vec4>("outputData");

    const std::string userShader = R"(
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    
    vec4 buffer = vec4(uv, 0.5, 1.0);

    fragColor = vec4(uv, uv);
}
)";

    gpu->kernel(userShader);
    ASSERT_FALSE(gpu->hasError().has_value()) << (gpu->hasError().has_value() ? gpu->hasError()->message : "");

    gpu->run(8, 8);
    ASSERT_FALSE(gpu->hasError().has_value()) << (gpu->hasError().has_value() ? gpu->hasError()->message : "");

    std::vector<gpgpu::vec4> res(64);
    outBuf.getTo(res);
    EXPECT_NEAR(res[0].w, 0.5f / 8.0f, 1e-2f);
}

// 6. Test exact user code snippet with texture(iChannel0, uv) and vec4 buffer
TEST_F(MultiPassChannelsTest, ExactUserSnippetWithTexture) {
    auto ch0 = gpu->createData<gpgpu::vec4>("_Buf_iCh0", std::vector<gpgpu::vec4>{gpgpu::vec4{0.1f, 0.2f, 0.3f, 1.0f}});
    auto ch1 = gpu->createData<gpgpu::vec4>("_Buf_iCh1", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch2 = gpu->createData<gpgpu::vec4>("_Buf_iCh2", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch3 = gpu->createData<gpgpu::vec4>("_Buf_iCh3", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto outBuf = gpu->createData<gpgpu::vec4>("_Buf_outputData");

    // Exact GLSL preamble generated by ShaderPass
    const std::string fullShader = R"(#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform _GPGPUPushConstants {
    uvec4 _u_dims;
    vec4  iResolution;
    vec4  iMouse;
    float iTime;
    float iTimeDelta;
    int   iFrame;
    uint  _u_flags;
} _pc;

#define iResolution _pc.iResolution
#define iTime _pc.iTime
#define iTimeDelta _pc.iTimeDelta
#define iFrame _pc.iFrame
#define iMouse _pc.iMouse

const vec4 iChannelResolution[4] = vec4[4](
    vec4(1.0, 1.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0)
);

layout(set = 0, binding = 0, std430) readonly buffer _Buf_iCh0 { vec4 _iChannel0_data[]; };
layout(set = 0, binding = 1, std430) readonly buffer _Buf_iCh1 { vec4 _iChannel1_data[]; };
layout(set = 0, binding = 2, std430) readonly buffer _Buf_iCh2 { vec4 _iChannel2_data[]; };
layout(set = 0, binding = 3, std430) readonly buffer _Buf_iCh3 { vec4 _iChannel3_data[]; };
layout(set = 0, binding = 4, std430) writeonly buffer _Buf_outputData { vec4 outputData[]; };

struct _Channel0_t { int id; } iChannel0;
vec4 texture(_Channel0_t ch, vec2 uv) { return _iChannel0_data[0]; }

// Shadertoy compatibility keyword remap
#define buffer _usr_buf_
#define sample _usr_sample_

#line 1
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    
    vec4 buffer = texture(iChannel0, uv);

    fragColor = vec4(uv, uv);
}

void main() {
    uvec3 _gid = gl_GlobalInvocationID;
    if (_gid.x >= _pc._u_dims.x || _gid.y >= _pc._u_dims.y) return;
    vec2 fragCoord = vec2(float(_gid.x) + 0.5, float(_gid.y) + 0.5);
    vec4 _st_fragColor = vec4(0.0);
    mainImage(_st_fragColor, fragCoord);
    uint _flat_idx = _gid.y * _pc._u_dims.x + _gid.x;
    outputData[_flat_idx] = _st_fragColor;
}
)";

    gpu->kernel(fullShader);
    ASSERT_FALSE(gpu->hasError().has_value()) << (gpu->hasError().has_value() ? gpu->hasError()->message : "");

    gpu->run(4, 4);
    ASSERT_FALSE(gpu->hasError().has_value()) << (gpu->hasError().has_value() ? gpu->hasError()->message : "");

    std::vector<gpgpu::vec4> result(16);
    outBuf.getTo(result);
    // uv.x for pixel (0,0) is 0.5 / 4.0 = 0.125
    EXPECT_NEAR(result[0].x, 0.125f, 1e-4f);
    EXPECT_NEAR(result[0].y, 0.125f, 1e-4f);
    EXPECT_NEAR(result[0].z, 0.125f, 1e-4f);
    EXPECT_NEAR(result[0].w, 0.125f, 1e-4f);
}

// 7. Test User Spectrogram Pipeline (BufferA ping-pong feedback + audio spectrum)
TEST_F(MultiPassChannelsTest, SpectrogramBufferFeedbackAndAudio) {
    const uint32_t W = 16;
    const uint32_t H = 16;

    // Simulated 512x2 audio buffer: row 0 (FFT frequency spectrum) has 0.8f
    std::vector<gpgpu::vec4> audioData(512 * 2, gpgpu::vec4{0.0f, 0.0f, 0.0f, 1.0f});
    for (int i = 0; i < 512; ++i) {
        audioData[i] = gpgpu::vec4{0.8f, 0.8f, 0.8f, 1.0f}; // FFT row
    }

    std::vector<gpgpu::vec4> bufPing(W * H, gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<gpgpu::vec4> bufPong(W * H, gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f});

    auto ch0 = gpu->createData<gpgpu::vec4>("_Buf_iCh0", bufPing);
    auto ch1 = gpu->createData<gpgpu::vec4>("_Buf_iCh1", audioData);
    auto ch2 = gpu->createData<gpgpu::vec4>("_Buf_iCh2", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto ch3 = gpu->createData<gpgpu::vec4>("_Buf_iCh3", std::vector<gpgpu::vec4>{gpgpu::vec4{0.0f, 0.0f, 0.0f, 0.0f}});
    auto outBuf = gpu->createData<gpgpu::vec4>("_Buf_outputData");

    const std::string spectrogramShader = R"(#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform _GPGPUPushConstants {
    uvec4 _u_dims;
    vec4  iResolution;
    vec4  iMouse;
    float iTime;
    float iTimeDelta;
    int   iFrame;
    uint  _u_flags;
} _pc;

#define iResolution _pc.iResolution
#define iTime _pc.iTime

const vec4 iChannelResolution[4] = vec4[4](
    vec4(16.0, 16.0, 1.0, 0.0),
    vec4(512.0, 2.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0)
);

layout(set = 0, binding = 0, std430) readonly buffer _Buf_iCh0 { vec4 _iChannel0_data[]; };
layout(set = 0, binding = 1, std430) readonly buffer _Buf_iCh1 { vec4 _iChannel1_data[]; };
layout(set = 0, binding = 2, std430) readonly buffer _Buf_iCh2 { vec4 _iChannel2_data[]; };
layout(set = 0, binding = 3, std430) readonly buffer _Buf_iCh3 { vec4 _iChannel3_data[]; };
layout(set = 0, binding = 4, std430) writeonly buffer _Buf_outputData { vec4 outputData[]; };

struct _Channel0_t { int id; } iChannel0;
struct _Channel1_t { int id; } iChannel1;

// BufferA: clamp, no vFlip
vec4 texture(_Channel0_t ch, vec2 uv) {
    vec2 u = clamp(uv, vec2(0.0), vec2(1.0));
    uvec2 p = clamp(uvec2(u * vec2(16.0, 16.0)), uvec2(0), uvec2(15));
    return _iChannel0_data[p.y * 16u + p.x];
}

// Audio: 512x2, clamp, no vFlip
vec4 texture(_Channel1_t ch, vec2 uv) {
    vec2 u = clamp(uv, vec2(0.0), vec2(1.0));
    uvec2 p = clamp(uvec2(u * vec2(512.0, 2.0)), uvec2(0), uvec2(511, 1));
    return _iChannel1_data[p.y * 512u + p.x];
}

#define buffer _usr_buf_
#define sample _usr_sample_

// Exact user Spectrogram code
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord)/iResolution.xy*1.0;
    vec2 dy = vec2(0, 2)/iResolution.xy;
    float f = uv.x;
    vec4 buffer = texture(iChannel0, uv+dy);   
    vec4 data = texture(iChannel1, vec2(f, 0.25)).brba;
    fragColor = 1.-uv.y<dy.y ? pow(data, vec4(1.9)) : buffer;
}

void main() {
    uvec3 _gid = gl_GlobalInvocationID;
    if (_gid.x >= _pc._u_dims.x || _gid.y >= _pc._u_dims.y) return;
    vec2 fragCoord = vec2(float(_gid.x) + 0.5, float(_gid.y) + 0.5);
    vec4 _st_fragColor = vec4(0.0);
    mainImage(_st_fragColor, fragCoord);
    uint _flat_idx = _gid.y * _pc._u_dims.x + _gid.x;
    outputData[_flat_idx] = _st_fragColor;
}
)";

    const std::string imageShader = R"(#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform _GPGPUPushConstants {
    uvec4 _u_dims;
    vec4  iResolution;
    vec4  iMouse;
    float iTime;
    float iTimeDelta;
    int   iFrame;
    uint  _u_flags;
} _pc;

#define iResolution _pc.iResolution

const vec4 iChannelResolution[4] = vec4[4](
    vec4(16.0, 16.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0)
);

layout(set = 0, binding = 0, std430) readonly buffer _Buf_iCh0 { vec4 _iChannel0_data[]; };
layout(set = 0, binding = 1, std430) readonly buffer _Buf_iCh1 { vec4 _iChannel1_data[]; };
layout(set = 0, binding = 2, std430) readonly buffer _Buf_iCh2 { vec4 _iChannel2_data[]; };
layout(set = 0, binding = 3, std430) readonly buffer _Buf_iCh3 { vec4 _iChannel3_data[]; };
layout(set = 0, binding = 4, std430) writeonly buffer _Buf_outputData { vec4 outputData[]; };

struct _Channel0_t { int id; } iChannel0;

vec4 texture(_Channel0_t ch, vec2 uv) {
    vec2 u = clamp(uv, vec2(0.0), vec2(1.0));
    uvec2 p = clamp(uvec2(u * vec2(16.0, 16.0)), uvec2(0), uvec2(15));
    return _iChannel0_data[p.y * 16u + p.x];
}

#define buffer _usr_buf_
#define sample _usr_sample_

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec4 buffer = texture(iChannel0, uv);
    fragColor = buffer;
}

void main() {
    uvec3 _gid = gl_GlobalInvocationID;
    if (_gid.x >= _pc._u_dims.x || _gid.y >= _pc._u_dims.y) return;
    vec2 fragCoord = vec2(float(_gid.x) + 0.5, float(_gid.y) + 0.5);
    vec4 _st_fragColor = vec4(0.0);
    mainImage(_st_fragColor, fragCoord);
    uint _flat_idx = _gid.y * _pc._u_dims.x + _gid.x;
    outputData[_flat_idx] = _st_fragColor;
}
)";

    const float expectedAudioVal = std::pow(0.8f, 1.9f);
    std::vector<gpgpu::vec4> finalImage(W * H);

    // Run 3 consecutive frames executing the full chain: BufferA -> Image on the same GPU instance
    for (int frame = 0; frame < 3; ++frame) {
        // 1. Pass: BufferA
        ch0.set(bufPing);
        ch1.set(audioData);
        gpu->kernel(spectrogramShader);
        ASSERT_FALSE(gpu->hasError().has_value());
        gpu->run(W, H);
        ASSERT_FALSE(gpu->hasError().has_value());
        outBuf.getTo(bufPong);
        std::swap(bufPing, bufPong);

        // 2. Pass: Image (reads newly rendered BufferA in bufPing)
        ch0.set(bufPing);
        gpu->kernel(imageShader);
        ASSERT_FALSE(gpu->hasError().has_value());
        gpu->run(W, H);
        ASSERT_FALSE(gpu->hasError().has_value());
        outBuf.getTo(finalImage);
    }

    // After 3 frames (2 rows scrolled per frame = 6 top rows filled: y = 15, 14, 13, 12, 11, 10)
    for (uint32_t y = 10; y <= 15; ++y) {
        EXPECT_NEAR(finalImage[y * W + 0].x, expectedAudioVal, 1e-3f) << "Row y=" << y;
    }
    // Lower rows (y = 0..9) must still be 0
    for (uint32_t y = 0; y <= 9; ++y) {
        EXPECT_NEAR(finalImage[y * W + 0].x, 0.0f, 1e-4f) << "Row y=" << y;
    }
}




