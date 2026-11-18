#include <gpgpu/gpgpu.hpp>
#include <gpgpu/gpgpu_c_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>

using namespace gpgpu;

// ============================================================================
// Phase 1 Unit Tests: Device Selection & Hardware Discovery
// ============================================================================

TEST(DeviceSelectionTest, EnumerateDevicesNonEmpty) {
    auto devices = GPU::devices();
    ASSERT_FALSE(devices.empty()) << "Expected at least one Vulkan device to be enumerated.";

    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        EXPECT_EQ(d.index, static_cast<uint32_t>(i));
        EXPECT_FALSE(d.name.empty());
        EXPECT_FALSE(d.vendor.empty());
        EXPECT_FALSE(d.driverVersion.empty());
        EXPECT_FALSE(d.apiVersion.empty());
        EXPECT_GT(d.totalMemoryBytes, 0u);
    }
}

TEST(DeviceSelectionTest, SelectDefaultDeviceAuto) {
    GPU gpu;
    EXPECT_FALSE(gpu.hasError()) << (gpu.hasError() ? gpu.hasError()->message : "");

    const DeviceInfo info = gpu.info();
    EXPECT_FALSE(info.name.empty());
    EXPECT_FALSE(info.vendor.empty());

    const Capabilities caps = gpu.capabilities();
    EXPECT_GT(caps.maxWorkGroupInvocations, 0u);
    EXPECT_GT(caps.maxWorkGroupSize[0], 0u);
    EXPECT_GT(caps.maxWorkGroupSize[1], 0u);
    EXPECT_GT(caps.maxWorkGroupSize[2], 0u);
    EXPECT_GT(caps.maxWorkGroupCount[0], 0u);
    EXPECT_GT(caps.maxStorageBufferRange, 0u);
}

TEST(DeviceSelectionTest, SelectDeviceByIndexZero) {
    GPU gpu(0);
    EXPECT_FALSE(gpu.hasError()) << (gpu.hasError() ? gpu.hasError()->message : "");
    EXPECT_EQ(gpu.info().index, 0u);
}

TEST(DeviceSelectionTest, SelectDeviceByInvalidIndexReturnsErrorWithoutCrashing) {
    GPU gpu(999);
    auto err = gpu.hasError();
    ASSERT_TRUE(err.has_value());
    EXPECT_EQ(err->code, GPUError::Code::DeviceNotFound);
    EXPECT_FALSE(err->message.empty());

    // Calling operations on invalid GPU should fail safely without crashing
    gpu.run(10, 10);
    EXPECT_TRUE(gpu.hasError().has_value());
}

TEST(DeviceSelectionTest, SelectDeviceByValidNameSubstring) {
    auto devices = GPU::devices();
    ASSERT_FALSE(devices.empty());

    // Use a substring of the first detected device's name
    std::string fullName = devices[0].name;
    std::string sub = fullName.substr(0, std::min<size_t>(fullName.size(), 4));

    GPU gpu(sub);
    EXPECT_FALSE(gpu.hasError()) << (gpu.hasError() ? gpu.hasError()->message : "");
    EXPECT_FALSE(gpu.info().name.empty());
}

TEST(DeviceSelectionTest, SelectDeviceByInvalidNameReturnsErrorWithoutCrashing) {
    GPU gpu("OlmayanKartAdi_NonExistentDevice_XYZ");
    auto err = gpu.hasError();
    ASSERT_TRUE(err.has_value());
    EXPECT_EQ(err->code, GPUError::Code::DeviceNotFound);
    EXPECT_FALSE(err->message.empty());
}

TEST(DeviceSelectionTest, C99DeviceEnumerationApi) {
    const int count = gpgpu_get_device_count();
    EXPECT_GT(count, 0);

    char nameBuf[256] = {0};
    int res = gpgpu_get_device_name(0, nameBuf, sizeof(nameBuf));
    EXPECT_EQ(res, 0);
    EXPECT_GT(std::strlen(nameBuf), 0u);

    // Invalid index check
    EXPECT_NE(gpgpu_get_device_name(9999, nameBuf, sizeof(nameBuf)), 0);
    EXPECT_NE(gpgpu_get_device_name(-1, nameBuf, sizeof(nameBuf)), 0);
}
