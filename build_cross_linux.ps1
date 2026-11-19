param(
    [ValidateSet("all", "x86_64", "aarch64", "armhf", "x86")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

$CmakeBin  = Join-Path $ProjectRoot "tools\cmake\bin\cmake.exe"
$NinjaBin  = Join-Path $ProjectRoot "tools\ninja\ninja.exe"
$PythonBin = Join-Path $ProjectRoot "tools\python\python.exe"
$ZigBin    = Join-Path $ProjectRoot "tools\zig\zig.exe"
$Toolchain = Join-Path $ProjectRoot "cmake\toolchains\linux-cross.cmake"

if (!(Test-Path $ZigBin)) {
    Write-Error "Zig compiler not found at $ZigBin. Please ensure tools/zig is extracted."
}

$TargetsToBuild = @()
if ($Target -eq "all") {
    $TargetsToBuild = @("x86_64", "aarch64", "armhf", "x86")
} else {
    $TargetsToBuild = @($Target)
}

$DeployLib = Join-Path $ProjectRoot "deploy\lib"
New-Item -ItemType Directory -Force -Path $DeployLib | Out-Null

foreach ($t in $TargetsToBuild) {
    $sysProcessor = $t
    $archFolder = "linux-$t"
    if ($t -eq "armhf") { $sysProcessor = "arm" }

    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "==> Cross-compiling GPGPU.VULKAN for Linux $t ($sysProcessor)..." -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Cyan

    $buildDir = Join-Path $ProjectRoot "build\cross-$t"
    $targetDeployDir = Join-Path $DeployLib $archFolder
    New-Item -ItemType Directory -Force -Path $targetDeployDir | Out-Null

    # Configure
    & $CmakeBin -B $buildDir -G Ninja `
        -DCMAKE_MAKE_PROGRAM="$NinjaBin" `
        -DCMAKE_TOOLCHAIN_FILE="$Toolchain" `
        -DCMAKE_SYSTEM_PROCESSOR="$sysProcessor" `
        -DCMAKE_BUILD_TYPE="Release" `
        -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON `
        -DPython3_EXECUTABLE="$PythonBin" `
        -DGPGPU_BUILD_SHARED=ON `
        -DGPGPU_BUILD_TESTS=OFF `
        -DGPGPU_BUILD_EXAMPLES=OFF `
        -DGPGPU_BUILD_BENCHMARKS=OFF
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Build
    & $CmakeBin --build $buildDir --target gpgpu_vulkan_shared
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Deploy
    $soFile = Join-Path $buildDir "libgpgpu_vulkan.so"
    if (!(Test-Path $soFile)) {
        $soFile = Join-Path $buildDir "gpgpu_vulkan.so"
    }

    Copy-Item $soFile (Join-Path $targetDeployDir "libgpgpu_vulkan.so") -Force
    if ($t -eq "x86_64") {
        Copy-Item $soFile (Join-Path $DeployLib "libgpgpu_vulkan.so") -Force
    }

    Write-Host "==> Successfully deployed: $(Join-Path $targetDeployDir "libgpgpu_vulkan.so")" -ForegroundColor Green
}

# Sync headers and bindings to deploy
Copy-Item (Join-Path $ProjectRoot "include\gpgpu\*") (Join-Path $ProjectRoot "deploy\include\gpgpu") -Recurse -Force
Copy-Item (Join-Path $ProjectRoot "bindings\*") (Join-Path $ProjectRoot "deploy\bindings") -Recurse -Force

Write-Host "==============================================================================" -ForegroundColor Green
Write-Host "==> All requested Linux shared libraries deployed successfully!" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
