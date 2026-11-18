param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

$CmakeBin  = Join-Path $ProjectRoot "tools\cmake\bin"
$NinjaBin  = Join-Path $ProjectRoot "tools\ninja"
$GccBin    = Join-Path $ProjectRoot "tools\w64devkit\bin"
$PythonBin = Join-Path $ProjectRoot "tools\python"

$env:PATH = "$CmakeBin;$NinjaBin;$GccBin;$PythonBin;$env:PATH"

$Preset = if ($Config -eq "Debug") { "standalone-windows-debug" } else { "standalone-windows-release" }
$BuildDir = Join-Path $ProjectRoot ("build\" + $Config.ToLower())

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning $BuildDir..."
    Remove-Item $BuildDir -Recurse -Force
}

Write-Host "==> Configuring GPGPU.VULKAN ($Preset)..." -ForegroundColor Cyan
& "$CmakeBin\cmake.exe" --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Building GPGPU.VULKAN ($Preset)..." -ForegroundColor Cyan
& "$CmakeBin\cmake.exe" --build --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Running Tests ($Preset)..." -ForegroundColor Cyan
& "$CmakeBin\ctest.exe" --preset $Preset --output-on-failure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Packaging Distributable SDK into deploy/..." -ForegroundColor Cyan
$DeployDir     = Join-Path $ProjectRoot "deploy"
$DeployLibDir  = Join-Path $DeployDir "lib"
$DeployIncDir  = Join-Path $DeployDir "include\gpgpu"
$DeployBindDir = Join-Path $DeployDir "bindings"

New-Item -ItemType Directory -Force -Path $DeployLibDir  | Out-Null
New-Item -ItemType Directory -Force -Path $DeployIncDir  | Out-Null
New-Item -ItemType Directory -Force -Path $DeployBindDir | Out-Null

# 1. Copy public C++20 and C99 headers
Copy-Item -Path (Join-Path $ProjectRoot "include\gpgpu\*") -Destination $DeployIncDir -Recurse -Force

# 2. Copy shared library (.dll) and import library (.dll.a)
$Win64Dir = Join-Path $DeployLibDir "windows-x64"
New-Item -ItemType Directory -Force -Path $Win64Dir | Out-Null
$DllPath = Join-Path $BuildDir "gpgpu_vulkan.dll"
if (Test-Path $DllPath) {
    Copy-Item -Path $DllPath -Destination (Join-Path $DeployLibDir "gpgpu_vulkan.dll") -Force
    Copy-Item -Path $DllPath -Destination (Join-Path $DeployLibDir "libgpgpu_vulkan.dll") -Force
    Copy-Item -Path $DllPath -Destination (Join-Path $Win64Dir "gpgpu_vulkan.dll") -Force
    Copy-Item -Path $DllPath -Destination (Join-Path $Win64Dir "libgpgpu_vulkan.dll") -Force
}
$ImpLibPath = Join-Path $BuildDir "libgpgpu_vulkan.dll.a"
if (Test-Path $ImpLibPath) {
    Copy-Item -Path $ImpLibPath -Destination (Join-Path $DeployLibDir "libgpgpu_vulkan.dll.a") -Force
    Copy-Item -Path $ImpLibPath -Destination (Join-Path $Win64Dir "libgpgpu_vulkan.dll.a") -Force
}

# 3. Create self-contained fat static archive (libgpgpu_vulkan_static.a) merging all internal static deps
$MergedStaticPath = (Join-Path $DeployLibDir "libgpgpu_vulkan_static.a") -replace '\\', '/'
if (Test-Path $MergedStaticPath) { Remove-Item $MergedStaticPath -Force }

$StaticArchives = @(
    (Join-Path $BuildDir "libgpgpu_vulkan_static.a"),
    (Join-Path $BuildDir "external\glslang\glslang\libglslang.a"),
    (Join-Path $BuildDir "external\glslang\glslang\libMachineIndependent.a"),
    (Join-Path $BuildDir "external\glslang\glslang\libGenericCodeGen.a"),
    (Join-Path $BuildDir "external\glslang\glslang\libglslang-default-resource-limits.a"),
    (Join-Path $BuildDir "external\glslang\glslang\OSDependent\Windows\libOSDependent.a"),
    (Join-Path $BuildDir "external\glslang\SPIRV\libSPIRV.a"),
    (Join-Path $BuildDir "external\SPIRV-Tools\source\opt\libSPIRV-Tools-opt.a"),
    (Join-Path $BuildDir "external\SPIRV-Tools\source\libSPIRV-Tools.a")
)

$MriLines = @("CREATE $MergedStaticPath")
foreach ($arch in $StaticArchives) {
    if (Test-Path $arch) {
        $normArch = $arch -replace '\\', '/'
        $MriLines += "ADDLIB $normArch"
    }
}
$MriLines += "SAVE"
$MriLines += "END"

$MriFile = Join-Path $BuildDir "merge_static.mri"
$MriLines -join "`n" | Set-Content -Path $MriFile -Encoding Ascii
cmd.exe /c "`"$GccBin\ar.exe`" -M < `"$MriFile`""
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& "$GccBin\ranlib.exe" $MergedStaticPath

# 4. Copy language bindings (Python, MATLAB, Java)
Copy-Item -Path (Join-Path $ProjectRoot "bindings\*") -Destination $DeployBindDir -Recurse -Force

Write-Host "==> Deploy package ready at: $DeployDir" -ForegroundColor Green
