% MATLAB komut penceresinde bir kez 
% addpath('deploy/lib');
% addpath('deploy/bindings/matlab');
% setenv('MW_MINGW64_LOC','C:\msys64\mingw64')
% clear mex
% mex -setup C++
% mex -Ideploy/include -Ldeploy/lib -lgpgpu_vulkan.dll deploy/bindings/matlab/gpgpu_mex.cpp
% projectDir = 'C:\Users\X\Desktop\_PROJECTS\GPU\GPGPU.VULKAN.MATLAB.TEST';
% dllDir = fullfile(projectDir, 'deploy', 'lib');
% setenv('PATH', [dllDir ';' getenv('PATH')]);

clear; clc;

filename = 'sdf3D.glsl';
kernel = fileread(filename);

tic
gpu = GPGPU(0);

remoteOut = gpu.createData('fragColor');

gpu.kernel(kernel);
gpu.run(8000,6000);

B = remoteOut.get();
toc
gpu.dispose();