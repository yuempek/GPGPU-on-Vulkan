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

gpu = GPGPU(1);
A = single( linspace(0, pi, 1000) );

remoteIn  = gpu.createData('varNameIn', A(:));
remoteOut = gpu.createData('outputData');

gpu.kernel('kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }');
gpu.run(800,600);

B = remoteOut.get();
gpu.dispose();