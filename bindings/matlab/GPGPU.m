classdef GPGPU < handle
    % GPGPU Object-Oriented MATLAB wrapper for GPGPU.VULKAN
    % Example usage:
    %   gpu = GPGPU(0);
    %   remoteIn = gpu.createData('varNameIn', single(rand(800, 600)));
    %   remoteOut = gpu.createData('outputData');
    %   gpu.kernel('kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }');
    %   gpu.run(800, 600);
    %   res = remoteOut.get();
    %   gpu.dispose();

    properties (Access = private)
        isInitialized = false;
    end

    methods
        function obj = GPGPU(device)
            if nargin < 1
                gpgpu_mex('init');
            else
                gpgpu_mex('init', device);
            end
            obj.isInitialized = true;
        end

        function data = createData(obj, name, localData, interp)
            if nargin < 3
                h = gpgpu_mex('createData', name);
            elseif nargin < 4
                h = gpgpu_mex('createData', name, single(localData));
            else
                h = gpgpu_mex('createData', name, single(localData), interp);
            end
            data = GPGPUData(h);
        end

        function kernel(obj, code)
            gpgpu_mex('kernel', code);
        end

        function run(obj, width, height, depth)
            if nargin < 3
                gpgpu_mex('run', width);
            elseif nargin < 4
                gpgpu_mex('run', width, height);
            else
                gpgpu_mex('run', width, height, depth);
            end
        end

        function err = hasError(obj)
            err = gpgpu_mex('hasError');
            if isempty(err)
                err = [];
            end
        end

        function dispose(obj)
            if obj.isInitialized
                gpgpu_mex('dispose');
                obj.isInitialized = false;
            end
        end

        function delete(obj)
            obj.dispose();
        end
    end
end
