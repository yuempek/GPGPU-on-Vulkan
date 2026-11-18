classdef GPGPUData < handle
    % GPGPUData Data handle representing GPU buffer in MATLAB

    properties (Access = private)
        handle = 0;
    end

    methods
        function obj = GPGPUData(h)
            obj.handle = h;
        end

        function out = get(obj)
            if obj.handle == 0
                error('Data handle is invalid or disposed.');
            end
            out = gpgpu_mex('get', obj.handle);
        end
    end
end
