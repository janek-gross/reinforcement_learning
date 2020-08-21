classdef f < handle
    % local function approximation base class 
    properties
        W % learnable function parameters
    end
    
    methods
        function obj = f()
        end
        
        function value = call(obj,x)
        end
        
        function grad_val = gradient(obj,x)
        end
    end
end