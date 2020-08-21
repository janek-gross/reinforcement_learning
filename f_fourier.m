classdef f_fourier < f
    % local function approximation by a linear combination of trigonometric
    % functions
    properties
        input_dim
        k=3; % number of frequencies
    end
    
    methods
        function obj = f_fourier(input_dim)
            obj.input_dim = input_dim;
            obj.W = zeros(obj.k,obj.k,obj.k,8); % learnable parameters
        end
        
        function value = call(obj, x)
            % three dimensional fourier basis 
            value = 0;
            for frequ_1 = 1:obj.k
                for frequ_2 = 1:obj.k
                    for frequ_3 = 1:obj.k
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,1)*cos(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,2)*cos(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,3)*cos(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,4)*cos(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,5)*sin(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,6)*sin(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,7)*sin(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        value = value + obj.W(frequ_1,frequ_2,frequ_3,8)*sin(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                    end
                end
            end
        end
        
        function grad_val = gradient(obj,x)
            % gradient function for gradient descent optimization
            grad_val = zeros(obj.k,obj.k,obj.k);
            for frequ_1 = 1:obj.k
                for frequ_2 = 1:obj.k
                    for frequ_3 = 1:obj.k
                        grad_val(frequ_1,frequ_2, frequ_3,1) = cos(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        grad_val(frequ_1,frequ_2, frequ_3,2) = cos(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                        grad_val(frequ_1,frequ_2, frequ_3,3) = cos(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        grad_val(frequ_1,frequ_2, frequ_3,4) = cos(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                        grad_val(frequ_1,frequ_2, frequ_3,5) = sin(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        grad_val(frequ_1,frequ_2, frequ_3,6) = sin(pi*x(1)*frequ_1)*cos(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                        grad_val(frequ_1,frequ_2, frequ_3,7) = sin(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*cos(pi*x(3)*frequ_3);
                        grad_val(frequ_1,frequ_2, frequ_3,8) = sin(pi*x(1)*frequ_1)*sin(pi*x(2)*frequ_2)*sin(pi*x(3)*frequ_3);
                    end
                end
            end
            
        end
    end
end