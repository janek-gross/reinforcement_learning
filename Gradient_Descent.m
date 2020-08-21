classdef Gradient_Descent < optimizer
    % gradient descent optimizer class
    properties
        learning_rate
    end
    methods
        function obj = Gradient_Descent(learning_rate)
            obj.learning_rate = learning_rate;
        end
        
        function update_W(obj, f, X, Y)
            % gradient step
            update =  +obj.learning_rate*(Y-f.call(X)).*f.gradient(X);
            if mean(abs(update)) > 1e9
                % sanity check to detect diverging training
                disp(X)
                disp(f.gradient(X))
                disp(Y)
                disp(f.W)
                error("very large update")
            else
                f.W = f.W +update;
                if f.W > 0.00001
                    % sanity check to detect impossible cost value
                    disp(f.W)
                    disp(X)
                    disp(Y)
                    error("positive cost")
                end
            end
        end
        
    end
end
    