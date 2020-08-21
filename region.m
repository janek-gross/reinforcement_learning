classdef region < handle
    % node class for the bipartide space tree
    properties
        region_min % region boundaries
        region_max
        var % variance of the state-action values influences the
            % learning rate in this region
        n   % number of state-action value samples for this region
        left = [] % left child node
        right = [] % right child node
        split_dim = 0 % dimension along which to biparte the state-action space
        fun % local action-value function
        points_history % can be used to save all samples.
        optimizer % optimizer for the local action-value function
    end
    
    methods
        function obj = region(region_min, region_max, n, optimizer)
            % constructor
            obj.region_min = region_min;
            obj.region_max = region_max;
            obj.n = n;
            obj.fun = f_fourier(length(obj.region_max));
            obj.var = f_fourier(length(obj.region_max));
           
            obj.points_history = [];
            obj.optimizer = optimizer;
        end
        
        function split(obj)
            % biparte function creates child regions. The state-action
            % space is biparted along the longest dimension.
            if obj.split_dim ~= 0
                error("Tried to split non-leaf node")
            end
            region_length = obj.region_max-obj.region_min;
            [~,split_dim] = max(region_length);
            obj.split_dim = split_dim;
            new_length = region_length(split_dim)/2;
            
            left_min = obj.region_min;
            left_max = obj.region_max;
            left_max(split_dim) = left_max(split_dim) - new_length;
            
            
            right_min = obj.region_min;
            right_min(split_dim) = right_min(split_dim) + new_length;
            right_max = obj.region_max;
       
            obj.left  = region(left_min, left_max, obj.n/2, Gradient_Descent(0.1));
            obj.left.fun.W = obj.fun.W;   
            obj.left.var.W = obj.var.W;
            obj.right = region(right_min, right_max, obj.n/2, Gradient_Descent(0.1));
            obj.right.fun.W = obj.fun.W;
            obj.left.var.W = obj.var.W;
            
            if isempty(obj.points_history)==false
                % distribute all samples of the parent node if samples are
                % saved
                left_history_rows = any(obj.points_history(:,split_dim)<=left_max(split_dim),2);
                left_history = obj.points_history(left_history_rows,:);
                obj.left.points_history = left_history;
                
                right_history_rows = any(obj.points_history(:,split_dim)>=right_min(split_dim),2);
                right_history = obj.points_history(right_history_rows,:);
                obj.right.points_history = right_history;
            end
        end
    end
end

