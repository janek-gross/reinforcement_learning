classdef Q < handle
    % Q function class.
    % The Q faunction assigns a value to each point in the state-action
    % space
    % 
    % This Q function uses Binary Space Partitioning on the state-action
    % space.
    % The local Q function in each partition is approximated by a set
    % fourier basis functions.
    % The weight of each basis function is learned.
    
    properties
        root_node % root node of the bipartide space graph
        domain_min % Domain boundaries need to match the statespace of the pendulum
        domain_max %
        domain_length %
        a_slice % helper variable for the retrieval of the best action of all actions in a given pendulum state
        Q_slice % helper variable for the retrieval of the highest state-action value in a given pendulum state
        region_slice = {} % helper variable to find the best state-action region in a given pendulum state
        n=1; % number of regions in the bipartide space tree
    end
    
    methods
        function obj = Q(region_division, domain_min, domain_max)
            % constructor
            % region_division starting regions
            % domain_min, domain_max as domain boundaries
            obj.domain_min = domain_min;
            obj.domain_max = domain_max;
            obj.domain_length = domain_max - domain_min;
            
            obj.root_node = region([0 0 0], [1 1 1], 0, Gradient_Descent(0.1));
            regions = [obj.root_node];
            for i = 1:region_division
                regions(i).split()
                obj.n = obj.n+1;
                regions(end+1) = regions(i).left;
                regions(end+1) = regions(i).right;
            end
        end
        
        function [s_out, a_out] = norm(obj, s, a)
            % Normalizes the state-action space to the interval [0,1].
            % While not the most efficient, a normalized state-action
            % spaces simplifies calculations and function approximation.
            vals = [s,a] - obj.domain_min;
            vals = vals./obj.domain_length;
            if any(vals < 0) || any(vals > 1)
                error("undefined state");
            end
            s_out = vals(1:2);
            a_out = vals(3);
        end
        
        function [s_out, a_out] = unnorm(obj, s, a)
            % rescale the interval [0,1] to the state-action space
            vals = [s,a].*obj.domain_length;
            vals = vals + obj.domain_min;
            if any(vals < obj.domain_min) || any(vals > obj.domain_max)
                disp(vals)
                error("undefined state");
            end
            s_out = vals(1:2);
            a_out = vals(3);
        end
        
        function split_region(~, region)
            region.split();
        end
        
        function region = region(obj,s,a)
            % tree traversal to retrieve the desired region in the
            % bipartide space graph
            [s, a] = obj.norm(s,a);
            point = [s,a];
            next = obj.root_node;
            while next.split_dim ~= 0
                if point(next.split_dim) >= next.left.region_min(next.split_dim) && point(next.split_dim) <= next.left.region_max(next.split_dim)
                    next = next.left;
                elseif point(next.split_dim) >= next.right.region_min(next.split_dim) && point(next.split_dim) <= next.right.region_max(next.split_dim)
                    next = next.right;
                else
                    disp(s)
                    disp(a)
                    disp(next)
                    disp(error("child node region not within parent node region"))
                end
            end
            region = next;
        end
        
        function [max_Q,argmax_Q] = max_Q(obj,s)
            % retreive the maximum state-action value max_Q and the
            % corresponding action given a state s
            [s,~] = obj.norm(s,0);
            obj.a_slice = [];
            obj.Q_slice = [];
            obj.region_slice = [];
            obj.slice(obj.root_node, s);
            [max_Q, arg] = max(obj.Q_slice);
            
            argmax_Q = obj.a_slice(arg);
            [~, argmax_Q] = obj.unnorm([0 0], argmax_Q);
            if max_Q > 1e9 || max_Q < -1e9
                disp(max_Q)
                error("very large Q")
            end
        end
        
        function slice(obj, region, s)
            % helper function that retrieves all state-actions regions, state-action values and actions given a state.
            if region.split_dim == 3
                obj.slice(region.left,  s);
                obj.slice(region.right, s);
            elseif region.split_dim > 0 && s(region.split_dim) >= region.left.region_min(region.split_dim) && s(region.split_dim) <= region.left.region_max(region.split_dim)
                obj.slice(region.left,  s);
            elseif region.split_dim > 0 && s(region.split_dim) >= region.right.region_min(region.split_dim) && s(region.split_dim) <= region.right.region_max(region.split_dim)
                obj.slice(region.right, s);
            elseif region.split_dim == 0
                obj.a_slice(end+1) = region.region_min(end)+ (region.region_max(end) - region.region_min(end))/2;% + rand()*(region.region_max(end)-region.region_min(end));
                obj.Q_slice(end+1) = region.fun.call([s,obj.a_slice(end)]);
                obj.region_slice{end+1} = region;
                return
            else
                disp(region)
                disp(s)
                error("something went wrong")
            end
        end
        
        function update_Q(obj,region,s,a,q)
            % optimization step to learn the correct state-action value q
            [s,a]=obj.norm(s,a);
            region.optimizer.update_W(region.fun,[s,a],q);        
        end
        
        
        function plot_region_2D(obj, region, dim1, dim2)
            % visualization of 2 dimensions of the bipartide space graph starting
            % at node region
            low = region.region_min;
            up  = region.region_max;
            if region.split_dim ~= 0                
                half_diff = (region.region_max(region.split_dim)-region.region_min(region.split_dim))/2;
                low(region.split_dim) = low(region.split_dim) + half_diff;
                up(region.split_dim)  =  up(region.split_dim) - half_diff;
                obj.plot_rectangle(low([dim1 dim2]), up([dim1 dim2]));
                obj.plot_region_2D(region.left,  dim1, dim2)
                obj.plot_region_2D(region.right, dim1, dim2)
            else
                diff = up-low;
                diff = diff([dim1 dim2]);
                verts = ([0 0;0 1;1 0;1 1]).*(diff/4) + low([dim1 dim2]) + 3/8*diff;
                face = [1 2 4 3];
                patch('Faces',face,'Vertices',verts,'FaceColor','black','EdgeColor','w', 'FaceAlpha', min(1,max(0,(region.mean/10)+1)))
            end
        end
        
        function plot_region_3D(obj, region)
            % 3D visualization of the bipartide space graph starting
            % at node region
            low = region.region_min;
            up  = region.region_max;
            if region.split_dim ~= 0
                half_diff = (region.region_max(region.split_dim)-region.region_min(region.split_dim))/2;
                low(region.split_dim) = low(region.split_dim) + half_diff;
                up(region.split_dim)  =  up(region.split_dim) - half_diff;
                obj.plot_cube(low, up);
                obj.plot_region_3D(region.left)
                obj.plot_region_3D(region.right)
            else
                diff = up-low;
                verts = ([0 0 0;0 1 0;1 1 0;1 0 0;0 0 1;0 1 1;1 1 1;1 0 1]).*(diff/4) + low + 3/8*diff;
                face = [1 2 3 4;5 6 7 8;3 4 8 7;1 2 6 5;2 3 7 6;1 4 8 5];
                patch('Faces',face,'Vertices',verts,'FaceColor','black','EdgeColor','w', 'FaceAlpha', min(1,max(0,(region.mean)+1)))
            end
        end
        
        function plot2D(obj, dim1, dim2)
            % plot helper function
            labels = ["angle", "angular velocity", "torque"];
            figure;
            obj.plot_region_2D(obj.root_node, dim1, dim2);
            xlabel(labels(dim1),'FontSize',16);
            ylabel(labels(dim2),'FontSize',16);
        end
        
        function plot3D(obj)
            % plot helper function
            region = obj.root_node;
            figure;
            obj.plot_region_3D(region);
            xlabel('angle','FontSize',16);
            ylabel('angular velocity','FontSize',16);
            zlabel('torque','FontSize',16);
        end
        function plot_rectangle(~, low, up)
            % plot helper function
            x = [low(1)  up(1);
                 low(1) low(1);
                  up(1) low(1);
                  up(1)  up(1)];
            y = [low(2) low(2);
                 low(2) up(2);
                  up(2) up(2);
                  up(2) low(2)];
            for i= 1:4
                plot(x(i,:), y(i,:), "black");
                hold on
            end
        end
        function plot_cube(~,low, up)
            % plot helper function
            x = [
             low(1)  up(1);
             low(1) low(1);
             low(1) low(1);
              up(1)  up(1);
              up(1) low(1);
              up(1)  up(1);
             low(1) low(1);
             low(1)  up(1);
             low(1) low(1);
              up(1)  up(1);
              up(1) low(1);
              up(1)  up(1)];

            y = [
              low(2) low(2);
              low(2)  up(2);
              low(2) low(2);
               up(2) low(2);
               up(2)  up(2);
               up(2)  up(2);
              low(2)  up(2);
              low(2) low(2);
              low(2) low(2);
               up(2) low(2);
               up(2)  up(2);
               up(2)  up(2)];

            z = [
              low(3) low(3);
              low(3) low(3);
              low(3)  up(3);
              low(3) low(3);
              low(3) low(3);
              low(3)  up(3);
               up(3)  up(3);
               up(3)  up(3);
               up(3) low(3);
               up(3)  up(3);
               up(3)  up(3);
               up(3) low(3)];
            for i= 1:12
                plot3(x(i,:), y(i,:), z(i,:), "black");
                hold on
            end
        end
        
    end
end
