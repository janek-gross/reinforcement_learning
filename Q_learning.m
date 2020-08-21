function reward_sum = Q_learning(learning, draw, Q, s, dt, mu, m, g, l, epsilon, gamma, n_steps, thr_n_i, thr_S, a_q,b_q,a_var,b_var)
    disp("###################")
    
    reward_sum=0;
    if ~learning
        epsilon=0; % no exploration
        disp('Executing learned strategy')
    else
        disp("#Running Q-learning")
    end
    
    [~,a] = Q.max_Q(s);
    %each step counts as 0.1 seconds
    for i = 1:n_steps
        
        %Select next action or exploration
        strat = rand();
        if strat < epsilon
            a = rand()*10-5; % random torque between -5 and 5
        end
        
        
        %Simulate next State each action has an interval of 0.1 sec
        %and the simulation has a dt of 0.001 sec
        snext=s;
        for sim_time=1:100
            snext = euler_sim(snext, dt, mu, m, l, g,a);
            
            if mod(sim_time,100)==0 && draw % visualization
                drawpend(snext,m,l);
            end
        end
        
        
        
        %Get reward
        r = -abs(snext(1));
        [Qmax,anext] = Q.max_Q(snext);

        if ~learning % skip the rest of the loop
            reward_sum = reward_sum + r;
            s=snext;
            a=anext;
            continue
        end
        
        %Calculate new state-action value q
        q = r + gamma*Qmax;

        %Update state-action value Q
        region = Q.region(s,a);
        region.n = region.n+1;
        
        % saving the history of all samples takes a lot of memory
        %region.points_history = [region.points_history;[s,a,q]];
        
        % adjust learning rate according to the number of samples in the
        % target region
        region.optimizer.learning_rate = 1/(a_q*region.n + b_q);
        
        % Learning step for the current region
        Q.update_Q(region,s,a,q);
        
        % adjust learning rate for the region variance
        alpha_var = 1/(a_var*region.n + b_var);
        % Learning step for the current region variance
        region.var.W =  region.var.W + alpha_var*((q - region.fun.call([s,a]))^2 - region.var.call([s,a]));

        % split regions with many samples and high variance
        split_criterion = abs(region.var.call([s,a])) > thr_S && region.n > thr_n_i;
  
        if split_criterion
            Q.split_region(region);
            Q.n = Q.n+1;         
        end
        
        s = snext;
        a = anext; 
    end
    
    disp("###################")
    fprintf("\n")
end