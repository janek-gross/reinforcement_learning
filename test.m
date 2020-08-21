clear
clc

%Problem parameters

dt = 0.001; % 1000 simulation steps per second
m = 1; % pendulum mass in kg
l = 1; % pendulum length
g = 9.8; % gravity constant
mu = 0.01; % friction
angle_range = [-pi,pi]; % at 0 the pendulum is in the upright position
velocity_range = [-2*pi,2*pi]; % pendulum velocity is capped at revolution per second
tau_range = [-5,5]; % maximum applicable torque is 5 Nm


%Q_learning parameters

a_q = 1; % learning rate parameter
b_q = 10; % learning rate parameter
a_var = 0.001; % learning rate parameter for variance
b_var = 5; % learning rate parameter for variance

epsilon = 0.2; % exploration rate
gamma = 0.8; % reward discount factor
n_steps = 500; % 500 steps=50 sec. per epoch. Each step takes 0.1s.
thr_Q = 0.1; % Q difference for splitting regions
thr_S_squared = 2; % variance threshold for splitting regions
thr_n_i = 30; % minimum number of samples in regions for splitting
initial_div = 50; %number of initial number regions in the state-action space
domain_min = [angle_range(1), velocity_range(1), tau_range(1)]; % domain limits of the state-action space
domain_max = [angle_range(2), velocity_range(2), tau_range(2)];



%------Q-learning------

%Training episode
draw_pend = false; % plotting slows down learning
n_training_epochs = 100;
n_evaluation_runs = 1;
r_sum_vector = []; % cumulative reward
n_parts = []; % 

r_sum_vector = [];
n_parts = [];
for n_e=1:n_evaluation_runs
    %initialize Q and s and tau
    Q_train = Q(initial_div, domain_min, domain_max);
    init_s = [pi 0];
    tau = 0;

    r_sum_vector(end+1, 1) = 0;
    n_parts(end+1, 1) = 0;
    for n_t=1:n_training_epochs
        fprintf('%d of %d training episodes_completed\n\n', n_t,n_training_epochs);

        learning = true;
        Q_learning(learning, draw_pend, Q_train, init_s, dt, mu, m, g, l, epsilon, gamma, n_steps, thr_n_i, thr_S_squared,a_q,b_q,a_var,b_var)
        Q_train

        learning = false; % evaluation run
        reward_sum = Q_learning(learning, draw_pend, Q_train, init_s, dt, mu, m, g, l, epsilon, gamma, n_steps, thr_n_i, thr_S_squared,a_q,b_q,a_var,b_var);
        draw_pend = false;
        %Save best strategy
        if reward_sum > max(r_sum_vector(end,:))
            disp('new best Q')
            save best_Q Q_train;
        end

        r_sum_vector(end,n_t) = reward_sum;
        n_parts(end,n_t) = Q_train.n;

    end
end

% %Plot Reward sum per training episode
% figure()
% plot(1:length(r_sum_vector), mean(r_sum_vector))
% hold on
% plot(1:length(r_sum_vector), mean(r_sum_vector)+std(r_sum_vector))
% plot(1:length(r_sum_vector), mean(r_sum_vector)-std(r_sum_vector))
% %title(function_approximator + "function approximator")
% xlabel("Average reward and standard deviation")
% hold off
% 
% %Plot Reward sum per training episode
% figure()
% plot(1:length(r_sum_vector), mean(n_parts))
% %title(function_approximator + "function approximator")
% xlabel("Averagy number of parts per episode")




% Somehow calling figure(5) only one time will result in the last figure being
% overwritten.
%figure(5)
%figure(5)

%Run learned policy
learning = false;
draw_pend = true;
n_steps = 500;
Q_learning(learning, draw_pend, Q_train, init_s, dt, mu, m, g, l, epsilon, gamma, n_steps, thr_n_i, thr_S_squared,a_q,b_q,a_var,b_var)


learning = false;
draw_pend = true;
n_steps = 500;
Q_struct = load('best_Q.mat');
best_Q = Q_struct.Q_train;
Q_learning(learning, draw_pend, best_Q, init_s, dt, mu, m, g, l, epsilon, gamma, n_steps, thr_n_i, thr_S_squared,a_q,b_q,a_var,b_var)

