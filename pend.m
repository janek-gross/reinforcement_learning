function ds = pend(s,mu,m,l,g,tau)
% pendulum dynamics function
% state s (angle and angular velocity)
% fricton mu
% mass m
% length l
% gravity constant g
% applied torque tau

ds(1) = s(2);
ds(2) = (-s(2)*mu)/(m*l*l) + (sin(s(1))*g)/l + tau/(m*l*l);
end