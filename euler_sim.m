function s = euler_sim(s, dt, mu, m, l, g,tau)
% pendulum simulation using the improved euler method
ds = pend(s,mu,m,l,g, tau);
s = s + dt*ds;
s(1) = s(1) + 0.5*dt*dt*ds(2);

if s(1) > pi
    s(1) = s(1)-2*pi;
end

if s(1) < -pi
    s(1) = s(1)+2*pi;
end

if s(2) < -2*pi
    s(2) = -2*pi;
end
if s(2) > 2*pi
    s(2) = 2*pi;
end

end

