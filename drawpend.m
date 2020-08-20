% function for the visualization of a pendulum in state s with mass m and
% length l
function drawpend(s,m,l)
th = s(1); % pendulum angle

mr = .3*sqrt(m); % pendulum radius dependends on pendulum mass

px = l*sin(th);
py = l*cos(th);

plot([-10 10],[0 0],'k','LineWidth',2) %horizon
hold on
plot([0 px],[0 py],'k','LineWidth',2) % pendulum

rectangle('Position',[px-mr/2,py-mr/2,mr,mr],'Curvature',1,'FaceColor',[.1 0.1 1]) % pendulum mass

xlim([-5 5]);
ylim([-2 2.5]);
set(gcf,'Position',[100 550 1000 400])
drawnow
hold off