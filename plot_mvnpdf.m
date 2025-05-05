function plot_mvnpdf(mux, muy, sigmax, sigmay, rho)
if abs(rho) >= 1
    error('rho must be in [-1, 1]')
end
mu = [mux muy];
sigma = [sigmax^2 rho*sigmax*sigmay; rho*sigmax*sigmay sigmay^2];
n = 100;
sigmaavg=mean([sigmax sigmay]);
mystartx = -10;
myendx = 10;
x = linspace(mystartx, myendx, n);
mystarty = -10;
myendy = 10;
y = linspace(mystarty, myendy, n);

z = zeros(n);

for k = 1:n
    z(k, :) = mvnpdf([x' y(k)*ones(n, 1)], mu, sigma);
end
% 
% subplot(221)
% plot(x, normpdf(x, mux, sigmax))
% xlabel('Density function of X')
% 
% subplot(222)
% plot(y, normpdf(y, muy, sigmay))
% xlabel('Density function of Y')
% 
subplot(121)
surf(x, y, z)
colormap('jet')
xlabel('X')
ylabel('Y')
title('Graph of the joint density of X and Y')

[X,Y] = meshgrid(x,y);
for i = 1:length(x)
    for j = 1:length(y)
        Z(i,j) = exp(-(1/(2*(1-rho^2)))*(((x(i)-mux)/sigmax)^2+...
            ((y(j)-muy)/sigmay)^2-...
            2*rho*(x(i)-mux)*(y(j)-muy)/(sigmax*sigmay)))/...
            (2*pi*sigmax*sigmay*sqrt(1-rho^2));
                 
    end
end
subplot(122)
contour(Y,X,Z)
xlabel('X')
ylabel('Y')
title('Contour plot of the joint density of X and Y')
end
