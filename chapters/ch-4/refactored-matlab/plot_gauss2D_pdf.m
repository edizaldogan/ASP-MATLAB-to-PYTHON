function plot_gauss2D_pdf(mu, sigma);

%    |plot_gauss2D_pdf(mu, sigma)| plots the mean and standard
%    deviation ellipsis of the 2D Gaussian process that has mean |mu|
%    and covariance matrix |sigma|, in a 2D plot.

mu = mu(:)';
stdev = sqrtm(sigma);

t = linspace(-pi, pi, 100);
t=t(:);
X = [cos(t) sin(t)] * stdev + repmat(mu,100,1);

line(X(:,1),X(:,2),'color',[0 1 1],'linew',2);
line(mu(1),mu(2),'marker','+','markersize',10,'color',[0 1 1],'linew',2);
