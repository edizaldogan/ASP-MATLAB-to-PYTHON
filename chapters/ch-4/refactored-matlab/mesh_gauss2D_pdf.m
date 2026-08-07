function mesh_gauss2D_pdf(mu,sigma,prior,gridx,gridy,ratioz);

%   |mesh_gauss2D_pdf(mu,sigma,weight,gridx,gridy,ratioz)| plots the PDF of
%   a 2D-Gaussian PDF in a 3D plot. |mu| (1x2) is the mean of the density,
%   |sigma| (2x2) is the covariance matrix of the density. |prior| is a
%   scalar used as a multiplicative factor on the value of the PSD. |gridx|
%   and |gridy| must be vectors of the type (x:y:z) |ratioz| is the
%   (scalar) aspect ratio on the Z axis;

[xx,yy] = meshgrid(gridx,gridy);
zz = gauss_pdf( [xx(:) yy(:)], mu, sigma)*prior;
zz = reshape(zz, size(xx));

% Gaussian mesh
hm = mesh(xx,yy,zz,'facealpha',0.7);
view(3);
cm=colormap(copper);
colormap(flipud(cm));

set(gca,'xlim', [gridx(1) gridx(end)], ...
     'ylim', [gridy(1) gridy(end)], ...
     'dataAspectRatio', [1 1 ratioz] );

hold on;
grid on;
rotate3d on;

% - Gaussian contour
[c,hc] = contour3(xx,yy,zz,'k-');
set(hc,'linewidth',2);

hold off;