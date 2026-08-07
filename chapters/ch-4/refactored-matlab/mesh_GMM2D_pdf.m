function mesh_GMM2D_pdf(mus,sigmas,weights,gridx,gridy,ratioz);

%   |mesh_GMM2D_pdf(mus,sigmas,weights,gridx,gridy,ratioz)| plots the PDF
%   of a 2D Gaussian Mixture Model PDF in a 3D plot. |mus| is a cell array
%   of the (1x2) means, |sigmas| is a cell array of the (2x2) covariance
%   matrices. |weights| is a vector of Gaussian weights. |gridx| and |gridy|
%   must be vectors of the type (x:y:z) |ratioz| is the (scalar) aspect
%   ratio on the Z axis;

[xx,yy] = meshgrid(gridx,gridy);
zz = GMM_pdf( [xx(:) yy(:)], mus, sigmas,weights);
zz = reshape(zz, size(xx));

% GMM mesh
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

% GMM contour
[c,hc] = contour3(xx,yy,zz,'k-');
set(hc,'linewidth',2);

hold off;