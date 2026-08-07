function [] = plot_GMM2D(data,means,covs)

% |plot_GMM2D(data, means, covs)| shows the standard deviation ellipsis of the Gaussian 
% components of a GMM defined by |means| and |covs|, on a 2D plot, together with |data| samples.

n_gaussians = length(means);
  
cla;
line(data(:,1),data(:,2), ...
    'linestyle','none','marker','.');
for j=1:n_gaussians
    plot_gauss2D_pdf(means{j},covs{j})
end;
