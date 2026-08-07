function likelihood = GMM_pdf(x,mus,sigmas,weights);

%    |GMM_pdf(x,mus,sigmas,weights)| returns the likelihood of sample
%    |x| (1xD) with respect to a Gaussian Mixture Model. |mus| is a cell
%    array of the (1xD) means, |sigmas| is a cell array of the (DxD)
%    covariance matrices. (1xD) |weight| is a vector of Gaussian weights. 
%    When a set of samples (NxD) is provided as input, a set of likelihoods 
%    is returned.

n_gaussians=size(mus,2);
likelihood=gauss_pdf(x,mus{1},sigmas{1})*weights(1);
for i=2:n_gaussians
    likelihood=likelihood+gauss_pdf(x,mus{i},sigmas{i})*weights(i);
end;
