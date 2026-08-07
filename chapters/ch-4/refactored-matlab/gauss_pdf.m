function likelihood = gauss_pdf(x,mu,sigma);

%    |gauss_pdf(x,mu,sigma)| returns the likelihood of sample |x| (1xD)
%    with respect to a Gaussian process with mean |mu| (1xD) and covariance
%    |sigma| (DxD). When a set of samples (NxD) is provided as input, a set of
%    likelihoods is returned.

[N,D] = size(x);
x = x-repmat(mu,N,1);
tmp = sum( (x* inv(sigma))  .*x ,2 ); % is a column vector Nx1
likelihood = exp(-0.5*tmp) / sqrt((2*pi)^D * det(sigma));