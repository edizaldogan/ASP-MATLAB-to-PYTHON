function  [new_means,new_covs,new_priors,total_loglike]= ...
    GMM_train(data,n_iterations,means,covs,priors)

%    |[new_means,new_sigmas,new_priors,total_loglike]= ...
%    GMM_train(data,n_iterations,n_gaussians)| returns the mean vectors,
%    covariance matrices, and priors of GMM Gaussian components obtained by
%    the EM training algorithm. |data| is the matrix of observations (one
%    observation per row) and |n_gaussians| is the desired number of
%    clusters. |total_loglike| is an array of values (one per iteration) of
%    the total likelihood of the data given the GMM model. GMMs are
%    initialized with a heuristic that spreads them randomly around
%    |mean(data)|. The algorithm iterates until convergence is reached or
%    the number of iterations exceeds |n_iterations|.    
%
%    |GMM_train(data,n_iterations,means,covs,priors)|, makes it possible to
%    initialize the means, covariance matrices, and priors of the GMM
%    components.
%
%    Example: for two gaussians
%     means{1} = [1 2]; means{2} = [3 4];
%     vars{1} = [2 0;0 2]; vars{2} = [1 0;0 1];
%     GMM_train(data,100,means,vars);

% Init GMMs
[numPts,dim] = size(data);
if ~iscell(means)
  n_gaussians=means;
  covdata=cov(data);
  meandata=mean(data);
  for i=1:n_gaussians,
    new_means{i} = randn(1,dim) * sqrtm(covdata) + meandata;
    new_covs{i}=covdata;
    new_priors(i) = 1 / n_gaussians;
   end;
else
  n_gaussians = length(means);
  new_means = means;
  new_covs = covs;
  new_priors = priors;
end;
  
% Init clusters
likelihood_before=-Inf;
total_loglike = [];  

% Iterate
for k=1:n_iterations

  % Screen output
  if rem(k,2) ==0
      fprintf('iteration %3d\n', k)
  end;
 
  % E step: update weights;
  for i=1:n_gaussians
      % Compute likelihood of the Gaussian given the data
      likelihood(i,:) = gauss_pdf(data,new_means{i},new_covs{i})*new_priors(i)';
  end;
  totLike = sum(log(sum(likelihood)));
  weights = ( likelihood ./ repmat( sum(likelihood) , n_gaussians , 1 ) )';
  sumWeights = sum(weights);
  total_loglike = [total_loglike totLike];
      
  % Check for convergence
  if (totLike-likelihood_before)<0.1
      fprintf('convergence reached\n')
      break;
  else   
      likelihood_before=totLike;
  end;

  % M step
  for i=1:n_gaussians
      W = repmat(weights(:,i),1,dim);
      new_means{i} = sum(data.*W) / sumWeights(i);
      x = ( data - repmat(new_means{i},numPts,1) );
      new_covs{i} = ((x.*W)' *  x) / sumWeights(i) ;
    end;
    new_priors = sumWeights / numPts;

end;