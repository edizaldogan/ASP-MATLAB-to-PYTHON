function  [new_means,new_covs,new_priors,distortion]= ...
    kmeans(data,n_iterations,means)

% KMEANS K-means algorithm 
%
%    |[new_means,new_covs,new_priors,distortion]= ...
%    kmeans(data,n_iterations,n_clusters)| , where  |data| is the matrix of
%    observations (one observation per row) and |n_clusters| is the desired
%    number of clusters, returns the mean vectors, covariance matrices, and
%    priors of k-means clusters. |distortion| is an array of values (one
%    per iteration) of sum of squared distances between the data and the
%    mean of their cluster. The clusters are initialized with a heuristic
%    that spreads them randomly around |mean(data)|. The algorithm iterates
%    until convergence is reached or the number of iterations exceeds
%    |n_iterations|. 
%
%    |kmeans(data,n_iterations,means)|, where |means| is a cell array
%    containing initial mean vectors, makes it possible to initialize
%    means. 
%
%    Example: for two clusters
%     means{1} = [1 2]; means{2} = [3 4]; kmeans(data,100,means);

% Init means
[numPts,dim] = size(data);
if ~iscell(means)
  numClust=means;
  for i=1:numClust,
    new_means{i} = randn(1,dim) * sqrtm(cov(data)) + mean(data);
  end;
else
  numClust = length(means);
  new_means = means;
end;
  
% Init clusters
whereBefore = zeros(1,numPts);
distortion = [];  

% Iterate
for k=1:n_iterations

  % Screen output
  if rem(k,2) ==0
      fprintf('iteration %3d\n', k)
  end;
 
  % Update clusters
  for i=1:numClust,
    x = data - repmat(new_means{i},numPts,1);
    sq_dist(i,:) = sum( (x .* x)' );
  end;
  [min_sq_dist,whereMin] = min(sq_dist);
  distortion = [distortion sum(min_sq_dist)];
      
  % Check for convergence
  if all(all(whereBefore == whereMin)),
      fprintf('convergence reached\n')
      break;
  else   
      whereBefore = whereMin;
  end;

  % Update means
  for i=1:numClust,
    subData = data(whereMin==i,:);
    new_means{i} = mean( subData );
    new_covs{i} = cov( subData );
    new_priors(i) = size(subData,1) / numPts;
  end;

end;