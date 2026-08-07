function new_hmm = HMM_train_FB(data,old_hmm,dmin,qmax)

%    |new_hmm = HMM_train_FB(data,old_hmm,dmin,qmax)| returns the Maximum
%    Likelihood re-estimation of a Gaussian Hidden Markov Model (i.e., a
%    single, possibly multivariate, Gaussian probability density function
%    per state) based on the forward-backward algorithm (aka. Baum-Welch
%    re-estimation formulas). Note that most operations are performed in
%    the log domain for accuracy. 
%   
%    |data|           : cell array of observation sequences for which the
%                       HMM is estimated. 
%    |old_hmm|        : structure of initial HMM parameters.
%    |old_hmm.means|  : (1xK) cell array of state-conditionnal (1xD) mean vectors.
%    |old_hmm.covs|   : (1xK) cell array of state-conditional (DxD)covariance matrices. 
%    |old_hmm.trans|  : (K+2)x(K+2) matrix of state transition probabilities 
%                       (including initial and final non-emitting states).
%    |dmin|           : minimum log-likelihood relative improvement until
%                       convergence. 
%    |qmax|           : maximum number of iterations until convergence.
%    |new_hmm|        : structure of final HMM estimate.
%    |new_hmm.means|  : (1xK) cell array of state-conditionnal (1xD) mean vectors.
%    |new_hmm.covs|   : (1xK) cell array of state-conditional (DxD)covariance matrices. 
%    |new_hmm.trans|  : (K+2)x(K+2) matrix of state transition probabilities 

warning('Off','MATLAB:log:logOfZero');

% Get dimensions
U = length(data);                   % number of observation sequences
D = length(old_hmm.means{2});       % dimension of observation vectors
K = size(old_hmm.trans,1) - 2;      % number of states (excluding I and F)

% Iterate until convergence
old_likelihood = -Inf;
new_likelihood = -Inf;
new_hmm = old_hmm;
for q =1:qmax

    % Initialize accumulators
    acc_means = cell(1,K);
    acc_covs = cell(1,K);
    for k=1:K
        acc_means{k} = zeros(1,D);
        acc_covs{k} = zeros(D,D);
    end
    acc_pi = zeros(1,K);
    acc_trans = zeros(K,K);
    acc_pf = zeros(1,K);
    acc_gamma = zeros(1,K);	

    % Perform E step
    for u=1:U
      N = size(data{u},1);
      emission = zeros(N,K);
      transition = log(old_hmm.trans);
      alpha = zeros(N,K);
      beta = zeros(N,K);
      xi = zeros(K,K,N-1);
      gamma = zeros(N,K);

      for k=1:K
          emission(:,k) = log(gauss_pdf(data{u},new_hmm.means{k+1},new_hmm.covs{k+1}));
      end
      [alpha,likelihood] = HMM_forward(transition,emission);
      new_likelihood = logsum([new_likelihood likelihood]);
      beta = HMM_backward(transition,emission);
      for n=1:N-1
          for k=1:K
              for l=1:K
                  xi(k,l,n) = alpha(n,k) + transition(k+1,l+1) + ... 
                      + emission(n,l) + beta(n+1,l); 
              end
          end
          tmp = xi(:,:,n);
          xi(:,:,n) = xi(:,:,n) - logsum(tmp(:));          
      end
      for n=1:N-1
          for k=1:K
              gamma(n,k) = logsum(xi(k,:,n)); 
          end
      end
      gamma(N,:) = alpha(N,:) + transition(2:K+1,K+2)' ...
          - logsum(alpha(N,:) + transition(2:K+1,K+2)');

      xi = exp(xi);
      gamma = exp(gamma);
      for k=1:K
          acc_means{k} = acc_means{k} + (gamma(:,k)' * data{u});
          tmp = data{u}-repmat(new_hmm.means{k+1},N,1);
          acc_covs{k} = acc_covs{k} + ((repmat(gamma(:,k),1,D) .* tmp)' * tmp);
      end
      acc_pi = acc_pi + gamma(1,:);
      acc_trans = acc_trans + sum(xi,3);
      acc_pf = acc_pf + gamma(N,:);
      acc_gamma = acc_gamma + sum(gamma,1); 
    end

    % Stop if convergence
    if (new_likelihood - old_likelihood < dmin*abs(old_likelihood))
        break;
    else
        old_likelihood = new_likelihood;
    end 
   
    % Perform M step
    for k=1:K
        new_hmm.means{k+1} = acc_means{k} ./ acc_gamma(k);
        new_hmm.covs{k+1} = acc_covs{k} ./ acc_gamma(k);
        new_hmm.trans(1,k+1) = acc_pi(k) / sum(acc_pi);
        new_hmm.trans(k+1,2:K+1) = acc_trans(k,:) ./ acc_gamma(k);
        new_hmm.trans(k+1,K+2) = acc_pf(k) / acc_gamma(k);
    end
    new_hmm.means{K+2} = [];
    new_hmm.covs{K+2} = [];
    new_hmm.trans(K+2,K+2) = 1;
    
    fprintf('iteration %3d / %3d (%4.2f)\n', q, qmax, new_likelihood);
end
