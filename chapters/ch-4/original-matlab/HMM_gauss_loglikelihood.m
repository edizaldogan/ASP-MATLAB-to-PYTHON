function log_likelihood = HMM_gauss_logpdf(data,hmm)

%    |log_likelihood = HMM_gauss_logpdf(data,hmm)| returns the full log
%    likelihood of a sequence of feature vectors for a Gaussian Hidden
%    Markov Model (i.e., a single, possibly multivariate, Gaussian
%    probability density function per state) based on the forward algorithm
%    (log version). Note that the backward algorithm can be used as well.
%
%    |data|           : (NxD) sequence of features vectors.
%    |hmm|            : structure of HMM parameters.
%    |hmm.means|      : (1xK) cell array of state-conditionnal (1xD) mean
%                        vectors. 
%    |hmm.covs|       : (1xK) cell array of state-conditional (DxD)
%                       covariance matrices.   
%    |hmm.trans|      : (K+2)x(K+2) matrix of state transition probabilities 
%                       (including initial and final non-emitting states).
%    |log_likelihood| : full log likelihood.

warning('Off','MATLAB:log:logOfZero'); 

N = size(data,1);
K = size(hmm.trans,1) - 2;      % number of states (excluding I and F)
transition = log(hmm.trans);
emission = zeros(N,K);
for k=1:K
    emission(:,k) = log(gauss_pdf(data,hmm.means{k+1},hmm.covs{k+1}));
end
[alpha,log_likelihood] = HMM_forward(transition,emission);
%[beta,log_likelihood] = HMM_backward(transition,emission);

