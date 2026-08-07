function [states,log_likehood]= HMM_gauss_viterbi(data,hmm)

%    |[states,log_likelihood] = HMM_gauss_viterbi(data,hmm)| returns the
%    best state sequence and the associated log likelihood of a sequence
%    of feature vectors for a Gaussian Hidden Markov Model (i.e., a single,
%    possibly multivariate, Gaussian probability density function per
%    state) based on the Viterbi alogrithm (log version). 
%
%    |data|           : (NxD) sequence of features vectors.
%    |hmm|            : structure of HMM parameters.
%    |hmm.means|      : (1xK) cell array of state-conditionnal (1xD) mean
%                       vectors. 
%    |hmm.covs|       : (1xK) cell array of state-conditional (DxD) 
%                       covariance matrices.  
%    |hmm.trans|      : (K+2)x(K+2) matrix of state transition probabilities 
%                       (including initial and final non-emitting states).
%    |states|         : (Nx1) vector of state-related indexes of best sequence. 
%    |log_likelihood| : best sequence likelihood.

warning('Off','MATLAB:log:logOfZero'); 

N = size(data,1);
K = size(hmm.trans,1) - 2;      % number of states (excluding I and F)
transition = log(hmm.trans);
emission = zeros(N,K);
for k=1:K
    emission(:,k) = log(gauss_pdf(data,hmm.means{k+1},hmm.covs{k+1}));
end
[states,log_likelihood] = HMM_viterbi(transition,emission);
