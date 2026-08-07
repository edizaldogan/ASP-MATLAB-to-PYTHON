function [beta,likelihood] = HMM_backward(transition,emission)  

%    |[beta,likehood] = HMM_backward(transition,emission)| performs
%    the backward recursion (log version) for a Hidden Markov Model.  
%
%    |transition| : (K+2)x(K+2) matrix of transition probabilities,
%                   first and last rows correspond to initial and
%                   final (non-emitting) states.
%    |emission|   : NxK matrix of state-conditional emission
%                   probabilities corresponding to a given sequence
%                   of observations of length N.
%    |beta|       : NxK matrix of state-related backward variables.
%    |likelihood| : full likelihood.

warning('Off','MATLAB:log:logOfZero');

% Initialization 
[N,K] = size(emission);
beta = log(zeros(N,K));
for k=1:K
    beta(N,k) = transition(k+1,K+2);
end

% Recursion
for n=N-1:-1:1
    for k=1:K
        beta(n,k) = logsum(transition(k+1,2:K+1) + emission(n+1,:) + beta(n+1,:));
    end     
end
    
% Termination
likelihood = logsum(transition(1,2:K+1) + emission(1,:) + beta(1,:));
