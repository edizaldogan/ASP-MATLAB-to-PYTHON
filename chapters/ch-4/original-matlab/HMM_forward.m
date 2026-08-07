function [alpha,likelihood] = HMM_forward(transition,emission)  

%    |[alpha,likelihood] = HMM_forward(transition,emission)| performs
%    the forward recursion (log version) for a Hidden Markov Model.
%
%    |transition| : (K+2)x(K+2) matrix of transition probabilities,
%                   first and last rows correspond to initial and
%                   final (non-emitting) states.
%    |emission|   : NxK matrix of state-conditional emission
%                   probabilities corresponding to a given sequence
%                   of observations of length N.
%    |alpha|      : NxK matrix of state-related forward variables.
%    |likelihood| : full likelihood.

warning('Off','MATLAB:log:logOfZero');

% Initialization 
[N,K] = size(emission);
alpha = log(zeros(N,K));
for k=1:K
    alpha(1,k) = transition(1,k+1) + emission(1,k);
end

% Recursion
for n=2:N
    for k=1:K
        alpha(n,k) = logsum(alpha(n-1,:) + transition(2:K+1,k+1)') + emission(n,k);
    end     
end

% Termination
likelihood = logsum(alpha(N,:) + transition(2:K+1,K+2)');
