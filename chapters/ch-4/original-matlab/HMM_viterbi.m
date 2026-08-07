function [state,likelihood] = HMM_viterbi(transition,emission)  

%    |[state,likelihood] = HMM_viterbi(transition,emission)| performs 
%    the Viterbi search (log version) of the best state sequence for 
%    a Hidden Markov Model. 
%
%    |transition| : (K+2)x(K+2) matrix of transition probabilities,
%                   first and last rows correspond to initial and
%                   final (non-emitting) states.
%    |emission|   : NxK matrix of state-conditional emission
%                   probabilities corresponding to a given sequence
%                   of observations of length N.
%    |state|      : (Nx1) vector of state-related indexes of best sequence. 
%    |likelihood| : best sequence likelihood.

warning('Off','MATLAB:log:logOfZero');

% Initialization 
[N,K] = size(emission);
delta = log(zeros(N,K));
psi = zeros(N,K);
for k=1:K
   delta(1,k) = transition(1,k+1) + emission(1,k); 
   psi(1,k) = 1;
end

% Recursion
for n=2:N
    for k=1:K
        [mx,id] = max(delta(n-1,:) + transition(2:K+1,k+1)');
        delta(n,k) = mx + emission(n,k);
        psi(n,k) = id;
    end     
end

% Termination
[mx,id] = max(delta(n,:) + transition(2:K+1,K+2)');
likelihood = mx;
state(N) = id;

% Backtracking
for n=N-1:-1:1
    state(n) = psi(n+1,state(n+1));
end
 