function [x,stateSeq] = HMM_generate(means,covs,transitions)

%    |[x,stateseq] = HMM_generate(means,covs,transitions)| returns a
%    D-dimensional sequence |x| (one observation per row) and a state
%    sequence |stateseq| drawn from a Markov model, given the |means| and
%    |covs| of the Gaussian multivariates describing its emission
%    probabilities (and stored in lists with empty matrices as first and
%    last elements to symbolize the entry and exit states), and given its
%    |transition| matrix.

dim = length(means{2});
numStates = length(means);

for i=2:(numStates-1),
  stDevs{i} = sqrtm(covs{i});
end;
stDevs{1} = [];
stDevs{numStates} = [];

% Generate the emitting states sequence
stateSeq(1) = 1; % Begin with entry state
t = 1;
while (stateSeq(t) ~= numStates),
  t = t+1;
  stateSeq(t) = pickState( transitions(stateSeq(t-1),:) );
end;

% Pick values in emitting states pdfs, omitting the entry state
for t = 2:(length(stateSeq)-1);
  x(t-1,:) = randn(1,dim) * stDevs{stateSeq(t)} + means{stateSeq(t)};
end;

stateSeq=stateSeq(2:end-1);

%=============================================
function [stat] = pickState(localTransitions)

cs = cumsum(localTransitions);
if ( (cs(end) - 1.0) > (eps/2) ),
  error('Bad transition probability values given to subroutine pickState.');
end;

unif = rand;
stat = 1;
while ( unif >= cs(stat) ),
  stat = stat+1;
end;
