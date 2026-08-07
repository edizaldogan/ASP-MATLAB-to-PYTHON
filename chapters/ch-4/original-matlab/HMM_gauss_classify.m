function class = HMM_gauss_classify(x,HMMs,priors);

%    |HMM_gauss_classify(x,HMMs,priors)| returns the class of the point
%    |x| with respect to HMM classes, using bayesian classification. HMM
%    states are modeled by a Gaussian multivariate. |x| {(NxD)} is a cell
%    array of test sequences. |priors| is a vector of class priors. The
%    function returns a vector of classes.

n_classes=length(HMMs);
n_sequences=length(x);

log_posterior=zeros(n_sequences,n_classes);
for i=1:n_classes
    for j=1:n_sequences
       log_posterior(j,i) = HMM_gauss_loglikelihood(x{j},HMMs{i})...
           +log(priors(i));
    end;
end;
[tmp,class]=max(log_posterior');
