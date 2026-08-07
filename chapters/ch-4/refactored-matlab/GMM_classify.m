function class = GMM_classify(x,GMMs,priors);

%    |GMM_classify(x,GMMs,priors)| returns the class of sample
%    |x| with respect to GMM classes, using Bayesian classification. 
%    |x| {(NxD)} is a cell array of test sequences.
%    |priors| is a vector of class priors. The function returns a 
%    vector of classes.

n_classes=length(GMMs);
n_sequences=length(x);

log_posterior=zeros(n_sequences,n_classes);
for i=1:n_classes
    for j=1:n_sequences
       log_posterior(j,i) = sum(log(GMM_pdf(x{j},...
           GMMs{i}.means,GMMs{i}.covs,GMMs{i}.priors)))...
           +log(priors(i));
    end;
end;
[tmp,class]=max(log_posterior');
