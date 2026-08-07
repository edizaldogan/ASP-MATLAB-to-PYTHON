function class = gauss_classify(x,mus,sigmas,priors);

%    |gauss_classify(x,mus,sigmas,priors)| returns the class of the point
%    |x| (1xD) with respect to Gaussian classes, using bayesian
%    classification. |mus| is a cell array of the (1xD) means, |sigmas| is
%    a cell array of the (DxD) covariance matrices. |priors| is a vector of
%    Gaussian priors. When a set of points (NxD) is provided as input, a
%    set of classes is returned.

n_classes=size(mus,2);
n_test=size(x,1);

likelihood_test=zeros(n_test,n_classes);
for j=1:n_classes
    likelihood_test(:,j) = gauss_pdf(x,mus{j},sigmas{j});
end;
bayesian_test=likelihood_test.*repmat(priors,n_test,1);

[tmp,class]=max(bayesian_test');
