%% Chapter 4 - How does a dictation machine recognize speech?
% This is a companion file to the book "Applied Signal Processing", 
% by T.Dutoit and F. Marques, Springer 2008.
% 
% It is supposed to be run cell-by-cell, using the cell mode of 
% MATLAB 6 and later versions. Search for "what are cells" in the 
% product help of MATLAB to know how to use them.
% 
% This file uses the SIGNAL_PROCESSING toolbox of MATLAB.
 
%%
% Although speech is by essence a non-stationary signal, and therefore
% calls for dynamic modeling, it is convenient to start this script by
% examining static modeling and classification of signals, seen as a
% statistical pattern recognition problem. We do this by using Gaussian
% multivariate models in Section 1 and extend it to Gaussian Mixture Models
% (GMM) in Section 2. We then examine, in Section 3, the more general
% dynamic modeling, using Hidden Markov Models (HMM) for isolated word
% classification. We follow in Section 4 by adding a simple bigram-based
% language model, implemented as a Markov model, to obtain a connected word
% classification system. We end in Section 5 by implementing a
% word-based speech recognition system, in which the system does not know
% in advance how many words each utterance contains.
%
% Copyright T. Dutoit, H. Bourlard (2007)

clear all;
set(0,'defaultFigureColor','w');
warning('Off','MATLAB:log:logOfZero'); 

%% 1. Gaussian modeling and Bayesian classification of vowels
%
% We will examine here how Gaussian multivariate models can be used for the
% classification of signals. 
%
% A good example is that of the classification of sustained vowels, i.e.,
% of the classification of incoming acoustic feature vectors into the
% corresponding phonemic classes. Acoustic feature vectors are generally
% highly multi-dimensional (as we shall see later), but we will work in a
% 2D space, so as to be able to plot our results. 
%
% In this script, we will work on a hypothetic language, whose phoneme set
% is only composed of four vowels {/a/,/e/,/i/,/u/}, and whose lexicon
% reduces to {why,you,we,are,hear,here} (supposedly produced as sequences
% of the aforementioned vowels). Every speech frame can then be represented 
% as a 2-dimensional vector of speech features in the form of pairs of 
% formant values (the first and the second spectral formants, F1 and F2). 
%
% Our first task will be to classify vowels, by using Gaussian class models 
% and Bayesian (MAP) decision. 
% Let us load a database of features extracted from the vowels and words of
% this language. Vowel samples are grouped in matrices of size N x 2, where 
% each of the N rows is a training example and each example is characterized 
% by a formant frequency pair. 
% Supposing that the whole database covers adequately our imaginary language, 
% it is easy to compute the prior probability P(qk) of each class qk 
% (k in {/a/,/e/,/i/,/u/}. The most common phoneme in the language is /e/.

load("data.mat"); % vowels={a,e,i,u};

N_samples=0;
for j=1:4
    N_samples = N_samples+size(vowels{j}.training,1); 
end;
for j=1:4
    prior(j) = size(vowels{j}.training,1)/N_samples;
end;
prior

%%
% As can be seen, our four vowel classes have serious overlap in the 2D
% vector space. 

plot(vowels{1}.training(:,1),vowels{1}.training(:,2),'k+'); hold on;
plot(vowels{2}.training(:,1),vowels{2}.training(:,2),'r*');
plot(vowels{3}.training(:,1),vowels{3}.training(:,2),'gp');
plot(vowels{4}.training(:,1),vowels{4}.training(:,2),'bs');
set(gca,'xlim',[0 1500],'ylim',[0 3000],'dataaspectratio',[1 1 1]);
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
legend('/a/','/e/','/i/','/u/');

%%
% Let us now assume that we are asked to identify an unknown vowel from its
% (F1,F2) features. One way of solving this problem is by performing
% multivariate Gaussian modeling of each class, i.e., finding the mean and
% covariance matrices of the data in each class. 
%
% *MATLAB function involved:*
% 
% * |plot_2Dgauss_pdf(mu, sigma)| plots the mean and standard
%    deviation ellipsis of the 2D Gaussian process that has mean |mu| and
%    covariance matrix |sigma|, in a 2D plot.

for j=1:4
    mu{j}=mean(vowels{j}.training);
    sigma{j}=cov(vowels{j}.training);
    plot_gauss2D_pdf(mu{j},sigma{j})
end;

%%
% As can be seen on the plot, /i/ has its mean F1 at 780 Hz and its mean F2
% at 1680 Hz. (NB: thes values are the ones fixed in our imaginary
% language; they do not correspond to those of English vowels at all). The
% covariance matrix for the /i/ class is almost diagonal (the scatter plot
% for the class has its principal axes almost parallel to the coordinate
% axes; see Appendix 1). Its diagonal elements are thus close to the square
% of the length of the halfmajor and halfminor axes of the standard
% deviation ellipsis: 76 Hz and 130 Hz, respectively.

mu{3}
sqrtm(sigma{3})

%%
% Let us estimate the likelihood of a test feature vector given
% the Gaussian model of class /e/, using the classical Gaussian PDF
% formula.

sample=[650 1903];
plot(sample(1),sample(2),'ko','linewidth',3)
x = sample-mu{2};
likelihood = exp(-0.5* x* inv(sigma{2}) *x') / sqrt((2*pi)^2 * det(sigma{2}))

%% 
% The likelihood of this vector is higher in class /i/ than in any other
% class (this is also intuitively obvious from the scatter plots shown
% previously).
%
% *MATLAB function involved:*
% 
% * |gauss_pdf(x,mu,sigma)| returns the likelihood of sample |x| (NxD)
%    with respect to a Gaussian process with mean |mu| (1xD) and covariance
%    |sigma| (DxD). When a set of samples is provided as input, a set of
%    likelihoods is returned.

for j=1:4
    likelihood(j) = gauss_pdf(sample,mu{j},sigma{j});
end;
likelihood

%%
% Likelihood values are generally very small. Since we will use products of
% them in the next paragraphs, we will systematically prefer their 
% log-likelihood estimates.% PDF values are vey small, and we will use 
% products of them in the next paragraphs, so we will systematically use 
% log-likelihood estimates.

log(likelihood)

%%
% Since not all phonemes have the same prior probability, Bayesian (MAP)
% classification of our test sample is not equivalent to finding the class
% with maximum likelihood. Posterior probabilities P(class|sample) must be
% estimated by multiplying the likelihood of the sample by the prior of
% each class, and dividing by the marginal likelihood of the sample
% (obtained by summing its likelihood for all classes). Again, for
% convenience, we compute the log of posterior probabilities. The result is
% that our sample gets classified as /e/ ratter than as /i/, because the
% prior probability of /e/ is much higher than that of /i/ in our imaginary
% language. 

marginal=sum(likelihood); % is a constant
log_posterior=log(likelihood)+log(prior)-log(marginal)

%% 
% Notice the marginal likelihood of the sample is not required for
% classifying it, as it is a subtractive constant for all log posterior
% probabilities. We will not compute it in the sequel.
%
% Multiplying likelihoods by priors can be seen as a weighting which 
% accounts for the intrinsic frequency of occurence of each class. 
% Plotting the posterior probability of classes in the (F1,F2) plane 
% gives a rough idea of how classes are delimited. Do not hesitate to
% rotate the plot.
%
% *MATLAB function involved:*
% 
% *|mesh_2Dgauss_pdf(mu,sigma,weight,gridx,gridy,ratioz)| plots the PDF of
%   a 2D-Gaussian PDF in a 3D plot. |mu| (1x2) is the mean of the density,
%   |sigma| (2x2) is the covariance matrix of the density. |weight| is a
%   scalar used as a multiplicative factor on the value of the PSD. |gridx|
%   and |gridy| must be vectors of the type (x:y:z) |ratioz| is the
%   (scalar) aspect ratio on the Z axis.

hold on;
for j=1:4
    mesh_gauss2D_pdf(mu{j},sigma{j},prior(j),0:50:1500, ...
        0:50:3000, 7e-9);
    hold on;
end;
hold off;

%% 
% One can easily compare the performance of max likelihood vs. max
% posterior classifiers on test data sets taken from our four vowels (and
% having the same prior distribution as from the training set).
%
% *MATLAB function involved:*
% 
% * |gauss_classify(x,mus,sigmas,priors)| returns the class of the point
%    |x| (1xD) with respect to Gaussian classes, using Bayesian
%    classification. |mus| is a cell array of the (1xD) means, |sigmas| is
%    a cell array of the (DxD) covariance matrices. |priors| is a vector of
%    Gaussian priors. When a set of points (NxD) is provided as input, a
%    set of classes is returned.

total=0;
errors_likelihood=0;
errors_bayesian=0;
for i=1:4
    n_test=size(vowels{i}.test,1);
    class_likelihood=gauss_classify(vowels{i}.test,mu,sigma,[1 1 1 1]);
    errors_likelihood=errors_likelihood+sum(class_likelihood'~=i);
    class_bayesian=gauss_classify(vowels{i}.test,mu,sigma,prior);
    errors_bayesian=errors_bayesian+sum(class_bayesian'~=i);
    total=total+n_test;
end;
likelihood_error_rate=errors_likelihood/total
bayesian_error_rate=errors_bayesian/total

%% 2. Gaussian Mixture Models (GMM)
% In the previous section, we have seen that Bayesian classification is
% based on the estimation of class PDFs. Up to now, we have modeled the PDF
% for each class /a/, /e/, /i/, /u/ as a Gaussian multivariate (one per
% class).  This implicitly assumes that the feature vectors in each class
% have a (uni-modal) normal distribution, as we used the |mean| and |cov|
% functions, which return the estimates of the mean and
% covariance matrix of supposedly Gaussian multivariate data samples. It
% turns out that the vowel data we used had actually been sampled according
% to Gaussian distributions, so that this hypothesis was satisfied. 
%
% Let us now try to classify the words of our imaginary language, using the
% same kind of approach as above. We will use 100 samples of the six words
% {why /uai/,you /iu/,we /ui/,are /ae/,hear /ie/,here /ie/} in our
% imaginary language, for which each speech frame is again characterized by
% an [F1,F2] feature vector. 

load data; % words={why,you,we,are,hear,here};

for i=1:6
    subplot(2,3,i)
    plot(words{i}.training_all(:,1),words{i}.training_all(:,2),'+'); 
    set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);
    xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
    title(words{i}.word);
    hold on;
end;

%%
% Notice that "you" and "we" have the same statistical distribution,
% because of their phonemic content (in our imaginary language) : /iu/ and
% /ui/. Notice also that "hear" and "here" also have the same distribution,
% because they have exactly the same phonemic transcription: /ie/. We will
% come back to this later.

%%
% We are now facing a PDF estimation problem: the PDF of the data in each
% class in no longer Gaussian. This is typical of practical ASR systems: in
% word-based ASR, each class accounts for the realization of several
% phonemes and is thus better described as a multi-modal distribution, i.e.
% a distribution with several maxima. The same holds for phoneme-based ASR
% as well. As a matter of fact, speech is very much submitted to
% coarticulation, which often results in several modes for the acoustic
% realization of each phoneme, as a function of the phonetic context in
% which it appears. 
%
% If we apply a uni-modal Gaussian model to word "why", for instance, we
% get a gross estimation of the PDF. This estimation does not correctly
% account for the fact that several areas in the (F1,F2) plane are more
% densely crowded. The maximum value of the Gaussian PDF is very low, since
% it spans more of the (F1,F2) space than it should (and the integral is
% constrained to one). 

training_set=words{1}.training_all;
test_set=words{1}.test_all;
mu_all=mean(training_set);
sigma_all=cov(training_set);

clf;
plot(training_set(:,1),training_set(:,2),'+'); 
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
hold on;
mesh_gauss2D_pdf(mu_all,sigma_all,...
        1, 0:50:1500, 0:50:2500,7e-9);

%%
% The total log likelihoods of the training and test data given this
% Gaussian model are obtained as sum of the log likelihoods of all feature
% vectors.

log_likelihood_training=sum(log(gauss_pdf(training_set,mu_all,sigma_all)))
log_likelihood_test=sum(log(gauss_pdf(test_set,mu_all,sigma_all)))

%%
% One way of estimating a multi-modal PDF is by clustering data, and then
% estimate a uni-modal PDF in each cluster. An efficient way to do this 
% is by using K-means clustering. Starting with k prototype vectors or 
% centroids, this algorithm first associates each feature vector in the 
% training set to its closest centroid. It then replaces every centroid 
% by the mean of all feature vectors that have been associated to it. 
% The algorithm iterates by re-associating each feature vector to one of 
% the newly found centroids, and so on until no further change occurs.
% 
% *MATLAB functions involved:*
% 
% * |[new_means,new_covs,new_priors,distortion]= ...
%    kmeans(data,n_iterations,n_clusters)| , where  |data| is the matrix of
%    observations (one observation per row) and |n_clusters| is the desired
%    number of clusters, returns the mean vectors, covariance matrices, and
%    priors of k-means clusters. |distortion| is an array of values (one
%    per iteration) of sum of squared distances between the data and the
%    mean of their cluster. The clusters are initialized with a heuristic
%    that spreads them randomly around |mean(data)|.The algorithm iterates
%    until convergence is reached or the number of iterations exceeds
%    |n_iterations|. |kmeans(data,n_iterations,means)|, where |means| is a
%    cell array containing initial mean vectors, makes it possible to
%    initialize means. 
%
% * |plot_kmeans2D(data,means)| plots the clusters associated 
%    with |means| in |data| samples, using a Euclidian distance. 
%    |means| is a cell array containing the prototype vectors.

% Initializing prototypes "randomly" around the mean
initial_means{1} = [0,1] * sqrtm(sigma_all) + mu_all;
initial_means{2} = [0,0] * sqrtm(sigma_all) + mu_all;
initial_means{3} = [1,2] * sqrtm(sigma_all) + mu_all;

[k_means,k_covs,k_priors,totalDist]=kmeans(training_set,1000,initial_means);

plot(training_set(:,1),training_set(:,2),'r+'); 
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);
plot_kmeans2D(training_set, k_means);

%% 
% The K-means algorithm converges monotonically to a (local) minimum of the
% global distortion defined as the sum of all distances between feature
% vectors and their associated centroids. 

plot(totalDist,'.-');
xlabel('Iteration'); ylabel('Global distortion'); grid on;

%%
% It is possible to see how clusters and prototype vectors evolve with
% iterations.

plot(training_set(:,1),training_set(:,2),'+'); 
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);

k_means=initial_means;
for j=1:length(totalDist)
    [k_means,k_covs,k_priors,]=kmeans(training_set,1,k_means);
    plot_kmeans2D(training_set, k_means);
    pause(0.5);
end;

%% 
% The resulting sub-classes, though, do not strictly correspond to the
% phonemes of "why". This is because the global criterion that is minimized
% by the algorithm is purely geometric. It would actually be very
% astonishing in these conditions to find the initial vowel sub-classes.
% This is not a problem, as what we are trying to do is to estimate the PDF
% of the data, not to classify it into "meaningful" sub-classes. Once
% clusters have been created, it is easy to compute the corresponding
% (supposedly uni-modal) Gaussian means and covariance matrices for each
% cluster (this is actually done inside our |kmeans| function), and to plot
% the sum of their PDFs, weighted by their priors. This produces an
% estimate of the PDF of our speech unit.
%
% *MATLAB function involved:*
% 
% *|mesh_GMM2D_pdf(mus,sigmas,weights,gridx,gridy,ratioz)| plots the PDF
%   of a 2D Gaussian Mixture Model PDF in a 3D plot. |mus| is a cell array
%   of the (1x2) means, |sigmas| is a cell array of the (2x2) covariance
%   matrices. |weight| is a vector of Gaussian weights. |gridx| and |gridy|
%   must be vectors of the type (x:y:z) |ratioz| is the (scalar) aspect
%   ratio on the Z axis;

plot(training_set(:,1),training_set(:,2),'+'); 
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
hold on;
mesh_GMM2D_pdf(k_means,k_covs,k_priors, ...
        0:50:1500, 0:50:2500,2e-8);
hold off;

%%
% The total log likelihoods of the training and test data are obtained as
% above, except we now consider that each feature vector "belongs" to each
% cluster with some weight equal to the prior of the cluster. Its
% likelihood is thus computed as a weighted sum of likelihoods (one per
% Gaussian)
%
% *MATLAB function involved:*
% 
% * |GMM_pdf(x,mus,sigmas,weights)| returns the likelihood of sample |x|
%    (1xD) with respect to a Gaussian Mixture Model. |mus| is a cell array
%    of the (1xD) means, |sigmas| is a cell array of the (DxD) covariance
%    matrices. |weight| is a vector of Gaussian weights. When a set of
%    samples (NxD) is provided as input, a set of likelihoods is returned.

log_likelihood_training=sum(log(GMM_pdf(training_set,...
    k_means,k_covs,k_priors)))
log_likelihood_test=sum(log(GMM_pdf(test_set,...
    k_means,k_covs,k_priors)))

%%
% The K-means approach used above is not optimal, in the sense that it is
% based on a purely geometric convergence criterion. The central algorithm
% for training GMMs is based on the EM (Expectation-Maximization) algorithm. 
% As opposed to K-means, EM truly maximizes the likelihood of the data given
% the GMM parameters (means, covariance matrices, and weights). Starting
% with k initial uni-modal Gaussians (one for each sub-class), it first
% estimates, for each feature vector, the probability of each sub-class
% given that vector. This is the Estimation step, which is based on "soft"
% classification: each feature vector belongs to all sub-classes, with some
% weights. In the Maximization step,  the mean and covariance of each
% sub-class is updated, using all feature vectors and taking those weights
% into account. The algorithm iterates on the E and M steps, until the
% total likelihood increase for the training data fall under some
% threshold. 
%
% The final estimate obtained by EM, however, only corresponds to a local
% maximum of the total likelihood of the data, whose value may be very
% sensitive to the initial uni-modal Gaussian estimates provided as input.
% A frequently used value for these initial estimates is precisely the one
% provided by the K-means algorithm.
%
% *MATLAB functions involved:*
% 
% * |[new_means,new_sigmas,new_priors,total_loglike]= ...
%    GMM_train(data,n_iterations,n_gaussians)| returns the mean vectors,
%    covariance matrices, and priors of GMM Gaussian components obtained by
%    the EM training algorithm. |data| is the matrix of observations (one
%    observation per row) and |n_gaussians| is the desired number of
%    clusters. |total_loglike| is an array of values (one per iteration) of
%    the total likelihood of the data given the GMM model. GMMs are
%    initialized with a heuristic that spreads them randomly around
%    |mean(data)|. The algorithm iterates until convergence is reached or
%    the number of iterations exceeds |n_iterations|.    
%
%    |GMM_train(data,n_iterations,means,covs,priors)|, makes it possible to
%    initialize the means, covariance matrices, and priors of the GMM
%    components.
%
% * |plot_GMM2D(data, means, covs)| shows the standard deviation ellipsis 
%    of the Gaussian components of a GMM defined by |means| and |covs|, 
%    on a 2D plot, together with |data| samples. 

% Using the output of K-means as initla conditions
[means,covs,priors,total_loglike]=GMM_train(training_set,100, ...
   k_means,k_covs,k_priors);

plot(training_set(:,1),training_set(:,2),'+'); 
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);
plot_GMM2D(training_set,means,covs);

%%
% Notice we do not set colors to vectors, as EM precisely does not strictly
% assign Gaussians to feature vectors. It is also possible to see how
% clusters and prototype vectors evolve with iterations.

plot(training_set(:,1),training_set(:,2),'+'); 
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);

means=k_means;
covs=k_covs;
priors=k_priors;
for j=1:length(total_loglike)
    [means,covs,priors]=GMM_train(training_set,1, ...
        means,covs,priors);
    plot_GMM2D(training_set,means,covs);
    pause(0.1);
end;

%%
% Applied to the sample feature vectors of "why", the EM algorithm
% converges monotonically, in 7 steps, from the K-means solution to a
% (local) maximum of the total likelihood of the sample data. 

plot(total_loglike,'.-');
xlabel('Iteration'); ylabel('Global Log Likelihood'); grid on;

%%
% The total log likelihoods of the training and test data given this GMM
% model are obtained as above. The increase compared to estimating the GMM
% parameters from K-means clustering is small, but this is due to the
% oversimplified PDF we are dealing with. GMMs are very much used in speech
% recognition.

log_likelihood_training=sum(log(GMM_pdf(training_set,means,covs,priors)))
log_likelihood_test=sum(log(GMM_pdf(test_set,means,covs,priors)))

%%
% Now let us try to recognize sample words in {why,you,we,are,hear,here}.
% We now use the sequence of feature vectors from our unknown signal
% (instead of a single vector as before), estimate the joint likelihood of
% all vectors in this sequence given each class, and obtain the posterior
% probabilities in the same way as above. If we assume that each sample in
% our sequence is independent from the others (which is in practice a
% rather bold claim, even for stationary signals; we will come back to this
% in the next section when introducing dynamic models), then the joint
% likelihood of the sequence is simply the product of the likelihoods of
% each sample. 
%
% We estimate a GMM for each word, , using 3 Gaussians per word. (When 2
% Gaussians are enough, one of the three ends up having very small weight).

for i=1:6
   [GMMs{i}.means,GMMs{i}.covs,GMMs{i}.priors,total_loglike]=...
       GMM_train(words{i}.training_all,100,3);
end;

for i=1:6
    subplot(2,3,i)
    plot(words{i}.training_all(:,1),words{i}.training_all(:,2),'+'); 
    set(gca,'xlim',[0 1500],'ylim',[0 3000],'dataaspectratio',[1 1 1]);
    title(words{i}.word);
    hold on;
    mesh_GMM2D_pdf(GMMs{i}.means,GMMs{i}.covs,GMMs{i}.priors, ...
         0:50:1500, 0:50:2500,8e-9);
end;
hold off;

%%
% Let us first try to recognize the first test sequence taken from "why".
% Since we do not know the priors of words in our imaginary language, we
% will set them all to 1/6. As expected, the maximum log likelihood is
% encountered for word "why": our first test word is correctly recognized. 

word_priors=ones(1,6)*1/6;
test_sequence=words{1}.test{1};
for i=1:6
    log_likelihood(i) = sum(log(GMM_pdf(test_sequence,...
        GMMs{i}.means,GMMs{i}.covs,GMMs{i}.priors)));
end;
log_posterior=log_likelihood+log(word_priors)
[maxlp,index]=max(log_posterior);
recognized=words{index}.word

clf;
plot(test_sequence);
xlabel('Frames'); ylabel('Freq [Hz]'); grid on;
legend('F1','F2');

%%
% Not all sequences are correctly classified, though. Sequence 2 is
% recognized as a "we".
test_sequence=words{1}.test{2};
for i=1:6
    log_likelihood(i) = sum(log(GMM_pdf(test_sequence,...
        GMMs{i}.means,GMMs{i}.covs,GMMs{i}.priors)));
end;
log_posterior=log_likelihood+log(word_priors)
[maxlp,index]=max(log_posterior);
recognized=words{index}.word

%%
% We may now compute the total word error rate. 
%
% *MATLAB function involved:*
% 
% * |GMM_classify(x,GMMs,priors)| returns the class of the point
%    |x| with respect to GMM classes, using bayesian classification. |x|
%    {(NxD)} is a cell array of test sequences. |priors| is a vector of
%    class priors. The function returns a vector of classes.

total=0;
errors=0;
for i=1:6
    n_test=length(words{i}.test);
    class=GMM_classify(words{i}.test,GMMs,word_priors);
    errors=errors+sum(class'~=i);
    class_error_rate(i)=sum(class'~=i)/n_test;
    total=total+n_test;

    subplot(2,3,i);
    hist(class,1:6);
    title(words{i}.word);
    set(gca,'xlim',[0 7],'ylim',[0 100]);
    
end;
overall_error_rate=errors/total
class_error_rate

%%
% Obviously, our static approach to word classification is not a success. 
% Only 70% of the words are correctly classified. The rather high error
% rates we obtain are not astonishing. Except for "why" and "are", which
% have fairly specific distributions, "here" and "here" have identical
% PDFs, as well as "you" and "we". 

%% 3. Hidden Markov Models (HMM)
% In the previous Sections, we have seen how to create a model, either
% Gaussian or GMM, for estimating the PDF of speech feature vectors, even
% with complicated distribution shapes, and have applied it to the
% classification of isolated words. The main drawback of such a static
% classification, as it stands, is that it does not take time into account.
% For instance, the posterior probability of a sequence of feature vectors
% does not change when the sequence is time-reversed, as in words /iu/ and
% /we/. This is due to the fact that our Bayesian classifier implicitly
% assumed that successive feature vectors are statistically independent.
%
% In this Section we will model each word in our imaginary language using a
% 2-state HMM (plus their initial and final states), except for "why",
% which will be modeled as a 3-state HMM. One should not conclude that
% word-based ASR systems set the number of interna HMM states for each word
% to the number of phonemes they contain. The number of states is usually
% higher than the number of phonemes, as phonemes are themselves realized
% in several articulatory steps which may each require a specific state.
% The reason for our choice is directly dictated by the fact that the test
% data we are using throughout this script was ransdomly generated by HMMs
% (see appendix 1) in which each phoneme was modeled by one HMM state
% modeled as a multivariate Gaussian. As a result, our test data virtually
% exhibits no coarticulation, and hence does not require more than one
% state per phoneme.
% 
% We will make one more simplification here: that of having access to a
% corpus of pre-segmented sentences, from which many examples of our 6
% words have been extracted. This will make it possible to train our word
% HMMs separately. In real ASR systems, segmentation (in words or phonemes)
% is not known. Sentence HMMS are thus created by concatenating word HMMs,
% and these sentence HMMs are trained. Words (or phoneme) segmentation is
% then obtained as a by-product of this training stage.

%% 
% We start by loading our training data and creating initial values for the
% left-right HMM of each word in our lexicon. Each state is modeled using a
% Gaussian multivariate whose mean feature vector is set to a random value
% close to the mean of all feature vectors in the word. The elements
% trans(i,j) of the transition matrix give the probability of going from
% state i to j (state 1 being the initial state). Transitions probabilities
% betweeen internal (emiting) states are set to a constant value of 0.8 for
% staying in the same state, and 0.2 for leaving to the next state.

load data; % words={I,hear,here,you,are,we,why};

% Initializing HMM parameters "why" is a spacial case: it has 3 states
mu=mean(words{1}.training_all);
sigma=cov(words{1}.training_all);
HMMs{1}.means = {[],mu,mu,mu,[]};
HMMs{1}.covs  = {[],sigma,sigma,sigma,[]};
HMMs{1}.trans = [ 0.0 1.0  0.0  0.0  0.0
                          0.0 0.8  0.2  0.0  0.0
                          0.0 0.0  0.8  0.2  0.0
                          0.0 0.0  0.0  0.8  0.2
                          0.0 0.0  0.0  0.0  1.0 ];
for i=2:6
    mu=mean(words{i}.training_all);
    sigma=cov(words{i}.training_all);
    HMMs{i}.means = {[],mu,mu,[]};
    HMMs{i}.covs  = {[],sigma,sigma,[]};
    HMMs{i}.trans = [0.0 1.0   0.0   0.0 
                            0.0 0.8   0.2   0.0
                            0.0 0.0   0.8   0.2
                            0.0 0.0   0.0    1   ];
end


%%
% Let us train our HMM models using the Baum-Welch (or Forward-Backward)
% algorithm, which is a particular implementation of the EM algorithm we
% already used for training our GMMs in the previous Section. This
% algorithm will adapt the parameters of our word HMMs so as to maximize the
% likelihood of each data set given each HMM model.
%
% *MATLAB function involved:*
% 
% * |new_hmm = HMM_train_FB(data,old_hmm,dmin,qmax)| returns the Maximum
%    Likelihood re-estimation of a Gaussian Hidden Markov Model (i.e., a
%    single, possibly multivariate, Gaussian probability density function
%    per state) based on the forward-backward algorithm (aka. Baum-Welch
%    re-estimation formulas). Note that most operations are performed in
%    the log domain for accuracy. |dmin| and |qmax| are respectively the
%    minimum log-likelihood relative improvement and the maximum number of
%    iterations until convergence.  

% In some MATLAB version, the HMM-related functions will flag lots of
% warnings. The next line disables them.
warning off all;

for i=1:6
     HMMs{i}=HMM_gauss_train(words{i}.training,HMMs{i},0.001,50);
end;
save word_hmms HMMs  % for use in the next Sections

% Notice it is possible to check the effect of training, compared to the
% exact HMMs used for generating the data, by uncommenting the following
% line:  
% HMMs=words;

%%
% The word "why" is now correctly modeled as a sequence of 3 states, each
% with a Gaussian multivariate PDF, which match those of the underlying
% phonemes in the word: /uai/.

for i=2:4
    subplot(1,3,i-1)
    plot(words{1}.training_all(:,1),words{1}.training_all(:,2),'+'); 
    set(gca,'xlim',[0 1500],'ylim',[0 3000],'dataaspectratio',[1 1 1]);
    title(['state ' num2str(i-1)]); % emiting states only
    hold on;
    mesh_gauss2D_pdf(HMMs{1}.means{i},HMMs{1}.covs{i},1, ...
         0:50:1500, 0:50:2500,1e-8);
end;
hold off;

%%
% The transition probabilities between the states of "why" have been
% updated by the Baum-Welch algorithm.

HMMs{1}.trans

%%
% As a result of this better modeling , the total likelihood of the data
% for word "why" is higher than with our previous static GMM model. The
% prevous model can actualy be seen as a single-state HMM, whose emission
% probabilities are modeled by a GMM. 

log_likelihood_training=0;
for i=1:length(words{1}.training)
    training_sequence=words{1}.training{i};
    log_likelihood_training=log_likelihood_training+...
        HMM_gauss_loglikelihood(training_sequence,HMMs{1});
end;

log_likelihood_test=0;
for i=1:length(words{1}.test)
    test_sequence=words{1}.test{i};
    log_likelihood_test=log_likelihood_test+...
        HMM_gauss_loglikelihood(test_sequence,HMMs{1});
end;

log_likelihood_training
log_likelihood_test

%%
% HMM-based isolated word recognition can now be achieved by finding the
% maximum of the posteriori probability of a sequence of feature vectors
% given all 6 HMM models. The 2nd test sequence for "why" (which was not
% correctly recognized using GMMs and a single state) now passes our
% classification test.

word_priors=ones(1,6)*1/6;

test_sequence=words{1}.test{2};
for i=1:6
   log_posterior(i) = HMM_gauss_loglikelihood(test_sequence,...
       HMMs{i})+log(word_priors(i));
end
log_posterior
[tmp,index]=max(log_posterior);
recognized=words{index}.word

%%
% The HMM model does not strictly assign states to feature vectors: each
% feature vector can be emitted by any state with a given probability. It
% is possible, though, to estimate the best path through the HMM given the
% data, by using the Viterbi algorithm.
%
% *MATLAB function involved:*
% 
% *|plot_HMM2D_timeseries(x,states)| plots a two-dimensional sequence |x|
%   (one observation per row) as two separate figures, one per dimension. 
%   It superposes the corresponding state sequence |stateSeq| as colored
%   dots on the observations. |x| and |stateSeq| must have the same length.

test_sequence=words{1}.test{1};
best_path=HMM_gauss_viterbi(test_sequence,HMMs{1});
clf;
plot_HMM2D_timeseries(test_sequence,best_path);
xlabel('Frames'); ylabel('Freq [Hz]'); grid on;
legend('F1','F2');

%%
% We may now compute the total word error rate again. 
%
% *MATLAB function involved:*
% 
% * |HMM_gauss_classify(x,HMMs,priors)| returns the class of the point
%    |x| with respect to HMM classes, using bayesian classification. HMM
%    states are modeled by a Gaussian multivariate. |x| {(NxD)} is a cell
%    array of test sequences. |priors| is a vector of class priors. The
%    function returns a vector of classes.

total=0;
errors=0;
for i=1:6
    n_test=length(words{i}.test);
    class=HMM_gauss_classify(words{i}.test,HMMs,word_priors);
    errors=errors+sum(class'~=i);
    class_error_rate(i)=sum(class'~=i)/n_test;
    total=total+n_test;

    subplot(2,3,i);
    hist(class,1:6);
    title(words{i}.word);
    set(gca,'xlim',[0 7], 'ylim', [0 100]);

    % Screen output
    fprintf('Word %3d\n', i)

end;
overall_error_rate=errors/total
class_error_rate

%%
% Notice the important improvement in the classification of "you" and "we",
% which are now modeled as HMMs with distinctive parameters. 84% of the
% (isolated) words are now recognized. The remaining he errors are due to
% the confusion between "here" and "hear".

%% 4. N-grams
% In the previous Section, we have used HMM models for the words of our
% imaginary language, which led to a great improvement in isolated word
% classification. It remains that "hear" and "here", having strictly
% identical PDFs, cannot be adequately distinguished. This kind of
% ambiguity can only be resolved, when words are embedded in a sentence, by
% using constraints imposed by the language on word sequences,i.e. by
% modeling the syntax of the language.
%
% We will now examine the more general problem of connected word
% classification, in which words are embedded in sentences. This task
% requires adding a language model on top of our isolated word
% classification system. For convenience, we will assume that our imaginary
% language imposes the same syntactic constraints as English. A sentence
% like "you are hear" is therefore impossible and should force the
% recognition of "you are here" wherever a doubt is possible. In this first
% step, we will also assume that word segmentation is known (this could
% easily be achieved, for instance, by asking the speaker to insert
% silences between words and detecting silences based on energy levels).  

%%
% Our data file contains a list of 150 such pre-segmented sentences. Let us
% plot the contents of the first one ("we hear why you are here").

load data; % sentences composed of words in {why,you,we,are,hear,here};
load word_hmms %word HMMs trained in the previous Section
warning off all;

clf;
for i=1:length(sentences{1}.test)
    subplot(2,3,i);
    test_sequence=sentences{1}.test{i}; % ith word
    plot(test_sequence(:,1),'+-');
    hold on;
    plot(test_sequence(:,2),'r*-');
    title(['Word' num2str(i)]);
    xlabel('Frames'); grid on;
end;
    
%%
% We model the syntactic constraints of our language by a bigram model,
% based on the probability of pairs of successive words in the language.  
% Such an approach reduces the language model to a simple Markov model. The
% component |bigram(i,j)| of its transition matrix gives P(wordi|wordj):
% the probability that the jth word in the lexicon is followed by the ith
% word.  Clearly, "You are hear" is made impossible by |bigrams(5,6)=0|. 

% states = I U {why,you,we,are,hear,here} U F where I and F stand for the
% begining and the end of a sentence

bigrams = ...
    [0  1/6 1/6  1/6  1/6  1/6  1/6  0  ;  % P(word|I)
     0  0    1/6  1/6  1/6  1/6  1/6  1/6; % P(word|"why")
     0  1/5  0     0    1/5  1/5  1/5  1/5; % P(word|"you") 
     0  0     0     0    1/4  1/4  1/4  1/4; % P(word|"we") 
     0  0     1/4  1/4  0     0    1/4  1/4; % P(word|"are") 
     0  1/4  1/4  0     0     0    1/4  1/4; % P(word|"hear") 
     0  0     1/4  1/4  1/4  0     0    1/4; % P(word|"here") 
     0  0     0     0     0     0    0     1];  % P(word|F) 

save bigrams bigrams; %  for use in the next Section

%% 
% Let us now try to classify a sequence of words taken from the test set. 
% We start by computing the log likelihood of each unknown word given the
% HMM model for each word in the lexicon. Each column of the log
% likelihoood matrix stands for a word in the sequence; each line stands
% for a word in the lexicon {why,you,we,are,hear,here}.

n_words=length(sentences{1}.test);
log_likelihoods=zeros(n_words,6);

for j=1:n_words
    unknown_word=sentences{1}.test{j};
    for k=1:6 % for each possible word HMM model
       log_likelihoods(j,k) = HMM_gauss_loglikelihood(unknown_word,HMMs{k});
    end;
end;
log_likelihoods

%%
% With the approach we used in the previous Section, we would classify this
% sentence as "we hear why you are hear" (by choosing the max likelihood
% candidate for each word independently of its neighbors). 

[tmp,indices]=max(log_likelihoods,[],2);
for j=1:n_words
    recognized_sequence{j}=words{indices(j)}.word;
end;
recognized_sequence

%% 
% We implement our language model as a Markov model on top of our
% word HMMs. The resulting model for the sequence to recognize is a
% discrete HMM, in which there are as many internal states as the number of
% words in the lexicon (six in our case). Each state can emit any of the
% |n_words| input words (which we will label as '1', '2', ... '|n_words|'),
% with emission probabilities equal to the likelihoods computed
% above. Bigrams are used as transition probabilities. Finding the best
% sequence of words from the lexicon given the sequence of observations [1,
% 2, ..., n_words] is obtained by looking for the best path in this model,
% using the Viterbi algorithm again. 
% 
% We now correctly classify our test sequence as "we hear why you are
% here".
%
% *MATLAB function involved:*
% 
% * |[state,likelihood] = HMM_viterbi(transition,emission)| performs 
%    the Viterbi search (log version) of the best state sequence for 
%    a discrete Hidden Markov Model. 
%    |transition| : (K+2)x(K+2) matrix of transition probabilities,
%                   first and last rows correspond to initial and
%                   final (non-emitting) states.
%    |emission|   : NxK matrix of state-conditional emission
%                   probabilities corresponding to a given sequence
%                   of observations of length N.
%    |state|      : (Nx1) vector of state-related indexes of best sequence. 
%    |likelihood| : best sequence likelihood.


best_path=HMM_viterbi(log(bigrams),log_likelihoods);
for j=1:n_words
    recognized_sequence{j}=words{best_path(j)}.word;
end;
recognized_sequence

%%
% We may finally compute the word error rate on our complete test data.

n_sentences=length(sentences);

total=0;
errors=0;
class_error_rate=zeros(1,6);
class=cell(6); % empty cells

for i=1:n_sentences

    n_words=length(sentences{i}.test);
    log_likelihoods=zeros(n_words,6);

    for j=1:n_words
        unknown_word=sentences{i}.test{j};
        for k=1:6 % for each possible word HMM model
            log_likelihoods(j,k) = HMM_gauss_loglikelihood(...
            unknown_word,HMMs{k});
        end;
    end;

    best_path=HMM_viterbi(log(bigrams),log_likelihoods);

    for j=1:n_words
        recognized_word=best_path(j);
        actual_word=sentences{i}.wordindex{j};
        class{actual_word}= [class{actual_word} recognized_word];
   
        if (recognized_word~=actual_word)
            errors=errors+1;
            class_error_rate(actual_word)=class_error_rate(actual_word)+1;
        end;
    end;

    total=total+n_words;

    % Screen output
     if rem(i, 10) == 0
          fprintf('Sentence %3d\n', i)
     end

end;

clf;
for i=1:6
    subplot(2,3,i);
    hist(class{i},1:6);
    title(words{i}.word);
    set(gca,'xlim',[0 7], 'ylim', [0 100]);
end;

overall_error_rate=errors/total
class_error_rate

%%
% We now have an efficient connected word classification system for our
% imaginary language. The final recognition rate is now 89.2%. It is still due to
% "here" being confused with "hear". This is due to the fact that our
% bigram model is not constrictive enough. It still allows non admissible
% sentences, such as in sentence #3: "why are you hear".

n_words=length(sentences{3}.test);
log_likelihoods=zeros(n_words,6); 

for j=1:n_words
    for k=1:6 % for each possible word HMM model
       unknown_word=sentences{3}.test{j};
       log_likelihoods(j,k) = HMM_gauss_loglikelihood(...
          unknown_word,HMMs{k});
    end;
end;

best_path=HMM_viterbi(log(bigrams),log_likelihoods);

recognized_sentence={};
for j=1:n_words
    recognized_sentence{j}=words{best_path(j)}.word;
end;
recognized_sentence

%%
% Bigrams cannot solve all "hear" vs. "here" ambiguities, because of the
% weaknesses of this poor language model. Trigrams could do a much better
% job ("are you hear", for instance, will be forbidden by a trigram
% language model), at the expense of additional complexity.

%%  5.Word-based continuous speech recognition
% For the last step of this Section, we will relax the pre-segmentation
% constraint, which will turn our classification system into a true
% word-based speech recognition system (albeit still in our imaginary
% language). 
%
% The discrete sentence HMM we used previously implicitly imposed
% initial and final states of word HMMs to fall after some specific feature
% vectors. The HMM therefore had to be changed for each new incoming
% sentence. When word segmentation is not known in advance, the initial and
% final states of all word HMMs must be erased, for the input feature
% vector sequence to be properly decoded into a sequence of words.
% 
% The resulting sentence HMM is a Gaussian HMM (Gaussian in our case, as
% each word HMM state is modeled as a Gaussian) composed of all the word
% HMM states, connected in a left-right topology inside word HMMs, and
% connected in an ergodic topology between word HMMs. For the six words of
% our language, this makes 13 internal states, plus the sentence-initial
% and sentence-final states. The transition probabilities between
% word-internal states are taken from the previously trained word HMMs,
% while the transition probabilities between word-final and word-initial
% states are taken from our bigram model. 

load data;
load word_hmms;
load bigrams;
warning off all;

sentence_HMM.trans=zeros(15,15);
word_i=[2 5 7 9 11 13 15]; % word-initial states, including sentence-final state;
word_f=[4 6 8 10 12 14]; % word-final states;

% P(word in sentence-initial position)
sentence_HMM.trans(1,word_i)=bigrams(1,2:8);

% copying trans. prob. for the 3 internal states of "why"
sentence_HMM.trans(2:4,2:4)=HMMs{1}.trans(2:4,2:4);

% distributing P(new word|state3,"why") to the first states of other word
% models, weighted by bigram probabilities.
sentence_HMM.trans(4,word_i)=...
    HMMs{1}.trans(4,5)*bigrams(2,2:8);

% same thing for the 2-state words
for i=2:6
   sentence_HMM.trans(word_i(i):word_f(i),word_i(i):word_f(i))=...
       HMMs{i}.trans(2:3,2:3);
   sentence_HMM.trans(word_f(i),word_i)=...
       HMMs{i}.trans(3,4)*bigrams(i+1,2:8);
end;

%%
% The emission probabilities of our sentence HMM are taken from the
% word-internal HMM states.

k=2;
sentence_HMM.means{1}=[]; % sentence-initial state
for i=1:6
    for j=2:length(HMMs{i}.means)-1
        sentence_HMM.means{k}=HMMs{i}.means{j};
        sentence_HMM.covs{k}=HMMs{i}.covs{j};
        k=k+1;
    end;
end;
sentence_HMM.means{k}=[]; % sentence-final state

%%
% The model is no longer sentence-dependent: the same HMM can be used to
% decode any incoming sequence of  feature vectors into a sequence of
% words. We search for the best path in our sentence HMM given the sequence
% of feature vectors of our test sequence, with the Viterbi algorithm.
%
% *MATLAB function involved:*
% 
% * |[states,log_likelihood] = HMM_gauss_viterbi(x,HMM)| returns the best
%    state sequence and the associated log likelihood of the sequence of
%    feature vectors |x| (one observation per row) with respect to a Markov
%    model |HMM| defined by: 
%       HMM.means HMM.covs HMM.trans
%    This function implements the forward recursion to estimate the
%    likelihood on the best path.

n_words=length(sentences{1}.test);
complete_sentence=[];
for i=1:n_words
    complete_sentence=[complete_sentence ; sentences{1}.test{i}];
end;

best_path=HMM_gauss_viterbi(complete_sentence,sentence_HMM);
clf;
plot_HMM2D_timeseries(complete_sentence,best_path);
xlabel('Frames'); ylabel('Freq [Hz]'); grid on;
legend('F1','F2');

state_sequence=best_path(diff([ 0 best_path])~=0)+1;
word_indices=state_sequence(ismember(state_sequence,word_i));
[tf,index]=ismember(word_indices,word_i);

recognized_sentence={};
for j=1:length(index)
    recognized_sentence{j}=words{index(j)}.word;
end;
recognized_sentence

%%
% We may finally compute the word error rate on our complete test data.

n_sentences=length(sentences);

total=0;
errors=0;
class_error_rate=zeros(1,6);
class=cell(6); % empty cells

for i=1:n_sentences

    n_words=length(sentences{i}.test);
    complete_sentence=[];
    for j=1:n_words
        complete_sentence=[complete_sentence ; sentences{i}.test{j}];
    end;

    best_path=HMM_gauss_viterbi(complete_sentence,sentence_HMM);
    plot_HMM2D_timeseries(complete_sentence,best_path);

    state_sequence=best_path(diff([ 0 best_path])~=0)+1;
    word_indices=state_sequence(ismember(state_sequence,word_i));
    [tf,index]=ismember(word_indices,word_i);

    for j=1:min(length(index),length(sentences{i}.wordindex))
        recognized_word=index(j);
        actual_word=sentences{i}.wordindex{j};
        class{actual_word}= [class{actual_word} recognized_word];

        if (recognized_word~=actual_word)
            errors=errors+1;
            class_error_rate(actual_word)=class_error_rate(actual_word)+1;
        end;
    end;

    total=total+n_words;

    % Screen output
     if rem(i, 10) == 0
          fprintf('Sentence %3d\n', i)
     end

end;

clf;
for i=1:6
    subplot(2,3,i);
    hist(class{i},1:6);
    title(words{i}.word);
    set(gca,'xlim',[0 7], 'ylim', [0 100]);
end;

overall_error_rate=errors/total
class_error_rate

%% 
% The final recognition rate of our word-based continuous speech recognizer is
% about 86.8%. This is only 2.4% less than when using pre-segmented words,
% which shows the efficiency of our sentence HMM model for both segmenting
% and classifying words. In practice, non segmented data is used for both
% training and testing, which could still slightly increase the word error
% rate.

%%
% Dictation machines still differ from this proof-of-concept in several
% ways. Mel Frequency Cepstral Coefficients (MFCCs) are used in place of
% formants for the acoustic model. Their first and second time-derivatives
% are added, as a simple way of accounting for the correlation between
% feature vectors within HMM states. Moreover, given the number of possible
% words (several tens of thousands) in natural languages, ASR systems
% involve one additional layer in the statistical description of sentences:
% that of phonemes. The word HMMs we have trained above are replaced by
% phoneme HMMs. Word HMMs are themselves composed of phoneme HMMs (in the
% same way as we have built our sentence HMM from word HMMs), and
% additional pruning mechanisms are used in the decoder to constrain the
% search for the best sequence of words from the input feature vector
% sequence.

%% Appendix 1: Generating the data
% In this Section we show how we generated the pseudo-speech data we use in
% the script. Generating samples distributed according to a Gaussian
% multivariate is very simple. Let us assume we want to produce 500 samples
% of (F1,F2) values for some vowel , supposedly centered on [700 1000] and
% with standard deviation matrix [50 0; 0 200]. Plotting the resulting
% samples shows that the length of the semimajor and semiminor axes of the
% standard deviation ellipsis are precisely the marginal standard
% deviations, 50 and 200, and that the imposed mean has moved the center of
% the cloud from the origin to the mean.

N=1000;
mu = [700 1000];
stdev = [50  0
           0  200];
x = randn(N,2) * stdev + repmat(mu,N,1);

clf;
plot(x(:,1),x(:,2),'+'); hold on;
set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
plot_gauss2D_pdf(mu,stdev*stdev)

%% 
% Applying some rotation of the Gaussian cloud around by an angle theta is
% obtained by multiplying the samples, while still centered on the origin,
% by some rotation matrix, and adding the imposed mean. This is equivalent
% to rotating the the standard deviation matrix.

mu2 = [750 1200];
theta=pi/4;
rotation=[cos(theta) sin(theta)
              -sin(theta)  cos(theta)];
stdev2=stdev*rotation;
x = randn(N,2) * stdev2 + repmat(mu2,N,1);

cla;
plot(x(:,1),x(:,2),'+'); hold on;
set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;
plot_gauss2D_pdf(mu2,stdev2*stdev2)

%% 
% Generating samples from a Gaussian Mixture Model (GMM) is
% straightforward. Let us produce samples from a mixture of two previous
% Gaussians, with weights 4/5 and 1/5.

x = [randn(N*4/5,2) * stdev + repmat(mu,N*4/5,1); ...
      randn(N*1/5,2) * stdev2 + repmat(mu2,N*1/5,1)] ;

x=x(randperm(N),:);

cla;
plot(x(:,1),x(:,2),'+'); hold on;
set(gca,'xlim',[0 1500],'ylim',[0 2500],'dataaspectratio',[1 1 1]);
xlabel('F1 [Hz]'); ylabel('F2 [Hz]'); grid on;

%%
% Generating samples according to an HMM model is quite straightforward,
% too. We give the example of the word "why" used in this script, and plot
% the generated data together with the corresponding states. (The content
% of HMM_generate is easy to understand). 
%
% *MATLAB function involved:*
% 
% * |[x,stateseq] = HMM_generate(means,covs,transitions)| returns a
%    D-dimensional sequence |x| (one observation per row) and a state
%    sequence |stateseq| drawn from a Markov model, given the |means| and
%    |covs| of the Gaussian multivariates describing its emission
%    probabilities (and stored in lists with empty matrices as first and
%    last elements to symbolize the entry and exit states), and given its
%    |transition| matrix.

% phonemes /u/, /a/, /i/
mu_u = [476 1050];
std_u = [92 40
         40 129];
var_u = std_u * std_u;

Pa = 0.15;
mu_a = [877 1183];
std_a = [95  20
            20  158];
var_a = std_a * std_a;

Pi = 0.15;
mu_i = [781 1681];
std_i = [76  -5
           -5  130];
var_i = std_i * std_i;

% left-right why=/uai/
why.means = {[],mu_u,mu_a,mu_i,[]};
why.covs = {[],var_u,var_a,var_i,[]};
why.trans = [ 0.0 1.0  0.0  0.0  0.0
                   0.0 0.95 0.05 0.0  0.0
	               0.0 0.0  0.95 0.05 0.0
                   0.0 0.0  0.0  0.95 0.05
	               0.0 0.0  0.0  0.0  1.0 ];

% Generate the emitting states sequence and the emitted vectors
[vectors,states]=HMM_generate(why.means, why.covs, why.trans);
clf;
plot_HMM2D_timeseries(vectors,states);

%%
% One can also plot the same data in the 2D feature space.
clf;
plot_HMM2D_featurespace(vectors,states);

%%
% Similar commands were used for generating all the training and test
% sequences used in this script, strored in gendata.m
