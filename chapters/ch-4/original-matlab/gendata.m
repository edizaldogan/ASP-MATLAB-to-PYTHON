% GENDATA Generation of simulation data for dictation_machine_ASP.m
%

% Specification of Gaussian statistics for the four vowels 

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

Pe = 0.4;
mu_e = [456 2098];
std_e = [84  -15
         -15  111];
var_e = std_e * std_e;

Pu = 0.3;
mu_u = [476 1050];
std_u = [92 40
         40 129];
var_u = std_u * std_u;

N=20000;
a.training = randn(N*Pa,2) * std_a + repmat(mu_a,N*Pa,1);
e.training = randn(N*Pe,2) * std_e + repmat(mu_e,N*Pe,1);
i.training = randn(N*Pi,2)   * std_i  + repmat(mu_i,N*Pi,1);
u.training = randn(N*Pu,2) * std_u + repmat(mu_u,N*Pu,1);
N=1000;
a.test = randn(N*Pa,2) * std_a + repmat(mu_a,N*Pa,1);
e.test = randn(N*Pe,2) * std_e + repmat(mu_e,N*Pe,1);
i.test  = randn(N*Pi,2) * std_i  + repmat(mu_i,N*Pi,1);
u.test = randn(N*Pu,2) * std_u + repmat(mu_u,N*Pu,1);

vowels={a,e,i,u};

% Specification of the word HMMs 

% left-right hear=/ie/
hear.word = 'hear';
hear.means = {[],mu_i,mu_e,[]};
hear.covs  = {[],var_i,var_e,[]};
hear.trans = [0.0 1.0  0.0  0.0 
                  0.0 0.95 0.05 0.0
                  0.0 0.0  0.95 0.05
                  0.0 0.0  0.0  1    ];

% left-right here=/ie/
here = hear;
here.word = 'here';

% left-right you=/iu/
you.word = 'you';
you.means = {[],mu_i,mu_u,[]};
you.covs  = {[],var_i,var_u,[]};
you.trans = [0.0 1.0  0.0  0.0 
                 0.0 0.95 0.05 0.0
                 0.0 0.0  0.95 0.05
                 0.0 0.0  0.0  1    ];

% left-right are=/ae/
are.word = 'are';
are.means = {[],mu_a,mu_e,[]};
are.covs  = {[],var_a,var_e,[]};
are.trans = [0.0 1.0  0.0  0.0 
                 0.0 0.95 0.05 0.0
                 0.0 0.0  0.95 0.05
                 0.0 0.0  0.0  1    ];

% left-right we=/ui/
we.word='we';
we.means = {[],mu_u,mu_i,[]};
we.covs  = {[],var_u,var_i,[]};
we.trans = [0.0 1.0  0.0  0.0 
               0.0 0.95 0.05 0.0
               0.0 0.0  0.95 0.05
               0.0 0.0  0.0  1    ];

% left-right why=/uai/
why.word='why';
why.means = {[],mu_u,mu_a,mu_i,[]};
why.covs  = {[],var_u,var_a,var_i,[]};
why.trans = [ 0.0 1.0  0.0  0.0  0.0
                   0.0 0.95 0.05 0.0  0.0
	               0.0 0.0  0.95 0.05 0.0
                   0.0 0.0  0.0  0.95 0.05
	               0.0 0.0  0.0  0.0  1.0 ];

% sampling word HMMs
hear.training_all=[];               
here.training_all=[];               
you.training_all=[];               
are.training_all=[];               
we.training_all=[];               
why.training_all=[];               
for i=1:100;
   hear.training{i}=HMM_generate(hear.means, hear.covs, hear.trans);
   hear.training_all=[hear.training_all; hear.training{i}];
   here.training{i}=HMM_generate(here.means, here.covs, here.trans);
   here.training_all=[here.training_all; here.training{i}];
   you.training{i}=HMM_generate(you.means, you.covs, you.trans);
   you.training_all=[you.training_all; you.training{i}];
   are.training{i}=HMM_generate(are.means, are.covs, are.trans);
   are.training_all=[are.training_all; are.training{i}];
   we.training{i}=HMM_generate(we.means, we.covs, we.trans);
   we.training_all=[we.training_all; we.training{i}];
   why.training{i}=HMM_generate(why.means, why.covs, why.trans);
   why.training_all=[why.training_all; why.training{i}];
end;
hear.test_all=[];               
here.test_all=[];               
you.test_all=[];               
are.test_all=[];               
we.test_all=[];               
why.test_all=[];               
for i=1:100;
   hear.test{i}=HMM_generate(hear.means, hear.covs, hear.trans);
   hear.test_all=[hear.test_all; hear.test{i}];
   here.test{i}=HMM_generate(here.means, here.covs, here.trans);
   here.test_all=[here.test_all; here.test{i}];
   you.test{i}=HMM_generate(you.means, you.covs, you.trans);
   you.test_all=[you.test_all; you.test{i}];
   are.test{i}=HMM_generate(are.means, are.covs, are.trans);
   are.test_all=[are.test_all; are.test{i}];
   we.test{i}=HMM_generate(we.means, we.covs, we.trans);
   we.test_all=[we.test_all; we.test{i}];
   why.test{i}=HMM_generate(why.means, why.covs, why.trans);
   why.test_all=[why.test_all; why.test{i}];
end;

words={why,you,we,are,hear,here};

% Using sample words to build admissible sentences
for i=1:30
    indices=fix(rand(1,6)*100+1);
    a=indices(1);b=indices(2);c=indices(3);
    d=indices(4);e=indices(5);f=indices(6);
    sentences{(i-1)*5+1}.test={we.test{a},hear.test{b},...
        why.test{c},you.test{d},are.test{e},here.test{f}};
    sentences{(i-1)*5+1}.wordindex={3,5,1,2,4,6};
    sentences{(i-1)*5+2}.test={why.test{a},we.test{b},...
        hear.test{c},you.test{d}};
    sentences{(i-1)*5+2}.wordindex={1,3,5,2};
    sentences{(i-1)*5+3}.test={why.test{b},are.test{b},...
        you.test{c},here.test{d}};
    sentences{(i-1)*5+3}.wordindex={1,4,2,6};
    sentences{(i-1)*5+4}.test={we.test{c},hear.test{d},...
        you.test{e},here.test{f}};
    sentences{(i-1)*5+4}.wordindex={3,5,2,6};
    sentences{(i-1)*5+5}.test={here.test{a},we.test{b},...
        are.test{c}};
    sentences{(i-1)*5+5}.wordindex={6,3,4};
end;

% Save the relevant variables
save data vowels words sentences