clear all

addpath('../../')
addpath('../../utilities/')

% load Sachs et al dataset (outliers removed)
dataset = load('X.mat');
names = dataset.names;

% dimensions
[d,n] = size(dataset.X);
p = d*(d+1)/2; 

% dataset
Z1 = dataset.X; % "correct" data
Z2 = 10*ones(d,n); % "incorrect" data
rnd = rand(1,n); % use same underlying randomness for all Bernoulli variables

% prior
Sig0 = eye(p,p); % prior "covariance matrix" for parameter vector (n.b. not 
                 % actually a covariance matrix, since the Gaussian prior will 
                 % be truncated)
mu0 = zeros(p,1); % prior "mean"                 
A0 = (1/2) * inv(Sig0); 
v0 = - 2 * A0 * mu0; 

% consider increasing levels of contamination
eps_levels = [0,0.01,0.02,0.05];
for level = 1:length(eps_levels)
    disp(['level = ',num2str(level,'%u')])
    
    eps = eps_levels(level); % contamination proportion
    contaminate = (rnd < eps);
    X = (contaminate==0) .* Z1 + (contaminate==1) .* Z2;

    % (non-robust) KSD-Bayes posterior
    disp('non-robust')
    robust = false;
    out = run_EGM(X,robust);
    beta = min(1,out.w);
    An = A0 + beta * out.An; 
    vn = v0 + beta * out.vn;
    Sign_KSD_Bayes{level} = (1/2) * inv(An);
    mun_KSD_Bayes{level} = -(1/2) * (An \ vn);
    
    % robust KSD-Bayes posterior
    disp('robust')
    robust = true;
    out = run_EGM(X,robust);
    beta = min(1,out.w);
    An = A0 + beta * out.An; 
    vn = v0 + beta * out.vn;
    Sign_KSD_Bayes_robust{level} = (1/2) * inv(An);
    mun_KSD_Bayes_robust{level} = -(1/2) * (An \ vn);

end

% save
save('results_EGM_robust.mat','Sign_KSD_Bayes','mun_KSD_Bayes', ...
                              'Sign_KSD_Bayes_robust','mun_KSD_Bayes_robust')                         
                          
% plotting networks
figure()
n_edges = 5; % number of edges to select
int_idx = interaction_index_EGM(d);
for level = 1:length(eps_levels)
    
    % (non-robust) KSD-Bayes network
    subplot(2,length(eps_levels),level)
    Z_KSD_Bayes = mun_KSD_Bayes{level} ./ (diag(Sign_KSD_Bayes{level}).^(1/2)); % number of standard deviations from 0    
    [~,I] = sort(Z_KSD_Bayes(int_idx(:,1)),'descend'); % threshold to obtain network
    edges = int_idx(I(1:n_edges),2:3); % edges selected
    num_correct = compare_Sachs(edges); % number of "correct" edges
    NodeTable = table(names,'VariableNames',{'Name'});
    EdgeTable = table(edges,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable,NodeTable);
    plot(G,'k-','layout','circle')
    title({'KSD-Bayes',['\epsilon = ',num2str(eps_levels(level))], ...
           ['Edges = ',num2str(num_correct,'%u'),'/',num2str(n_edges,'%u')]})
    set(gca,'XColor', 'none','YColor','none')
    
    % robust KSD-Bayes network
    subplot(2,length(eps_levels),length(eps_levels) + level)
    Z_KSD_Bayes_robust = mun_KSD_Bayes_robust{level} ./ (diag(Sign_KSD_Bayes_robust{level}).^(1/2)); % number of standard deviations from 0    
    [~,I] = sort(Z_KSD_Bayes_robust(int_idx(:,1)),'descend'); % threshold to obtain network
    edges = int_idx(I(1:n_edges),2:3); % edges selected
    num_correct = compare_Sachs(edges); % number of "correct" edges
    NodeTable = table(names,'VariableNames',{'Name'});
    EdgeTable = table(edges,'VariableNames',{'EndNodes'});
    G = graph(EdgeTable,NodeTable);
    plot(G,'k-','layout','circle')
    title({'Robust KSD-Bayes',['\epsilon = ',num2str(eps_levels(level))], ...
           ['Edges = ',num2str(num_correct,'%u'),'/',num2str(n_edges,'%u')]})
    set(gca,'XColor', 'none','YColor','none')

end


























