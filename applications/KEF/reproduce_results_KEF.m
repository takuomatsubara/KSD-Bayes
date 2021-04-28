clear all
rng(0)

addpath('../../')

% load galaxy dataset
dataset = load('X.mat'); 
center = mean(dataset.X);
scale = 0.5 * std(dataset.X);

% dimensions
[d,n] = size(dataset.X);

% center and scale
Z1 = (dataset.X - center) / scale; % "correct" data
D = 10; % location for displaced data
Z2 = D + 0.1*randn(1,n);
rnd = rand(1,n); % use same underlying randomness for all Bernoulli variables

% prior
p = 25; % number of basis functions to use
Sig0 = 100 * diag((1:p).^(-1.1)); % prior covariance matrix
mu0 = zeros(p,1); % prior mean             
A0 = (1/2) * inv(Sig0); 
v0 = - 2 * A0 * mu0; 

% width of the reference measure
L = 3; 

% consider increasing levels of contamination
eps_levels = [0,0.1,0.2];
for level = 1:length(eps_levels)
    
    eps = eps_levels(level); % contamination proportion
    contaminate{level} = (rnd < eps);
    X{level} = (contaminate{level}==0) .* Z1 + (contaminate{level}==1) .* Z2;

    % (non-robust) KSD-Bayes posterior
    robust = false;
    out = run_KEF(X{level},p,L,robust);
    beta = min(1,out.w);
    An = A0 + beta * out.An; % posterior precision
    vn = v0 + beta * out.vn;
    Sign_KSD_Bayes{level} = (1/2) * nearestSPD(inv(An)); % ensures numerical SPD
    mun_KSD_Bayes{level} = -(1/2) * (An \ vn);

    % robust KSD-Bayes posterior
    robust = true;
    out = run_KEF(X{level},p,L,robust);
    beta = min(1,out.w);
    An = A0 + beta * out.An; % posterior precision
    vn = v0 + beta * out.vn;
    Sign_KSD_Bayes_robust{level} = (1/2) * nearestSPD(inv(An)); % ensures numerical SPD
    mun_KSD_Bayes_robust{level} = -(1/2) * (An \ vn);

end

% plotting
figure()
X_grid = linspace(-3*L,max(3*L,D+3),1000); % plotting grid
n_samples = 10; % number of samples to plot
for level = 1:length(eps_levels)
   
    %% plot dataset
    subplot(3,length(eps_levels),level)
    hold on
    histogram(center+scale*X{level}(:,contaminate{level}==1), ...
              'BinWidth',1000,'FaceColor','k')
    histogram(center+scale*X{level}(:,contaminate{level}==0), ...
              'BinWidth',1000,'FaceColor','w')
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'xlim',[0,50000])
    if level == length(eps_levels)
        legend({'contaminated','original'})
    end
    
    title({'Dataset',['\epsilon = ',num2str(eps_levels(level))]})
    box on
    
    %% (non-robust) KSD-Bayes
    subplot(3,length(eps_levels),length(eps_levels)+level)
    hold on
    
    % plot (non-robust) KSD-Bayes posterior mean
    pdf_vals = pdf_KEF(X_grid,mun_KSD_Bayes{level},L);
    plot(center+scale*X_grid,pdf_vals/scale,'k-')
    
    % plot (non-robust) KSD-Bayes posterior samples
    for i = 1:n_samples
        coeff = mvnrnd(mun_KSD_Bayes{level}',Sign_KSD_Bayes{level})';
        pdf_vals = pdf_KEF(X_grid,coeff,L);
        plot(center+scale*X_grid,pdf_vals/scale,'k:')
    end
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    if level ~= 1
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
    end
    title('KSD-Bayes')
    set(gca,'xlim',[0,50000])
    box on
    
    %% Robust KSD-Bayes
    subplot(3,length(eps_levels),2*length(eps_levels) + level)
    hold on
    
    % plot robust KSD-Bayes posterior mean
    pdf_vals = pdf_KEF(X_grid,mun_KSD_Bayes_robust{level},L);
    plot(center+scale*X_grid,pdf_vals/scale,'k-')
    
    % plot KSD-Bayes posterior samples
    for i = 1:n_samples
        coeff = mvnrnd(mun_KSD_Bayes_robust{level}',Sign_KSD_Bayes_robust{level})';
        pdf_vals = pdf_KEF(X_grid,coeff,L);
        plot(center+scale*X_grid,pdf_vals/scale,'k:')
    end
    if level ~= 1
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
    end
    title('Robust KSD-Bayes')
    set(gca,'xlim',[0,50000])
    box on
    if level == length(eps_levels)
        legend({'mean','samples'})
    end
    
end
    



