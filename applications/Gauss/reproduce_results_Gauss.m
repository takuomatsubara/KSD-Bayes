clear all

addpath('../../')

% simulate a dataset
rng(0)
n = 100;
Sig = 1; % measurement variance
Z1 = 1 + Sig^(1/2) * randn(1,n); % "correct" data
Z2 = Sig^(1/2) * randn(1,n); % "incorrect" data (add the mean, phi, below)
rnd = rand(1,n); % use same underlying randomness for all Bernoulli variables

% prior
Sig0 = 1; % prior standard deviation
mu0 = 0; % prior mean             
A0 = (1/2) * inv(Sig0); 
v0 = - 2 * A0 * mu0; 

%% consider increasing levels of contamination (epsilon), for fixed displacement (y)
y = 10;
eps_levels = [0,0.1,0.2];
for level = 1:length(eps_levels)
    
    eps = eps_levels(level); % contamination proportion
    contaminate = (rnd < eps);
    X = (contaminate==0) .* Z1 + (contaminate==1) .* (y + Z2);

    % standard posterior
    Sign_Bayes(level) = 1 / ((1/Sig0) + (n/Sig));
    mun_Bayes(level) = Sign_Bayes(level) * ((mu0/Sig0) + (sum(X)/Sig));

    % (non-robust) KSD-Bayes posterior
    scalar = true;
    robust = false;
    out = run_Gauss(X,scalar,robust);
    beta = min(1,out.w);
    An = A0 + beta * out.An; % posterior precision
    vn = v0 + beta * out.vn;
    Sign_KSD_Bayes(level) = (1/2) * inv(An);
    mun_KSD_Bayes(level) = -(1/2) * (An \ vn);

    % robust KSD-Bayes posterior
    robust = true;
    out = run_Gauss(X,scalar,robust);
    beta = min(1,out.w);
    An = A0 + beta * out.An; % posterior precision
    vn = v0 + beta * out.vn;
    Sign_KSD_Bayes_robust(level) = (1/2) * inv(An);
    mun_KSD_Bayes_robust(level) = -(1/2) * (An \ vn);

end

% plotting
figure()
styles = {'-','--',':','-.'};
leg = cell(length(eps_levels),1);
interval = [0,3.5];
for level = 1:length(eps_levels)
    
    leg{level} = ['\epsilon = ',num2str(eps_levels(level),'%u')];
    
    % standard posterior
    subplot(2,3,1)
    hold on
    fplot(@(theta) normpdf(theta,mun_Bayes(level),Sign_Bayes(level)^(1/2)) , ...
          interval,['k',styles{level}])
    title('Bayes')
    ylabel(['Pseudo Posterior (y = ',num2str(y),')'])
    set(gca,'ylim',[0,5])
    set(gca,'xticklabel',[])
    box on
    
    % (non-robust) KSD-Bayes posterior
    subplot(2,3,2)
    hold on
    fplot(@(theta) normpdf(theta,mun_KSD_Bayes(level),Sign_KSD_Bayes(level)^(1/2)) , ...
          interval, ['k',styles{level}])
    title('KSD-Bayes')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    set(gca,'xticklabel',[])
    box on
    
    % robust KSD-Bayes posterior
    subplot(2,3,3)
    hold on
    fplot(@(theta) normpdf(theta,mun_KSD_Bayes_robust(level),Sign_KSD_Bayes_robust(level)^(1/2)) , ...
          interval, ['k',styles{level}])   
    title('Robust KSD-Bayes')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    set(gca,'xticklabel',[])
    box on
    
end
legend(leg)

%% consider increasing displacement (phi), for fixed levels of contamination (epsilon)
eps = 0.1;
y_levels = [1,10,20];
for level = 1:length(y_levels)
    
    y = y_levels(level); % contamination displacement
    contaminate = (rnd < eps);
    X = (contaminate==0) .* Z1 + (contaminate==1) .* (y + Z2);

    % standard posterior
    Sign_Bayes(level) = 1 / ((1/Sig0) + (n/Sig));
    mun_Bayes(level) = Sign_Bayes(level) * ((mu0/Sig0) + (sum(X)/Sig));

    % (non-robust) KSD-Bayes posterior
    scalar = true;
    robust = false;
    out = run_Gauss(X,scalar,robust);
    An = A0 + out.w * out.An; % posterior precision
    vn = v0 + out.w * out.vn;
    Sign_KSD_Bayes(level) = (1/2) * inv(An);
    mun_KSD_Bayes(level) = -(1/2) * (An \ vn);

    % robust KSD-Bayes posterior
    robust = true;
    out = run_Gauss(X,scalar,robust);
    An = A0 + out.w * out.An; % posterior precision
    vn = v0 + out.w * out.vn;
    Sign_KSD_Bayes_robust(level) = (1/2) * inv(An);
    mun_KSD_Bayes_robust(level) = -(1/2) * (An \ vn);

end

% plotting
styles = {'-','--',':','-.'};
leg = cell(length(y_levels),1);
interval = [0,3.5];
for level = 1:length(y_levels)
    
    leg{level} = ['y = ',num2str(y_levels(level),'%u')];
    
    % standard posterior
    subplot(2,3,4)
    hold on
    fplot(@(theta) normpdf(theta,mun_Bayes(level),Sign_Bayes(level)^(1/2)) , ...
          interval,['k',styles{level}])
    xlabel('\theta')
    ylabel(['Pseudo Posterior (\epsilon = ',num2str(eps),')'])
    set(gca,'ylim',[0,5])
    box on
    
    % (non-robust) KSD-Bayes posterior
    subplot(2,3,5)
    hold on
    fplot(@(theta) normpdf(theta,mun_KSD_Bayes(level),Sign_KSD_Bayes(level)^(1/2)) , ...
          interval, ['k',styles{level}])
    xlabel('\theta')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    box on
    
    % robust KSD-Bayes posterior
    subplot(2,3,6)
    hold on
    fplot(@(theta) normpdf(theta,mun_KSD_Bayes_robust(level),Sign_KSD_Bayes_robust(level)^(1/2)) , ...
          interval, ['k',styles{level}])   
    xlabel('\theta')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    box on
    
end
legend(leg)











