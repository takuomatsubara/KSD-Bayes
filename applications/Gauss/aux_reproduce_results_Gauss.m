clear all

addpath('../../')
addpath('../../utilities/')

%% Sensitivity to kernel parameters
disp('Sensitivity to kernel parameters')

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

% consider increasing levels of contamination (epsilon), for fixed displacement (y)
y = 10;
eps_levels = [0,0.1,0.2];
for level = 1:length(eps_levels)
    
    eps = eps_levels(level); % contamination proportion
    contaminate = (rnd < eps);
    X = (contaminate==0) .* Z1 + (contaminate==1) .* (y + Z2);

    % consider different kernel parameters
    sigma_vals = [0.5,1,2];
    gamma_vals = [0.1,0.5,0.9];
    
    for sig_ix = 1:length(sigma_vals)
        for gam_ix = 1:length(gamma_vals)
            
            sigma = sigma_vals(sig_ix);
            gamma = gamma_vals(gam_ix);
            a = 1;
            b = 0;

            % robust KSD-Bayes posterior
            scalar = true;
            robust = true;
            out = aux_run_Gauss(X,scalar,robust,sigma,gamma,a,b);
            beta = min(1,out.w);
            An = A0 + beta * out.An; % posterior precision
            vn = v0 + beta * out.vn;
            Sign_KSD_Bayes(level,sig_ix,gam_ix) = (1/2) * inv(An);
            mun_KSD_Bayes(level,sig_ix,gam_ix) = -(1/2) * (An \ vn);

        end
    end

end

% plotting
figure()
styles = {'-','--',':','-.'};
leg = cell(length(eps_levels),1);
interval = [0,2];
for level = 1:length(eps_levels)
    
    leg{level} = ['\epsilon = ',num2str(eps_levels(level),'%u')];
    
    for sig_ix = 1:length(sigma_vals)
        for gam_ix = 1:length(gamma_vals)
            
            sigma = sigma_vals(sig_ix);
            gamma = gamma_vals(gam_ix);

            % (non-robust) KSD-Bayes posterior
            subplot(3,3,3*(gam_ix-1)+sig_ix)
            hold on
            fplot(@(theta) normpdf(theta,mun_KSD_Bayes(level,sig_ix,gam_ix), ...
                                   Sign_KSD_Bayes(level,sig_ix,gam_ix)^(1/2)) , ...
                  interval, ['k',styles{level}])
            if (sig_ix == 2) && (gam_ix == 1)
                title({'Robust KSD-Bayes',['\sigma = ',num2str(sigma)]})
            elseif gam_ix == 1
                title(['\sigma = ',num2str(sigma)])
            end
            if sig_ix == 1
                ylabel(['\gamma = ',num2str(gamma)])
            end
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'ylim',[0,5])
            if gam_ix ~= 3
                set(gca,'xticklabel',[])
            end
            box on
    
        end
    end       
end
legend(leg)


%% Sampling distribution of beta
disp('Sampling distribution of beta')

n_sims = 100; % number of datasets to simulate

% prior
Sig0 = 1; % prior standard deviation
mu0 = 0; % prior mean             
A0 = (1/2) * inv(Sig0); 
v0 = - 2 * A0 * mu0; 

n_vals = [10,50,100]; % number of samples in dataset
for sim_ix = 1:n_sims
    disp(['Sim ',num2str(sim_ix,'%u'),' of ',num2str(n_sims,'%u')])
    
    for n_ix = 1:length(n_vals)
    
        % simulate a dataset
        n = n_vals(n_ix);
        Sig = 1; % measurement variance
        Z1 = 1 + Sig^(1/2) * randn(1,n); % "correct" data
        Z2 = Sig^(1/2) * randn(1,n); % "incorrect" data (add the mean, phi, below)
        rnd = rand(1,n); % use same underlying randomness for all Bernoulli variables

        % consider increasing levels of contamination (epsilon), for fixed displacement (y)
        y = 10;
        eps_levels = [0,0.1,0.2];
        for level = 1:length(eps_levels)

            eps = eps_levels(level); % contamination proportion
            contaminate = (rnd < eps);
            X = (contaminate==0) .* Z1 + (contaminate==1) .* (y + Z2);

            % (non-robust) KSD-Bayes posterior
            scalar = true;
            robust = false;
            out = run_Gauss(X,scalar,robust);
            beta(sim_ix,n_ix,level) = min(1,out.w);

        end
    end
end

% plotting
figure()
data = {squeeze(beta(:,1,:)),squeeze(beta(:,2,:)),squeeze(beta(:,3,:))}; 
boxplotGroup(data, 'PrimaryLabels', {'\epsilon = 0','\epsilon = 0.1','\epsilon = 0.2'}, ...
  'SecondaryLabels',{'n=10','n=50','n=100'}, 'GroupLabelType', 'Vertical')


%% Efficiency vs robustness
disp('Efficiency vs robustness')

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

% consider increasing levels of contamination (epsilon), for fixed displacement (y)
y = 10;
eps_levels = [0,0.1,0.2];
for level = 1:length(eps_levels)
    
    eps = eps_levels(level); % contamination proportion
    contaminate = (rnd < eps);
    X = (contaminate==0) .* Z1 + (contaminate==1) .* (y + Z2);

    % consider different preconditioner parameters
    a_vals = [0.1,1,10];
    b_vals = [-5,0,5];
    
    for a_ix = 1:length(a_vals)
        for b_ix = 1:length(b_vals)
            
            a = a_vals(a_ix);
            b = b_vals(b_ix);

            % robust KSD-Bayes posterior
            scalar = true;
            robust = true;
            sigma = sqrt(regscm(X','verbose','off'));
            gamma = 0.5;
            out = aux_run_Gauss(X,scalar,robust,sigma,gamma,a,b);
            beta = min(1,out.w);
            An = A0 + beta * out.An; % posterior precision
            vn = v0 + beta * out.vn;
            Sign_KSD_Bayes(level,a_ix,b_ix) = (1/2) * inv(An);
            mun_KSD_Bayes(level,a_ix,b_ix) = -(1/2) * (An \ vn);

        end
    end

end

% plotting
figure()
styles = {'-','--',':','-.'};
leg = cell(length(eps_levels),1);
interval = [-2,3];
for level = 1:length(eps_levels)
    
    leg{level} = ['\epsilon = ',num2str(eps_levels(level),'%u')];
    
    for a_ix = 1:length(a_vals)
        for b_ix = 1:length(b_vals)
            
            a = a_vals(a_ix);
            b = b_vals(b_ix);

            % (non-robust) KSD-Bayes posterior
            subplot(3,3,3*(a_ix-1)+b_ix)
            hold on
            fplot(@(theta) normpdf(theta,mun_KSD_Bayes(level,a_ix,b_ix), ...
                                   Sign_KSD_Bayes(level,a_ix,b_ix)^(1/2)) , ...
                  interval, ['k',styles{level}])
            if (a_ix == 1) && (b_ix == 2)
                title({'Robust KSD-Bayes',['b = ',num2str(b)]})
            elseif a_ix == 1
                title(['b = ',num2str(b)])
            end
            if b_ix == 1
                ylabel(['a = ',num2str(a)])
            end
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'ylim',[0,5])
            if a_ix ~= 3
                set(gca,'xticklabel',[])
            end
            box on
    
        end
    end       
end
legend(leg)


%% Comparison to power posteriors and MMD Bayes

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

% consider increasing levels of contamination (epsilon), for fixed displacement (y)
y = 10;
eps_levels = [0,0.1,0.2];
for level = 1:length(eps_levels)
    
    eps = eps_levels(level); % contamination proportion
    contaminate = (rnd < eps);
    X = (contaminate==0) .* Z1 + (contaminate==1) .* (y + Z2);

    % robust KSD-Bayes posterior
    scalar = true;
    robust = true;
    out = run_Gauss(X,scalar,robust);
    beta = min(1,out.w);
    An = A0 + beta * out.An; % posterior precision
    vn = v0 + beta * out.vn;
    Sign_KSD_Bayes_robust(level) = (1/2) * inv(An);
    mun_KSD_Bayes_robust(level) = -(1/2) * (An \ vn);

    % power posterior
    power_posterior_pdf{level} = aux_power_posterior_Bayes(X);
    
    % MMD Bayes posterior
    MMD_Bayes_posterior_pdf{level} = aux_MMD_Bayes(X);
    
end

% plotting
figure()
styles = {'-','--',':','-.'};
leg = cell(length(eps_levels),1);
interval = [0,3.5];
for level = 1:length(eps_levels)
    
    leg{level} = ['\epsilon = ',num2str(eps_levels(level),'%u')];
      
    % robust KSD-Bayes posterior
    subplot(2,3,1)
    hold on
    fplot(@(theta) normpdf(theta,mun_KSD_Bayes_robust(level),Sign_KSD_Bayes_robust(level)^(1/2)) , ...
          interval, ['k',styles{level}])   
    title('Robust KSD-Bayes')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    set(gca,'xticklabel',[])
    box on
    
    % power posterior
    subplot(2,3,2)
    hold on
    fplot(@(theta) power_posterior_pdf{level}(theta) , ...
          interval, ['k',styles{level}])   
    title('Power Posterior')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    set(gca,'xticklabel',[])
    box on
    
    % MMD Bayes
    subplot(2,3,3)
    hold on
    fplot(@(theta) MMD_Bayes_posterior_pdf{level}(theta) , ...
          interval, ['k',styles{level}])   
    title('MMD Bayes')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    set(gca,'xticklabel',[])
    box on
    
end
legend(leg)

% consider increasing displacement (phi), for fixed levels of contamination (epsilon)
eps = 0.1;
y_levels = [1,10,20];
for level = 1:length(y_levels)
    
    y = y_levels(level); % contamination displacement
    contaminate = (rnd < eps);
    X = (contaminate==0) .* Z1 + (contaminate==1) .* (y + Z2);

    % robust KSD-Bayes posterior
    scalar = true;
    robust = true;
    out = run_Gauss(X,scalar,robust);
    An = A0 + out.w * out.An; % posterior precision
    vn = v0 + out.w * out.vn;
    Sign_KSD_Bayes_robust(level) = (1/2) * inv(An);
    mun_KSD_Bayes_robust(level) = -(1/2) * (An \ vn);
    
    % power posterior
    power_posterior_pdf{level} = aux_power_posterior_Bayes(X);
    
    % MMD Bayes posterior
    MMD_Bayes_posterior_pdf{level} = aux_MMD_Bayes(X);

end

% plotting
styles = {'-','--',':','-.'};
leg = cell(length(y_levels),1);
interval = [0,3.5];
for level = 1:length(y_levels)
    
    leg{level} = ['y = ',num2str(y_levels(level),'%u')];
    
    % robust KSD-Bayes posterior
    subplot(2,3,4)
    hold on
    fplot(@(theta) normpdf(theta,mun_KSD_Bayes_robust(level),Sign_KSD_Bayes_robust(level)^(1/2)) , ...
          interval, ['k',styles{level}])   
    xlabel('\theta')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    box on
    
    % power posterior
    subplot(2,3,5)
    hold on
    fplot(@(theta) power_posterior_pdf{level}(theta) , ...
          interval, ['k',styles{level}])   
    title('Power Posterior')
    xlabel('\theta')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    box on
    
    % MMD Bayes
    subplot(2,3,6)
    hold on
    fplot(@(theta) MMD_Bayes_posterior_pdf{level}(theta) , ...
          interval, ['k',styles{level}])   
    title('MMD Bayes')
    xlabel('\theta')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'ylim',[0,5])
    box on
    
end
legend(leg)





