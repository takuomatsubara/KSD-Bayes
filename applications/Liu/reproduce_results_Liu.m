clear all

addpath('../../')
addpath('../../utilities/')

% simulate a dataset
rng(0)
Prec = [1,    -0.6, -0.2, -0.2, -0.2; ...
        -0.6, 1,    0,    0,    0   ; ...
        -0.2, 0,    1,    0,    0   ; ...
        -0.2, 0,    0,    1,    0   ; ...
        -0.2, 0,    0,    0,    1   ]; % precision matrix
Inv_Prec = inv(Prec);    
n = 500;
Z1 = mvnrnd(zeros(1,5),Inv_Prec,n)'; % "correct" data
Z2 = mvnrnd(10 * ones(1,5),Inv_Prec,n)'; % "incorrect" data
rnd = rand(1,n); % use same underlying randomness for all Bernoulli variables

% prior
Sig0 = 100*eye(2,2); % prior covariance matrix
mu0 = zeros(2,1); % prior mean             
A0 = (1/2) * inv(Sig0); 
v0 = - 2 * A0 * mu0; 

% normalisation constant via Gauss-Hermite cubature
order = 10; % order of the Gauss-Hermite rule
[nodes_1D,weights_1D] = GaussHermite(order);
Sig_45 = Inv_Prec(4:5,4:5);
Sig_45_chol = chol(Sig_45,'lower');
nodes_2D = [repelem(nodes_1D,order,1), repmat(nodes_1D,order,1)]; % tensor product
nodes_2D = nodes_2D * Sig_45_chol; % correlation structure
weights_2D = (2*pi)^(-1) * repelem(weights_1D,order,1) .* repmat(weights_1D,order,1); % normalised for integrals wrt N(0,I_2)
r_mod = @(t1,t2,W) exp(t1*tanh(W(:,1)) + t2*tanh(W(:,2)));
log_r = @(t1,t2,X) t1*tanh(X(4,:)) + t2*tanh(X(5,:));
C = @(t1,t2) (det(Sig_45_chol))^(-1) * weights_2D' * r_mod(t1,t2,nodes_2D);

% log-likelihood
log_lik = @(t1,t2,X) sum( log( mvnpdf(X',zeros(1,5),Inv_Prec) ) ...
                          + log_r(t1,t2,X)' - log( C(t1,t2) ) ) ;

% log (un-normalised) posterior 
log_post = @(t1,t2,X) log_lik(t1,t2,X) + log( mvnpdf([t1,t2],mu0',Sig0) );

% consider increasing levels of contamination
eps_levels = [0,0.1,0.2];
for level = 1:length(eps_levels)
    
    eps = eps_levels(level); % contamination proportion
    contaminate = (rnd < eps);
    X{level} = (contaminate==0) .* Z1 + (contaminate==1) .* Z2;

    % (non-robust) KSD-Bayes posterior
    robust = false;
    out = run_Liu(X{level},robust);
    beta(level) = min(1,out.w);
    An = A0 + beta(level) * out.An; % posterior precision
    vn = v0 + beta(level) * out.vn;
    Sign_KSD_Bayes{level} = (1/2) * inv(An);
    mun_KSD_Bayes{level} = -(1/2) * (An \ vn);

    % robust KSD-Bayes posterior
    robust = true;
    out = run_Liu(X{level},robust);
    beta_robust(level) = min(1,out.w);
    An = A0 + beta_robust(level) * out.An; % posterior precision
    vn = v0 + beta_robust(level) * out.vn;
    Sign_KSD_Bayes_robust{level} = (1/2) * inv(An);
    mun_KSD_Bayes_robust{level} = -(1/2) * (An \ vn);

end

% plotting
figure()
interval = [-0.5,3];
for level = 1:length(eps_levels)
    
    leg{level} = ['\epsilon = ',num2str(eps_levels(level),'%u')];
    
    % standard Bayes posterior
    subplot(length(eps_levels),3,3*level-2)
    lp = log_post(0,0,X{level}); % baseline for avoiding numerical underflow
    fcontour(@(t1,t2) exp( log_post(t1,t2,X{level}) - lp ), interval, ...
             'LineWidth', 0.5, 'LineColor', 'black')
    if level == 1
        title({'Bayes',['\epsilon = ',num2str(eps_levels(level))]})
    else
        title(['\epsilon = ',num2str(eps_levels(level))])
    end
    if level == length(eps_levels)
        xlabel('\theta_1')
    else
        set(gca,'Xticklabel',[])
    end
    ylabel('\theta_2')
    box on
    hline(0,'k')
    vline(0,'k')
       
    % (non-robust) KSD-Bayes posterior
    subplot(length(eps_levels),3,3*level-1)
    fcontour(@(t1,t2) mvnpdf([t1,t2],mun_KSD_Bayes{level}',Sign_KSD_Bayes{level}),interval, ...
             'LineWidth', 0.5, 'LineColor', 'black')
    if level == 1
        title({'KSD-Bayes',['\epsilon = ',num2str(eps_levels(level))]})
    else
        title(['\epsilon = ',num2str(eps_levels(level))])
    end
    if level == length(eps_levels)
        xlabel('\theta_1')
    else
        set(gca,'Xticklabel',[])
    end
    set(gca,'Yticklabel',[])
    box on
    hline(0,'k')
    vline(0,'k')
    
    % robust KSD-Bayes posterior
    subplot(length(eps_levels),3,3*level)
    fcontour(@(t1,t2) mvnpdf([t1,t2],mun_KSD_Bayes_robust{level}',Sign_KSD_Bayes_robust{level}),interval, ...
             'LineWidth', 0.5, 'LineColor', 'black')
    if level == 1
        title({'Robust KSD-Bayes',['\epsilon = ',num2str(eps_levels(level))]})
    else
        title(['\epsilon = ',num2str(eps_levels(level))])
    end
    if level == length(eps_levels)
        xlabel('\theta_1')
    else
        set(gca,'Xticklabel',[])
    end
    set(gca,'Yticklabel',[])
    box on
    hline(0,'k')
    vline(0,'k')
    
end











