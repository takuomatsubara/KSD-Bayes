function out = aux_power_posterior_Bayes(X)
% out = aux_power_posterior_Bayes(X)
%
% Perform power posterior Bayesian inference for the normal location model.
%
% Input:
% X = 1xn dataset.
%
% Output:
% out = function handle to pdf of the generalised Bayes posterior.

n = length(X);

% power posterior 
beta = sqrt( (2 + mean(X)^2) / (1 + mean(X.^2)) );
mu = (beta * n / (1 + beta * n)) * mean(X);
sigma = sqrt(1 / (1 + beta * n));

% generalised posterior pdf
out = @(theta) normpdf(theta,mu,sigma);