function out = aux_MMD_Bayes(X)
% out = aux_MMD_Bayes(X)
%
% Perform MMD-Bayesian inference for the normal location model.
%
% Input:
% X = 1xn dataset.
%
% Output:
% out = function handle to pdf of the generalised Bayes posterior.

n = length(X);

% MMD squared
MMD2 = @(theta) (1/3) * exp(-theta^2/6) ...
                - (2/n) * sqrt(1/3) * sum(exp(-(theta-X).^2/3)) ...
                + (1/n)^2 * sum(sum(exp(-pdist2(X,X).^2)));

% learning rate
%beta = exp(n) / n; % the rate stated in Appendix H of the paper
beta = 1; % a more appropriate rate
            
% un-normalised pdf            
q = @(theta) normpdf(theta) * exp(- beta * n * MMD2(theta));
            
% approximate normalisation constant
n_grid = 10000; % grid resolution
L_grid = [-5,5]; % grid limits
x_grid = linspace(L_grid(1),L_grid(2),n_grid);
q_grid = zeros(1,n_grid);
for i = 1:n_grid
    q_grid(i) = q(x_grid(i));
end
C = trapz(x_grid,q_grid);

% generalised posterior pdf
out = @(theta) q(theta) / C;
         