function out = run_Liu(X,robust)
% run_Liu(X,robust)
%
% Perform KSD-Bayesian inference for the Liu et al experiment.
%
% Input:
% X = 5xn dataset, each column one sample.
% robust = logical, indicating whether or not to be bias-robust.
%
% Output:
% out.An = 2x2 matrix An, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.vn = 2x1 vector v_n, s.t. nKSD^2 = theta' An theta + vn' theta + C.

% use faster computation with scalar kernel and preconditioner
scalar = true;

% KSD-Bayesian inference for the Liu et al experiment
out = KSD_Bayes(X, @(x) grad_T_Liu(x), ...
                   @(x) grad_b_Liu(x), ...
                   @(x) M_Liu(x,scalar,robust), ...
                   @(x,y) K_Liu(x,y,scalar), ...
                   scalar );               
               
end               

