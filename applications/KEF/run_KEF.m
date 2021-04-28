function out = run_KEF(X,p,L,robust)
% run_KEF(X,robust)
%
% Perform KSD-Bayesian inference for the kernel exponential family.
%
% Input:
% X      = 1xn dataset, each column one sample.
% p      = 1x1, number of basis functions to use.
% L      = scalar, width of the reference measure. 
% robust = logical, indicating whether or not to be bias-robust.
%
% Output:
% out.An = pxp matrix An, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.vn = px1 vector v_n, s.t. nKSD^2 = theta' An theta + vn' theta + C.

% use faster computation with scalar kernel and preconditioner
scalar = true;

% KSD-Bayesian inference for the Liu et al experiment
out = KSD_Bayes(X, @(x) grad_T_KEF(x,p), ...
                   @(x) grad_b_KEF(x,L), ...
                   @(x) M_KEF(x,robust), ...
                   @(x,y) K_KEF(x,y), ...
                   scalar ); 
               
end               

