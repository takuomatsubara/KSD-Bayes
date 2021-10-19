function out = run_EGM(X,robust)
% run_EGM(X,robust)
%
% Perform KSD-Bayesian inference for the exponential graphical model.
%
% Input:
% X = dxn dataset, each column one sample.
% robust = logical, indicating whether or not to be bias-robust.
%
% Output:
% out.An = pxp matrix An, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.vn = px1 vector v_n, s.t. nKSD^2 = theta' An theta + vn' theta + C.

% use faster computation with scalar kernel and preconditioner
scalar = true;

% kernel preconditioner
S = regscm(X','verbose','off');

% KSD-Bayesian inference for exponential graphical model
out = KSD_Bayes(X, @(x) grad_T_EGM(x), ...
                   @(x) grad_b_EGM(x), ...
                   @(x) M_EGM(x,scalar,robust), ...
                   @(x,y) K_EGM(x,y,scalar,S), ...
                   scalar );            
               
end               

