function out = aux_run_Gauss(X,scalar,robust,sigma,gamma,a,b)
% aux_run_Gauss(X,scalar,robust,sigma,gamma)
%
% Perform KSD-Bayesian inference for the Gaussian location model.
%
% Input:
% X = dxn dataset, each row one sample.
% scalar = logical, indicating whether scalar kernels and preconditioners
%          are being provided
% robust = logical, indicating whether or not to be bias-robust.
% sigma  = scalar bandwidth parameter.
% gamma  = scalar exponent parameter.
% a      = scalar parameter.
% b      = scalar parameter.
%
% Output:
% out.An = pxp matrix An, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.vn = px1 vector v_n, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.w  = approximation of the optimal weight for correct frequentist
%          coverage

% KSD-Bayesian inference for Gaussian location model
out = KSD_Bayes(X, @(x) grad_T_Gauss(x), ...
                   @(x) grad_b_Gauss(x), ...
                   @(x) aux_M_Gauss(x,scalar,robust,a,b), ...
                   @(x,y) aux_K_Gauss(x,y,scalar,sigma,gamma) , ...
                   scalar );                          
               
end               

