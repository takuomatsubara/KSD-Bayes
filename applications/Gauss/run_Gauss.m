function out = run_Gauss(X,scalar,robust)
% run_Gauss(X,robust)
%
% Perform KSD-Bayesian inference for the Gaussian location model.
%
% Input:
% X = dxn dataset, each row one sample.
% scalar = logical, indicating whether scalar kernels and preconditioners
%          are being provided
% robust = logical, indicating whether or not to be bias-robust.
%
% Output:
% out.An = pxp matrix An, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.vn = px1 vector v_n, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.w  = approximation of the optimal weight for correct frequentist
%          coverage

% KSD-Bayesian inference for Gaussian location model
out = KSD_Bayes(X, @(x) grad_T_Gauss(x), ...
                   @(x) grad_b_Gauss(x), ...
                   @(x) M_Gauss(x,scalar,robust), ...
                   @(x,y) K_Gauss(x,y,scalar) , ...
                   scalar );                          
               
end               

