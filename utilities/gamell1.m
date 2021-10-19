function [gammahat,Csgn,muhat,gammahat0] = gamell1(X,is_centered,muhat)
% Computes the Ell1-estimator of sphericity.
%
% Inputs:
%
%   X               data matrix of size n x p (rows are observations)
%
% Optional inputs:
%
%   is_centered     (logical) is the X already centered. Defaul=false
%   muhat           p x 1 vector (e.g., spatial median of the data)
%
% Outputs:
%
%   gammahat        Estimator of sphericity proposed by Cheng et al (2019)
%                   The sphericty estimator gammahat has a correction factor 
%                   that improved gammahat0 estimator when p/n is large and 
%                   hence it is the recommended estimator to use. 
%   Csgn            Spatial sign covariance matrix (p x p matrix)
%   muhat           Spatial median (p x 1 vector)
%   gammahat0       Estimator of sphericity we proposed in our paper,
%                   Ollila and Raninen (2019).
%
% Toolbox: 
%
%   RegularizedSCM ({esa.ollila,elias.raninen}@aalto.fi)
%
% References:
% 
%   Cheng, G., Liu, B., Peng, L., Zhang, B., & Zheng, S. (2019). Testing  
%       the equality of two high‐dimensional spatial sign covariance 
%       matrices. Scandinavian Journal of Statistics, 46(1), 257-271.
%
%   Ollila, E., & Raninen, E. (2019). Optimal shrinkage covariance matrix 
%       estimation under random sampling from elliptical distributions. 
%       IEEE Transactions on Signal Processing, 67(10), 2707-2719.
% 
%--------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print_info = false;
[n,p] = size(X);

if nargin < 2
    is_centered = false;
end

if ~is_centered
    if print_info, fprintf('centering the data'); end 
    if nargin < 3
        muhat = spatmed(X);
    end
    X = X -  repmat(muhat,n,1);
else
   if print_info, fprintf('Not centering the data'); end 
end

d = sqrt(sum(X.*conj(X),2));
X = X(d~=0,:); % eliminate observations that have zero length
d = d(d~=0);
n = size(X,1);
X = X.*repmat(1./d,1,p);

if isreal(X) 
    m2 = mean(d.^(-2));
    m1 = mean(1./d);
    ratio = m2/(m1^2);
    delta= (1/n^2)*(2- 2*ratio + ratio^2);
else
    delta = 0;
end


if p/n < 10 
    Csgn = (1/n)*(X')*X; % Sign covariance matrix
    gammahat0 = (p*n/(n-1))*(trace(Csgn^2) - 1/n);
    gammahat0 = real(gammahat0);
    gammahat = gammahat0 - p*delta;
else
    l = svd(X,'econ'); 
    eigvals = l.^2/n;
    traceCsgn_sq = sum(eigvals.^2);
    gammahat0 = (p*n/(n-1))*(traceCsgn_sq - 1/n);
    gammahat0 = real(gammahat0);
    gammahat = gammahat0 - p*delta;
end

gammahat0 = min(p,max(1,gammahat0)); % NOTE:\gamma in [1, p];
gammahat = min(p,max(1,gammahat)); % NOTE:\gamma in [1, p];