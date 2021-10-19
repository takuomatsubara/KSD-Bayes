function ret = compute_etas(X,is_centered,norm_factor)
% COMPUTE_ETAS compute the scale and sphericity statistics based on the
% sample covariance matrix (SCM) 
% ret = compute_etas(X,...)
%
% Function computes the scale statistic eta1 = trace(S)/p and 
% eta2 = trace(S^2)/p % and sphericity statistic gamma = eta2/eta1^2, where
% S denotes the unbiased SCM of the n x p data matrix X
%
% inputs:
%   X               data matrix of size n x p (rows are observations)
% Optional inputs:
%   is_centered     (logical) is data X already centered? Defaul=false
%   norm_factor     (logical) flag. If true, then compute the biased SCM. 
%                    Default = false (i.e., compute the unbiased SCM)
% toolbox: RegularizedSCM 
%--------------------------------------------------------------------------

[n,p] = size(X);
denom = n-1;

if nargin < 3
    norm_factor = false;
end

if ~islogical(norm_factor) 
    error('''norm_factor'' needs to be logical'); 
end

if norm_factor
    denom = n;
end

if nargin == 1 
    is_centered = false;
end

if ~islogical(is_centered) 
     error('input ''is_centered'' needs to be logical'); 
end

if ~is_centered 
   ret.xbar = mean(X);
    X = X - repmat(ret.xbar,n,1);
end

if p/n < 10 
% if p is order(s) of magnitude larger than n, then use svd
    ret.S = X'*X/denom;
    ret.eta(1) = trace(ret.S)/p;
    ret.eta(2) = trace(ret.S^2)/p;    
else 
    [~,L,ret.V]= svd(X,'econ'); 
    rnkX = rank(X); 
    l = diag(L);
    ret.V = ret.V(:,1:rnkX);
    ret.eig = l(1:rnkX).^2/denom;
    ret.eta(1) = sum(ret.eig)/p;
    ret.eta(2) = sum(ret.eig.^2)/p;
end
ret.gamma  = ret.eta(2)/ret.eta(1)^2;
