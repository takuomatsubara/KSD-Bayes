function [kappahat, xbar] = ellkurt(X,xbar,is_centered,print_info)
% ELLKURT computes the estimate of the elliptical kurtosis parameter of a
% p-dimensional distribution given the data set X
% [kappahat, xbar] = ellkurt(X,...)
%
% inputs:
%   X               data matrix of size n x p (rows are observations)
% Optional inputs:
%   xbar            sample mean vector of the data X 
%   is_centered     (logical) is the X already centered. Defaul=false
%   print_info      (logical) verbose flag. Default=false
%
% toolbox: RegularizedSCM ({esa.ollila,elias.raninen}@aalto.fi)
%--------------------------------------------------------------------------

[n,p] = size(X);

if isreal(X) 
    ka_lb = -2/(p+2); % theoretical lower bound for the kurtosis parameter
else 
    ka_lb = -1/(p+1); % theoretical lower bound for kurtosis parameter
end

if nargin < 4 || isempty(print_info)
    print_info = false;
end

if nargin < 3 || isempty(is_centered)
    is_centered = false;
end

if nargin < 2 || isempty(xbar)
    xbar = mean(X);
end
       
if ~is_centered
    if print_info, fprintf('ellkurt: centering the data...'); end
    X = X - repmat(xbar,n,1);
end
vari =  mean(abs(X).^2);

indx = (vari==0);
if any(indx)
    if print_info
        fprintf('ellkurt: found a variable with a zero sample variance\n');
        fprintf('         ...ignoring the variable in the calculation\n');
    end
end

if isreal(X) 
    kurt1n = (n-1)/((n-2)*(n-3));
    g2 = mean(X(:,~indx).^4)./(vari(~indx).^2)-3;
    G2 = kurt1n*((n+1)*g2 + 6);
    kurtest = mean(G2);
    kappahat = (1/3)*kurtest;
else
    g2 = mean(abs(X(:,~indx)).^4)./(vari(~indx).^2)-2;
    kurtest = mean(g2);
    kappahat = (1/2)*kurtest;
end

if kappahat > 1e6
    error('ellkurt: something is worong, too large value for kurtosis\n');
end

if kappahat <= ka_lb + (abs(ka_lb))/40
      kappahat = ka_lb + (abs(ka_lb))/40;
end


