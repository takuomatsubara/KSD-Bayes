function [RSCM, invRSCM, stats] = regscm(X,varargin)
% REGSCM Computes different regularized (shrinkage SCM) estimators given a
% data matrix X (rows are observations)
% [RSCM, invRSCM, stats] = regscm(X,...) 
%
% REGSCM can be called with numerous optional arguments. Optional
% arguments are given in parameter pairs, so that first argument is
% the name of the parameter and the next argument is the value for
% that parameter. Optional parameter pairs can be given in any order.
%
% Optional Parameters:
%demo
% Parameter name        Values and description
%==========================================================================
% --Basic parameter is the choise of the estimator to use
%
% 'approach'            (string) which estimate of the MMSE shrinkage pe- 
%                       nalty parameter beta to use
%                       'ell1' (default)    uses ELL1-RSCM estimator
%                       'ell2'              uses ELL2-RSCM estimator
%                       'lw'                uses Ledoit-Wolf approach
%==========================================================================
% -- It is also possible to give other values for the parameters that are
% needed by Ell1- and Ell2-RSCM estimators. For example, it is possible to
% specify the value of elliptical kurtosis parameter if one has some a
% priori knowledge of its value (e.g., for Gaussian-like data, one
% case set kappa to a value equal to 0) or if the value of the elliptical
% kurtosis has been computed previously using *ellkurt* function.
% 
% 'kappa'               (number) elliptical kurtosis parameter to use. Must
%                       be larger than kappa_lowerb = -2/(p+2)
% 'gamma'               (number) sphericity measure to use. Must be a value
%                       between [1,p]. If gamma is not given, then approach 
%                       parameter determines the sphericity estimator used  
% 'inverse'             (logical) true or false. If true then one computes 
%                       also the inverse of the regularized SCM. Computation 
%                       of the inverse is optimized for speed and computed 
%                       via SVD when p > 10*n, i.e., when p is order(s) of 
%                       magnitude larger than the sample size. 
% 'verbose'             (string) Either 'on' (default) or 'off'. When 'on' 
%                       then prints progress of the function in text format.
% 'centered'            (logical) if true then X is already centered, and
%                       omits centering of the data. If false (default), 
%                       then the data will be centered. 
%
% See also ELLKURT, COMPUTE_ETAS
%
% toolbox: RegularizedSCM
% version: 1.1 
%   changes to 1.0: changed the gamma^Ell1 estimator to include the
%   correction term suggested by Cheng et al (2019). For details see
%   GAMELL1
%
% authors: Copyright by Esa Ollila and Elias Raninen. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-- Check requirements on the input data matrix X
argin = inputParser;
valX = @(X) assert(isreal(X) && (numel(size(X))==2),['''X'' must be a ' ...
    'real-valued matrix having at least two columns (variables)']);
addRequired(argin,'X',valX);

if nargin == 0
  error ('You must supply the data matrix  as input argument.');
end

if any (any (isnan (X)))
  error ('Input data contains NaN''s.');
end

if ~isa (X, 'double')
  X = double(X);
end

[n,p] = size(X); % data matrix dimensions
kappa_lowerb    = -2/(p+2); % theoretical lower bound for ellipt. kurtosis 

%-- optional input parsing rules
valKappa = @(x) assert(isscalar(x) && x > kappa_lowerb && isreal(x), ... 
    ['''kappa'' must be a real-valued scalar and larger than the ' ...
    'lowerbound ', num2str(kappa_lowerb)]);
addParameter(argin,'kappa',[], valKappa);

valGamma = @(x) assert(isreal(x) && x >= 1 && x < p,['''gamma'' must be ' ...
    'real-valued and in the range [1,' num2str(p) ')']);
addParameter(argin,'gamma',[], valGamma);

valVerbose = @(x) assert(any(strcmpi(x,{'on','off'})),['''verbose'' must' ...
    ' be a string equal to ''on'' or ''off''']);
addParameter(argin,'verbose','on',valVerbose);

addParameter(argin,'approach','ell1', ... 
    @(x) logical(sum(strcmp(x,{'ell1','ell2','lw'}))));
% must be ell1, ell2 or lw

addParameter(argin,'inverse',false,@islogical);
addParameter(argin,'centered',false,@islogical);

%---  parse inputs
parse(argin,X,varargin{:});
compute_inv = argin.Results.inverse;
is_centered = argin.Results.centered;
print_info = strcmpi(argin.Results.verbose,'on');
compute_gamma = isempty(argin.Results.gamma);
compute_kappa = isempty(argin.Results.kappa);
approach = argin.Results.approach;

%-- print information about data
if print_info
  fprintf('Number of variables           : %d\n', p);
  fprintf('Number of samples             : %d\n', n);
end

%-- Compute the values of eta_1 = trace(S)/p and eta(2)=trace(S^2)/p
if any(strcmpi(approach,{'ell1','ell2'})) 
    ret = compute_etas(X,is_centered);
else
    ret = compute_etas(X,is_centered,true);
end

stats.eta = ret.eta;

%-- Now compute the optimal shrinkage parameter value
if any(strcmpi(approach,{'ell1','ell2'})) 
% Ell1 and Ell2 requires estimates of elliptical kurtosis and sphericity
    if compute_kappa
        if ~is_centered 
            kappahat = ellkurt(X,ret.xbar);
        else
            if print_info, fprintf('skipping the centering of the data\n'); end
            kappahat = ellkurt(X,[],is_centered,print_info);
        end
        if print_info
            fprintf('Computed elliptical kurtosis  : %.3f\n', kappahat);
        end
    else
        kappahat = argin.Results.kappa;
        if print_info
            fprintf('Using the elliptical kurtosis value %.3f\n', kappahat);
        end
    end

    if compute_gamma
        switch approach 
            case 'ell1'
                if print_info 
                    fprintf('computing Ell1-RSCM estimator...\n');
                end
                gammahat = gamell1(X);
            case 'ell2'
                if print_info
                    fprintf('computing Ell2-RSCM estimator...\n'); 
                end
                gammahat = gamell2(X,kappahat,ret.eta,is_centered);
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized parameter: ''' approach '''']);  
        end  
    else 
        gammahat = argin.Results.gamma;
        if print_info 
              fprintf('using given gamma value       : %.7f\n',gammahat);    
        end        
    end
    T = gammahat - 1;
    be = T/(T + kappahat*(2*gammahat+p)/n + (gammahat + p)/(n-1));
    be = min(1,max(0,be));
    
    stats.kappa = kappahat; 
    stats.gamma = gammahat; 
else
   if print_info 
             fprintf('computing the LW-RSCM estimator...\n');
   end
    % we compute the Ledoit-Wolf estimate
    if ~is_centered 
        X = X - repmat(ret.xbar,n,1);
    end
    T = n*(ret.gamma - 1);
    c1 = mean(sum(X.*X,2).^2)/p;
    c2  = (c1 - ret.eta(2))/(T*ret.eta(1)^2); 
    be = max(0,1-c2);
    be = min(be,1);
end

stats.beta  = be;

% regularized sample covariance matrix estimate
if p/n < 10
    RSCM =  be*ret.S + (1-be)*ret.eta(1)*eye(p);
else
    RSCM =  be*(ret.V*diag(ret.eig)*ret.V') + (1-be)*ret.eta(1)*eye(p);
    stats.eigval = be*[ret.eig; zeros(p-n,1)] + (1-be)*ret.eta(1);
end

if compute_inv 
    if print_info, fprintf('Computing the inverse RSCM...'); end
    if p/n >= 10
        tmp = (1-be)*ret.eta(1);
        dd = ( 1./(ret.eig*be + tmp)) - 1/tmp ;
        invRSCM = ret.V*diag(dd)*ret.V' + (1/tmp)*eye(p);
    else 
        invRSCM = RSCM \ eye(p);
    end
    if print_info, fprintf('done!\n'); end
else
    invRSCM  = []; 
end







