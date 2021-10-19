function out = KSD_Bayes(X,grad_T,grad_b,M,K,scalar)
% out = KSD_Bayes(X,grad_T,grad_b,M,K,scalar)
%
% Perform KSD-Bayesian inference.
%
% Input:
% X      = dxn dataset, each column one sample.
% grad_T = function handle, returning the gradient matrix of the natural
%          sufficient statistic T(x) in the exponential family.
% grad_b = function handle, returning the gradient vector for the function
%          b(x) in the exponential family.
% M      = function handle, returning the preconditioner.
% K      = bivariate function handle, returning a kernel object.
% scalar = logical, indicating whether scalar kernels and preconditioners
%          are being provided
%
% Output:
% out.An = pxp matrix An, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.vn = px1 vector v_n, s.t. nKSD^2 = theta' An theta + vn' theta + C.
% out.w  = approximation of the optimal weight for correct frequentist
%          coverage

% dimensions
[~,n] = size(X);
[p,~] = size(grad_T(X(:,1)));

% terms required to compute KSD
if scalar
    phi = @(x) bsxfun(@times,M(x).m,grad_T(x)');
    A = @(x,y) K(x,y).k * (phi(x)' * phi(y));
    v_perp = @(x,y) K(y,x).k * (grad_b(y)') * bsxfun(@times,M(y).m,phi(x)) ...
                    + K(x,y).k * (grad_b(x)') * bsxfun(@times,M(x).m,phi(y)) ...
                    + divx_MK(x,y,M,K,scalar) * phi(y) ...
                    + divy_MK(x,y,M,K,scalar) * phi(x);                
else
    phi = @(x) (M(x).M') * (grad_T(x)');
    A = @(x,y) (phi(x)') * K(x,y).K * phi(y);
    v_perp = @(x,y) (grad_b(y)') * M(y).M * K(y,x).K * phi(x) ...
                    + (grad_b(x)') * M(x).M * K(x,y).K * phi(y) ...
                    + divx_MK(x,y,M,K,scalar) * phi(y) ...
                    + divy_MK(x,y,M,K,scalar) * phi(x);
end
            
% return the parts required to construct a Gaussian "likelihood"
A_cache = zeros(p,p,n); % cache value of (1/n) \sum_i A(x,x_i)
v_cache = zeros(p,1,n); % cache computed values of (1/n) \sum_i v_perp(x,x_i)
out.An = zeros(p,p);
out.vn = zeros(p,1);
out.w = 1; % need to instantiate all fields to avoid issues with Matlab Coder
for i = 1:n
    for j = 1:n
        A_ij = A(X(:,i),X(:,j));
        v_ij = v_perp(X(:,i),X(:,j))';
        A_cache(:,:,i) = A_cache(:,:,i) + (1/n) * A_ij;
        v_cache(:,:,i) = v_cache(:,:,i) + (1/n) * v_ij;
        out.An = out.An + (1/n) * A_ij;
        out.vn = out.vn + (1/n) * v_ij;        
    end
end

% the minimum KSD estimator
theta = - (1/2) * (out.An \ out.vn); % minimum KSD estimator

% information-type matrices
H = (2/n) * out.An;
J = zeros(p,p);
for k = 1:n 
    tmp = 2 * A_cache(:,:,k) * theta + v_cache(:,:,k);
    J = J + (1/n) * (tmp * tmp');         
end

% estimated optimal weight
out.w = trace(H*(J\H)) / trace(H);

end


%% helper functions

function out = divx_M(x,M,scalar)
% \nabla_x \cdot M(x)

% dimension
d = length(x);

if scalar
    out = diag(M(x).mx)';
else
    out = zeros(1,d); 
    for i = 1:d
        out = out + M(x).Mx(i,:,i);
    end
end

end

function out = divx_MK(x,y,M,K,scalar)
% \nabla_x \cdot (M(x) K(x,y))

% dimension
d = length(x);

if scalar
    out = K(x,y).k * divx_M(x,M,scalar) ...
          + (M(x).m') .* (K(x,y).kx');
else
    out = divx_M(x,M,scalar) * K(x,y).K; 
    for i = 1:d
        out = out + M(x).M(i,:) * K(x,y).Kx(:,:,i);
    end
end

end

function out = divy_MK(x,y,M,K,scalar)
% \nabla_y \cdot (M(y) K(x,y))

% dimension
d = length(x);

if scalar
    out = K(x,y).k * divx_M(y,M,scalar) ...
          + (M(y).m') .* (K(x,y).ky);
else
    out = divx_M(y,M,scalar) * K(x,y).K;
    for i = 1:d
        out = out + M(y).M(i,:) * K(x,y).Ky(:,:,i);
    end
end

end













