function out = ksd2(X,G)
% out = ksd2(X,G)
%
% Computes KSD^2 using a pre-conditioned IMQ kernel
%
% Inputs:
% X = dxn array, the dataset.
% G = dxn array, each column the gradient of the log density.
%
% Output:
% out = scalar, the KSD^2 between the dataset and the model.

% dimensions
[d,n] = size(X);
X = X';
G = G';

% kernel parameters
Gam = 1; % length scale
C = 1; % amplitude

% vectorised computation 
tmp0 = d / Gam;
tmp1 = dot(repmat(G,n,1),repelem(G,n,1),2);
tmp2 = repmat(X,n,1) - repelem(X,n,1);
tmp3 = repmat(G,n,1) - repelem(G,n,1);
tmp4 = - 3 * (dot(tmp2,tmp2,2)/(Gam^2)) ./ ((1 + dot(tmp2,tmp2,2)/Gam).^(5/2)) ...
       + (tmp0 + dot(tmp3,tmp2,2)/Gam) ./ ((1 + dot(tmp2,tmp2,2)/Gam).^(3/2)) ...
       + tmp1 ./ ((1 + dot(tmp2,tmp2,2)/Gam).^(1/2));
K = reshape(tmp4,n,n);

% compute KSD^2
out = C * (1/(n^2)) * ones(1,n) * K * ones(n,1);

end
