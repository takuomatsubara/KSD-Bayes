function out = aux_M_Gauss(x,scalar,robust,a,b)
% out = aux_M_Gauss(x,scalar,robust,a,b)
%
% Preconditioner matrix M(x) for use in the Gaussian location model.
%
% Input:
% x      = dx1 state vector.
% scalar = logical, indicating whether M(x) = diag(m_1(x),...,m_d(x)).
% robust = logical, indicating whether or not to be bias-robust.
% a      = scalar parameter.
% b      = scalar parameter.
%
% Output:
% out     = structured object.
% out.m   = dx1 vector m(x).
% out.mx  = dxd matrix with entries d(m_i)/d(x_j).
% out.M   = dxd matrix M(x).
% out.Mx  = dxdxd matrix with entries d(M_[i,j])/d(x_k).

% dimension
d = length(x);

if ~robust
    % identity preconditioner for the Langevin Stein discrepancy
    out.m = ones(d,1);
    out.mx = zeros(d,d);
    if ~scalar
        out.M = eye(d);
        out.Mx = zeros(d,d,d);
    end
else
    % robust preconditioner for diffusion Stein discrepancy
    out.m = a^2 * (a^2+(x-b).^2).^(-1/2);
    out.mx = a^2 * diag(-(1/2) * (a^2+(x-b).^2).^(-3/2) .* (2*(x-b)));
    if ~scalar
        out.M = diag(out.m);
        out.Mx = zeros(d,d,d);
        for i = 1:d
            out.Mx(i,i,:) = a^2 * -(1/2) * (a^2+(x(i)-b)^2)^(-3/2) * 2 * (x(i)-b);
        end
    end
end


end
