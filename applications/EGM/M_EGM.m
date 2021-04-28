function out = M_EGM(x,scalar,robust)
% out = M_EGM(x)
%
% Preconditioner matrix M(x) for use in the exponential graphical model.
%
% Input:
% x      = dx1 state vector.
% scalar = logical, indicating whether M(x) = diag(m_1(x),...,m_d(x)).
% robust = logical, indicating whether or not to be bias-robust.
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
    out.m = exp(-x);
    out.mx = diag(-exp(-x));
    if ~scalar
        out.M = diag(out.m);
        out.Mx = zeros(d,d,d);
        for i = 1:d
            out.Mx(i,i,:) = - exp(-x(i));
        end
    end
end


end
