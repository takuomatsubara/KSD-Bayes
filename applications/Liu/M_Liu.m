function out = M_Liu(x,scalar,robust)
% out = M_Liu(x)
%
% Preconditioner matrix M(x) for use in the Liu et al experiment.
%
% Input:
% x      = 5x1 state vector.
% scalar = logical, indicating whether M(x) = diag(m_1(x),...,m_5(x)).
% robust = logical, indicating whether or not to be bias-robust.
%
% Output:
% out     = structured object.
% out.m   = 5x1 vector m(x).
% out.mx  = 5x5 matrix with entries d(m_i)/d(x_j).
% out.M   = 5x5 matrix M(x).
% out.Mx  = 5x5x5 matrix with entries d(M_[i,j])/d(x_k).

% dimension
d = 5;

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
    out.m = [(1 + x(1)^2 + x(2)^2 + x(3)^2 + x(4)^2 + x(5)^2)^(-1/2); ...
             (1 + x(1)^2 + x(2)^2)^(-1/2); ...
             (1 + x(1)^2 + x(3)^2)^(-1/2); ...
             (1 + x(1)^2 + x(4)^2)^(-1/2); ...
             (1 + x(1)^2 + x(5)^2)^(-1/2)]; 
    out.mx = [- x' * (1 + x(1)^2 + x(2)^2 + x(3)^2 + x(4)^2 + x(5)^2)^(-3/2); ...
              - [x(1), x(2), 0, 0, 0] * (1 + x(1)^2 + x(2)^2)^(-3/2); ...
              - [x(1), 0, x(3), 0, 0] * (1 + x(1)^2 + x(2)^2)^(-3/2); ...
              - [x(1), 0, 0, x(4), 0] * (1 + x(1)^2 + x(2)^2)^(-3/2); ...
              - [x(1), 0, 0, 0, x(5)] * (1 + x(1)^2 + x(2)^2)^(-3/2)];
    if ~scalar
        out.M = diag(out.m);
        out.Mx = zeros(d,d,d);
        for i = 1:d
            out.Mx(i,i,:) = out.mx(i,:);
        end
    end
end


end
