function out = M_KEF(x,robust)
% out = M_KEF(x)
%
% Preconditioner matrix M(x) for use in the kernel exponential family.
%
% Input:
% x      = 1x1 state vector.
% robust = logical, indicating whether or not to be bias-robust.
%
% Output:
% out     = structured object.
% out.m   = scalar m(x).
% out.mx  = scalar dm/dx.

% dimension
d = 1;

if ~robust
    % identity preconditioner for the Langevin Stein discrepancy
    out.m = ones(d,1);
    out.mx = zeros(d,d);
else
    % robust preconditioner for diffusion Stein discrepancy
    C = 1;
    out.m = (1 + C*x^2)^(-1/2); 
    out.mx = - C * x * (1 + C*x^2)^(-3/2);
end


end
