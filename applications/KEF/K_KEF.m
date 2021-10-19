function out = K_KEF(x,y,S)
% out = K_KEF(x,y,S)
%
% Kernel K(x,y) for use in the kernel exponential family.
%
% Input:
% x      = 1x1 state vector.
% y      = 1x1 state vector.
% S      = dxd positive definite matrix.
%
% Output:
% out     = structured object.
% out.k   = scalar k(x,y). 
% out.kx  = scalar dk/dx.
% out.ky  = scalar dk/dy.
% out.kxy = scalar d2(k)/dxdy.

% dimension
d = 1;

% kernel amplitude
C = 1; 

% univariate kernel for diffusion Stein discrepancy
out.k = C * (1 + (x-y)'*(S\(x-y)))^(-1/2);
out.kx = C * (-1/2) * (1 + (x-y)'*(S\(x-y)))^(-3/2) * 2 * (S\(x-y));
out.ky = C * (-1/2) * (1 + (x-y)'*(S\(x-y)))^(-3/2) * 2 * (S\(y-x))';
out.kxy = C * (-1/2) * (-3/2) * (1 + (x-y)'*(S\(x-y)))^(-5/2) * 2 * (S\(x-y)) * 2 * (S\(y-x))' ...
                + C * (-1/2) * (1 + (x-y)'*(S\(x-y)))^(-3/2) * 2 * (-inv(S));           

end