function out = K_KEF(x,y)
% out = K_KEF(x,y)
%
% Kernel K(x,y) for use in the kernel exponential family.
%
% Input:
% x      = 1x1 state vector.
% y      = 1x1 state vector.
%
% Output:
% out     = structured object.
% out.k   = scalar k(x,y). 
% out.kx  = scalar dk/dx.
% out.ky  = scalar dk/dy.
% out.kxy = scalar d2(k)/dxdy.

% dimension
d = 1;

% kernel length-scale
ell = 1;

% kernel amplitude
C = 1; 

% univariate kernel for diffusion Stein discrepancy
out.k = C * (1 + norm((x-y)/ell)^2)^(-1/2);
out.kx = C * (-1/2) * (1 + norm((x-y)/ell)^2)^(-3/2) * 2 * ((x-y)/ell) * (1/ell);
out.ky = C * (-1/2) * (1 + norm((x-y)/ell)^2)^(-3/2) * 2 * ((y-x)/ell)' * (1/ell);
out.kxy = C * (-1/2) * (-3/2) * (1 + norm((x-y)/ell)^2)^(-5/2) * 2 * ((x-y)/ell) * 2 * ((y-x)/ell)' * (1/ell)^2 ...
                + C * (-1/2) * (1 + norm((x-y)/ell)^2)^(-3/2) * 2 * (1/ell)^2 * (-eye(d,d));         

end