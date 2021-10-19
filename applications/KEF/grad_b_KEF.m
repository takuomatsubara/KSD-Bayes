function out = grad_b_KEF(x,L)
% out = grad_b_KEF(x)
%
% Gradient of the function b(x) from the kernel exponential family.
%
% Input:
% x     = 1x1 state vector.
% L     = scalar, width of the reference measure. 
%
% Output:
% out = 1x1 vector containing db/dx.

out = - x / (L^2);

end
