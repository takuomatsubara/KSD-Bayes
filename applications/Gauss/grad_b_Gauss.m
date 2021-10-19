function out = grad_b_Gauss(x)
% out = grad_b_Gauss(x)
%
% Gradient of the function b(x) from the Gaussian location model.
%
% Input:
% x = dx1 state vector.
%
% Output:
% out = dx1 vector containing d(b)/d(x_i).

d = length(x);
out = -x;

end
