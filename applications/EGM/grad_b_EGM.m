function out = grad_b_EGM(x)
% out = grad_b_EGM(x)
%
% Gradient of the function b(x) from the exponential graphical model.
%
% Input:
% x = dx1 state vector.
%
% Output:
% out = dx1 vector containing d(b)/d(x_i).

d = length(x);
out = ones(d,1);

end
