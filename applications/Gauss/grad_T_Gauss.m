function out = grad_T_Gauss(x)
% out = grad_T_Gauss(x)
%
% Gradient of the natural sufficient statistic T(x) from the Gaussian
% location model.
%
% Input:
% x = dx1 state vector.
%
% Output:
% out = pxd matrix containing the d(T_i)/d(x_j).

d = length(x);
out = eye(d,d);

end

