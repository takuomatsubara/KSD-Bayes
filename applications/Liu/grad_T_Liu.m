function out = grad_T_Liu(x)
% out = grad_T_Liu(x)
%
% Gradient of the natural sufficient statistic T(x) from the Liu et al
% experiment.
%
% Input:
% x = 5x1 state vector.
%
% Output:
% out = 2x5 matrix containing the d(T_i)/d(x_j).

out = [0, 0, 0, 1 - tanh(x(4))^2, 0; ...
       0, 0, 0, 0, 1 - tanh(x(5))^2];

end

