function out = grad_b_Liu(x)
% out = grad_b_Liu(x)
%
% Gradient of the function b(x) from the Liu et al experiment.
%
% Input:
% x = 5x1 state vector.
%
% Output:
% out = 5x1 vector containing d(b)/d(x_i).

out = [-x(1) + 0.6*x(2) + 0.2*(x(3)+x(4)+x(5)); ...
       -x(2) + 0.6*x(1); ...
       -x(3) + 0.2*x(1); ...
       -x(4) + 0.2*x(1); ...
       -x(5) + 0.2*x(1)];

end
