function out = grad_T_KEF(x,p)
% out = grad_T_KEF(x)
%
% Gradient of the natural sufficient statistic T(x) from the kernel
% exponential family.
%
% Input:
% x = 1x1 state vector.
% p = 1x1, number of basis functions to use.
%
% Output:
% out = px1 matrix containing the d(T_i)/d(x_j).

out = (factorial((1:p)')).^(-1/2) ...
      .* ((x.^((0:(p-1))))') ...
      .* ((1:p)' - (x^2)) ...
      * exp(-x^2/2);
   
end

