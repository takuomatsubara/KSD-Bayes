function out = grad_T_EGM(x)
% out = grad_T_EGM(x)
%
% Gradient of the natural sufficient statistic T(x) from the exponential
% graphical model.
%
% Input:
% x = dx1 state vector.
%
% Output:
% out = pxd matrix containing the d(T_i)/d(x_j).

d = length(x);
p = d*(d+1)/2;
out = zeros(p,d);

for i = 1:d
    out(i,i) = - exp(x(i));
end
lin_idx = param_index_EGM(d); % linear indexing for interaction terms
for i = 1:d
    for j = (i+1):d
        idx = lin_idx(i,j);
        out(idx,i) = - exp(x(i) + x(j));
        out(idx,j) = - exp(x(i) + x(j));
    end
end

end

