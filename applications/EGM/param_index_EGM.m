function out = param_index_EGM(d)
% out = param_index(d)
%
% Linear indexing of model parameters.
% theta_i is given linear index i.
% theta_{i,j} is given linear index out(i,j).
%
% Input:
% d = dimension of the state vector.
%
% Output:
% out = matrix containing the linear index of (i,j).

out = zeros(d,d);
counter = d;
for i = 1:d
    for j = (i+1):d
        counter = counter + 1;
        out(i,j) = counter;
    end
end


end