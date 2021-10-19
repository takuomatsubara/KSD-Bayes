function out = interaction_index_EGM(d)
% out = interaction_index(d)
%
% Linear index of the interaction parameters theta_{i,j} with i ~= j
%
% Input:
% d = dimension of the state vector.
%
% Output:
% out = d(d-1)/2 x 3 matrix, the first column containing the linear index 
%       of theta_{i,j}, and the second and third columns containing i and j.

out = zeros(d*(d-1)/2,3);
counter = 0;
for i = 1:d
    for j = (i+1):d
        counter = counter + 1;
        out(counter,:) = [d+counter,i,j];
    end
end


end