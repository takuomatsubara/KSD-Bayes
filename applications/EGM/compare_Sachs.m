function out = compare_Sachs(edges)
% out = compare_Sachs(edges)
%
% Computes the number of entries in edges that are present in the network
% reported in Sachs et al. "Causal Protein-Signaling Networks Derived from
% Multiparameter Single-Cell Data", Science 2005.
%
% Input:
% edges = n_edges x 2 matrix, each row containing the vertices of an edge,
%         indexed in {1,2,...,11} using the same convention as Sachs et al.
%
% Output:
% out = integer, the number of rows in "edges" that appear in Sachs et al.

% compare networks to Sachs
A = zeros(11,11); % Sach's network
     %praf %pmek %plcg %PIP2 %PIP3 %p44/42 %pakts473 %PKA %PKC %P38 %pjnk
A = [  0  ,  1  ,  0  ,  0  ,  0  ,   0   ,     0   , 0  , 0  , 0  ,  0  ; ... %praf
       0  ,  0  ,  0  ,  0  ,  0  ,   1   ,     0   , 0  , 0  , 0  ,  0  ; ... %pmek
       0  ,  0  ,  0  ,  1  ,  1  ,   0   ,     0   , 0  , 1  , 0  ,  0  ; ... %plcg
       0  ,  0  ,  0  ,  0  ,  0  ,   0   ,     0   , 0  , 1  , 0  ,  0  ; ... %PIP2
       0  ,  0  ,  0  ,  1  ,  0  ,   0   ,     1   , 0  , 0  , 0  ,  0  ; ... %PIP3
       0  ,  0  ,  0  ,  0  ,  0  ,   0   ,     1   , 0  , 0  , 0  ,  0  ; ... %p44/42
       0  ,  0  ,  0  ,  0  ,  0  ,   0   ,     0   , 0  , 0  , 0  ,  0  ; ... %pakts473
       1  ,  1  ,  0  ,  0  ,  0  ,   1   ,     1   , 0  , 0  , 1  ,  1  ; ... %PKA
       1  ,  1  ,  0  ,  0  ,  0  ,   0   ,     0   , 1  , 0  , 1  ,  1  ; ... %PKC
       0  ,  0  ,  0  ,  0  ,  0  ,   0   ,     0   , 0  , 0  , 0  ,  0  ; ... %P38
       0  ,  0  ,  0  ,  0  ,  0  ,   0   ,     0   , 0  , 0  , 1  ,  0  ];    %pjnk
   
% check each edge
out = 0;
for i = 1:size(edges,1)
    if (A(edges(i,1),edges(i,2)) ~= 0) || (A(edges(i,2),edges(i,1)) ~= 0)
        out = out + 1;
    end
end
   
end