function out = K_Liu(x,y,scalar,S)
% out = K_Liu(x,y,scalar,S)
%
% Matrix-valued kernel K(x,y) for use in the Liu et al experiment.
%
% Input:
% x      = 5x1 state vector.
% y      = 5x1 state vector.
% scalar = logical, indicating whether K(x,y) = k(x,y) * eye(5,5)
% S      = 5x5 positive definite matrix.
%
% Output:
% out     = structured object.
% out.k   = scalar k(x,y). 
% out.kx  = 5x1 vector with entries dk/d(x_i).
% out.ky  = 1x5 vector with entries dk/d(y_j).
% out.kxy = 5x5 matrix with entries d2(k)/d(x_i)d(y_j).
% out.K   = 5x5 matrix K(x,y). 
% out.Kx  = 5x5x5 matrix with entries d(K_[i,j])/d(x_k).
% out.Ky  = 5x5x5 matrix with entries d(K_[i,j])/d(y_k).
% out.Kxy = 5x5x5x5 matrix with entries d2(K_[i,j])/d(x_k)d(y_l).

% dimension
d = 5;

% kernel amplitude
C = 1; 

% univariate kernel for diffusion Stein discrepancy
out.k = C * (1 + (x-y)'*(S\(x-y)))^(-1/2);
out.kx = C * (-1/2) * (1 + (x-y)'*(S\(x-y)))^(-3/2) * 2 * (S\(x-y));
out.ky = C * (-1/2) * (1 + (x-y)'*(S\(x-y)))^(-3/2) * 2 * (S\(y-x))';
out.kxy = C * (-1/2) * (-3/2) * (1 + (x-y)'*(S\(x-y)))^(-5/2) * 2 * (S\(x-y)) * 2 * (S\(y-x))' ...
                + C * (-1/2) * (1 + (x-y)'*(S\(x-y)))^(-3/2) * 2 * (-inv(S));      

% matrix-valued kernel            
if scalar == false
    
    out.K = zeros(d,d);
    out.Kx = zeros(d,d,d);
    out.Ky = zeros(d,d,d);
    out.Kxy = zeros(d,d,d,d);

    out.K = out.k * eye(d,d);
    for i = 1:d
        out.Kx(i,i,:) = out.kx;
        out.Ky(i,i,:) = out.ky;
        for j = 1:d
            out.Kxy(i,i,:,:) = out.kxy;
        end
    end

end



end