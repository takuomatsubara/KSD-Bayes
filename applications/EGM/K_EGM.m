function out = K_EGM(x,y,scalar,S)
% out = K_EGM(x,y,scalar,S)
%
% Matrix-valued kernel K(x,y) for use in the exponential graphical model.
%
% Input:
% x      = dx1 state vector.
% y      = dx1 state vector.
% scalar = logical, indicating whether K(x,y) = k(x,y) * eye(d,d).
% S      = dxd positive definite matrix.
%
% Output:
% out     = structured object.
% out.k   = scalar k(x,y). 
% out.kx  = dx1 vector with entries dk/d(x_i).
% out.ky  = 1xd vector with entries dk/d(y_j).
% out.kxy = dxd matrix with entries d2(k)/d(x_i)d(y_j).
% out.K   = dxd matrix K(x,y). 
% out.Kx  = dxdxd matrix with entries d(K_[i,j])/d(x_k).
% out.Ky  = dxdxd matrix with entries d(K_[i,j])/d(y_k).
% out.Kxy = dxdxdxd matrix with entries d2(K_[i,j])/d(x_k)d(y_l).

% dimension
d = length(x);

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