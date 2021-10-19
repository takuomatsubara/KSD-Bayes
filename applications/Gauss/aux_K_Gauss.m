function out = aux_K_Gauss(x,y,scalar,sigma,gamma)
% out = aux_K_Gauss(x,y,scalar,sigma,gamma)
%
% Matrix-valued kernel K(x,y) for use in the Gaussian location model.
%
% Input:
% x      = dx1 state vector.
% y      = dx1 state vector.
% scalar = logical, indicating whether K(x,y) = k(x,y) * eye(d,d).
% sigma  = scalar bandwidth parameter.
% gamma  = scalar exponent parameter.
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
out.k = C * (1 + (norm(x-y)/sigma)^2)^(-gamma);
out.kx = C * (-gamma) * (1 + (norm(x-y)/sigma)^2)^(-gamma-1) * 2 * (x-y) / sigma;
out.ky = C * (-gamma) * (1 + (norm(x-y)/sigma)^2)^(-gamma-1) * 2 * (y-x)' / sigma;
out.kxy = C * (-gamma) * (-gamma-1) * (1 + (norm(x-y)/sigma)^2)^(-gamma-2) * 2 * (x-y) * 2 * (y-x)' / (sigma^2) ...
                + C * (-gamma) * (1 + (norm(x-y)/sigma)^2)^(-gamma-1) * 2 * (-eye(d,d)) / sigma;         

% matrix-valued kernel            
if ~scalar    
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