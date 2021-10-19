function out = pdf_KEF(X,coeff,L)
% out = pdf_KEF(x,coeff)
%
% Un-normalised probability density function for the kernel exponential
% family.
%
% Input:
% X     = 1xn vector, each column a state x_i.
% coeff = px1 vector of coefficients.
% L     = scalar, width of the reference measure. 
%
% Output:
% out = 1xn vector, each column the un-normalised p.d.f. p(x_i).

% dimensions
p = length(coeff);

% Vandermonde matrix
mat = (factorial((0:(p-1))')).^(-1/2) .* bsxfun(@power,X,(0:(p-1))') ...
      .* exp(-X.^2/2);

% log of un-normalised p.d.f.
log_pdf = - (1/2) * (X/L).^2 + coeff' * mat;

% prevent overflow
log_pdf = log_pdf - max(log_pdf);

% output un-normalised p.d.f.
out = exp(log_pdf);

% numerical normalisation
out = out / trapz(X,out);

end

