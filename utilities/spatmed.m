function  smed = spatmed(X,print_info)
% SPATMED computes the spatial median of the data set X
% 
% inputs:
%   X               data matrix of size n x p (rows are observations)
% Optional inputs:
%   print_info      (logical) verbose flag. Default=false
%
% toolbox: RegularizedSCM ({esa.ollila,elias.raninen}@aalto.fi)
%--------------------------------------------------------------------------


if nargin ==1
    print_info = false;
end 

if ~islogical(print_info) 
    error('Input ''print_info'' needs to be logical'); 
end

len = sum(X.*conj(X),2); 
X = X(len~=0,:);
n = size(X,1);

if isreal(X)
    smed0 = median(X);
else
    smed0 = mean(X);
end
    
norm0 = norm(smed0);

iterMAX = 500;
EPS = 1.0e-4;
TOL = 1.0e-6;   

for iter = 1:iterMAX 

   Xc = bsxfun(@minus,X,smed0);
   len = sqrt(sum(Xc.*conj(Xc),2)); 
   len(len<EPS)= EPS;
   Xpsi = bsxfun(@rdivide, Xc, len);
   update = sum(Xpsi)/sum(1./len);
   smed = smed0 + update; 
   
   dis = norm(update)/norm0;  
   %fprintf('At iter = %3d, dis=%.6f\n',iter,dis);
   
   if (dis<=TOL) 
       break;             
   end
   smed0 = smed;
   norm0 = norm(smed);
   
end

if print_info
   fprintf('spatmed::convergence at iter = %3d, dis=%.10f\n',iter,dis);
end

% DONE