function y=wintakeall(x)
% Copyright  Radu Dogaru, Natural Computing Lab., University "Politehnica" of Bucharest
% radu_d@ieee.org 
% http://atm.neuro.pub.ro/radu_d/ 
% Last update, February 2015

[m, n]=size(x);
if m>1
   ix=find(x==max(x));
   y=-ones(m,1); 
   y(ix(1))=1;
else
   y=sign(x);
end

