function [T,startinds,endinds]=hbf_TM_Phi_LC(D,ci,co,zerolevel)
% HBF_TM_PHI_LC makes a BEM transfer matrix for potential using LC approach
%
% [T,startinds,endinds]=HBF_TM_PHI_LC(D,ci,co,zerolevel)
%   D:  D matrix, cell [N(meshes) x N(meshes)]
%   ci: conductivity inside each boundary surface, [N(meshes) x 1]
%   co: conductivity outside each boundary surface, [N(meshes) x 1]
%   zerolevel (optional):  index to the mesh, on which the mean potential 
%       (over vertices) is set to zero; use 0 to omit 
%   T: BEM transfer matrix, N(vertices in the whole model)^2
%   startinds, endinds: indices that point to rows/columns of T that
%       correspond to each BEM boundary surface;
%       inds{1}=startinds(1):endinds(1)
% 
%   If the model is embedded in infinite conductor (that is, co(end)>0),
%   zero of the potential is uniquely defined and 'zerolevel' argument is
%   not needed. In a finite model (co(end)=0), the zero of the potential
%   needs to be specified. In this BEM solver, the default choice is that
%   the mean of the potential over the outer boundary of the model is zero.
%
% v160302 Matti Stenroos

Nsurf=size(D,1);
defvals=zeros(Nsurf);
if nargin==4 && zerolevel>0
     defvals=DefValuesD(D,zerolevel);
elseif nargin<4 && co(end)==0
    defvals=DefValuesD(D,Nsurf);
end

c=zeros(Nsurf);
b=zeros(Nsurf,1);
startinds=zeros(Nsurf,1);
endinds=zeros(Nsurf,1);
Nop=0;
for I=1:Nsurf,
    startinds(I)=Nop+1;
    endinds(I)=startinds(I)+size(D{I,1},1)-1;
    Nop=endinds(I);
    b(I)=2/(ci(I)+co(I));
    for J=1:Nsurf
        c(I,J)=2*(ci(J)-co(J))/(ci(I)+co(I));
    end
end

Tinv=zeros(Nop);
for I=1:Nsurf,
    for J=1:Nsurf,
        Tinv(startinds(I):endinds(I),startinds(J):endinds(J))=c(I,J)*(D{I,J}+defvals(I,J));
    end
end

for I=1:Nop,
    Tinv(I,I)=Tinv(I,I)+1;
end

fprintf('Inverting %d x %d matrix...',Nop,Nop);
T=inv(Tinv);
fprintf('OK.\n');

for I=1:Nsurf,
     T(:,startinds(I):endinds(I))=T(:,startinds(I):endinds(I))*b(I);
end

function defvals=DefValuesD(D,defsurf)
% function defvals=DefValuesD(D,defsurf);
% Calculates deflation values that set the zero level to surface "defsurf".

Nsurf=size(D,1);
N_defsurf=size(D{defsurf,defsurf},1);
defvals=zeros(size(D));
for S=1:Nsurf,
    defvals(S,defsurf)=1/N_defsurf;
end