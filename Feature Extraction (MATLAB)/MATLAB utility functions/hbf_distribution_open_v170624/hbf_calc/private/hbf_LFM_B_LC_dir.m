function LFMm=hbf_LFM_B_LC_dir(bmeshes,coils,TB,spos,sdir)
% HBF_LFM_B_LC_DIR builds magnetic lead field matrix based on directed 
%   current dipoles.
%
% LFM=HBF_LFM_B_LC_DIR(meshes,coils,TB,spos,sdir)
%   meshes: BEM geometry, cell array of hbf structs
%   coils:  coil description, hbf struct
%   TB:     TB matrix built with the hbf BEM solver
%   spos:   source positions, [M x 3]
%   sdir:   source orientations (unit-length), [M x 3]
%
%   LFM:   lead field matrix, [Number of coils (field points) x M]
%           [l_1 ... l_M]
%
% You can also compute magnetic field due to any set of directed dipoles by 
% giving the dipole moments (with amplitude) in the 'sdir' argument.
%
% v160229 Matti Stenroos

[sind,eind]=NodeIndices(bmeshes);
Nof=size(TB,2);
endsurf=find(eind==Nof);
fp=zeros(Nof,3);
for M=1:endsurf,
    fp(sind(M):eind(M),:)=bmeshes{M}.p;
end
%Bn due tue volume currents
phiinf=hbf_Phiinf_dir(fp,spos,sdir);
Bnvol=TB*phiinf;

if isfield(coils,'QP')
    if isfield(coils,'QtoC')
        QtoC=coils.QtoC;
    elseif isfield(coils,'QPinds')
        QtoC=QpToCoilsMatrix(coils);
    else
        QtoC=1;
    end
    % Bninf in sensor integration points
    Bninf=hbf_Binf_dir(coils.QP,coils.QN,spos,sdir);
    LFMm=QtoC*Bninf+Bnvol;
else
    % Bninf in positions 'p'
    Bninf=hbf_Binf_dir(coils.p,coils.n,spos,sdir);
    LFMm=Bninf+Bnvol;
end
% LFMm=QtoC*Bninf-Bnvol;
%Change of sign convention in Bnvol (compared to earlier codes and papers)
%TB must then be made with the new convention, using hbf_TM_B_Linear.m

function [startinds,endinds]=NodeIndices(meshes)
Nsurf=length(meshes);
startinds=zeros(Nsurf,1);
endinds=zeros(Nsurf,1);
Nop=0;
for I=1:Nsurf,
    startinds(I)=Nop+1;
    endinds(I)=startinds(I)+size(meshes{I}.p,1)-1;
    Nop=endinds(I);
end
function res=QpToCoilsMatrix(coils)
% function QpToCoils=QpToCoilsMatrix(coils)
coilinds=coils.QPinds;
Nc=size(coilinds,1);
Nqp=coilinds(end,2);
res=zeros(Nc,Nqp);
for I=1:Nc,
    inds=(coilinds(I,1):coilinds(I,2));
    w=coils.QW(inds);
    res(I,inds)=w;
end
res=sparse(res);