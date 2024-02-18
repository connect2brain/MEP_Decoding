function LFMm=hbf_LFM_B_LC_xyz(meshes,coils,TB,spos)
% HBF_LFM_B_LC_XYZ builds magnetic lead field matrix based on xyz-oriented
%   unit-current dipoles
%
% LFMm=HBF_LFM_B_LC_XYZ(meshes,coils,TB,spos)
%   meshes: BEM geometry, cell array of hbf structs
%   coils:  coil description, hbf struct
%   TB:     TB matrix built with the hbf BEM solver
%   spos:   source positions, [M x 3]
%
%   LFMm:   lead field matrix, [Number of coils x 3M]
%       [l_1x l_1y l1_z ... l_Mx l_My l_Mz]
%
% v160229 Matti Stenroos

[sind,eind]=NodeIndices(meshes);
Nof=size(TB,2);
endsurf=find(eind==Nof);

fp=zeros(Nof,3);
for M=1:endsurf,
    fp(sind(M):eind(M),:)=meshes{M}.p;
end

phiinf=hbf_Phiinf_xyz(fp,spos);
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
    Bninf=hbf_Binf_xyz(coils.QP,coils.QN,spos);
    LFMm=QtoC*Bninf+Bnvol;
else
    % Bninf in positions 'p'
    Bninf=hbf_Binf_xyz(coils.p,coils.n,spos);
    LFMm=Bninf+Bnvol;
end
% LFMm=QtoC*Bninf-Bnvol;
%Change of sign convention in Bnvol (compared to earlier codes and papers)
%TB must then be made with the new convention, using hbf_TM_B_Linear.m
function [startinds,endinds,Nop]=NodeIndices(meshes)
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