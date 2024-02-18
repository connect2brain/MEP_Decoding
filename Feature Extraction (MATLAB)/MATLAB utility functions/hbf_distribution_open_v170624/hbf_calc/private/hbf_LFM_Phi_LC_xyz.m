function LFM=hbf_LFM_Phi_LC_xyz(meshes,Tphi,spos,flag_averef)
% HBF_LFM_PHI_LC_XYZ builds electric lead field matrix based on xyz-oriented
% unit-current dipoles
%
% LFM=HBF_LFM_PHI_LC_XYZ(meshes,Tphi,spos,flag_averef)
%   meshes: BEM geometry, cell array of hbf structs
%   Tphi:   Tphi matrix built with the hbf BEM solver
%   spos:   source positions, [M x 3]
%   flag_averef (optional, default value 1): give 0, if you do not want to
%           use average reference
%
%   LFM:   lead field matrix, [Number of electrodes x 3M]
%       [l_1x l_1y l1_z ... l_Mx l_My l_Mz]
%
% This function assumes average reference by default. If 'flag_averef' is
% selected 0, the potential is computed againts the reference chosen when
% building the BEM matrix.
%
% v160229 Matti Stenroos
[sind,eind]=NodeIndices(meshes);
Nof=size(Tphi,2);
endsurf=find(eind==Nof);
fp=zeros(Nof,3);
for M=1:endsurf,
    fp(sind(M):eind(M),:)=meshes{M}.p;
end
LFMinf=hbf_Phiinf_xyz(fp,spos);
LFM=Tphi*LFMinf;

if nargin<4 || flag_averef==1,
    m=mean(LFM,1);
    LFM=LFM-ones(size(LFM,1),1)*m;
end

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