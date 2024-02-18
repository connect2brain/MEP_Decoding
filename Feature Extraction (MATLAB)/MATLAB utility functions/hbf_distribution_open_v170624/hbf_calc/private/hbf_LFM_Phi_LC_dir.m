function LFM=hbf_LFM_Phi_LC_dir(bmeshes,Tphi,spos,sdir,flag_averef)
% HBF_LFM_PHI_LC_DIR builds electric lead field matrix based on directed 
% current dipoles.
%
% LFM=HBF_LFM_PHI_LC_DIR(meshes,Tphi,spos,sdir,flag_averef)
%   meshes: BEM geometry, cell array of hbf structs
%   Tphi:   Tphi matrix built with the hbf BEM solver
%   spos:   source positions, [M x 3]
%   sdir:   source orientations (unit-length), [M x 3]
%   flag_averef (optional, default value 1): give 0, if you do not want to
%           use average reference
%
%   LFM:   lead field matrix, [Number of electrodes x M]
%       [l_1 ... l_M]
%
% You can also compute phi due to any set of directed dipoles by 
% giving the dipole moments (with amplitude) in the 'sdir' argument.
%
% This function assumes average reference by default. If 'flag_averef' is
% selected 0, the potential is computed againts the reference chosen when
% building the BEM matrix.
%
% v160229 Matti Stenroos

[sind,eind]=NodeIndices(bmeshes);
Nof=size(Tphi,2);
endsurf=find(eind==Nof);
fp=zeros(Nof,3);
for M=1:endsurf,
    fp(sind(M):eind(M),:)=bmeshes{M}.p;
end
LFMinf=hbf_Phiinf_dir(fp,spos,sdir);
LFM=Tphi*LFMinf;

if nargin<5 || flag_averef==1,
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