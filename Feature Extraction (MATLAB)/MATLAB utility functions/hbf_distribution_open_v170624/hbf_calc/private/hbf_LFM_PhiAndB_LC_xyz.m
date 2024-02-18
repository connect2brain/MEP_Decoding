function [LFMphi,LFMb]=hbf_LFM_PhiAndB_LC_xyz(bmeshes,coils,Tphi,TB,spos,flag_averef)
% HBF_LFM_PHIANDB_LC_XYZ builds electric and magnetic lead field matrices
%   based on xyz-oriented unit-current dipoles.
%
% [LFMphi,LFMm]=HBF_LFM_PHIANDB_LC_XYZ(meshes,coils,Tphi,TB,spos,flag_averef)
%   meshes: BEM geometry, cell array of hbf structs
%   coils:  coil description, hbf struct
%   TB:     TB matrix built with the hbf BEM solver
%   Tphi:   Tphi matrix built with the hbf BEM solver
%   spos:   source positions, [M x 3]
%   flag_averef (optional, default value 1): give 0, if you do not want to
%           use average reference in LFMphi
%
%   LFMphi: electric lead field matrix, [Number of electrodes x M]
%       [lphi_1x lphi_1y lphi_1z ... lphi_Mx lphi_My lphi_Mz]
%   LFMb: magnetic lead field matrix, [Number of coils x M]
%       [lb_1x lb_1y lb_1z ... lb_Mx lb_My lb_Mz]
%
%
% This function assumes average reference by default. If 'flag_averef' is
% selected 0, the potential is computed againts the reference chosen when
% building the BEM matrix.
%
% v160303 Matti Stenroos
[sind,eind]=NodeIndices(bmeshes);
Nof=size(TB,2);
endsurf=find(eind==Nof);
fp=zeros(Nof,3);
for M=1:endsurf,
    fp(sind(M):eind(M),:)=bmeshes{M}.p;
end
phiinf=hbf_Phiinf_xyz(fp,spos);

LFMphi=Tphi*phiinf;
%Set average reference. If you do not want to set the reference, comment the
%following two lines.
if nargin<7 || flag_averef==1,
    m=mean(LFMphi,1);
    LFMphi=LFMphi-ones(size(LFMphi,1),1)*m;
end
%Bn due tue volume currents
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
    LFMb=QtoC*Bninf+Bnvol;
else
    % Bninf in positions 'p'
    Bninf=hbf_Binf_xyz(coils.p,coils.n,spos);
    LFMb=Bninf+Bnvol;
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