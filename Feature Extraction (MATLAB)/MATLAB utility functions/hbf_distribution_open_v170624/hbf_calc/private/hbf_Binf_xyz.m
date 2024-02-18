function Binf=hbf_Binf_xyz(fp,fpdir,spos)
% HBF_BINF_XYZ  computes an arbitrary component of the magnetic field due to 
%   xyz-oriented unit-current-dipole triplet in infinite non-magnetic medium 
%
% Binf=HBF_BINF_XYZ(fp,fpdir,spos)
%   fp = field points (where the field is computed), [N x 3]
%   fpdir = field orientations (which component is computed), [N x 3]
%   spos  = source positions, [M x 3]
%
%   Binf  = resulting magnetic field for each field point and dipole, [N x 3M]
%       [B_1x B_1y B_1z ... B_Mx B_My B_Mz]
% v160229 Matti Stenroos

Nfp=size(fp,1);
Nsp=size(spos,1);
Binf=zeros(Nfp,3*Nsp);

K=1e-7;%mu0/(4pi)
zerov=zeros(Nfp,1);
for S=1:Nsp,
    R=fp-ones(Nfp,1)*spos(S,:);
    absR=sqrt(sum(R.*R,2));
    absRm3 = 1./(absR.*absR.*absR);
    sourcex=sum([zerov -R(:,3) R(:,2)].*fpdir,2);
    sourcey=sum([R(:,3) zerov -R(:,1)].*fpdir,2);
    sourcez=sum([-R(:,2) R(:,1) zerov].*fpdir,2);
    Binf(:,3*S-2:3*S)= K*[sourcex sourcey sourcez].*(absRm3*[1 1 1]);
end