function phiinf=hbf_Phiinf_xyz(fp,spos)
% HBF_PHIINF_XYZ  computes potential due to a xyz-oriented unit-current-dipole 
%   triplet in infinite homogeneous conductor that has unit conductivity
%
% phiinf=HBF_PHIINF_XYZ(fp,spos)
%   fp = field points (where the field is computed), [N x 3]
%   spos  = source positions, [M x 3]
%
%   phiinf = resulting potential for each field point and dipole, [N x 3M]
%     [phiinf_1x phiinf_1y phiinf_1z ... phiinf_Mx phiinf_My phiinf_Mz]
%
% v160229 Matti Stenroos
Nfp=size(fp,1);
Nsp=size(spos,1);
phiinf=zeros(Nfp,3*Nsp);

K=1/(4*pi);
for S=1:Nsp,
    R=fp-ones(Nfp,1)*spos(S,:);
    absR=sqrt(sum(R.*R,2));
    absRm3 = 1./(absR.*absR.*absR);
    phiinf(:,3*S-2:3*S)= K*R.*(absRm3*[1 1 1]);
end
