function phiinf=hbf_Phiinf_dir(fp,spos,sdir)
% HBF_PHIINF_DIR  computes potential due to a current dipole in infinite
%   homogeneous conductor that has unit conductivity
%
% phiinf=HBF_PHIINF_DIR(fp,spos,sdir)
%   fp = field points (where the field is computed), [N x 3]
%   spos  = source positions, [M x 3]
%   spdir = dipole moments (or, if unit-norm, dipole orientations), [M x 3]
%
%   phiinf = resulting potential for each field point and dipole, [N x M]
%
% v160229 Matti Stenroos
Nfp=size(fp,1);
Nsp=size(spos,1);
phiinf=zeros(Nfp,Nsp);

K=1/(4*pi);
for S=1:Nsp,
    R=fp-ones(Nfp,1)*spos(S,:);
    absR=sqrt(sum(R.*R,2));
    absRm3 = 1./(absR.*absR.*absR);
    source=K*ones(Nfp,1)*sdir(S,:);
    phiinf(:,S)= sum(source.*R,2).*absRm3;    
end
