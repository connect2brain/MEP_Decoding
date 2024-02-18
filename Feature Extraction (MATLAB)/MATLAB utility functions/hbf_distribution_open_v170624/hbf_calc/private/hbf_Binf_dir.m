function Binf=hbf_Binf_dir(fp,fpdir,spos,sdir)
% HBF_BINF_DIR  computes an arbitrary component of the magnetic field due to a 
%   current dipole in infinite non-magnetic medium 
%
% Binf=HBF_BINF_DIR(fp,fpdir,spos,sdir)
%   fp = field points (where the field is computed), [N x 3]
%   fpdir = field orientations (which component is computed), [N x 3]
%   spos  = source positions, [M x 3]
%   spdir = dipole moments (or, if unit-norm, dipole orientations), [M x 3]
%
%   Binf  = resulting magnetic field for each field point and dipole, [N x M]
% v160229 Matti Stenroos
Nfp=size(fp,1);
Nsp=size(spos,1);
Binf=zeros(Nfp,Nsp);

K=1e-7;%mu0/(4pi)
for S=1:Nsp,
    R=fp-ones(Nfp,1)*spos(S,:);
    absR=sqrt(sum(R.*R,2));
    absRm3 = 1./(absR.*absR.*absR);
    Binf(:,S)=K*triple(ones(Nfp,1)*sdir(S,:),R,fpdir).*absRm3;

end
function R=triple(R1,R2,R3)
%dot(R1,cross(R2,R3)=dot(cross(R1,R2),R3)
    R=R3(:,1).*(R1(:,2).*R2(:,3)-R1(:,3).*R2(:,2))+...
    R3(:,2).*(R1(:,3).*R2(:,1)-R1(:,1).*R2(:,3))+...
    R3(:,3).*(R1(:,1).*R2(:,2)-R1(:,2).*R2(:,1));

