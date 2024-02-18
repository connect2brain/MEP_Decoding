function omegas = hbf_SolidAngles(triangles,vertices,fieldpoints)
%HBF_SOLIDANGLES calculates solid angles spanned by triangles of a mesh in
%   one point or a pointset
%
%omega=HBF_SOLIDANGLES(mesh,fieldpoints)
%omega=HBF_SOLIDANGLES(elements,vertices,fieldpoints)
%   mesh:   hbf mesh struct
%   elements: triangle description, [N(triangles) x 3]
%   vertices:   mesh vertices, [N(vertices) x 3]
%   fieldpoint: point, in which the angles are evaluated, [N(fieldpoints) x 3]
%
%   omegas: solid angles, [N(fieldpoints) x N(triangles]
% Eq. 8 of vanOosterom & Strackee, IEEE TBME 1983
%
% This function assumes, by tradition, that the mesh has CW orientation.
%   When used with CCW meshes (like meshes is Helsinki BEM framework), the
%   angles thus have wrong sign: a point inside a closed mesh sees omega of
%   -4pi instead of correct 4pi.
% v160419 Matti Stenroos
if nargin==2,
    fieldpoints=vertices;
    vertices=triangles.p;
    triangles=triangles.e;
end
e1=triangles(:,1);
e2=triangles(:,2);
e3=triangles(:,3);
Nof=size(fieldpoints,1);
omegas=zeros(Nof,size(e1,1));
for F=1:Nof,
    p=vertices-ones(length(vertices),1)*fieldpoints(F,:);
    absp=sqrt(p(:,1).^2+p(:,2).^2+p(:,3).^2);
    
    
    absp1=absp(e1);
    absp2=absp(e2);
    absp3=absp(e3);
    
    p1=p(e1,:);
    p2=p(e2,:);
    p3=p(e3,:);
    
    dot12=dotp(p1, p2);
    dot23=dotp(p2, p3);
    dot13=dotp(p1, p3);
    
    nom=triple(p1,p2,p3);
    den=absp1.*absp2.*absp3 + dot12.*absp3 + dot13.*absp2 + dot23.*absp1;
    omegas(F,:)=-2*atan2(nom,den);
end
function dot=dotp(R1,R2)
    dot=R1(:,1).*R2(:,1)+R1(:,2).*R2(:,2)+R1(:,3).*R2(:,3);
function R=triple(R1,R2,R3)
%dot(R1,cross(R2,R3)=dot(cross(R1,R2),R3)
R=R3(:,1).*(R1(:,2).*R2(:,3)-R1(:,3).*R2(:,2))+...
    R3(:,2).*(R1(:,3).*R2(:,1)-R1(:,1).*R2(:,3))+...
    R3(:,3).*(R1(:,1).*R2(:,2)-R1(:,2).*R2(:,1));