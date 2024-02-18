function DB=hbf_DB_Linear(fieldpoints,dir,sourcemesh)
% HBF_DB_Linear makes a DB matrix between a mesh and field points.
%   The matrix is made for an arbitrary field component and linear basis.
%
% DB=HBF_DB_LINEAR(fieldpoints,fielddir,sourcemesh)
%   fieldpoints: a set of points, where DB is computed, [N x 3]
%   fielddir: field directions (unit length), [N x 3]
%   sourcemesh: mesh, where the double layer is spanned, hbf struct
%
%   DB: DB matrix, [N(field points) x N(source vertices)]
%
% v160229 Matti Stenroos


elements=sourcemesh.e;
points=sourcemesh.p;
noe=size(elements,1);
nop=size(points,1);
nof=size(fieldpoints,1);

e1=elements(:,1);
e2=elements(:,2);
e3=elements(:,3);

p1=points(e1,:);
p2=points(e2,:);
p3=points(e3,:);

%for gammas
y21=p1-p2;
y32=p2-p3;
y13=p3-p1;
absy21=sqrt(y21(:,1).*y21(:,1)+y21(:,2).*y21(:,2)+y21(:,3).*y21(:,3));
absy32=sqrt(y32(:,1).*y32(:,1)+y32(:,2).*y32(:,2)+y32(:,3).*y32(:,3));
absy13=sqrt(y13(:,1).*y13(:,1)+y13(:,2).*y13(:,2)+y13(:,3).*y13(:,3));
absym1=1./[absy21 absy32 absy13];
gnom=zeros(noe,3);
gden=zeros(noe,3);
gammas=zeros(noe,3);

% tempnormals=cross(p2-p1,p3-p1);
tempnormals=cross(y13,y21);
doubleareas=sqrt(sum(tempnormals.*tempnormals,2));
normals=tempnormals./(doubleareas*[1 1 1]);
S1dir=zeros(nof,noe);
S2dir=zeros(nof,noe);
S3dir=zeros(nof,noe);

DB=zeros(nof,nop);

for I=1:nof,
    fp=fieldpoints(I,:);
    fpdir=dir(I,:);
   
    ys=ones(nop,1)*fp-points;
    y1=ys(e1,:);
    y2=ys(e2,:);
    y3=ys(e3,:);
    
    abs_y=sqrt(ys(:,1).*ys(:,1)+ys(:,2).*ys(:,2)+ys(:,3).*ys(:,3));
    abs_y1=abs_y(e1);
    abs_y2=abs_y(e2);
    abs_y3=abs_y(e3);
    
    %solid angles
    dot12=dots(y1, y2);
    dot23=dots(y2, y3);
    dot13=dots(y1, y3);
    onom=triple(y1,y2,y3);
    oden=abs_y1.*abs_y2.*abs_y3 + dot12.*abs_y3 + dot13.*abs_y2 + dot23.*abs_y1;
%     omegas=2*atan2(onom,oden);%with this fp, this would be omega, but we want -omega..   
    omegas=-2*atan2(onom,oden);
    
    %gammas
    gnom(:,1)=abs_y2 .* absy21 + dots(y2,y21);
    gden(:,1)=abs_y1 .* absy21 + dots(y1,y21);
    
    gnom(:,2)=abs_y3 .* absy32 + dots(y3,y32);
    gden(:,2)=abs_y2 .* absy32 + dots(y2,y32);
    
    gnom(:,3)=abs_y1 .* absy13 + dots(y1,y13);
    gden(:,3)=abs_y3 .* absy13 + dots(y3,y13);
    
    test=abs(gnom)<1e-12 | abs(gden)<1e-12;
    if any(test(:))
        nonsingular=~test;
        gammas(nonsingular)=log(gnom(nonsingular)./gden(nonsingular)).*absym1(nonsingular);
    else
        gammas=log(gnom./gden).*absym1;
    end
    
    %project fp to surface of each triangle;
    fpm=ones(noe,1)*fp;
    vproj=dots(normals,y1);
    Amatrix=fpm-[vproj vproj vproj].*normals;
    cmatrix=Amatrix-fpm;
    
    c1vec=p1-Amatrix;
    c2vec=p2-Amatrix;
    c3vec=p3-Amatrix;
    c1x2n=triple(normals,c1vec,c2vec);
    c2x3n=triple(normals,c2vec,c3vec);
    c3x1n=triple(normals,c3vec,c1vec);
    
    ndotc=dots(normals,cmatrix);
    omegasw=ndotc.*omegas;
    s1=gammas(:,1).*c1x2n;
    s2=gammas(:,2).*c2x3n;
    s3=gammas(:,3).*c3x1n;
    G=(s1+s2+s3-omegasw);
    Gtemp=(G./(doubleareas))*[1 1 1];    
    S1dir(I,:)=-dots(y32.*Gtemp,ones(noe,1)*fpdir)';
    S2dir(I,:)=-dots(y13.*Gtemp,ones(noe,1)*fpdir)';
    S3dir(I,:)=-dots(y21.*Gtemp,ones(noe,1)*fpdir)';

end
for J=1:noe,
    DB(:,e1(J))=DB(:,e1(J))+S1dir(:,J);
    DB(:,e2(J))=DB(:,e2(J))+S2dir(:,J);
    DB(:,e3(J))=DB(:,e3(J))+S3dir(:,J);

end

function R=cross(R1,R2)
R(:,1)=R1(:,2).*R2(:,3)-R1(:,3).*R2(:,2);
R(:,2)=R1(:,3).*R2(:,1)-R1(:,1).*R2(:,3);
R(:,3)=R1(:,1).*R2(:,2)-R1(:,2).*R2(:,1);
function dot=dots(R1,R2)
dot=R1(:,1).*R2(:,1)+R1(:,2).*R2(:,2)+R1(:,3).*R2(:,3);

function R=triple(R1,R2,R3)
%dot(R1,cross(R2,R3)=dot(cross(R1,R2),R3)
    R=R3(:,1).*(R1(:,2).*R2(:,3)-R1(:,3).*R2(:,2))+...
    R3(:,2).*(R1(:,3).*R2(:,1)-R1(:,1).*R2(:,3))+...
    R3(:,3).*(R1(:,1).*R2(:,2)-R1(:,2).*R2(:,1));
