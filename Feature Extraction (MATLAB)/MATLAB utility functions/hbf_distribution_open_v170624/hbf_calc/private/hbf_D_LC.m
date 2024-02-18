function D=hbf_D_LC(fieldmesh,sourcemesh,verbose)
% HBF_DI_LC makes a LC double-layer matrix between two meshes 
%
% D=HBF_D_LC(fieldmesh,sourcemesh,verbose)
%   fieldmesh: mesh, where double-layer potential is evaluated, hbf struct
%   sourcemesh: mesh, where the double layer is spanned, hbf struct
%   verbose (optional, default 0): give 1 for intermediate output
%
%   D: Double-layer matrix, [N(field vertices) x N(source vertices)]
%
% v160229 Matti Stenroos


if nargin==2
    verbose=0;
end

elements=sourcemesh.e;
nodes=sourcemesh.p;
nop=size(nodes,1);
noe=size(elements,1);

e1=elements(:,1);
e2=elements(:,2);
e3=elements(:,3);

fp=fieldmesh.p;
nof=size(fp,1);

shapes1=zeros(nof,noe);
shapes2=zeros(nof,noe);
shapes3=zeros(nof,noe);

%for gammas
gnom=zeros(noe,3);
gden=zeros(noe,3);

gammas=zeros(noe,3);
y21=nodes(e1,:)-nodes(e2,:);
y32=nodes(e2,:)-nodes(e3,:);
y13=nodes(e3,:)-nodes(e1,:);
absy21=sqrt(y21(:,1).*y21(:,1)+y21(:,2).*y21(:,2)+y21(:,3).*y21(:,3));
absy32=sqrt(y32(:,1).*y32(:,1)+y32(:,2).*y32(:,2)+y32(:,3).*y32(:,3));
absy13=sqrt(y13(:,1).*y13(:,1)+y13(:,2).*y13(:,2)+y13(:,3).*y13(:,3));
absym1=1./[absy21 absy32 absy13];

normals=cross(-y21,y13);
areas=sqrt(sum(normals.*normals,2))/2;
sden=1./(4*areas.*areas);


if verbose
    disp('Creating DL matrix - calculating shape functions');
    dispstep=round(nof/10);
end

for I=1:nof,
    if verbose && ~mod(I,dispstep)
        fprintf('%2d percents done\n',round(100*I/nof));
    end
    %
    %%shapes=D_Shapes_LC_3(fp(I,:),triangles,nodes,normals,areas);
    
    ys=ones(nop,1)*fp(I,:)-nodes;
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
    omegas=2*atan2(onom,oden);
    
    
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
    
    dmat=[onom onom onom];
    
    s1vec=-y32.*dmat;
    s2vec=-y13.*dmat;
    s3vec=-y21.*dmat;
        
    thirdmat=-y21.*[gammas(:,1) gammas(:,1) gammas(:,1)]-...
        y32.*[gammas(:,2) gammas(:,2) gammas(:,2)]-...
        y13.*[gammas(:,3) gammas(:,3) gammas(:,3)];
    
    f1temp=triple(normals,y2,y3);
    f2temp=triple(normals,y3,y1);
    f3temp=triple(normals,y1,y2);
    
    f1vec=f1temp.*omegas;
    f2vec=f2temp.*omegas;
    f3vec=f3temp.*omegas;
    
    shapes1(I,:)=((f1vec+dots(s1vec,thirdmat)).*sden)';
    shapes2(I,:)=((f2vec+dots(s2vec,thirdmat)).*sden)';
    shapes3(I,:)=((f3vec+dots(s3vec,thirdmat)).*sden)';
    
end
D=zeros(nof,nop);
for J=1:noe,
    D(:,e1(J))=D(:,e1(J))+shapes1(:,J);
    D(:,e2(J))=D(:,e2(J))+shapes2(:,J);
    D(:,e3(J))=D(:,e3(J))+shapes3(:,J);
end
D=D/(4*pi);

function R=cross(R1,R2)
R=zeros(size(R1));
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