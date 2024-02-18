function [tfd]=rid_rihaczek4(x,fbins)
% This function computes reduced interference Rihaczek distribution;
% Input: x: signal, fbins=required frequency bins
% Output: tfd = Generated reduced interference Rihaczek distribution

tbins = length(x);
amb = zeros(tbins);

for tau = 1:tbins
    amb(tau,:) = ifft( conj(x) .* x([tau:tbins 1:tau-1]) );
end

% tbins is the total number of time bins (in this case 1001)
midpoint = ceil(tbins / 2); % Use ceil to ensure the index is an integer that is not odd
ambTemp = [amb(:, midpoint+1:tbins) amb(:, 1:midpoint)];
amb1 = [ambTemp(midpoint+1:tbins,:); ambTemp(1:midpoint,:)];

%ambTemp = [amb(:,tbins/2+1:tbins) amb(:,1:tbins/2)];
%amb1 = [ambTemp(tbins/2+1:tbins,:); ambTemp(1:tbins/2,:)];

D=(-1:2/(tbins-1):1)'*(-1:2/(tbins-1):1);
L=D;
K=chwi_krn(D,L,0.00001);
[s,d]=size(amb1);
df=K(1:s,1:d);
ambf = amb1 .* df;

A = zeros(fbins,tbins);
tbins=tbins-1;
if tbins ~= fbins
    for tt = 1:tbins
        A(:,tt) = datawrap(ambf(:,tt), fbins);
    end
else
    A = ambf; 
end

tfd = fft2(A);