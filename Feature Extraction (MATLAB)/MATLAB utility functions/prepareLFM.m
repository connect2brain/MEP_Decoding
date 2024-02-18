function [LnrN]=prepareLFM(headmodel, normalOrientations, scalingB)
%
% lead-field matrix generation
%
% input:
% headmodel - headmodel struct
% chanlocs0 - chanlocs struct
% normalOrientations - choose normal orientations
% badCh - bad channels -vector. boolean valued: 0/false is good chan, 
% 1/true bad chan
% scalingB - depth scaling such that weak (deep) sources get more more
% weight.  Otherwise, there is a strong depth bias with general linear 
% modelling (GLM) such that superficial sources get stronger amplitudes. 
% Does not help/affect beamforming.
% 
% output:
% LnrN - lead-field matrix
% chanlocs - chanlocs struct where bad channels where removed from
%

if normalOrientations
    LFM=headmodel.leadfield;
else
    LFM=headmodel.leadfield3D;
end


Lnr=LFM;

if ~normalOrientations
    LnrN=reshape(Lnr,[size(Lnr,1),3,size(Lnr,2)/3]);
else
    LnrN=reshape(Lnr,[size(Lnr,1),1,size(Lnr,2)]);
end

if scalingB
      
    vals=sqrt(sum(sum(LnrN.^2))); %norm of each source orient. The lfm is 3D 
    med=3*median(vals); temp=vals; temp(temp>med)=med; %values exceeding 
    % 3 x median are outliers are rescaled in the following to 3x median 
    LnrN=LnrN./vals.*temp;
    
    scalings=(sum(LnrN.^2)).^.2; % depth scaling such that the power of each source is scaled closer to each other but not exactly
    % note: if exponent is .5, the source powers  (norms) are set exactly
    % the same. If exponent is 0, there is no scaling at all
    LnrN=LnrN./scalings;
end

LnrN=squeeze(LnrN); %squeezing in case of fixed orientations (where 2nd dimension is redundant)
% lassoglm-based source locs with 3D-orientations