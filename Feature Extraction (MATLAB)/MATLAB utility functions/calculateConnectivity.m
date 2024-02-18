function [iPLV, PLI, wPLI, Coh, imCoh, lagCoh, oPEC] = calculateConnectivity(preprocessedData, padsize, indxcmb)
    % Calculate various connectivity measures
    numPairs = size(indxcmb, 1);
    numTrials = size(preprocessedData, 3); % Assuming third dimension is trials
    
    % Initialize connectivity matrices
    iPLV = zeros(numPairs, numTrials);
    PLI = zeros(numPairs, numTrials);
    wPLI = zeros(numPairs, numTrials);
    Coh = zeros(numPairs, numTrials);
    imCoh = zeros(numPairs, numTrials);
    lagCoh = zeros(numPairs, numTrials);
    oPEC = zeros(numPairs, numTrials);
    
    % Start parallel pool if not already started
    if isempty(gcp('nocreate'))
        parpool; % Start with default settings
    end

    % Using parfor loop for parallel processing
    parfor trial = 1:numTrials
        % Initialize temporary variables for each connectivity measure
        temp_iPLV = zeros(numPairs, 1);
        temp_PLI = zeros(numPairs, 1);
        temp_wPLI = zeros(numPairs, 1);
        temp_Coh = zeros(numPairs, 1);
        temp_imCoh = zeros(numPairs, 1);
        temp_lagCoh = zeros(numPairs, 1);
        temp_oPEC = zeros(numPairs, 1);


        for pairIdx = 1:numPairs
            % Extract signals for the pair
            signal1 = preprocessedData((padsize+1):(end-padsize), indxcmb(pairIdx, 1), trial);
            signal2 = preprocessedData((padsize+1):(end-padsize), indxcmb(pairIdx, 2), trial);

            %% oPEC
           
            % Orthogonalization 
            signal1Orth = signal1 - (signal2 * (signal2' * signal1) / (signal2' * signal2));
            signal2Orth = signal2 - (signal1 * (signal1' * signal2) / (signal1' * signal1));

            % Compute power envelopes
            pow1Orth = abs(signal1Orth).^2;
            pow2Orth = abs(signal2Orth).^2;

            % Log-transform
            lnPow1Orth = log(pow1Orth + eps);
            lnPow2Orth = log(pow2Orth + eps);

            % Standardize (z-score) the log-transformed power
            zLnPow1Orth = (lnPow1Orth - mean(lnPow1Orth)) / std(lnPow1Orth);
            zLnPow2Orth = (lnPow2Orth - mean(lnPow2Orth)) / std(lnPow2Orth);

            % Compute Pearson correlation
            corrVal = corr(zLnPow1Orth, zLnPow2Orth);

            % Compute phase difference
            cs = signal1 .* conj(signal2);
            lags = angle(cs);
            Cohy = mean(cs) / sqrt(mean(abs(signal1).^2) * mean(abs(signal2).^2));

            % Store results in temporary variables
            temp_iPLV(pairIdx) = abs(mean(imag(exp(1i * lags))));
            temp_PLI(pairIdx) = abs(mean(sign(lags)));
            temp_wPLI(pairIdx) = abs(sum(imag(cs)) / sum(abs(imag(cs))));
            temp_Coh(pairIdx) = abs(Cohy);
            temp_imCoh(pairIdx) = abs(imag(Cohy));
            temp_lagCoh(pairIdx) = (imag(mean(cs))^2) / ((mean(abs(signal1).^2) * mean(abs(signal2).^2)) - (real(mean(cs))^2));
            temp_oPEC(pairIdx) = 0.5 * log((1 + corrVal) / (1 - corrVal)); % Fisher's Z-Transform
        end

        % Assign temporary results to final matrices
        iPLV(:, trial) = temp_iPLV;
        PLI(:, trial) = temp_PLI;
        wPLI(:, trial) = temp_wPLI;
        Coh(:, trial) = temp_Coh;
        imCoh(:, trial) = temp_imCoh;
        lagCoh(:, trial) = temp_lagCoh;
        oPEC(:, trial) = temp_oPEC;
    end
end
