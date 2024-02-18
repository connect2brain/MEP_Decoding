function preprocessedData = preprocessData(data, b, ARord, padsize)
    % Preprocess the data for connectivity analysis
    [numpts, numchan] = size(data);
    if numpts < (ARord + length(b) - 1)
        error('Data must have length greater or equal to sum of the model order and the FIR filter order');
    end

    % Data filtering
    data = filter(b, 1, data);
    data = data(length(b)-1:end, :);

    % Add leading and trailing points before Hilbert transform
    coefficients = zeros(ARord, numchan);
    for i = 1:numchan
        a = aryule(data(:, i), ARord);
        coefficients(:, i) = -1 * flip(a(:, 2:end)');
    end
    datapadded = cat(1, ones(padsize, numchan), data, ones(padsize, numchan));
    
    % Data padding
    for i = padsize:-1:1
        datapadded(end-i+1, :) = sum(coefficients .* datapadded((end-i-ARord+1):(end-i), :));
        datapadded(i, :) = sum(coefficients .* datapadded((i+ARord):-1:(i+1), :));
    end

    % Apply Hilbert transform
    preprocessedData = hilbert(datapadded);
end
