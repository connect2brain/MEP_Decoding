function [filteredData] = spatialFilter(subjData, ch_names, channel_group, original_weights)
    % SPATIALFILTER Applies given weights to selected EEG channels in a 3D data array.
    %
    % Inputs:
    %   subjData       - A 3D array of EEG data (channels x time x trials).
    %   ch_names       - Cell array of channel names corresponding to the rows in subjData.
    %   channel_group  - Cell array of channel names for the spatial filter.
    %   original_weights - Original array of weights for the full channel group.
    %
    % Output:
    %   filteredData   - The spatially filtered EEG data.

    assert(ndims(subjData) == 3, 'subjData must be a 3D array.');

    % Initialize filteredData with zeros
    [numChannels, numTimePoints, numTrials] = size(subjData);
    filteredData = zeros(numTimePoints, numTrials);

    % Apply weights to available channels
    for i = 1:length(channel_group)
        channel = channel_group{i};
        weight = original_weights(i);
        channelIndex = find(strcmp(ch_names, channel));

        if ~isempty(channelIndex)
            % Add weighted channel data to filteredData
            filteredData = filteredData + weight * squeeze(subjData(channelIndex, :, :));
        end
    end
end



