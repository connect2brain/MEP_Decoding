function brain_region = categorize_channels_rl(ch_names)
    brain_region = containers.Map('KeyType', 'char', 'ValueType', 'char');
    for i = 1:length(ch_names)
        ch = ch_names{i};
        if startsWith(ch, 'F') || startsWith(ch, 'AF') || startsWith(ch, 'FF') || startsWith(ch, 'AFp')
            brain_region(ch) = 'Frontal';
        elseif startsWith(ch, 'C') || contains(ch, 'FC') || contains(ch, 'CP')
            brain_region(ch) = 'Central';
        elseif startsWith(ch, 'P') || contains(ch, 'PO') || contains(ch, 'PPO')
            brain_region(ch) = 'Parietal';
        elseif startsWith(ch, 'T') || contains(ch, 'FT') || contains(ch, 'TP') || contains(ch, 'FFT') || contains(ch, 'TTP') || contains(ch, 'OI')
            brain_region(ch) = 'Temporal';
        elseif startsWith(ch, 'O') || contains(ch, 'POO') || contains(ch, 'OI') || contains(ch, 'Iz')
            brain_region(ch) = 'Occipital';
        else
            brain_region(ch) = 'Other';
        end
    end
end
