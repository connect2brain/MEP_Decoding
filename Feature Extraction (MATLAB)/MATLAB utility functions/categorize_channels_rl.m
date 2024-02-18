function brain_side = categorize_channels_rl(ch_names)
    brain_side = containers.Map('KeyType', 'char', 'ValueType', 'char');
    for i = 1:length(ch_names)
        ch = ch_names{i};

        % Identify the character to use for side determination
        if ch(end) == 'h'
            % Use the second-to-last character if last character is 'h'
            relevant_char = ch(end-1);
        else
            % Use the last character otherwise
            relevant_char = ch(end);
        end

        % Categorize based on the relevant character
        if ismember(relevant_char, {'1', '3', '5', '7', '9'})
            brain_side(ch) = 'Left';
        elseif ismember(relevant_char, {'2', '4', '6', '8', '0'})
            brain_side(ch) = 'Right';
        else
            brain_side(ch) = 'Mid';
        end
    end
end

