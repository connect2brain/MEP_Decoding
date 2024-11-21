% % Function to calculate asymmetry
% function asymmetry_value = calculateAsymmetry(sub_df, Left_chs, Right_chs)
%     ln_R_power = mean(log(sub_df.PSD(ismember(sub_df.Channel, Right_chs))));
%     ln_L_power = mean(log(sub_df.PSD(ismember(sub_df.Channel, Left_chs))));
%     asymmetry_value = ln_R_power - ln_L_power;
% end

function asymmetry_value = calculateAsymmetry(sub_df, Left_chs, Right_chs)
    % Extract PSD values for right and left channels
    R_PSD = sub_df.PSD(ismember(sub_df.Channel, Right_chs));
    L_PSD = sub_df.PSD(ismember(sub_df.Channel, Left_chs));

    % Check if either right or left channel data is missing
    if isempty(R_PSD) || isempty(L_PSD)
        % Skip calculation or assign a default value (e.g., 0)
        asymmetry_value = NaN; % or use 0, or any other placeholder value
    else
        % Calculate asymmetry
        ln_R_power = mean(log(R_PSD));
        ln_L_power = mean(log(L_PSD));
        asymmetry_value = ln_R_power - ln_L_power;
    end
end
