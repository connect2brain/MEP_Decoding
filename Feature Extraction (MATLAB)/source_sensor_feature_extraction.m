%% SOURCE AND SENSOR FEATURE EXTRACTION

clear 
clc

%% Preparation

addpath '';

% Set paths to relevant toolboxes
TOOLBOXPATH = '';
addpath(fullfile(TOOLBOXPATH, 'eeglab2023.1'));
addpath(fullfile(TOOLBOXPATH, 'fieldtrip-20230716'));
addpath(fullfile(TOOLBOXPATH, 'hbf_distribution_open_v170624'));
addpath(fullfile(TOOLBOXPATH, 'plotroutines_v180921'));
addpath(fullfile(TOOLBOXPATH, 'plotroutines_v170706'));

% Initialize fieldtrip
ft_defaults

input_directory = '';
output_directory = '';

RdBuReversed = load(fullfile(input_directory, 'RdBuReversed.mat'));
RdBuReversed = RdBuReversed.RdBuReversed;

dataTable = readtable(fullfile(input_directory, 'subjects_list.xlsx'), 'Basic', 1);

subjects = [0:49];

% Feature Extraction
for subnum = subjects
    fprintf('Subject %d ', subnum)
    dataTableSub = dataTable(subnum, :);
    load(char(dataTableSub.headmodel))
    load(char(dataTableSub.cleandata))
    
    epochs = epochs2;

    % Structure of epochs2:
    % epochs2: structure containing cleaned and aligned EEG/TMS data with fields:
    %   .trial      : [1000×400×ch double] - 3D matrix of timepoints × trials × channels
    %   .time       : [1×1000 double] - time vector in seconds with respect to TMS pulse, -1.0050 to -0.0060.
    %   .dimord     : 'time_rpt_chan' - data dimensions (.trial) are timepoints x trials x channels
    %   .label      : {113×1 cell} - channel labels included in analysis
    %   .cfg        : [1×1 struct] - information about previous preprocessing operations
    %   .sampleinfo : [400×2 double] - sample boundaries (start/end) for each trial
    %   .trialSorting: [400×1 double] - original trial indices
    %   .trialLabels : [400×1 double] - binary labels (1: high MEPs[1-200], 0: low MEPs[201-400])
    %   .mepsize     : [400×1 double] - MEP amplitudes in mV
    %   .fsample     : 1000 - sampling rate in Hz

    %       ______   ______  __    __ _______   ______  ________      _______  ________  ______   ______  __    __  ______  ________ _______  __    __  ______  ________ ______  ______  __    __ 
    %  /      \ /      \|  \  |  \       \ /      \|        \    |       \|        \/      \ /      \|  \  |  \/      \|        \       \|  \  |  \/      \|        \      \/      \|  \  |  \
    % |  ▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓  | ▓▓ ▓▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓    | ▓▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓\ | ▓▓  ▓▓▓▓▓▓\\▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓\ ▓▓  | ▓▓  ▓▓▓▓▓▓\\▓▓▓▓▓▓▓▓\▓▓▓▓▓▓  ▓▓▓▓▓▓\ ▓▓\ | ▓▓
    % | ▓▓___\▓▓ ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓__| ▓▓ ▓▓   \▓▓ ▓▓__        | ▓▓__| ▓▓ ▓▓__   | ▓▓   \▓▓ ▓▓  | ▓▓ ▓▓▓\| ▓▓ ▓▓___\▓▓  | ▓▓  | ▓▓__| ▓▓ ▓▓  | ▓▓ ▓▓   \▓▓  | ▓▓    | ▓▓ | ▓▓  | ▓▓ ▓▓▓\| ▓▓
    %  \▓▓    \| ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓    ▓▓ ▓▓     | ▓▓  \       | ▓▓    ▓▓ ▓▓  \  | ▓▓     | ▓▓  | ▓▓ ▓▓▓▓\ ▓▓\▓▓    \   | ▓▓  | ▓▓    ▓▓ ▓▓  | ▓▓ ▓▓        | ▓▓    | ▓▓ | ▓▓  | ▓▓ ▓▓▓▓\ ▓▓
    %  _\▓▓▓▓▓▓\ ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓▓▓▓▓▓\ ▓▓   __| ▓▓▓▓▓       | ▓▓▓▓▓▓▓\ ▓▓▓▓▓  | ▓▓   __| ▓▓  | ▓▓ ▓▓\▓▓ ▓▓_\▓▓▓▓▓▓\  | ▓▓  | ▓▓▓▓▓▓▓\ ▓▓  | ▓▓ ▓▓   __   | ▓▓    | ▓▓ | ▓▓  | ▓▓ ▓▓\▓▓ ▓▓
    % |  \__| ▓▓ ▓▓__/ ▓▓ ▓▓__/ ▓▓ ▓▓  | ▓▓ ▓▓__/  \ ▓▓_____     | ▓▓  | ▓▓ ▓▓_____| ▓▓__/  \ ▓▓__/ ▓▓ ▓▓ \▓▓▓▓  \__| ▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓__/ ▓▓ ▓▓__/  \  | ▓▓   _| ▓▓_| ▓▓__/ ▓▓ ▓▓ \▓▓▓▓
    %  \▓▓    ▓▓\▓▓    ▓▓\▓▓    ▓▓ ▓▓  | ▓▓\▓▓    ▓▓ ▓▓     \    | ▓▓  | ▓▓ ▓▓     \\▓▓    ▓▓\▓▓    ▓▓ ▓▓  \▓▓▓\▓▓    ▓▓  | ▓▓  | ▓▓  | ▓▓\▓▓    ▓▓\▓▓    ▓▓  | ▓▓  |   ▓▓ \\▓▓    ▓▓ ▓▓  \▓▓▓
    %   \▓▓▓▓▓▓  \▓▓▓▓▓▓  \▓▓▓▓▓▓ \▓▓   \▓▓ \▓▓▓▓▓▓ \▓▓▓▓▓▓▓▓     \▓▓   \▓▓\▓▓▓▓▓▓▓▓ \▓▓▓▓▓▓  \▓▓▓▓▓▓ \▓▓   \▓▓ \▓▓▓▓▓▓    \▓▓   \▓▓   \▓▓ \▓▓▓▓▓▓  \▓▓▓▓▓▓    \▓▓   \▓▓▓▓▓▓ \▓▓▓▓▓▓ \▓▓   \▓▓
    % 
    % SOURCE RECONSTRUCTION

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ▄▄▀██ ▄▄▄ █▄ ▄██████ ▄▄▄█▄▀█▀▄█▄▄ ▄▄██ ▄▄▀█ ▄▄▀██ ▄▄▀█▄▄ ▄▄█▄ ▄██ ▄▄▄ ██ ▀██ 
    % ██ ▀▀▄██ ███ ██ ███████ ▄▄▄███ █████ ████ ▀▀▄█ ▀▀ ██ ██████ ████ ███ ███ ██ █ █ 
    % ██ ██ ██ ▀▀▀ █▀ ▀██████ ▀▀▀█▀▄█▄▀███ ████ ██ █ ██ ██ ▀▀▄███ ███▀ ▀██ ▀▀▀ ██ ██▄ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % ROI EXTRACTION
    
    % Get vertex indices of interest
    atlasGLASSER = load('...\Glasser_parcellation.mat');
    atlasGLASSER = atlasGLASSER.atlasGLASSER;
   
    % Define the base ROIs as individual strings
    base_roi_labels = {
        '4_ROI',         % M1 - precentral gyrus/ BA 4
        '1_ROI',         % S1 - postcentral gyrus/ BA 1
        '2_ROI',         % S1 - postcentral gyrus/ BA 2
        '3a_ROI',        % S1 - postcentral gyrus/ BA 3a
        '6ma_ROI',       % SMA - medial surface of the frontal lobe
        '6d_ROI',        % PMC - anterior to M1
        '9m_ROI',        % DLPFC: BA 9
        '46_ROI',        % DLPFC: BA 46
        '7PL_ROI',       % PPC
        '7PC_ROI',       % PPC
        '7Am_ROI',       % PPC
        '8C_ROI',        % FEF
        '8BM_ROI'        % FEF
    };

    % Initialize an empty cell array for the combined ROI labels
    roi_labels = {};
    
    % Loop over the base ROI labels to add left and right hemisphere prefixes
    for i = 1:length(base_roi_labels)
        roi_labels{end + 1} = ['L_L_' base_roi_labels{i}];
        roi_labels{end + 1} = ['R_R_' base_roi_labels{i}];
    end

    % Initialize a structure to hold the coordinates for each ROI 
    roi_coordinates = struct();
    
    % Loop through each ROI label to extract coordinates
    for i = 1:length(roi_labels)
        label = roi_labels{i};
        label_indices = find(ismember(atlasGLASSER.parcellationlabel, label));
        vertex_indices = find(ismember(atlasGLASSER.parcellation, label_indices));
        roi_coordinates.(label) = headmodel.smesh.p(vertex_indices, :); 
    end

    % Store relevant ROI indices 
    % Initialize a structure to hold the vertex indices for each ROI
    roi_vertex_indices = struct();
    
    % Loop through each ROI label to extract vertex indices
    for i = 1:length(roi_labels)
        label = roi_labels{i};
        label_indices = find(ismember(atlasGLASSER.parcellationlabel, label));
        vertex_indices = find(ismember(atlasGLASSER.parcellation, label_indices));
    
        % Store the vertex indices for each ROI in the structure
        roi_vertex_indices.(label) = vertex_indices; 
    end

    % Extract indices into a cell array
    indices_cell_array = struct2cell(roi_vertex_indices);
    
    % Ensure each array in the cell array is a column vector
    indices_cell_array = cellfun(@(x) x(:), indices_cell_array, 'UniformOutput', false);
    
    % Flatten the cell array to obtain a single array of indices
    DIPOLES_OF_INTEREST = vertcat(indices_cell_array{:})';

    % Extract ROI names into a cell array
    roi_cell_array = repelem(fieldnames(roi_vertex_indices),cell2mat(cellfun(@(x) numel(x), indices_cell_array, 'UniformOutput', false)'));

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ █████ ▄▄▀██ ▄▀▄ ██ ███ ████ ▄▄▀██ ▄▄▄█ ▄▄▀██ ▄▀▄ ██ ▄▄▄██ ▄▄▄ ██ ▄▄▀██ ▄▀▄ █▄ ▄██ ▀██ ██ ▄▄ ██
    % ██ █████ █████ █ █ ███ █ █████ ▄▄▀██ ▄▄▄█ ▀▀ ██ █ █ ██ ▄▄███ ███ ██ ▀▀▄██ █ █ ██ ███ █ █ ██ █▀▀██
    % ██ ▀▀ ██ ▀▀▄██ ███ ███▄▀▄█████ ▀▀ ██ ▀▀▀█ ██ ██ ███ ██ █████ ▀▀▀ ██ ██ ██ ███ █▀ ▀██ ██▄ ██ ▀▀▄██
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % LCMV BEAMFORMING 
                                                                                                                                                                                        
    % Match the leadfield with the available channels
    [~, elec_indx]=ismember(epochs.label, headmodel.label); % position of leadfield channel in data channel
    
    if any(elec_indx == 0)
        warning('Removing channels from data that are not in the headmodel')
        epochs.trial(:,:,elec_indx == 0) = [];
        epochs.label(elec_indx == 0) = [];
        [~, elec_indx]=ismember(epochs.label, headmodel.label); % recalculate
    end
    
    % Reorder the leadfield matrix to match the electrode order in the data
    L_reordered = headmodel.leadfield(elec_indx,:);
    headmodel.leadfield=L_reordered;
    headmodel.label = epochs.label;

    % Sanity Check
    % [LnrN]=prepareLFM(headmodel, 1, []);
    % sensitivity_profile=zeros(size(L_reordered,1),1);
    % sensitivity_profile(ismember(headmodel.label, 'C3'))=1;
    % subj_smesh = headmodel.smesh;
    % PlotDataOnMesh(subj_smesh,sensitivity_profile'*LnrN,'colormap', jet ,'colorbar', 0, 'view', [-90 50]);
  
    % Estimate the covariance matrix
    Cov = cov(reshape(epochs.trial, [], size(epochs.trial, 3)));
    close all;

    % Set parameters for the number of dipoles and sensors based on the sizes of L_reordered and Covariance matrix
    nr_dipoles = size(L_reordered,2);
    nr_sensors = size(Cov,2);
    
    % Start timing the execution
    tic
    
    % Create a range of lambda values for regularization
    lambdarange = unique([0.001:0.01:1 1:0.1:10 10:5:100]);
    numiter = numel(lambdarange);
    
    % Initialize a table to store the sensitivity profiles for Z-scored data
    test_sens_profZ = zeros([length(DIPOLES_OF_INTEREST) numiter+2]);
    test_sens_profZ(:,1) = DIPOLES_OF_INTEREST';
    test_sens_profZ = array2table(test_sens_profZ);
    test_sens_profZ.Properties.VariableNames{1} = 'dip Idx';
    test_sens_profZ.Properties.VariableNames{2} = 'roi';
    test_sens_profZ.("roi") = roi_cell_array;
    
    % Duplicate the structure for raw data
    test_sens_profraw = test_sens_profZ;
    
    % Loop through each lambda value
    for l = 1:numiter
        % Regularize the covariance matrix
        lambda = lambdarange(l);
        Cov_regularized = Cov + lambda * eye(size(Cov));
        invCy = pinv(Cov_regularized);
    
        % Initialize the weights matrix
        weights = zeros(nr_dipoles, nr_sensors);
    
        % Compute filter weights for each dipole
        for i=1:nr_dipoles
            lf1 = L_reordered(:,i);
            filt = pinv(lf1' * invCy * lf1) * lf1' * invCy;
            weights(i,:) = filt;            
        end
        
        % Update the sensitivity profiles for each dipole of interest
        for k = 1:numel(DIPOLES_OF_INTEREST)
            d = DIPOLES_OF_INTEREST(k);
            tmp     = weights(d,:) * L_reordered;
            tmp2    = normalize(tmp);
            test_sens_profZ{k,l+2} = tmp2(d);
            test_sens_profraw{k,l+2} = tmp(d); % should all be 1
        end
        % Set column names for the current lambda value
        test_sens_profZ.Properties.VariableNames{l+2} = ['L' num2str(lambda)];
        test_sens_profraw.Properties.VariableNames{l+2} = ['L' num2str(lambda)];
    end
    
    % Extract column names that contain lambda values
    lvar = test_sens_profZ.Properties.VariableNames(contains(test_sens_profZ.Properties.VariableNames, 'L'));
    
    % Compute summary statistics for Z-scored and raw data
    summary_test_sens_profZ = grpstats(test_sens_profZ(:,horzcat({'roi'}, lvar)),'roi');
    summary_test_sens_profraw = grpstats(test_sens_profraw(:,horzcat({'roi'}, lvar)),'roi');
    
    % Calculate mean sensitivity profile across lambdas
    meanByLambdaZ = mean(summary_test_sens_profZ(:,3:end));
    meanByLambdaraw = mean(summary_test_sens_profraw(:,3:end));
    
    % % Plotting results
    % figure
    % hold on
    % p = plot(lambdarange, summary_test_sens_profZ{:,3:end},'linew',2);
    % plot(lambdarange, meanByLambdaZ{1,:},'k','linew',4)
    % l = legend([summary_test_sens_profZ.roi', {'mean'}], Interpreter='none',Location='none');
    % hold off
    % for lab = 1:numel(p)
    %     text(p(lab).XData(find(p(lab).YData == max(p(lab).YData),1)), max(p(lab).YData) + 0.01*range(ylim), p(lab).DisplayName, HorizontalAlignment="center",Interpreter="none")
    % end
    % l.Visible = 'off';
    % box
    % axis padded
    % set(p,'LineStyle','none')
    % set(p,'Marker','o')
    % title(sprintf('Best lambda is %.3f at a mean Z score of %.3f', lambdarange(find(meanByLambdaZ{:,:} == max(meanByLambdaZ{1,:}))) , meanByLambdaZ{1,find(meanByLambdaZ{:,:} == max(meanByLambdaZ{1,:}))}))
    
    % Stop timing the execution
    toc

    % Optimized lambda weights calculation
    lambda = lambdarange(meanByLambdaZ{:,:} == max(meanByLambdaZ{1,:}));
    Cov_regularized = Cov + lambda * eye(size(Cov));
    invCy = pinv(Cov_regularized);
    weights = zeros(nr_dipoles, nr_sensors);
    for i=1:nr_dipoles
        lf1 = L_reordered(:,i);
        filt = pinv(lf1' * invCy * lf1) * lf1' * invCy;
        weights(i,:) = filt;            
    end
  
    % %Plot sensitivity profile
    % figure
    % dip2plot = 14920;
    % points                  = headmodel.smesh.p(dip2plot,:);
    % the_view                = [0 90];
    % sens_prof               = weights(dip2plot,:) * L_reordered ; 
    % sens_prof               = normalize(sens_prof);
    % % sens_profl              = sens_prof(:,1:end/2);
    % % sens_profr              = sens_prof(:,end/2+1:end);
    % hold on
    % SensProfPlot            = PlotDataOnMesh(headmodel.smesh,sens_prof,'view',the_view,'colormap',RdBuReversed, 'colorbar', 0);  
    % plot3(points(:,1),points(:,2),points(:,3),'Color', 'black', 'Marker', '.', 'MarkerSize',20)
    % scatter3(headmodel.elec_ft.chanpos(:,1),headmodel.elec_ft.chanpos(:,2),headmodel.elec_ft.chanpos(:,3), 20,'blue','filled')
    % hold off
    % title(sprintf('Sensitivity profile with lambda = %.2f for dipole %d = %.3f', lambda, dip2plot, sens_prof(dip2plot)))
    % colorbar

    clear('elec_indx', 'L_reordered', 'lambda', 'Cov_regularized', 'nr_sensors', 'nr_dipoles', 'invCy', 'i', 'lf1', 'filt')
   
    
    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ██ ██ ▄▄▄█ ▄▄▀██ ▄▄▀██ ▄▀▄ ██ ▄▄▄ ██ ▄▄▀██ ▄▄▄██ █████████ ███ █▄ ▄██ ▄▄▄ ██ ██ █ ▄▄▀██ ████▄ ▄██ ▄▄▄ █ ▄▄▀█▄▄ ▄▄█▄ ▄██ ▄▄▄ ██ ▀██ 
    % ██ ▄▄ ██ ▄▄▄█ ▀▀ ██ ██ ██ █ █ ██ ███ ██ ██ ██ ▄▄▄██ ██████████ █ ███ ███▄▄▄▀▀██ ██ █ ▀▀ ██ █████ ███▄▄▄▀▀█ ▀▀ ███ ████ ███ ███ ██ █ █ 
    % ██ ██ ██ ▀▀▀█ ██ ██ ▀▀ ██ ███ ██ ▀▀▀ ██ ▀▀ ██ ▀▀▀██ ▀▀ ███████▄▀▄██▀ ▀██ ▀▀▀ ██▄▀▀▄█ ██ ██ ▀▀ █▀ ▀██ ▀▀▀ █ ██ ███ ███▀ ▀██ ▀▀▀ ██ ██▄ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % HEADMODEL VISUALISATION

    % % Plot all ROIs based on the selected indices
    % % Get subject smesh and bmeshes
    % bmeshes = {headmodel.bmeshes(1) headmodel.bmeshes(2) headmodel.bmeshes(3)}; 
    % subj_smesh = headmodel.smesh;
    % 
    % % Close all existing figures and set the figure window style
    % close all;
    % set(0, 'DefaultFigureWindowStyle', 'docked');
    % roi_colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k', [1 0.5 0], [0.5 0 1], [0 1 0.5], [0.5 1 0], [1 0 0.5], [0 0.5 1]};
    % 
    % % Create a new figure
    % figure;
    % 
    % % Define a view for the plot
    % view_angle = [-125 35]; % Adjust this as needed
    % 
    % % Plot the brain mesh
    % hp = PlotMesh(subj_smesh, 'facealpha', 1, 'facecolor', [248 255 192]./255, 'view', view_angle);
    % ax = hp.Parent;
    % hold(ax, 'on');
    % 
    % % Iterate through each unique ROI and plot selected indices
    % unique_roi_labels = unique(strrep(roi_labels, 'L_L_', ''), 'stable');
    % unique_roi_labels = unique(strrep(unique_roi_labels, 'R_R_', ''), 'stable');
    % 
    % for i = 1:length(unique_roi_labels)
    %     roi_name = unique_roi_labels{i};
    %     left_indices = roi_vertex_indices.(['L_L_' roi_name]);
    %     right_indices = roi_vertex_indices.(['R_R_' roi_name]);
    % 
    %     % Combine left and right indices
    %     combined_indices = [left_indices; right_indices];
    % 
    %     % Determine the color for this ROI
    %     color = roi_colors{i};
    % 
    %     % Plot points for this ROI
    %     points = subj_smesh.p(combined_indices, :);
    %     plot3(points(:, 1), points(:, 2), points(:, 3), 'Color', color, 'Marker', '.', 'MarkerSize', 20, 'Parent', ax);
    % end
    % 
    % hold(ax, 'off');
    % 
    % % Set the view angle and enable 3D rotation
    % view(ax, view_angle);
    % rotate3d on;

    %% ENTIRE HEADMODEL AND ELECTRODES
    % fig = figure('Name', datafile_filename); axes; hold on
    % 
    % patch('faces', headmodel.bmeshes(end-2).e, 'vertices', headmodel.bmeshes(end-2).p, 'facealpha', 0.1, 'edgecolor', 'none'); % csf to bone
    % patch('faces', headmodel.bmeshes(end-1).e, 'vertices', headmodel.bmeshes(end-1).p, 'facecolor', [248 232 192]./255, 'facealpha', 0.2, 'edgecolor', 'none'); % bone to scalp
    % patch('faces', headmodel.bmeshes(end).e, 'vertices', headmodel.bmeshes(end).p, 'facecolor', [1 0.7 0.7], 'facealpha', 0.2, 'edgecolor', [0.9 0.9 0.9], 'edgealpha', 0.05); % scalp to air
    % 
    % %add the sensors
    % elec_xyz = mat2cell(headmodel.elec.pproj, length(headmodel.elec.pproj), [1 1 1]);
    % %elec_xyz = mat2cell(headmodel.hbf_elecs.pproj, length(headmodel.hbf_elecs.pproj), [1 1 1]);
    % plot3(elec_xyz{:}, 'Color', [0.5 0.5 0.5], 'Marker', '.', 'LineStyle', 'none', 'MarkerSize', 17.5)
    % 
    % %and check original locations
    % elec_xyz = mat2cell(headmodel.elec.porig, length(headmodel.elec.pproj), [1 1 1]);
    % %elec_xyz = mat2cell(headmodel.hbf_elecs.porig, length(headmodel.hbf_elecs.pproj), [1 1 1]);
    % plot3(elec_xyz{:}, 'Color', [0.9 0.5 0.5], 'Marker', '.', 'LineStyle', 'none', 'MarkerSize', 10)
    % 
    % % add the cortical mesh
    % patch('faces', headmodel.smesh.e, 'vertices', headmodel.smesh.p, 'facecolor', [0.7 0.6 0.6], 'facealpha', 0.75, 'edgecolor', 'none');
    % 
    % view([-90 0]);
    % axis tight equal off; material dull; lighting gouraud; camlight
    % 
    % clear('elec_xyz')
    % drawnow
    % 
    % hold on
    % 
    % %Create mesh_mask from preselected indices
    % mesh_mask = false(size(headmodel.smesh.p, 1), 1);
    % mesh_mask(all_indices_combined) = true;
    % 
    % %Extract coordinates and orientations using mesh_mask
    % dipoles_xyz = mat2cell(headmodel.smesh.p(mesh_mask, :), sum(mesh_mask), [1 1 1]);
    % dipoles_uvw = mat2cell(headmodel.smesh.nn(mesh_mask, :), sum(mesh_mask), [1 1 1]);
    % 
    % plot3(dipoles_xyz{:}, 'Color', 'red', 'Marker', '.', 'LineStyle', 'none', 'MarkerSize', 20)
    % quiver3(dipoles_xyz{:}, dipoles_uvw{:}, 1, 'Color', 'red', 'LineWidth', 2.5)
    % rotate3d

    % % Sanity check
    % [LnrN]=prepareLFM(headmodel, 1, []); 
    % sensitivity_profile=zeros(size(headmodel.leadfield,1),1);
    % sensitivity_profile(ismember(headmodel.label, 'C3'))=1;
    % PlotDataOnMesh(headmodel.smesh,sensitivity_profile'*LnrN,'colormap', jet ,'colorbar', 0, 'view', [-90 50]);
    
    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ▄▄▀██ ▄▄▄ █▄ ▄███▄▄ ▄▄█▄ ▄██ ▄▀▄ ██ ▄▄▄██ ▄▄▀██ ▄▄▄ ██ ██ ██ ▄▄▀██ ▄▄▄ ██ ▄▄▄██ ▄▄▄ 
    % ██ ▀▀▄██ ███ ██ ██████ ████ ███ █ █ ██ ▄▄▄██ █████ ███ ██ ██ ██ ▀▀▄██▄▄▄▀▀██ ▄▄▄██▄▄▄▀▀
    % ██ ██ ██ ▀▀▀ █▀ ▀█████ ███▀ ▀██ ███ ██ ▀▀▀██ ▀▀▄██ ▀▀▀ ██▄▀▀▄██ ██ ██ ▀▀▀ ██ ▀▀▀██ ▀▀▀ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % ROI TIMECOURSES

    % Calculate dipole timecourses
    dipole_timecourses = nan([size(epochs.trial, [1 2]) numel(DIPOLES_OF_INTEREST)]); % [time x trial x dipole]
    for i = 1:size(epochs.trial, 2)
        dipole_timecourses(:,i,:) = squeeze((epochs.trial(:,i,:))) * weights(DIPOLES_OF_INTEREST,:)';
    end
    % 
    % % Plot dipole timecourses
    % ix = 13; % which of the DIPOLE_OF_INTEREST
    % 
    % figure
    % plot(epochs.time, dipole_timecourses(:,9,ix))
    % title(sprintf('Multiple Trials for Dipole #%i', DIPOLES_OF_INTEREST(ix)))
    % xlabel('time (s)'), ylabel('node normal amplitude (au)')
    % 
    % ix = 9; %which trial
    % 
    % figure
    % plot(epochs.time, squeeze(dipole_timecourses(:,ix,11:20)))
    % title(sprintf('Multiple Dipoles for Trial #%i', ix))
    % xlabel('time (s)'), ylabel('node normal amplitude (au)')
    % xlim([-1 0])
   
    %
    % Initialize a structure to store ROI timecourses
    roi_timecourses = struct(); % times x trials x sources
    
    % Iterate over each field (ROI label) in roi_vertex_indices
    field_names = fieldnames(roi_vertex_indices);
    for i = 1:numel(field_names)
        roi_name = field_names{i};
        
        % Extract the vertex indices for this ROI
        roi_indices = roi_vertex_indices.(roi_name);
        
        % Initialize a matrix to store timecourses for this ROI
        num_trials = size(dipole_timecourses, 2);
        num_samples = size(dipole_timecourses, 1);
        num_dipoles = numel(roi_indices);
        roi_timecourses.(roi_name) = zeros(num_samples, num_trials, num_dipoles); % [time x trial x dipole]
    
        % Loop through ROI indices and extract dipole timecourses
        for j = 1:num_dipoles
            dipole_idx = roi_indices(j);
            % Find the position of dipole_idx in DIPOLES_OF_INTEREST
            dipole_pos_in_DIPOLES = find(DIPOLES_OF_INTEREST == dipole_idx);
            % Extract the corresponding dipole timecourse
            roi_timecourses.(roi_name)(:,:,j) = dipole_timecourses(:,:,dipole_pos_in_DIPOLES);
        end
    end
   
    % Initialize a structure to store averaged timecourses for each ROI
    avg_roi_timecourses = struct();

    % Iterate over each ROI
    field_names = fieldnames(roi_timecourses);

    for i = 1:numel(field_names)
        roi_name = field_names{i};

        % Average timecourses across sources within this ROI
        averaged_timecourses = mean(roi_timecourses.(roi_name), 3); % [1001 x 400]

        % Store the averaged timecourses in the structure
        avg_roi_timecourses.(roi_name) = averaged_timecourses;
    end

    % Convert the structure to a cell array
    avg_roi_timecourses = struct2cell(avg_roi_timecourses);

    % Combine the averaged timecourses for all ROIs into data_matrix
    data_matrix = cat(3, avg_roi_timecourses{:}); % The resulting data_matrix will have dimensions 1001 x 400 x 26
    data_matrix = permute(data_matrix, [3, 1, 2]); % Now it will be 26 x 1001 x 400
    data_matrix1 = data_matrix;

    %       ______   ______  __    __ _______   ______  ________      ________ ________  ______  ________ __    __ _______  ________  ______  
    %  /      \ /      \|  \  |  \       \ /      \|        \    |        \        \/      \|        \  \  |  \       \|        \/      \ 
    % |  ▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓  | ▓▓ ▓▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓    | ▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓\\▓▓▓▓▓▓▓▓ ▓▓  | ▓▓ ▓▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓\
    % | ▓▓___\▓▓ ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓__| ▓▓ ▓▓   \▓▓ ▓▓__        | ▓▓__   | ▓▓__   | ▓▓__| ▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓__| ▓▓ ▓▓__   | ▓▓___\▓▓
    %  \▓▓    \| ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓    ▓▓ ▓▓     | ▓▓  \       | ▓▓  \  | ▓▓  \  | ▓▓    ▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓    ▓▓ ▓▓  \   \▓▓    \ 
    %  _\▓▓▓▓▓▓\ ▓▓  | ▓▓ ▓▓  | ▓▓ ▓▓▓▓▓▓▓\ ▓▓   __| ▓▓▓▓▓       | ▓▓▓▓▓  | ▓▓▓▓▓  | ▓▓▓▓▓▓▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓▓▓▓▓▓\ ▓▓▓▓▓   _\▓▓▓▓▓▓\
    % |  \__| ▓▓ ▓▓__/ ▓▓ ▓▓__/ ▓▓ ▓▓  | ▓▓ ▓▓__/  \ ▓▓_____     | ▓▓     | ▓▓_____| ▓▓  | ▓▓  | ▓▓  | ▓▓__/ ▓▓ ▓▓  | ▓▓ ▓▓_____|  \__| ▓▓
    %  \▓▓    ▓▓\▓▓    ▓▓\▓▓    ▓▓ ▓▓  | ▓▓\▓▓    ▓▓ ▓▓     \    | ▓▓     | ▓▓     \ ▓▓  | ▓▓  | ▓▓   \▓▓    ▓▓ ▓▓  | ▓▓ ▓▓     \\▓▓    ▓▓
    %   \▓▓▓▓▓▓  \▓▓▓▓▓▓  \▓▓▓▓▓▓ \▓▓   \▓▓ \▓▓▓▓▓▓ \▓▓▓▓▓▓▓▓     \▓▓      \▓▓▓▓▓▓▓▓\▓▓   \▓▓   \▓▓    \▓▓▓▓▓▓ \▓▓   \▓▓\▓▓▓▓▓▓▓▓ \▓▓▓▓▓▓ 
                                                                                                                                    
    % IRASA PLOTTING
    % Parameters
    roiNames = fieldnames(roi_vertex_indices); % Get all ROI names
    numROIs = length(roiNames);
    numTimePoints = size(data_matrix, 2); 
    numTrials = size(data_matrix, 3);
    fsample = epochs.fsample;

    % Initialize the structures to store results
    results = struct();

    % Start a parallel pool (if not already started)
    if isempty(gcp('nocreate'))
        parpool; % starts the default parallel pool
    end

    % Parallel loop over ROIs
    parfor roi = 1:numROIs
        roiName = roiNames{roi};

        % Select the data for the current ROI
        roiData = squeeze(data_matrix(roi, :, :));
        numSamples = size(roiData, 1); 
        numTrials = size(roiData, 2);

        % Initialize arrays to store results for each trial
        SNRArray = cell(1, numTrials);
        PowerSpectrumArray = cell(1, numTrials);
        FractalComponentArray = cell(1, numTrials);

        % Create FieldTrip data structure for all trials
        data_signal = [];
        data_signal.label = {roiName};
        data_signal.trial = mat2cell(roiData', ones(1, numTrials), numSamples);
        data_signal.time = repmat({(0:(numSamples-1)) / fsample}, 1, numTrials);
        data_signal.sampleinfo = [(1:numTrials)' * numSamples - numSamples + 1, (1:numTrials)' * numSamples];

        % Initialize configuration for preprocessing
        cfg = [];

        % Detrend and Demean
        cfg.detrend = 'yes'; % Remove linear trends
        cfg.demean = 'yes'; % Remove the mean from each trial

        % Apply preprocessing
        data_signal = ft_preprocessing(cfg, data_signal);

        % Frequency analysis for fractal and original components with keeptrials
        cfg_fractal = [];
        cfg_fractal.foilim = [0.5 100];
        cfg_fractal.pad = 'nextpow2';
        cfg_fractal.method = 'irasa';
        cfg_fractal.output = 'fractal';
        cfg_fractal.keeptrials = 'yes';  % Keep trials
        fractal = ft_freqanalysis(cfg_fractal, data_signal);

        cfg_original = cfg_fractal;
        cfg_original.output = 'original';
        original = ft_freqanalysis(cfg_original, data_signal);

        % Calculate the number of frequency points
        numFreqPoints = size(fractal.powspctrm, 3);  % Assuming the third dimension is frequency

        % Initialize arrays to store metrics across trials
        SNRArray = zeros(numFreqPoints, numTrials);
        PowerSpectrumArray = zeros(numFreqPoints, numTrials);
        FractalComponentArray = zeros(numFreqPoints, numTrials);

        % Calculate SNR, Power Spectrum, Fractal Component for all trials
        cfg               = [];
        cfg.parameter     = 'powspctrm';
        cfg.operation     = 'x2-x1'; 
        cfg.keeptrials = 'yes'; 
        oscillatory_w = ft_math(cfg, fractal, original);

        % % Remove the singleton dimension and average across trials
        % avg_oscillatory = mean(squeeze(oscillatory_w.powspctrm), 1);
        % avg_original = mean(squeeze(original.powspctrm), 1);
        % avg_fractal = mean(squeeze(fractal.powspctrm), 1);
        % 
        % % Log-log plot for the averaged metrics across trials
        % figure();
        % hold on;
        % plot(log(oscillatory_w.freq), log(avg_oscillatory), 'b', 'LineWidth', 2);
        % plot(log(original.freq), log(avg_original), 'r', 'LineWidth', 2);
        % plot(log(fractal.freq), log(avg_fractal), 'g', 'LineWidth', 2);
        % xlabel('log-freq');
        % ylabel('log-power');
        % legend({'oscillatory', 'original', 'fractal'}, 'Location', 'southwest');
        % 
        % % Convert power spectra to decibels for each trial and frequency point
        % % Add a small constant eps to ensure positivity
        % SNRArray = 10 * log10(squeeze(oscillatory_w.powspctrm(:, 1, :)) + eps);
        % PowerSpectrumArray = 10 * log10(squeeze(original.powspctrm(:, 1, :)) + eps);
        % FractalComponentArray = 10 * log10(squeeze(fractal.powspctrm(:, 1, :)) + eps);

        % Store the results for the ROI
        results(roi).SNR = SNRArray;
        results(roi).PowerSpectrum = PowerSpectrumArray;
        results(roi).FractalComponent = FractalComponentArray;
        
        % Average across trials for each frequency point
        avgSNR = mean(SNRArray, 1);
        avgPowerSpectrum = mean(PowerSpectrumArray, 1);
        avgFractalComponent = mean(FractalComponentArray, 1);

        % % Plotting for the current roi (linear scale)
        % figure;
        % yyaxis left; % Left y-axis for SNR
        % plot(oscillatory_w.freq, avgSNR, 'b', 'LineWidth', 2);
        % ylabel('SNR (dB)', 'Color', 'b');
        % grid on;
        % 
        % yyaxis right; % Right y-axis for Power Spectrum and Fractal Component
        % plot(original.freq, avgPowerSpectrum, 'r', 'LineWidth', 2);
        % hold on;
        % plot(oscillatory_w.freq, avgFractalComponent, 'g', 'LineWidth', 2);
        % xlabel('Frequency (Hz)');
        % ylabel('Power (dB)', 'Color', 'r');
        % title(sprintf('ROI %s - SNR, Power Spectrum, and Fractal Component', roiName));
        % legend({'SNR', 'Power Spectrum', 'Fractal Component'}, 'Location', 'NorthEast');
        % 
        % % Adjust y-axis label positions
        % ax = gca;
        % ax.YAxis(1).Label.Position(1) = ax.YAxis(1).Label.Position(1) - 5;
        % ax.YAxis(2).Label.Position(1) = ax.YAxis(2).Label.Position(1) + 5;
     end
    
    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ▄▄▄ ██ ▄▄▄ ██ ██ ██ ▄▄▀██ ▄▄▀██ ▄▄▄████ ▄▄▄██ ▄▄▄█ ▄▄▀█▄▄ ▄▄██ ██ ██ ▄▄▀██ ▄▄▄██ ▄▄▄ ███▄ ▄██ ▄▄▀█ ▄▄▀██ ▄▄▄ █ ▄▄▀
    % ██▄▄▄▀▀██ ███ ██ ██ ██ ▀▀▄██ █████ ▄▄▄████ ▄▄███ ▄▄▄█ ▀▀ ███ ████ ██ ██ ▀▀▄██ ▄▄▄██▄▄▄▀▀████ ███ ▀▀▄█ ▀▀ ██▄▄▄▀▀█ ▀▀ 
    % ██ ▀▀▀ ██ ▀▀▀ ██▄▀▀▄██ ██ ██ ▀▀▄██ ▀▀▀████ █████ ▀▀▀█ ██ ███ ████▄▀▀▄██ ██ ██ ▀▀▀██ ▀▀▀ ███▀ ▀██ ██ █ ██ ██ ▀▀▀ █ ██ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

    % SOURCE FEATURES IRASA
    freq_bands = {
        'delta', 0.5, 4.0;
        'theta', 4.0, 8.0;
        'alpha', 8.0, 12.0;
        'low_beta', 13.0, 18.0;
        'high_beta', 19.0, 35.0;
        'low_gamma', 36.0, 58.0;
        'high_gamma', 58.0, 100.0
    };

    % Define the base ROI labels with specific Brodmann area distinctions where applicable
    base_roi_labels = {
        'M1',         % precentral gyrus/ BA 4
        'S1_BA1',     % postcentral gyrus/ BA 1
        'S1_BA2',     % postcentral gyrus/ BA 2
        'S1_BA3a',    % postcentral gyrus/ BA 3a
        'SMA',        % medial surface of the frontal lobe
        'PMC',        % premotor cortex - anterior to M1
        'DLPFC_BA9',  % dorsolateral prefrontal cortex: BA 9
        'DLPFC_BA46', % dorsolateral prefrontal cortex: BA 46
        'PPC_7PL',    % posterior parietal cortex
        'PPC_7PC',    % posterior parietal cortex
        'PPC_7Am',    % posterior parietal cortex
        'FEF_8C',     % frontal eye field
        'FEF_8BM'     % frontal eye field
    };
    
    % Parameters
    roiNames = fieldnames(roi_vertex_indices); % Get all ROI names
    numROIs = length(roiNames);
    numTimePoints = size(data_matrix, 2); 
    numTrials = size(data_matrix, 3); 
    fsample = epochs.fsample;

    % Initialize the structures to store results
    results = struct();

    roi_labels = cell(1, length(base_roi_labels) * 2);
    for i = 1:length(base_roi_labels)
        roi_labels{(i - 1) * 2 + 1} = ['left_' base_roi_labels{i}];
        roi_labels{i * 2} = ['right_' base_roi_labels{i}];
    end
    
    % Initialize the table for the entire dataset
    extractedFeaturesTable = cell2table(cell(0, 13), 'VariableNames', {'ROI', 'Trial_index', 'Condition', 'Freq_band', 'Peak_Freq', 'Peak_SNR_dB', 'Mean_SNR', 'AUC_SNR', 'Abs_Power', 'Rel_Power', 'Fractal_Activity', 'Fractal_Slope', 'Fractal_Offset'});
   
    % Pre-allocate a cell array for temporary tables
    tempTables = cell(1, numROIs);
    
    % Ensure parallel pool is active
    if isempty(gcp('nocreate'))
        parpool;
    end
 
    parfor roiIdx = 1:numROIs
        % Select the data for the current ROI
        roiName = roiNames{roiIdx};
        roiData = squeeze(data_matrix(roiIdx, :, :));
        numSamples = size(roiData, 1); 
        numTrials = size(roiData, 2);
    
        % Initialize arrays to store results for each trial
        SNRArray = cell(1, numTrials);
        PowerSpectrumArray = cell(1, numTrials);
        FractalComponentArray = cell(1, numTrials);

        % Create FieldTrip data structure for all trials
        data_signal = [];
        data_signal.label = {roiName};
        data_signal.trial = mat2cell(roiData', ones(1, numTrials), numSamples);
        data_signal.time = repmat({(0:(numSamples-1)) / fsample}, 1, numTrials);
        data_signal.sampleinfo = [(1:numTrials)' * numSamples - numSamples + 1, (1:numTrials)' * numSamples];

        % Initialize configuration for preprocessing
        cfg = [];
        
        % Detrend and Demean
        cfg.detrend = 'yes'; % Remove linear trends
        cfg.demean = 'yes'; % Remove the mean from each trial
        
        % Apply preprocessing
        data_signal = ft_preprocessing(cfg, data_signal);

        % Frequency analysis for fractal and original components with keeptrials
        cfg_fractal = [];
        cfg_fractal.foilim = [0.5 100];
        cfg_fractal.pad = 'nextpow2';
        cfg_fractal.method = 'irasa';
        cfg_fractal.output = 'fractal';
        cfg_fractal.keeptrials = 'yes';  % Keep trials
        fractal = ft_freqanalysis(cfg_fractal, data_signal);
    
        cfg_original = cfg_fractal;
        cfg_original.output = 'original';
        original = ft_freqanalysis(cfg_original, data_signal);
    
        % Calculate the number of frequency points
        numFreqPoints = size(fractal.powspctrm, 3);  % Assuming the third dimension is frequency
    
        % Initialize arrays to store metrics across trials
        SNRArray = zeros(numFreqPoints, numTrials);
        PowerSpectrumArray = zeros(numFreqPoints, numTrials);
        FractalComponentArray = zeros(numFreqPoints, numTrials);
    
        % Calculate SNR, Power Spectrum, Fractal Component for all trials
        cfg               = [];
        cfg.parameter     = 'powspctrm';
        cfg.operation     = 'x2-x1'; 
        cfg.keeptrials = 'yes'; 
        oscillatory_w = ft_math(cfg, fractal, original);
        SNRArray = 10 * log10(squeeze(oscillatory_w.powspctrm(:, 1, :)) + eps);
        PowerSpectrumArray = 10 * log10(squeeze(original.powspctrm(:, 1, :)) + eps);
        FractalComponentArray = 10 * log10(squeeze(fractal.powspctrm(:, 1, :)) + eps);

        % % Store the results for the ROI
        % results(roiIdx).SNR = SNRArray;
        % results(roiIdx).PowerSpectrum = PowerSpectrumArray;
        % results(roiIdx).FractalComponent = FractalComponentArray;

        %  % Access the SNR, Power Spectrum, and Fractal data for the current ROI
        % SNRArray = results(roiIdx).SNR;
        % PowerSpectrumArray = results(roiIdx).PowerSpectrum;
        % FractalComponentArray = results(roiIdx).FractalComponent;

        % Temporary table for each ROI
        tempFeaturesTable = cell2table(cell(0, 13), 'VariableNames', {'ROI', 'Trial_index', 'Condition', 'Freq_band', 'Peak_Freq', 'Peak_SNR_dB', 'Mean_SNR', 'AUC_SNR', 'Abs_Power', 'Rel_Power', 'Fractal_Activity', 'Fractal_Slope', 'Fractal_Offset'});
    
        for trial = 1:size(roiData, 2)
            % Determine condition
            condition = 'low';
            if trial <= numTrials/2
                condition = 'high';
            end

            for band = 1:size(freq_bands, 1)
                bandName = freq_bands{band, 1};
                bandStart = freq_bands{band, 2};
                bandEnd = freq_bands{band, 3};
            
                % Extract frequency indices for the current band
                freqIndices = find(oscillatory_w.freq >= bandStart & oscillatory_w.freq <= bandEnd); 

                % Extract SNR, Power Spectrum, and Fractal data for the current band and trial
                snrData = real(SNRArray(trial, freqIndices)); % Ensure snrData is real
                psData = PowerSpectrumArray(trial, freqIndices);
                fcData = FractalComponentArray(trial, freqIndices);

                % Check if there are enough data points for findpeaks
                if length(freqIndices) >= 3
                    % Find peaks in the SNR data
                    [peakValues, peakIndices] = findpeaks(snrData);
            
                    % Determine the peak frequency and peak SNR
                    if ~isempty(peakValues)
                        [peakSNR_dB, idxOfMaxPeak] = max(peakValues);
                        peakFreqIndex = peakIndices(idxOfMaxPeak);
                        peakFreq = real(oscillatory_w.freq(freqIndices(peakFreqIndex)));
                    else
                        % Use max-peak approach if no peaks are found
                        peakSNR_dB = max(snrData);
                        peakFreq = real(oscillatory_w.freq(freqIndices(snrData == peakSNR_dB)));
                    end
                else
                    % Use max-peak approach if not enough data points
                    peakSNR_dB = max(snrData);
                    peakFreq = real(oscillatory_w.freq(freqIndices(snrData == peakSNR_dB)));
                end
                
                % Calculate other SNR metrics
                meanSNR = mean(snrData); % Mean SNR
                aucSNR = trapz(oscillatory_w.freq(freqIndices), snrData); % AUC for SNR
                aucSNR = real(aucSNR); % Ensure aucSNR is real
                
                % Calculate absolute and relative power for the Power Spectrum
                absPower = sum(psData);
                totalPower = sum(PowerSpectrumArray(trial, :)); % Total power for the trial
                relPower = sum(psData) / sum(PowerSpectrumArray(trial, :)); % Relative power for the band
                
                % Fractal Activity
                fractalActivity = trapz(oscillatory_w.freq(freqIndices), fcData); % Fractal Activity

                % Perform log-log transformation
                logFreq = log10(oscillatory_w.freq(freqIndices));
                logFCData = fcData;
            
                % Perform linear fit
                linearFit = polyfit(logFreq, logFCData, 1);  % First-degree polynomial fit
                fractalSlope = linearFit(1);  % Slope of the line
                fractalOffset = linearFit(2);  % Y-intercept of the line

                % Append results to the table
                newRow = {roiName, trial, condition, bandName, peakFreq, peakSNR_dB, meanSNR, aucSNR, absPower, relPower, fractalActivity, fractalSlope, fractalOffset};

                tempFeaturesTable = [tempFeaturesTable; newRow];
                % Store the temporary table in the pre-allocated cell array
                tempTables{roiIdx} = tempFeaturesTable;
            end
        end
    end
    
    % Combine the temporary tables into the main table after the loop
    extractedFeaturesTable = vertcat(tempTables{:});
    
    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = extractedFeaturesTable.Properties.VariableNames;  % Extract column names
    extractedFeaturesTable = [headers; table2cell(extractedFeaturesTable)];  % Combine headers and data

    % Save the extractedFeaturesTable as a .mat file
    % Define output filename
    output_filename = sprintf('Source_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    
    % Save the extractedFeaturesTable as a .mat file
    save(output_path, 'extractedFeaturesTable')

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ▄▄▄ ██ ▄▄▄ ██ ██ ██ ▄▄▀██ ▄▄▀██ ▄▄▄████ ▄▄▀██ ▄▄▄ ██ ▀██ ██ ▀██ ██ ▄▄▄██ ▄▄▀█▄▄ ▄▄█▄ ▄██ ███ █▄ ▄█▄▄ ▄▄██ ███ 
    % ██▄▄▄▀▀██ ███ ██ ██ ██ ▀▀▄██ █████ ▄▄▄████ █████ ███ ██ █ █ ██ █ █ ██ ▄▄▄██ ██████ ████ ████ █ ███ ████ ████▄▀▀▀▄
    % ██ ▀▀▀ ██ ▀▀▀ ██▄▀▀▄██ ██ ██ ▀▀▄██ ▀▀▀████ ▀▀▄██ ▀▀▀ ██ ██▄ ██ ██▄ ██ ▀▀▀██ ▀▀▄███ ███▀ ▀███▄▀▄██▀ ▀███ ██████ ██
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % SOURCE CONNECTIVITY

    numROIs = length(avg_roi_timecourses);
    numTrials = size(avg_roi_timecourses{1}, 2);
    numTimePoints = size(avg_roi_timecourses{1}, 1);
    indxcmb = nchoosek(1:numROIs, 2); % All pairwise combinations of ROIs
    fsample = epochs.fsample; % sampling frequency in Hz
    ARord = 30; % AR model order for padding

    % Preallocate arrays for connectivity results
    iPLV_results = cell(length(freq_bands), 1);
    PLI_results = cell(length(freq_bands), 1);
    wPLI_results = cell(length(freq_bands), 1);
    Coh_results = cell(length(freq_bands), 1);
    imCoh_results = cell(length(freq_bands), 1);
    lagCoh_results = cell(length(freq_bands), 1);
    oPEC_results = cell(length(freq_bands), 1); 
    
    for bandIdx = 1:size(freq_bands, 1)
        % Band-specific settings
        bandName = freq_bands{bandIdx, 1};
        bpfreq = [freq_bands{bandIdx, 2}, freq_bands{bandIdx, 3}];
        n = round((0.15*fsample)/2)*2; % filter order
        b = fir1(n, bpfreq/(fsample/2)); % filter coefficients
        padsize = round(2*fsample/mean(bpfreq)); % padding size
    
        % Initialize arrays for this band
        iPLV_band = zeros(size(indxcmb, 1), numTrials);
        PLI_band = zeros(size(indxcmb, 1), numTrials);
        wPLI_band = zeros(size(indxcmb, 1), numTrials);
        Coh_band = zeros(size(indxcmb, 1), numTrials);
        imCoh_band = zeros(size(indxcmb, 1), numTrials);
        lagCoh_band = zeros(size(indxcmb, 1), numTrials);
        oPEC_band = zeros(size(indxcmb, 1), numTrials);
    
        for trial = 1:numTrials
            % Reshape data for this trial
            trialData = zeros(numTimePoints, numROIs);
            for roi = 1:numROIs
                trialData(:, roi) = avg_roi_timecourses{roi}(:, trial);
            end
    
            % Preprocess the data
            preprocessedData = preprocessData(trialData, b, ARord, padsize);
    
            % Calculate connectivity measures
            [iPLV, PLI, wPLI, Coh, imCoh, lagCoh, oPEC] = calculateConnectivity(preprocessedData, padsize, indxcmb);
    
            % Store results for this trial and band
            iPLV_band(:, trial) = iPLV;
            PLI_band(:, trial) = PLI;
            wPLI_band(:, trial) = wPLI;
            Coh_band(:, trial) = Coh;
            imCoh_band(:, trial) = imCoh;
            lagCoh_band(:, trial) = lagCoh;
            oPEC_band(:, trial) = oPEC;
        end
    
        % Store results for this frequency band
        iPLV_results{bandIdx} = iPLV_band;
        PLI_results{bandIdx} = PLI_band;
        wPLI_results{bandIdx} = wPLI_band;
        Coh_results{bandIdx} = Coh_band;
        imCoh_results{bandIdx} = imCoh_band;
        lagCoh_results{bandIdx} = lagCoh_band;
        oPEC_results{bandIdx} = oPEC_band;
    end

    % Create source connectivity dataframes
    numROIs = length(avg_roi_timecourses);
    indxcmb = nchoosek(1:numROIs, 2); % All pairwise combinations of ROIs
    roiCombLabels = arrayfun(@(x) sprintf('%s-%s', roi_labels{indxcmb(x, 1)}, roi_labels{indxcmb(x, 2)}), 1:size(indxcmb, 1), 'UniformOutput', false);
    numTrials = size(avg_roi_timecourses{1}, 2);
    freq_bands = {'delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma'};
    conditions = [repmat({'high'}, numTrials/2, 1); repmat({'low'}, numTrials/2, 1)];
    
    % Preallocate cell array for parallel results
    parallelResults = cell(length(freq_bands), 1);
    
    % Loop through each frequency band in parallel
    parfor bandIdx = 1:length(freq_bands)
        bandTable = table();
        for trialIdx = 1:numTrials
            for roiPairIdx = 1:length(roiCombLabels)
                newRow = {trialIdx, conditions{trialIdx}, freq_bands{bandIdx}, roiCombLabels{roiPairIdx}, ...
                          iPLV_results{bandIdx}(roiPairIdx, trialIdx), PLI_results{bandIdx}(roiPairIdx, trialIdx), ...
                          wPLI_results{bandIdx}(roiPairIdx, trialIdx), Coh_results{bandIdx}(roiPairIdx, trialIdx), ...
                          imCoh_results{bandIdx}(roiPairIdx, trialIdx), lagCoh_results{bandIdx}(roiPairIdx, trialIdx), ...
                          oPEC_results{bandIdx}(roiPairIdx, trialIdx)};
                bandTable = [bandTable; newRow];
            end
        end
        parallelResults{bandIdx} = bandTable;
    end
    
    % Concatenate all tables
    sourceconnectivity_df = vertcat(parallelResults{:});
    sourceconnectivity_df.Properties.VariableNames = {'Trial_index', 'Condition', 'Freq_band', 'ROI_comb', ...
                                                      'iPLV', 'PLI', 'wPLI', 'Coh', 'imCoh', 'lagCoh', 'oPEC'};

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = sourceconnectivity_df.Properties.VariableNames;  % Extract column names
    sourceconnectivity_df = [headers; table2cell(sourceconnectivity_df)];  % Combine headers and data


    % Save the sourceconnectivity_df as a .mat file
    % Define output filename
    output_filename = sprintf('SourceCon_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    
    % Save the extractedFeaturesTable as a .mat file
    save(output_path, 'sourceconnectivity_df')

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ▄▄▄ ██ ▄▄▄ ██ ██ ██ ▄▄▀██ ▄▄▀██ ▄▄▄████ ▄▄ █ ▄▄▀██ ▄▄▀
    % ██▄▄▄▀▀██ ███ ██ ██ ██ ▀▀▄██ █████ ▄▄▄████ ▀▀ █ ▀▀ ██ ███
    % ██ ▀▀▀ ██ ▀▀▀ ██▄▀▀▄██ ██ ██ ▀▀▄██ ▀▀▀████ ████ ██ ██ ▀▀▄
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % Define frequency ranges
    freq_bands = {
        'delta', 0.5, 4.0;
        'theta', 4.0, 8.0;
        'alpha', 8.0, 12.0;
        'low_beta', 13.0, 18.0;
        'high_beta', 19.0, 35.0;
        'low_gamma', 36.0, 58.0;
        'high_gamma', 58.0, 100.0
    };
    
    % Initialize PAC results storage
    PAC_results = cell(26, 1); % One cell for each ROI
    
    % Sampling frequency
    Fs = 1000; % Assuming a sampling frequency of 1000 Hz
    
    % Define the combinations for PAC calculation
    PAC_combinations = {
        'theta', 'low_gamma';
        'theta', 'high_gamma';
        'alpha', 'low_gamma';
        'alpha', 'high_gamma';
        'low_beta', 'low_gamma';
        'low_beta', 'high_gamma';
        'high_beta', 'low_gamma';
        'high_beta', 'high_gamma'
    };

    
    % Start a parallel pool if not already started
    if isempty(gcp('nocreate'))
        parpool;
    end
    
    % Loop over ROIs
    for roiIdx = 1:numROIs
        roiData = avg_roi_timecourses{roiIdx}; % Extract data for current ROI
        numTrials = size(roiData, 2);
        numCombinations = size(PAC_combinations, 1);
        roiPAC = zeros(numTrials, numCombinations); % Initialize PAC matrix for current ROI
    
        % Loop over PAC combinations
        for pacIdx = 1:numCombinations
            phase_band_name = PAC_combinations{pacIdx, 1};
            amp_band_name = PAC_combinations{pacIdx, 2};
            
            % Find the row in freq_bands that matches the phase_band_name and amp_band_name
            phase_band_row = freq_bands(strcmp(freq_bands(:,1), phase_band_name), :);
            amp_band_row = freq_bands(strcmp(freq_bands(:,1), amp_band_name), :);
        
            % Extract the frequency range for phase and amplitude bands
            phase_band_range = [phase_band_row{1, 2}, phase_band_row{1, 3}];
            amp_band_range = [amp_band_row{1, 2}, amp_band_row{1, 3}];
        
            % Extract start and end frequencies for both bands
            phase_freq_low = phase_band_range(1);
            phase_freq_high = phase_band_range(2);
            amp_freq_low = amp_band_range(1);
            amp_freq_high = amp_band_range(2);
     
            % Parallel loop over trials
            parfor trialIdx = 1:numTrials
                x = roiData(:, trialIdx); % Extract data for current trial
                % Calculate PAC
                [tf_canolty] = tfMVL(x, [amp_freq_low, amp_freq_high], [phase_freq_low, phase_freq_high], Fs);
                roiPAC(trialIdx, pacIdx) = tf_canolty;
            end
        end
        PAC_results{roiIdx} = roiPAC; % Store results for current ROI
    end

    
    % Preallocate cell array to store parallel results
    parallelResults = cell(length(roi_labels), 1);
    
    % Parallel loop over ROIs
    parfor roiIdx = 1:length(roi_labels)
        roiName = roi_labels{roiIdx};
        pacData = PAC_results{roiIdx}; % PAC data for the current ROI
        roiTable = table(); % Initialize a table for each ROI
    
        % Loop over trials
        for trialIdx = 1:numTrials
            % Determine condition based on trial index
            condition = 'high';
            if trialIdx > numTrials/2
                condition = 'low';
            end
    
            % Loop over frequency-pair combinations
            for freqPairIdx = 1:size(PAC_combinations, 1)
                freqPairName = sprintf('%s-%s', PAC_combinations{freqPairIdx, 1}, PAC_combinations{freqPairIdx, 2});
                pacValue = pacData(trialIdx, freqPairIdx);
    
                % Create a new row for the table
                newRow = {trialIdx, condition, roiName, freqPairName, pacValue};
                roiTable = [roiTable; newRow];
            end
        end
    
        % Store the table for the current ROI in the cell array
        parallelResults{roiIdx} = roiTable;
    end
    
    % Combine all ROI tables into one
    pac_df = vertcat(parallelResults{:});
    
    % Set the column names for the DataFrame
    pac_df.Properties.VariableNames = {'Trial_index', 'Condition', 'ROI', 'Freq_combinations', 'PAC_value'};

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = pac_df.Properties.VariableNames;  % Extract column names
    pac_df = [headers; table2cell(pac_df)];  % Combine headers and data

    % Save the pac_df as a .mat file
    % Define output filename
    output_filename = sprintf('pac_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    
    % Save the pac_df as a .mat file
    save(output_path, 'pac_df')

    %       ______  ________ __    __  ______   ______  _______       ________ ________  ______  ________ __    __ _______  ________  ______  
    %  /      \|        \  \  |  \/      \ /      \|       \     |        \        \/      \|        \  \  |  \       \|        \/      \ 
    % |  ▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓ ▓▓\ | ▓▓  ▓▓▓▓▓▓\  ▓▓▓▓▓▓\ ▓▓▓▓▓▓▓\    | ▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓\\▓▓▓▓▓▓▓▓ ▓▓  | ▓▓ ▓▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓\
    % | ▓▓___\▓▓ ▓▓__   | ▓▓▓\| ▓▓ ▓▓___\▓▓ ▓▓  | ▓▓ ▓▓__| ▓▓    | ▓▓__   | ▓▓__   | ▓▓__| ▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓__| ▓▓ ▓▓__   | ▓▓___\▓▓
    %  \▓▓    \| ▓▓  \  | ▓▓▓▓\ ▓▓\▓▓    \| ▓▓  | ▓▓ ▓▓    ▓▓    | ▓▓  \  | ▓▓  \  | ▓▓    ▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓    ▓▓ ▓▓  \   \▓▓    \ 
    %  _\▓▓▓▓▓▓\ ▓▓▓▓▓  | ▓▓\▓▓ ▓▓_\▓▓▓▓▓▓\ ▓▓  | ▓▓ ▓▓▓▓▓▓▓\    | ▓▓▓▓▓  | ▓▓▓▓▓  | ▓▓▓▓▓▓▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓▓▓▓▓▓\ ▓▓▓▓▓   _\▓▓▓▓▓▓\
    % |  \__| ▓▓ ▓▓_____| ▓▓ \▓▓▓▓  \__| ▓▓ ▓▓__/ ▓▓ ▓▓  | ▓▓    | ▓▓     | ▓▓_____| ▓▓  | ▓▓  | ▓▓  | ▓▓__/ ▓▓ ▓▓  | ▓▓ ▓▓_____|  \__| ▓▓
    %  \▓▓    ▓▓ ▓▓     \ ▓▓  \▓▓▓\▓▓    ▓▓\▓▓    ▓▓ ▓▓  | ▓▓    | ▓▓     | ▓▓     \ ▓▓  | ▓▓  | ▓▓   \▓▓    ▓▓ ▓▓  | ▓▓ ▓▓     \\▓▓    ▓▓
    %   \▓▓▓▓▓▓ \▓▓▓▓▓▓▓▓\▓▓   \▓▓ \▓▓▓▓▓▓  \▓▓▓▓▓▓ \▓▓   \▓▓     \▓▓      \▓▓▓▓▓▓▓▓\▓▓   \▓▓   \▓▓    \▓▓▓▓▓▓ \▓▓   \▓▓\▓▓▓▓▓▓▓▓ \▓▓▓▓▓▓ 
    % 
    fsample = epochs.fsample; % sampling frequency in Hz                                                                                                                                   
    n_trials = numTrials/2;
    n_samples = fsample +1;

    % Initialize the data structures for high and low conditions ->>
    data_high = [];
    data_low = [];
    
    % Copy the common data fields
    data_high.label = epochs.label; % channel labels
    data_high.fsample = epochs.fsample; % sampling frequency

    data_low.label = epochs.label; % channel labels
    data_low.fsample = epochs.fsample; % sampling frequency
   
    % Initialize trial and time fields
    data_high.trial = cell(1, n_trials);
    data_high.time = cell(1, n_trials);
    data_low.trial = cell(1, n_trials);
    data_low.time = cell(1, n_trials);
 
    % Extract trials and corresponding time data for high and low conditions
    for trialIdx = 1:n_trials % Assuming the first n_trials are 'high'
        % Extract the trial data and transpose to 'channels x times'
        data_high.trial{trialIdx} = permute(squeeze(epochs.trial(:, trialIdx, :)), [2, 1]);
        data_high.time{trialIdx} = epochs.time; 
    end
    
    for trialIdx = n_trials + 1:numTrials % Assuming the next n_trials trials are 'low'
        % Extract the trial data and transpose to 'channels x times'
        data_low.trial{trialIdx - n_trials} = permute(squeeze(epochs.trial(:, trialIdx, :)), [2, 1]);
        data_low.time{trialIdx - n_trials} = epochs.time; 
    end

    % Add sampleinfo 
    data_high.sampleinfo = [(1:n_trials)' * n_samples - n_samples + 1, (1:n_trials)' * n_samples];
    data_low.sampleinfo = [(1:n_trials)' * n_samples - n_samples + 1, (1:n_trials)' * n_samples];


    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ▄▄▄ ██ ▄▄▄██ ▀██ ██ ▄▄▄ ██ ▄▄▄ ██ ▄▄▀████ ▄▄ ██ ▄▄▄ ██ ▄▄▀
    % ██▄▄▄▀▀██ ▄▄▄██ █ █ ██▄▄▄▀▀██ ███ ██ ▀▀▄████ ▀▀ ██▄▄▄▀▀██ ██ 
    % ██ ▀▀▀ ██ ▀▀▀██ ██▄ ██ ▀▀▀ ██ ▀▀▀ ██ ██ ████ █████ ▀▀▀ ██ ▀▀ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % SENSOR PSD
    
    % Categorize channels based on Brain regions and Hemisphere
    brain_region_map = categorize_channels(data_high.label);
    brain_side_map = categorize_channels_rl(data_high.label);
    
    % Initialize cell arrays for brain regions and sides
    brain_regions = cell(1, length(data_high.label));
    brain_sides = cell(1, length(data_high.label));
  
    % Populate brain_regions and brain_sides based on the maps
    for i = 1:length(data_high.label)
        ch = data_high.label{i};
        brain_regions{i} = brain_region_map(ch);
        brain_sides{i} = brain_side_map(ch);
    end

    % Initialize results structures
    results_high = struct();
    results_low = struct();
    
    % Loop through each frequency band
    for i = 1:size(freq_bands, 1)
        band = freq_bands{i, 1};
        low_freq = freq_bands{i, 2};
        high_freq = freq_bands{i, 3};
    
        % Set taper based on frequency band
        if strcmp(band, 'low_gamma') || strcmp(band, 'high_gamma')
            taper = 'dpss';
            tapsmofrq = 2; %for gamma
            step_size = 1; % Larger step for higher frequency bands
        else
            taper = 'hanning';
            tapsmofrq = []; % Not needed for hanning taper
            step_size = 1; % Smaller step for lower frequency bands
        end
    
        % Configuration for ft_freqanalysis
        cfg = [];
        cfg.method = 'mtmfft';
        cfg.output = 'pow';
        cfg.keeptrials = 'yes';
        cfg.channel = data_high.label; 
        cfg.pad = 'nextpow2';
        % Determine frequencies of interest for this band
        cfg.foi = low_freq:step_size:high_freq;
        cfg.taper = taper;
        if ~isempty(tapsmofrq)
            cfg.tapsmofrq = tapsmofrq;
        end
    
        % Perform frequency analysis for high and low conditions
        temp_freq_high = ft_freqanalysis(cfg, data_high);
        temp_freq_low = ft_freqanalysis(cfg, data_low);
    
        % Store results
        results_high.(band) = temp_freq_high;
        results_low.(band) = temp_freq_low;
    end

    % Initialize results table
    n_trials = length(data_high.trial);
    n_channels = length(data_high.label);
    n_freq_bands = size(freq_bands, 1);
    n_conditions = 2; % 'Low' and 'High'
    n_quant_status = 2; % 'Absolute' and 'Relative'
    n_rows = n_trials * n_channels * n_freq_bands * n_conditions * n_quant_status;

    % Precompute conditions array
    conditions_arr = ["high", "low"];
    
    % Initialize the results table
    column_names = {'Trial_index', 'Condition', 'Channel', 'Brain_region', 'Brain_side', 'Freq_band', 'Quant_status', 'PSD'};
    results_table = table('Size', [n_rows, numel(column_names)], 'VariableTypes', {'double', 'categorical', 'string', 'categorical', 'categorical', 'categorical', 'categorical', 'double'}, 'VariableNames', column_names);
    
    % Parallel loop
    results_parts = cell(1, n_trials);
    parfor trialIdx = 1:n_trials
        temp_results = table('Size', [n_channels * n_freq_bands * n_conditions * n_quant_status, numel(column_names)], 'VariableTypes', {'double', 'categorical', 'string', 'categorical', 'categorical', 'categorical', 'categorical', 'double'}, 'VariableNames', column_names);
        temp_row = 1;
    
        for conditionIdx = 1:n_conditions
            condition = conditions_arr(conditionIdx);  % Use the precomputed array
            trial_index_adjusted = trialIdx + (conditionIdx == 1) * 200; % Add 200 if 'high' condition
    
            for channelIdx = 1:n_channels
                channel_label = data_high.label(channelIdx);
                brain_region = brain_regions{channelIdx};
                brain_side = brain_sides{channelIdx};

                for bandIdx = 1:n_freq_bands
                    band = freq_bands{bandIdx, 1};
                    low_freq = freq_bands{bandIdx, 2};
                    high_freq = freq_bands{bandIdx, 3};
    
                    % Access the freq analysis results for the current band and condition
                    if strcmp(condition, 'high')
                        freq_analysis = results_high.(band);
                    else
                        freq_analysis = results_low.(band);
                    end

                    % Extract the PSD data and frequency values
                    band_data = freq_analysis.powspctrm(trialIdx, channelIdx, :);
                    freq_values = freq_analysis.freq;
    
                    % Find indices of the frequency range
                    freq_indices = find(freq_values >= low_freq & freq_values <= high_freq);
                    
                    % Extract the relevant PSD data for bandpower calculation
                    relevant_psd_data = squeeze(band_data(:, freq_indices));
                    freq_range = freq_values(freq_indices);

                    % Calculate integrated power using summing approach
                    integrated_power = sum(relevant_psd_data);
                    
                    % Convert to dB
                    abs_power_db = 10 * log10(integrated_power);
                    
                    % Calculate total power across all bands for relative power
                    total_power = 0;
                    for b = 1:n_freq_bands
                        band_name = freq_bands{b, 1};
                        if strcmp(condition, 'high')
                            total_band_analysis = results_high.(band_name);
                        else
                            total_band_analysis = results_low.(band_name);
                        end
                        total_band_data = total_band_analysis.powspctrm(trialIdx, channelIdx, :);
                        
                        % Use summing approach for total power calculation
                        total_power = total_power + sum(total_band_data(:));
                    end
                    
                    % Calculate relative power
                    rel_power = integrated_power / total_power;
     
    
                    % Fill the temporary table
                    temp_results(temp_row, :) = {trial_index_adjusted, condition, channel_label, brain_region, brain_side, band, 'Absolute', abs_power_db};
                    temp_row = temp_row + 1;
                    temp_results(temp_row, :) = {trial_index_adjusted, condition, channel_label, brain_region, brain_side, band, 'Relative', rel_power};
                    temp_row = temp_row + 1;
                end
            end
        end
        results_parts{trialIdx} = temp_results;
    end
        
    % Concatenate results from all parts
    results_table = vertcat(results_parts{:});
    
    % Rename the table to power_df
    power_df = results_table;
    
    % Convert categorical columns to cell arrays of character vectors
    power_df.Condition = cellstr(power_df.Condition);
    power_df.Brain_region = cellstr(power_df.Brain_region);
    power_df.Brain_side = cellstr(power_df.Brain_side);
    power_df.Freq_band = cellstr(power_df.Freq_band);
    power_df.Quant_status = cellstr(power_df.Quant_status);
    power_df.Channel = cellstr(power_df.Channel);

    % Sort the final table by 'Trial_index'
    power_df = sortrows(power_df, 'Trial_index');

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = power_df.Properties.VariableNames;  % Extract column names
    power_df = [headers; table2cell(power_df)];  % Combine headers and data

    % Save the power_df as a .mat file
    % Define output filename
    output_filename = sprintf('Power_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
   
    % Save the power_df as a .mat file
    save(output_path, 'power_df');

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ▄▄▄ ██ ▄▄▄██ ▀██ ██ ▄▄▄ ██ ▄▄▄ ██ ▄▄▀███▄ ▄██ ▄▄▀█ ▄▄▀██ ▄▄▄ █ ▄▄▀
    % ██▄▄▄▀▀██ ▄▄▄██ █ █ ██▄▄▄▀▀██ ███ ██ ▀▀▄████ ███ ▀▀▄█ ▀▀ ██▄▄▄▀▀█ ▀▀ 
    % ██ ▀▀▀ ██ ▀▀▀██ ██▄ ██ ▀▀▀ ██ ▀▀▀ ██ ██ ███▀ ▀██ ██ █ ██ ██ ▀▀▀ █ ██ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % SENSOR IRASA

    % Initialize results structures for IRASA
    results_fractal_high = struct();
    results_fractal_low = struct();
    results_oscillatory_high = struct();
    results_oscillatory_low = struct();

    temp_fractal_high = cell(size(freq_bands, 1), 1);
    temp_fractal_low = cell(size(freq_bands, 1), 1);
    temp_oscillatory_high = cell(size(freq_bands, 1), 1);
    temp_oscillatory_low = cell(size(freq_bands, 1), 1);

    % Configurate irasa
    cfg_irasa_common = [];
    cfg_irasa_common.method = 'irasa';
    cfg_irasa_common.keeptrials = 'yes';
    cfg_irasa_common.pad = 'nextpow2';
    %cfg_irasa_common.channel = {data_high.label};

    % Loop through all frequency bands
    parfor i = 1:size(freq_bands, 1)
        % Initialize configuration for this iteration
        cfg_irasa = cfg_irasa_common;
        cfg_irasa.foilim = [freq_bands{i, 2}, freq_bands{i, 3}];
    
        % Perform IRASA analysis for high condition
        cfg_irasa.output = 'fractal';
        fractal_high_temp = ft_freqanalysis(cfg_irasa, data_high);
        cfg_irasa.output = 'original';
        original_high_temp = ft_freqanalysis(cfg_irasa, data_high);
    
        % Perform IRASA analysis for low condition
        cfg_irasa.output = 'fractal';
        fractal_low_temp = ft_freqanalysis(cfg_irasa, data_low);
        cfg_irasa.output = 'original';
        original_low_temp = ft_freqanalysis(cfg_irasa, data_low);
    
        % Subtract fractal from original to get oscillatory component
        cfg_math = [];
        cfg_math.parameter = 'powspctrm';
        cfg_math.operation = 'x2-x1'; 
        oscillatory_high_temp = ft_math(cfg_math, original_high_temp, fractal_high_temp);
        oscillatory_low_temp = ft_math(cfg_math, original_low_temp, fractal_low_temp);

        % Store temporary results
        temp_fractal_high{i} = fractal_high_temp;
        temp_fractal_low{i} = fractal_low_temp;
        temp_oscillatory_high{i} = oscillatory_high_temp;
        temp_oscillatory_low{i} = oscillatory_low_temp;
    end
    
    % Consolidate results
    for i = 1:size(freq_bands, 1)
        band = freq_bands{i, 1};
        results_fractal_high.(band) = temp_fractal_high{i};
        results_fractal_low.(band) = temp_fractal_low{i};
        results_oscillatory_high.(band) = temp_oscillatory_high{i};
        results_oscillatory_low.(band) = temp_oscillatory_low{i};
    end

    % Initialize results table
    n_trials = length(data_high.trial);
    n_channels = length(data_high.label);
    n_freq_bands = size(freq_bands, 1);
    n_conditions = 2; % 'Low' and 'High'
    n_quant_status = 2; % 'Absolute' and 'Relative'
    n_components = 2; % 'Oscillatory' and 'Fractal'
    n_rows = n_trials * n_channels * n_freq_bands * n_conditions * n_quant_status;

    % Precompute conditions array
    conditions_arr = ["high", "low"];
    
    % Initialize the results table
    column_names = {'Trial_index', 'Condition', 'Channel', 'Brain_region', 'Brain_side', 'Freq_band', 'Quant_status', 'Component', 'Metric_Value'};
    results_table = table('Size', [n_rows, numel(column_names)], 'VariableTypes', {'double', 'categorical', 'string', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'double'}, 'VariableNames', column_names);
    
    % Parallel loop
    results_parts = cell(1, n_trials);

    parfor trialIdx = 1:n_trials
        temp_results = table('Size', [n_channels * n_freq_bands * n_conditions * n_quant_status* n_components, numel(column_names)], 'VariableTypes', {'double', 'categorical', 'string', 'categorical', 'categorical', 'categorical', 'categorical','categorical', 'double'}, 'VariableNames', column_names);
        temp_row = 1;
    
        for conditionIdx = 1:n_conditions
            condition = conditions_arr(conditionIdx);
            trial_index_adjusted = trialIdx + (conditionIdx == 1) * 200;

            for channelIdx = 1:n_channels
                channel_label = data_high.label{channelIdx};
                brain_region = brain_regions{channelIdx};
                brain_side = brain_sides{channelIdx};

                for bandIdx = 1:n_freq_bands
                    band = freq_bands{bandIdx, 1};
                    low_freq = freq_bands{bandIdx, 2};
                    high_freq = freq_bands{bandIdx, 3};
            
                    % Access IRASA results for the current band and condition
                    if strcmp(condition, 'high')
                        fractal_analysis = results_fractal_high.(band);
                        oscillatory_analysis = results_oscillatory_high.(band);
                    else
                        fractal_analysis = results_fractal_low.(band);
                        oscillatory_analysis = results_oscillatory_low.(band);
                    end
            
                    % Extract fractal and oscillatory components for the frequency band
                    freq_values = fractal_analysis.freq;
                    fractal_data_band = fractal_analysis.powspctrm(trialIdx, channelIdx, :);
                    oscillatory_data_band = oscillatory_analysis.powspctrm(trialIdx, channelIdx, :);

                    % Find indices of the frequency range
                    freq_indices = find(freq_values >= low_freq & freq_values <= high_freq);

                    % Extract the relevant data for bandpower calculation
                    relevant_fractal_data = squeeze(fractal_data_band(:, freq_indices));
                    relevant_oscillatory_data = squeeze(oscillatory_data_band(:, freq_indices));
               
                    % Calculate integrated (summed) values for absolute measures
                    integrated_fractal = sum(relevant_fractal_data);
                    integrated_oscillatory = sum(relevant_oscillatory_data);

                    % Convert to dB and extract the real part
                    abs_fractal_db = real(10 * log10(integrated_fractal + eps));
                    abs_oscillatory_db = real(10 * log10(integrated_oscillatory + eps));
                    
                    % Calculate total power across all bands for relative measures
                    total_fractal_power = 0;
                    total_oscillatory_power = 0;
                    for b = 1:n_freq_bands
                        band_name = freq_bands{b, 1};
                        if strcmp(condition, 'high')
                            total_fractal_analysis = results_fractal_high.(band_name);
                            total_oscillatory_analysis = results_oscillatory_high.(band_name);
                        else
                            total_fractal_analysis = results_fractal_low.(band_name);
                            total_oscillatory_analysis = results_oscillatory_low.(band_name);
                        end
                        total_fractal_power = total_fractal_power + sum(total_fractal_analysis.powspctrm(trialIdx, channelIdx, :));
                        total_oscillatory_power = total_oscillatory_power + sum(total_oscillatory_analysis.powspctrm(trialIdx, channelIdx, :));
                    end
                    
                    % Calculate relative power
                    rel_fractal_power = integrated_fractal / total_fractal_power;
                    rel_oscillatory_power = integrated_oscillatory / total_oscillatory_power;
                    
                    temp_results(temp_row, :) = {trial_index_adjusted, condition, channel_label, brain_region, brain_side, band, 'Absolute', 'Fractal', abs_fractal_db};
                    temp_row = temp_row + 1;
                    temp_results(temp_row, :) = {trial_index_adjusted, condition, channel_label, brain_region, brain_side, band, 'Relative', 'Fractal', rel_fractal_power};
                    temp_row = temp_row + 1;
                    temp_results(temp_row, :) = {trial_index_adjusted, condition, channel_label, brain_region, brain_side, band, 'Absolute', 'Oscillatory', abs_oscillatory_db};
                    temp_row = temp_row + 1;
                    temp_results(temp_row, :) = {trial_index_adjusted, condition, channel_label, brain_region, brain_side, band, 'Relative', 'Oscillatory', rel_oscillatory_power};
                    temp_row = temp_row + 1;
                end
            end
        end
        results_parts{trialIdx} = temp_results;
    end
    
    % Concatenate results from all parts
    results_table = vertcat(results_parts{:});
    
    % Rename the table to irasa_df
    irasa_df = results_table;
    
    % Convert categorical columns to cell arrays of character vectors
    irasa_df.Condition = cellstr(irasa_df.Condition);
    irasa_df.Brain_region = cellstr(irasa_df.Brain_region);
    irasa_df.Brain_side = cellstr(irasa_df.Brain_side);
    irasa_df.Freq_band = cellstr(irasa_df.Freq_band);
    irasa_df.Quant_status = cellstr(irasa_df.Quant_status);
    irasa_df.Component = cellstr(irasa_df.Component);
    irasa_df.Channel = cellstr(irasa_df.Channel);
    
    % Sort the final table by 'Trial_index'
    irasa_df = sortrows(irasa_df, 'Trial_index');

   % Convert Table to Cell Array with Headers (easier to load in python)
    headers = irasa_df.Properties.VariableNames;  % Extract column names
    irasa_df = [headers; table2cell(irasa_df)];  % Combine headers and data

    % Save the irasa_df as a .mat file
    % Define output filename
    output_filename = sprintf('irasa_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
   
    % Save the irasa_df as a .mat file
    save(output_path, 'irasa_df');

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ████ ▄▄ ██ ▄▄▄ ██ ███ ██ ▄▄▄██ ▄▄▀█████ ▄▄▀██ ▄▄▄ ██ ███ ██ ▄▀▄ ██ ▄▀▄ ██ ▄▄▄█▄▄ ▄▄██ ▄▄▀██ ███ 
    % ████ ▀▀ ██ ███ ██ █ █ ██ ▄▄▄██ ▀▀▄█████ ▀▀ ██▄▄▄▀▀██▄▀▀▀▄██ █ █ ██ █ █ ██ ▄▄▄███ ████ ▀▀▄██▄▀▀▀▄
    % ████ █████ ▀▀▀ ██▄▀▄▀▄██ ▀▀▀██ ██ █████ ██ ██ ▀▀▀ ████ ████ ███ ██ ███ ██ ▀▀▀███ ████ ██ ████ ██
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % POWER ASYMMETRY

    % Convert the cell array to a table
    columnNames = power_df(1, :); % Assuming the first row contains column names
    power_df_table = cell2table(power_df(2:end, :), 'VariableNames', columnNames);
    
    % Filter for absolute PSD values
    qq = "Absolute";
    abs_power_df = power_df_table(strcmp(power_df_table.Quant_status, qq), :);
    
    % Convert dB to Raw Power
    abs_power_df.PSD = 10 .^ (abs_power_df.PSD / 10);
    
    % Create lists for Left and Right channels
    Left_chs = brain_side_map.keys();
    Left_chs = Left_chs(strcmp(values(brain_side_map, Left_chs), 'Left'));
    Right_chs = brain_side_map.keys();
    Right_chs = Right_chs(strcmp(values(brain_side_map, Right_chs), 'Right'));
    
    % Group and calculate asymmetry
    groupVars = {'Condition', 'Trial_index', 'Freq_band', 'Brain_region'};
    [grouped_tbl, group_ids] = findgroups(abs_power_df(:, groupVars));
    asymmetry_values = arrayfun(@(g) calculateAsymmetry(abs_power_df(grouped_tbl == g, :), Left_chs, Right_chs), unique(grouped_tbl));
    
    % Combine group IDs and asymmetry values
    asymmetry_data = [table2cell(group_ids), num2cell(asymmetry_values)];
    
    % Create the result table
    asymmetry_df = cell2table(asymmetry_data, 'VariableNames', [groupVars, {'Asymmetry_score'}]);

    % Sort the result table
    asymmetry_df = sortrows(asymmetry_df, {'Trial_index', 'Condition', 'Freq_band', 'Brain_region'});

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = asymmetry_df.Properties.VariableNames;  % Extract column names
    asymmetry_df = [headers; table2cell(asymmetry_df)];  % Combine headers and data

    % Save the asymmetry_df as a .mat file
    % Define output filename
    output_filename = sprintf('Asymmetry_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
   
    % Save the power_df as a .mat file
    save(output_path, 'asymmetry_df');

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ██ █████ ██ ▄▄▄ ██ ▄▄▀█▄▄ ▄▄██ ██ ████ ▄▄ ██ ▄▄▄ ██ ███ ██ ▄▄▄██ ▄▄▀
    % ██ ▄▄ █████ ██ ███ ██ ▀▀▄███ ████ ▄▄ ████ ▀▀ ██ ███ ██ █ █ ██ ▄▄▄██ ▀▀▄
    % ██ ██ ██ ▀▀ ██ ▀▀▀ ██ ██ ███ ████ ██ ████ █████ ▀▀▀ ██▄▀▄▀▄██ ▀▀▀██ ██ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % HJORTH POWER

    % Define the channel groups
    channels_C3 = {'C3', 'FC1', 'FC5', 'CP1', 'CP5'};
    channels_C4 = {'C4', 'FC2', 'FC6', 'CP2', 'CP6'};
    channels_FCC3h = {'FCC3h', 'FFC1h', 'FFC5h', 'CCP1h', 'CCP5h'};
    channels_FCC4h = {'FCC4h', 'FFC2h', 'FFC6h', 'CCP2h', 'CCP6h'};
    
    % Define weights
    weights = [1, -1/4, -1/4, -1/4, -1/4];

    % Define the frequency range for bandpower calculation
    freq_range = [8 12]; %mu 
    
    % Initialize variables to store results
    % Number of trials for high and low conditions
    n_trials_high = numTrials/2;
    n_trials_low = numTrials/2;
    
    % Extracting eeg_high and eeg_low from epochs.trial
    % Reshaping to channels x times x trials format
    eeg_high = permute(epochs.trial(:, 1:n_trials_high, :), [3, 1, 2]);
    eeg_low = permute(epochs.trial(:, (n_trials_high+1):(n_trials_high+n_trials_low), :), [3, 1, 2]);

    n_trials = size(eeg_high, 3); 
    conditions = {'High', 'Low'};
    eeg_data = {eeg_low, eeg_high};
    hjorth_power_results = [];
    
    % Loop over conditions
    for cond_idx = 1:numel(conditions)
        condition = conditions{cond_idx};
        data = eeg_data{cond_idx};
    
        % Loop over trials
        for trial = 1:n_trials
            % Apply the spatial filter for each channel group
            signal_C3 = spatialFilter(data, data_high.label, channels_C3, weights);
            signal_C4 = spatialFilter(data, data_high.label, channels_C4, weights);
            signal_FCC3h = spatialFilter(data, data_high.label, channels_FCC3h, weights);
            signal_FCC4h = spatialFilter(data, data_high.label, channels_FCC4h, weights);

            % Calculate bandpower for each channel group
            power_C3 = bandpower(detrend(signal_C3(:, trial)), fsample, freq_range);
            power_C4 = bandpower(detrend(signal_C4(:, trial)), fsample, freq_range);
            power_FCC3H = bandpower(detrend(signal_FCC3h(:, trial)), fsample, freq_range);
            power_FCC4H = bandpower(detrend(signal_FCC4h(:, trial)), fsample, freq_range);

            % Store results
            hjorth_power_results = [hjorth_power_results; {trial + n_trials_high * (cond_idx - 1), condition, power_C3, power_C4, power_FCC3H, power_FCC4H}];
        end
    end
    
    % Convert results to table
    hjorth_power_table = cell2table(hjorth_power_results, 'VariableNames', {'Trial_index', 'Condition', 'Power_C3', 'Power_C4', 'Power_FCC3H', 'Power_FCC4H'});

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = hjorth_power_table.Properties.VariableNames;  % Extract column names
    hjorth_power_table = [headers; table2cell(hjorth_power_table)];  % Combine headers and data

    % Save the hjorth_power_table as a .mat file
    output_filename = sprintf('Hjorth_Power_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    save(output_path, 'hjorth_power_table');
 
    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ██ █████ ██ ▄▄▄ ██ ▄▄▀█▄▄ ▄▄██ ██ ████ ▄▄ ██ ██ █ ▄▄▀██ ▄▄▄ ██ ▄▄▄
    % ██ ▄▄ █████ ██ ███ ██ ▀▀▄███ ████ ▄▄ ████ ▀▀ ██ ▄▄ █ ▀▀ ██▄▄▄▀▀██ ▄▄▄
    % ██ ██ ██ ▀▀ ██ ▀▀▀ ██ ██ ███ ████ ██ ████ █████ ██ █ ██ ██ ▀▀▀ ██ ▀▀▀
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % HJORTH PHASE

    clear filter;
 
    % Define the conditions and corresponding data
    conditions = {'Low', 'High'};
    eeg_data = {eeg_low, eeg_high};  
    
    % Define parameters for phase estimation
    arOrder = 65;
    edge = 25;
    hilbertwindow = 128;
    offset_correction = 4;

    % Initialize filters
    bpfreq = [8 12];
    D_bandpass = designfilt('bandpassfir', 'FilterOrder', 75, 'CutoffFrequency1', bpfreq(1) , 'CutoffFrequency2', bpfreq(2), 'SampleRate', 500, 'DesignMethod', 'window');
    D_nyquist = designfilt('lowpassfir', 'FilterOrder', 50, 'CutoffFrequency', 0.5, 'SampleRate', 500);
    
    % Initialize arrays for storing results
    trial_indices = [];
    condition_list = [];
    phase_C3_list = [];
    phase_C4_list = [];
    phase_FCC3h_list = [];
    phase_FCC4h_list = [];
    
    % Loop over conditions
    for cond_idx = 1:numel(conditions)
        condition = conditions{cond_idx};
        data = eeg_data{cond_idx}; % data is 3D: channels x times x trials
    
        % Apply spatial filtering for the entire condition data
        signal_C3 = spatialFilter(data, data_high.label, channels_C3, weights);
        signal_C4 = spatialFilter(data, data_high.label, channels_C4, weights);
        signal_FCC3h = spatialFilter(data, data_high.label, channels_FCC3h, weights);
        signal_FCC4h = spatialFilter(data, data_high.label, channels_FCC4h, weights);
    
        % Number of trials in the current condition
        num_trials = size(data, 3);
    
        % Process each trial
        for trial = 1:num_trials
            % Adjust trial index based on condition
            if strcmp(condition, 'High')
                trial_index = trial;
            else % For 'Low' condition
                trial_index = trial + numTrials/2;
            end
    
    
            % Extract trial data from each spatially filtered signal
            trial_data_C3 = signal_C3(:, trial);
            trial_data_C4 = signal_C4(:, trial);
            trial_data_FCC3h = signal_FCC3h(:, trial);
            trial_data_FCC4h = signal_FCC4h(:, trial);
    
            % Initialize lists to store phase for each signal
            phase_list = cell(1, 4);
    
            % Detrend, filter, and phase estimation for each signal
            for i = 1:4
                % Select the appropriate trial data
                switch i
                    case 1
                        current_signal = trial_data_C3;
                    case 2
                        current_signal = trial_data_C4;
                    case 3
                        current_signal = trial_data_FCC3h;
                    case 4
                        current_signal = trial_data_FCC4h;
                end
    
                % Detrend 
                detrended_signal = detrend(current_signal);

                % Downsample to 500 Hz
                downsampled_signal = detrended_signal(1:2:end);

                % Filter
                filtered_signal = filter(D_nyquist, downsampled_signal);
    
                % Phase estimation
                [phase, ~] = phastimate(filtered_signal, D_bandpass, edge, arOrder, hilbertwindow, offset_correction);
    
                % Store the last phase value
                phase_list{i} = phase(end);
               
            end
    
            % Update phase lists and trial_indices, condition_list
            phase_C3_list = [phase_C3_list; phase_list{1}];
            phase_C4_list = [phase_C4_list; phase_list{2}];
            phase_FCC3h_list = [phase_FCC3h_list; phase_list{3}];
            phase_FCC4h_list = [phase_FCC4h_list; phase_list{4}];
            trial_indices = [trial_indices; trial_index];
            condition_list = [condition_list; {condition}];
    
        end
    end

     % Calculate sine and cosine for each value in the phase lists
    Sin_Phase_C3H = sin(phase_C3_list);
    Cos_Phase_C3H = cos(phase_C3_list);
    Sin_Phase_C4H = sin(phase_C4_list);
    Cos_Phase_C4H = cos(phase_C4_list);
    Sin_Phase_FCC3H = sin(phase_FCC3h_list);
    Cos_Phase_FCC3H = cos(phase_FCC3h_list);
    Sin_Phase_FCC4H = sin(phase_FCC4h_list);
    Cos_Phase_FCC4H = cos(phase_FCC4h_list);

    % Create the table with additional sine and cosine columns
    hjorth_phase_df = table(trial_indices, condition_list, ...
                            phase_C3_list, phase_C4_list, phase_FCC3h_list, phase_FCC4h_list, ...
                            Sin_Phase_C3H, Cos_Phase_C3H, ...
                            Sin_Phase_C4H, Cos_Phase_C4H, ...
                            Sin_Phase_FCC3H, Cos_Phase_FCC3H, ...
                            Sin_Phase_FCC4H, Cos_Phase_FCC4H, ...
                            'VariableNames', {'Trial_index', 'Condition', ...
                                              'Phase_C3H', 'Phase_C4H', 'Phase_FCC3H', 'Phase_FCC4H', ...
                                              'Sin_Phase_C3H', 'Cos_Phase_C3H', ...
                                              'Sin_Phase_C4H', 'Cos_Phase_C4H', ...
                                              'Sin_Phase_FCC3H', 'Cos_Phase_FCC3H', ...
                                              'Sin_Phase_FCC4H', 'Cos_Phase_FCC4H'});


    
    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = hjorth_phase_df.Properties.VariableNames;  % Extract column names
    hjorth_phase_df = [headers; table2cell(hjorth_phase_df)];  % Combine headers and data

    % Save the hjorth_phase_df as a .mat file
    output_filename = sprintf('Hjorth_Phase_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    save(output_path, 'hjorth_phase_df');

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ████ ██ █████ ██ ▄▄▄ ██ ▄▄▀█▄▄ ▄▄██ ██ ████ ▄▄ ██ ▄▄▄ ██ ███ ██ ▄▄▄██ ▄▄▀█████ ▄▄▀██ ▄▄▄ ██ ███ ██ ▄▀▄ ██ ▄▀▄ ██ ▄▄▄█▄▄ ▄▄██ ▄▄▀██ ███ 
    % ████ ▄▄ █████ ██ ███ ██ ▀▀▄███ ████ ▄▄ ████ ▀▀ ██ ███ ██ █ █ ██ ▄▄▄██ ▀▀▄█████ ▀▀ ██▄▄▄▀▀██▄▀▀▀▄██ █ █ ██ █ █ ██ ▄▄▄███ ████ ▀▀▄██▄▀▀▀▄
    % ████ ██ ██ ▀▀ ██ ▀▀▀ ██ ██ ███ ████ ██ ████ █████ ▀▀▀ ██▄▀▄▀▄██ ▀▀▀██ ██ █████ ██ ██ ▀▀▀ ████ ████ ███ ██ ███ ██ ▀▀▀███ ████ ██ ████ ██
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % HJORTH POWER ASYMMETRY 

    % Convert the cell array to a table
    columnNames = hjorth_power_table(1, :); % Assuming the first row contains column names
    hjorth_power_df = cell2table(hjorth_power_table(2:end, :), 'VariableNames', columnNames);
    
    % Compute the Power Asymmetry for C3 and C4
    hjorth_power_df.Power_Asymmetry_C = ...
        (hjorth_power_df.Power_C3 - hjorth_power_df.Power_C4) ./ ...
        (hjorth_power_df.Power_C3 + hjorth_power_df.Power_C4);
    
    % Compute the Power Asymmetry for FCC3H and FCC4H
    hjorth_power_df.Power_Asymmetry_FCC = ...
        (hjorth_power_df.Power_FCC3H - hjorth_power_df.Power_FCC4H) ./ ...
        (hjorth_power_df.Power_FCC3H + hjorth_power_df.Power_FCC4H);
    
    % Create a new table containing only the Power Asymmetry values for both M1 and S1
    M1_S1_power_asymmetry_df = hjorth_power_df(:, {'Trial_index', 'Condition', 'Power_Asymmetry_C', 'Power_Asymmetry_FCC'});
  
    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = M1_S1_power_asymmetry_df.Properties.VariableNames;  % Extract column names
    M1_S1_power_asymmetry_df = [headers; table2cell(M1_S1_power_asymmetry_df)];  % Combine headers and data
    
    % Save the new table
    output_filename = sprintf('M1_S1_power_asymmetry_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    save(output_path, 'M1_S1_power_asymmetry_df');

    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ██ █████ ██ ▄▄▄ ██ ▄▄▀█▄▄ ▄▄██ ██ ████ ▄▄▀██ ▄▄▄ ██ ▀██ ██ ▀██ ██ ▄▄▄██ ▄▄▀█▄▄ ▄▄█▄ ▄██ ███ █▄ ▄█▄▄ ▄▄██ ███ 
    % ██ ▄▄ █████ ██ ███ ██ ▀▀▄███ ████ ▄▄ ████ █████ ███ ██ █ █ ██ █ █ ██ ▄▄▄██ ██████ ████ ████ █ ███ ████ ████▄▀▀▀▄
    % ██ ██ ██ ▀▀ ██ ▀▀▀ ██ ██ ███ ████ ██ ████ ▀▀▄██ ▀▀▀ ██ ██▄ ██ ██▄ ██ ▀▀▀██ ▀▀▄███ ███▀ ▀███▄▀▄██▀ ▀███ ██████ ██
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % HJORTH CONNECTIVITY

    eeg = permute(epochs.trial(:, 1:numTrials, :), [3, 1, 2]);

    % Loop over trials
    for trial = 1:numTrials
        % Apply the spatial filter for each channel group
        signal_C3 = spatialFilter(eeg, data_high.label, channels_C3, weights);
        signal_C4 = spatialFilter(eeg, data_high.label, channels_C4, weights);
        signal_FCC3h = spatialFilter(eeg, data_high.label, channels_FCC3h, weights);
        signal_FCC4h = spatialFilter(eeg, data_high.label, channels_FCC4h, weights);
    end    
    % signals have the shape times x trials 

    % Define the signals and their labels
    signals = {signal_C3, signal_C4, signal_FCC3h, signal_FCC4h};
    signal_labels = {'C3', 'C4', 'FCC3h', 'FCC4h'};
    signal_combinations = nchoosek(1:length(signals), 2); % All pairwise combinations of signals
    
    % Prepare the other variables as before
    conditions = [repmat({'high'}, numTrials/2, 1); repmat({'low'}, numTrials/2, 1)];
    freq_bands = {'delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma'};

    numSignals = length(signal_labels);
    numTrials = size(signal_C3, 2);
    numTimePoints = size(signal_C3, 1);
    indxcmb = nchoosek(1:length(signals), 2); % All pairwise combinations of signals
    fsample = epochs.fsample; % sampling frequency in Hz
    ARord = 30; % AR model order for padding

    freq_bands = {
    'delta', 0.5, 4.0;
    'theta', 4.0, 8.0;
    'alpha', 8.0, 12.0;
    'low_beta', 13.0, 18.0;
    'high_beta', 19.0, 35.0;
    'low_gamma', 36.0, 58.0;
    'high_gamma', 58.0, 100.0
    };

    % Preallocate arrays for connectivity results
    iPLV_results = cell(length(freq_bands), 1);
    PLI_results = cell(length(freq_bands), 1);
    wPLI_results = cell(length(freq_bands), 1);
    Coh_results = cell(length(freq_bands), 1);
    imCoh_results = cell(length(freq_bands), 1);
    lagCoh_results = cell(length(freq_bands), 1);
    oPEC_results = cell(length(freq_bands), 1); 
    
    for bandIdx = 1:size(freq_bands, 1)
        % Band-specific settings
        bandName = freq_bands{bandIdx, 1};
        bpfreq = [freq_bands{bandIdx, 2}, freq_bands{bandIdx, 3}];
        n = round((0.15*fsample)/2)*2; % filter order
        b = fir1(n, bpfreq/(fsample/2)); % filter coefficients
        padsize = round(2*fsample/mean(bpfreq)); % padding size
    
        % Initialize arrays for this band
        iPLV_band = zeros(size(indxcmb, 1), numTrials);
        PLI_band = zeros(size(indxcmb, 1), numTrials);
        wPLI_band = zeros(size(indxcmb, 1), numTrials);
        Coh_band = zeros(size(indxcmb, 1), numTrials);
        imCoh_band = zeros(size(indxcmb, 1), numTrials);
        lagCoh_band = zeros(size(indxcmb, 1), numTrials);
        oPEC_band = zeros(size(indxcmb, 1), numTrials);

        % Loop over trials and signal combinations
        for trial = 1:numTrials
            for combIdx = 1:size(signal_combinations, 1)
                signal1 = signals{signal_combinations(combIdx, 1)}(:, trial); %has to be 1001 x 1
                signal2 = signals{signal_combinations(combIdx, 2)}(:, trial);
                
                % Preprocess the data 
                preprocessedData = preprocessData([signal1, signal2], b, ARord, padsize);

                % Calculate connectivity measures
                [iPLV, PLI, wPLI, Coh, imCoh, lagCoh, oPEC] = calculateConnectivity(preprocessedData, padsize, [1, 2]);

                % Store results for this trial and band
                iPLV_band(combIdx, trial) = iPLV;
                PLI_band(combIdx, trial) = PLI;
                wPLI_band(combIdx, trial) = wPLI;
                Coh_band(combIdx, trial) = Coh;
                imCoh_band(combIdx, trial) = imCoh;
                lagCoh_band(combIdx, trial) = lagCoh;
                oPEC_band(combIdx, trial) = oPEC;
            end
        end 
        % Store results for this frequency band
        iPLV_results{bandIdx} = iPLV_band;
        PLI_results{bandIdx} = PLI_band;
        wPLI_results{bandIdx} = wPLI_band;
        Coh_results{bandIdx} = Coh_band;
        imCoh_results{bandIdx} = imCoh_band;
        lagCoh_results{bandIdx} = lagCoh_band;
        oPEC_results{bandIdx} = oPEC_band;
    end

    % Create source connectivity dataframe
    % Start a parallel pool (if not already started)
    if isempty(gcp('nocreate'))
        parpool; % Start with default settings
    end

    % Define signal pair labels
    signalPairLabels = arrayfun(@(x) sprintf('%s-%s', signal_labels{signal_combinations(x, 1)}, signal_labels{signal_combinations(x, 2)}), 1:size(signal_combinations, 1), 'UniformOutput', false);
    
    % Preallocate cell array for parallel results
    parallelResults = cell(length(freq_bands), 1);
    
    % Loop through each frequency band in parallel
    parfor bandIdx = 1:length(freq_bands)
        bandTable = table();
        for trialIdx = 1:numTrials
            for signalPairIdx = 1:length(signalPairLabels)
                newRow = {trialIdx, conditions{trialIdx}, freq_bands{bandIdx}, signalPairLabels{signalPairIdx}, ...
                          iPLV_results{bandIdx}(signalPairIdx, trialIdx), PLI_results{bandIdx}(signalPairIdx, trialIdx), ...
                          wPLI_results{bandIdx}(signalPairIdx, trialIdx), Coh_results{bandIdx}(signalPairIdx, trialIdx), ...
                          imCoh_results{bandIdx}(signalPairIdx, trialIdx), lagCoh_results{bandIdx}(signalPairIdx, trialIdx), ...
                          oPEC_results{bandIdx}(signalPairIdx, trialIdx)};
                bandTable = [bandTable; newRow];
            end
        end
        parallelResults{bandIdx} = bandTable;
    end
    
    % Concatenate all tables
    hjorthconnectivity_df = vertcat(parallelResults{:});
    hjorthconnectivity_df.Properties.VariableNames = {'Trial_index', 'Condition', 'Freq_band', 'Signal_comb', ...
                                                      'iPLV', 'PLI', 'wPLI', 'Coh', 'imCoh', 'lagCoh', 'oPEC'};

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = hjorthconnectivity_df.Properties.VariableNames;  % Extract column names
    hjorthconnectivity_df = [headers; table2cell(hjorthconnectivity_df)];  % Combine headers and data

    % Save the hjorthconnectivity_df as a .mat file
    % Define output filename
    output_filename = sprintf('HjorthCon_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    
    % Save the hjorthconnectivity_df as a .mat file
    save(output_path, 'hjorthconnectivity_df')
    
    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ██ █████ ██ ▄▄▄ ██ ▄▄▀█▄▄ ▄▄██ ██ ████ ▄▄ █ ▄▄▀██ ▄▄▀
    % ██ ▄▄ █████ ██ ███ ██ ▀▀▄███ ████ ▄▄ ████ ▀▀ █ ▀▀ ██ ███
    % ██ ██ ██ ▀▀ ██ ▀▀▀ ██ ██ ███ ████ ██ ████ ████ ██ ██ ▀▀▄
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % HJORTH PAC 

    % Define your signals and their labels
    signals = {signal_C3, signal_C4, signal_FCC3h, signal_FCC4h};
    signal_labels = {'C3', 'C4', 'FCC3h', 'FCC4h'};
    
    % Initialize PAC results storage
    PAC_results = cell(length(signals), 1); % One cell for each signal
    Fs = epochs.fsample;
    
    % Define the combinations for PAC calculation
    PAC_combinations = {
        'theta', 'low_gamma';
        'theta', 'high_gamma';
        'alpha', 'low_gamma';
        'alpha', 'high_gamma';
        'low_beta', 'low_gamma';
        'low_beta', 'high_gamma';
        'high_beta', 'low_gamma';
        'high_beta', 'high_gamma'
    };
    
    % Loop over each signal
    for signalIdx = 1:length(signals)
        % Initialize PAC matrix for current signal
        numTrials = size(signals{signalIdx}, 2); % Number of trials for the current signal
        numCombinations = size(PAC_combinations, 1);
        signalPAC = zeros(numTrials, numCombinations);
        
        % Loop over PAC combinations
        for pacIdx = 1:numCombinations
            phase_band_name = PAC_combinations{pacIdx, 1};
            amp_band_name = PAC_combinations{pacIdx, 2};
            
            % Extract frequency range for phase and amplitude bands
            phase_band_range = freq_bands{strcmp(freq_bands(:,1), phase_band_name), 2:3};
            amp_band_range = freq_bands{strcmp(freq_bands(:,1), amp_band_name), 2:3};
            
            % Parallel loop over trials
            parfor trialIdx = 1:numTrials
                x = signals{signalIdx}(:, trialIdx); % Extract data for current trial
                
                % Calculate PAC
                [tf_canolty] = tfMVL(x, amp_band_range, phase_band_range, Fs);
                signalPAC(trialIdx, pacIdx) = tf_canolty;
            end
        end
        PAC_results{signalIdx} = signalPAC; % Store results for the current signal
    end

    % Save PAC results in df
    % Define signal names
    signal_names = {'C3', 'C4', 'FCC3h', 'FCC4h'};
    
    % Preallocate cell array to store parallel results
    parallelResults = cell(length(signal_names), 1);
    
    % Parallel loop over signals
    parfor signalIdx = 1:length(signal_names)
        signalName = signal_names{signalIdx};
        pacData = PAC_results{signalIdx}; % PAC data for the current signal
        signalTable = table(); % Initialize a table for each signal
    
        % Loop over trials
        for trialIdx = 1:numTrials
            % Determine condition based on trial index
            condition = 'high';
            if trialIdx > numTrials/2
                condition = 'low';
            end
    
            % Loop over frequency-pair combinations
            for freqPairIdx = 1:size(PAC_combinations, 1)
                freqPairName = sprintf('%s-%s', PAC_combinations{freqPairIdx, 1}, PAC_combinations{freqPairIdx, 2});
                pacValue = pacData(trialIdx, freqPairIdx);
    
                % Create a new row for the table
                newRow = {trialIdx, condition, signalName, freqPairName, pacValue};
                signalTable = [signalTable; newRow];
            end
        end
    
        % Store the table for the current signal in the cell array
        parallelResults{signalIdx} = signalTable;
    end
    
    % Combine all signal tables into one
    hjorth_pac_df = vertcat(parallelResults{:});
    
    % Set the column names for the DataFrame
    hjorth_pac_df.Properties.VariableNames = {'Trial_index', 'Condition', 'Signal', 'Freq_combinations', 'PAC_value'};

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = hjorth_pac_df.Properties.VariableNames;  % Extract column names
    hjorth_pac_df = [headers; table2cell(hjorth_pac_df)];  % Combine headers and data
    
    % Save the hjorth_pac_df as a .mat file
    % Define output filename
    output_filename = sprintf('hjorth_pac_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    
    % Save the hjorth_pac_df as a .mat file
    save(output_path, 'hjorth_pac_df');
  
    % ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    % ██ ██ █████ ██ ▄▄▄ ██ ▄▄▀█▄▄ ▄▄██ ██ ████ ▄▄▄██ ▄▄▀█ ▄▄▀██ ▄▄▀█▄▄ ▄▄█ ▄▄▀██ ███████ ▄▄▄██ ▄▄▄█ ▄▄▀█▄▄ ▄▄██ ██ ██ ▄▄▀██ ▄▄▄██ ▄▄▄ 
    % ██ ▄▄ █████ ██ ███ ██ ▀▀▄███ ████ ▄▄ ████ ▄▄███ ▀▀▄█ ▀▀ ██ ██████ ███ ▀▀ ██ ███████ ▄▄███ ▄▄▄█ ▀▀ ███ ████ ██ ██ ▀▀▄██ ▄▄▄██▄▄▄▀▀
    % ██ ██ ██ ▀▀ ██ ▀▀▀ ██ ██ ███ ████ ██ ████ █████ ██ █ ██ ██ ▀▀▄███ ███ ██ ██ ▀▀ ████ █████ ▀▀▀█ ██ ███ ████▄▀▀▄██ ██ ██ ▀▀▀██ ▀▀▀ 
    % ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    % HJORTH FRACTAL FEATURES

    % Extracting dimensions from epochs.trial
    [n_times, n_trials, n_channels] = size(epochs.trial);
    
    % Initialize the data matrix
    data = zeros(n_channels, n_times, n_trials);
    
    % Extracting and reorganizing data from epochs.trial
    for trial = 1:n_trials
        % Transpose and permute each trial's data to match the desired format
        trial_data = squeeze(epochs.trial(:, trial, :))'; % Transpose to get [channels × time]
        data(:, :, trial) = trial_data;
    end
    
    % Define the channel groups
    channels_C3 = {'C3', 'FC1', 'FC5', 'CP1', 'CP5'};
    channels_C4 = {'C4', 'FC2', 'FC6', 'CP2', 'CP6'};
    channels_FCC3h = {'FCC3h', 'FFC1h', 'FFC5h', 'CCP1h', 'CCP5h'};
    channels_FCC4h = {'FCC4h', 'FFC2h', 'FFC6h', 'CCP2h', 'CCP6h'};
    
    % Define weights
    weights = [1, -1/4, -1/4, -1/4, -1/4];
    
    % Apply spatial filter
    filtered_data1 = spatialFilter(data, epochs.label, channels_C3, weights);
    filtered_data2 = spatialFilter(data, epochs.label, channels_C4, weights);
    filtered_data3 = spatialFilter(data, epochs.label, channels_FCC3h, weights);
    filtered_data4 = spatialFilter(data, epochs.label, channels_FCC4h, weights);

    % Define the new labels for the filtered data
    new_labels = {'C3_Hjorth', 'C4_Hjorth', 'FCC3h_Hjorth', 'FCC4h_Hjorth'};
    
    % Initialize the new data to be appended
    new_data = zeros(n_times, n_trials, length(new_labels));
    
    % Populate the new data matrix with the filtered data
    % Populate the new data matrix with the filtered data
    new_data(:, :, 1) = filtered_data1; % For C3_Hjorth
    new_data(:, :, 2) = filtered_data2; % For C4_Hjorth
    new_data(:, :, 3) = filtered_data3; % For FCC3h_Hjorth
    new_data(:, :, 4) = filtered_data4; % For FCC4h_Hjorth

    % Replace the existing trials with the new data
    epochs.trial = new_data;
    
    % Replace the existing labels with the new labels
    epochs.label = new_labels;

    % Initialize the data structures 
    data = [];
   
    % Copy the common data fields
    data.label = epochs.label; % channel labels
    data.fsample = epochs.fsample; % sampling frequency

    data.time = cell(1, n_trials);
    data.trial = cell(1, n_trials);
    n_samples = n_times;
 
    % Extract trials and corresponding time data for high and low conditions
    for trialIdx = 1:n_trials %the first 200 trials are 'high'
        % Extract the trial data and transpose to 'channels x times'
        data.trial{trialIdx} = permute(squeeze(epochs.trial(:, trialIdx, :)), [2, 1]);
        data.time{trialIdx} = epochs.time; % Assuming time is the same for all trials
    end
    
    % Add sampleinfo 
    data.sampleinfo = [(1:n_trials)' * n_samples - n_samples + 1, (1:n_trials)' * n_samples];

    % SENSOR IRASA
    % Define frequency bands 
    freq_bands = {
        'delta', 0.5, 4.0;
        'theta', 4.0, 8.0;
        'alpha', 8.0, 12.0;
        'low_beta', 13.0, 18.0;
        'high_beta', 19.0, 35.0;
        'low_gamma', 36.0, 58.0;
        'high_gamma', 58.0, 100.0
    };
    
    % Initialize results structures for IRASA
    results_fractal = struct();
    temp_fractal = cell(size(freq_bands, 1), 1);

    % Configurate irasa
    cfg_irasa_common = [];
    cfg_irasa_common.method = 'irasa';
    cfg_irasa_common.keeptrials = 'yes';
    cfg_irasa_common.pad = 'nextpow2';

    % Initialize the parallel pool (if not already initialized)
    if isempty(gcp('nocreate'))
        parpool; 
    end

    % Loop through all frequency bands
    parfor i = 1:size(freq_bands, 1)
        % Initialize configuration for this iteration
        cfg_irasa = cfg_irasa_common;
        cfg_irasa.foilim = [freq_bands{i, 2}, freq_bands{i, 3}];
    
        % Perform IRASA analysis
        cfg_irasa.output = 'fractal';
        fractal_temp = ft_freqanalysis(cfg_irasa, data);
  
        % Store temporary results
        temp_fractal{i} = fractal_temp;
    end
    
    % Consolidate results
    for i = 1:size(freq_bands, 1)
        band = freq_bands{i, 1};
        results_fractal.(band) = temp_fractal{i};
    end

    % Initialize results table
    n_trials = length(data.trial);
    n_channels = length(data.label);
    n_freq_bands = size(freq_bands, 1);
    n_rows = n_trials * n_channels * n_freq_bands;

    % Categorize channels based on Brain regions and Hemisphere
    brain_region_map = categorize_channels(data.label);
    brain_side_map = categorize_channels_rl(data.label);
    
    % Initialize cell arrays for brain regions and sides
    brain_regions = cell(1, length(data.label));
    brain_sides = cell(1, length(data.label));
    
    % Populate brain_regions and brain_sides based on the maps
    for i = 1:length(data.label)
        ch = data.label{i};
        brain_regions{i} = brain_region_map(ch);
        brain_sides{i} = brain_side_map(ch);
    end
    
    % Include new columns in column_names
    column_names = {'Trial_index', 'Channel', 'Freq_band', 'Exponent', 'Offset'};
    
    % Update results_table initialization
    results_table = table('Size', [n_rows, numel(column_names)], 'VariableTypes', {'double', 'categorical',  'categorical', 'double', 'double'}, 'VariableNames', column_names);
    
    % Row index for inserting into results_table
    row_index = 1;
    
    % Loop over all trials, channels, and frequency bands
    for trialIdx = 1:n_trials
        for chanIdx = 1:n_channels
            for freqIdx = 1:n_freq_bands
                band = freq_bands{freqIdx, 1};
                fractal_data = results_fractal.(band).powspctrm(trialIdx, chanIdx, :);
    
                % Log-log transformation
                log_freqs = log10(results_fractal.(band).freq);
                log_power = log10(squeeze(fractal_data));
    
                % Linear regression
                linear_model = fitlm(log_freqs, log_power);
                exponent = linear_model.Coefficients.Estimate(2); % Slope
                offset = linear_model.Coefficients.Estimate(1); % Intercept
    
                % Store results
                results_table.Trial_index(row_index) = trialIdx;
                results_table.Channel(row_index) = categorical({data.label{chanIdx}});
                results_table.Freq_band(row_index) = categorical({band});
                results_table.Exponent(row_index) = exponent;
                results_table.Offset(row_index) = offset;
    
                % Increment row index
                row_index = row_index + 1;
            end
        end
    end

    % Calculate the total number of rows needed
    total_rows = n_trials * n_channels * n_freq_bands;
    
    % Expand brain_regions and brain_sides to match the total number of rows
    expanded_brain_regions = repmat(brain_regions, 1, total_rows / length(brain_regions));
    expanded_brain_sides = repmat(brain_sides, 1, total_rows / length(brain_sides));
    
    % Add Brain_region and Brain_Side to the results_table
    results_table.Brain_region = categorical(expanded_brain_regions');
    results_table.Brain_side = categorical(expanded_brain_sides');

    % Initialize the 'Condition' column in the results_table as a cell array
    results_table.Condition = cell(n_rows, 1);
    
    % Assign 'high' or 'low' to the 'Condition' column based on the trial index
    for row = 1:n_rows
        trialIdx = results_table.Trial_index(row);
        if trialIdx <= n_trials/2
            results_table.Condition{row} = 'high';
        else
            results_table.Condition{row} = 'low';
        end
    end
    
    % Convert the 'Condition' cell array to categorical
    results_table.Condition = categorical(results_table.Condition);

    % Rename the table to fractal_df
    fractal_df = results_table;

    % Convert categorical columns to cell arrays of character vectors
    fractal_df.Condition = cellstr(fractal_df.Condition);
    fractal_df.Brain_region = cellstr(fractal_df.Brain_region);
    fractal_df.Brain_side = cellstr(fractal_df.Brain_side);
    fractal_df.Freq_band = cellstr(fractal_df.Freq_band);
    fractal_df.Channel = cellstr(fractal_df.Channel);

    % Sort the final table by 'Trial_index'
    fractal_df = sortrows(fractal_df, 'Trial_index');

    % Convert Table to Cell Array with Headers (easier to load in python)
    headers = fractal_df.Properties.VariableNames;  % Extract column names
    fractal_df = [headers; table2cell(fractal_df)];  % Combine headers and data

    % Save the fractal_df as a .mat file
    output_filename = sprintf('hjorth_fractal_df_Subject_%d.mat', subnum);
    output_path = fullfile(output_directory, output_filename);
    % Save the file
    save(output_path, 'fractal_df');
end


