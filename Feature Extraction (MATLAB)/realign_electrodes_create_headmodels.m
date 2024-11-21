% HEADMODELS REFTEP++

%% Preparation

% Set paths to relevant toolboxes
TOOLBOXPATH = '';
addpath(fullfile(TOOLBOXPATH, 'eeglab2023.1'));
addpath(fullfile(TOOLBOXPATH, 'fieldtrip-20230716'));
addpath(fullfile(TOOLBOXPATH, 'hbf_distribution_open_v170624'));
addpath(fullfile(TOOLBOXPATH, 'plotroutines_v180921'));
addpath(fullfile(TOOLBOXPATH, 'plotroutines_v170706'));
addpath(genpath(''));

ft_defaults;

% Define input and output directories
input_directory = '';
savePath = '';

dataTable = readtable(fullfile(input_directory, 'subject_list.xlsx'), 'Basic', 1);

subjects = [18:29]; %select subjects with individual headmodels (18-29)


% Feature Extraction
for subnum = subjects
    fprintf(' ProcessingSubject %d ', subnum)

    dataTableSub = dataTable(subnum, :);
    load(char(dataTableSub.headmodel))

    % Define the base path
    baseFolder = sprintf('', subnum);

    % Find the unique identifier folder
    reftepDirs = dir([baseFolder '*']);
    reftepDirs = reftepDirs([reftepDirs.isdir]);

    if isempty(reftepDirs)
        warning('No unique identifier directory found for subject %d. Skipping...', subnum);
        %continue;
    end

    % Assuming there's only one unique identifier directory per subject
    uniqueFolder = reftepDirs(1).name;
    sessionBaseFolder = fullfile('', uniqueFolder, 'Sessions');

    % Get session folder
    sessionDirs = dir([sessionBaseFolder filesep 'Session_*']);
    sessionDirs = sessionDirs([sessionDirs.isdir]);

    if isempty(sessionDirs)
        warning('No session directory found for subject %d. Skipping...', subnum);
        %continue;
    end

    % Assuming there's only one session directory
    sessionFolder = sessionDirs(1).name;
    sourcedataFolder = fullfile(sessionBaseFolder, sessionFolder);
    files = dir([sourcedataFolder filesep 'EEG']);
    idx = contains({files.name}, 'EEGMarkers') & ~contains({files.name}, '._');
    markers = files(idx);
    [~, maxidx] = max([markers.datenum]');
    eegmarkersFile = markers( maxidx ).name;
    eegmarkersPath = [sourcedataFolder filesep  'EEG' filesep eegmarkersFile];

    % Load electrode locations
    elec = ft_read_sens(eegmarkersPath); %RAS coordinates (from original T1 nii)
    
    % Get bmeshes and smesh
    bmeshes={headmodel.bmeshes(1) headmodel.bmeshes(2) headmodel.bmeshes(3)};
    subj_smesh=headmodel.smesh;
   
    % % Extracting NAS, LTR, and RTR positions from elec.chanpos
    % NAS = elec.chanpos(1, :);
    % RTR = elec.chanpos(2, :);
    % LTR = elec.chanpos(3, :);

    %% APPROXIMATE THE FIDUCIALS 
    % NAS
    hold on
    PlotMeshes(bmeshes(3),'facealpha',1,'facecolor',[.9 .8 .6], 'view', [-180 0]);
    if exist('NAS'); PlotPoints(NAS,'r.',20); end
    if exist('RTR'); PlotPoints(NAS,'r.',20); end
    if exist('LTR'); PlotPoints(NAS,'r.',20); end
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3 0.3 0.3 0.3]);
    datacursormode on
    dcm_obj = datacursormode(gcf);
    set(dcm_obj,'DisplayStyle','datatip',...
        'SnapToDataVertex','off','Enable','on')
    disp('Click line to display a data tip, then press Return.')
    % Wait while the user does this.
    pause 
    c_info = getCursorInfo(dcm_obj);
    NAS=c_info.Position;
    close gcf
    
    %% RTR
    hold on
    PlotMeshes(bmeshes(3),'facealpha',1,'facecolor',[.9 .8 .6], 'view', [90 0]);
    if exist('NAS'); PlotPoints(NAS,'r.',20); end
    if exist('RTR'); PlotPoints(NAS,'r.',20); end
    if exist('LTR'); PlotPoints(NAS,'r.',20); end
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3 0.3 0.3 0.3]);
    datacursormode on
    dcm_obj = datacursormode(gcf);
    set(dcm_obj,'DisplayStyle','datatip',...
        'SnapToDataVertex','off','Enable','on')
    disp('Click line to display a data tip, then press Return.')
    % Wait while the user does this.
    pause 
    c_info = getCursorInfo(dcm_obj);
    RTR=c_info.Position;
    close gcf
    
    %% LTR
    hold on
    PlotMeshes(bmeshes(3),'facealpha',1,'facecolor',[.9 .8 .6], 'view', [-90 0]);
    if exist('NAS'); PlotPoints(NAS,'r.',20); end
    if exist('RTR'); PlotPoints(NAS,'r.',20); end
    if exist('LTR'); PlotPoints(NAS,'r.',20); end
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3 0.3 0.3 0.3]);
    datacursormode on
    dcm_obj = datacursormode(gcf);
    set(dcm_obj,'DisplayStyle','datatip',...
        'SnapToDataVertex','off','Enable','on')
    disp('Click line to display a data tip, then press Return.')
    % Wait while the user does this.
    pause 
    c_info = getCursorInfo(dcm_obj);
    LTR=c_info.Position;
    close gcf
    
    %% Plot the approximated fiducials
    hold on
    PlotMeshes(bmeshes(3),'facealpha',0.7,'facecolor',[.9 .8 .6], 'view', [-180 0]);
    PlotPoints(NAS,'r.',50);
    PlotPoints(LTR,'b.',50);
    PlotPoints(RTR,'g.',50);
    rotate3d
    
    % Define fiducials structure
    fid.elecpos = [NAS; LTR; RTR];
    fid.label = {'NAS', 'LTR', 'RTR'};
    fid.unit = 'mm';
    
    % Configuration for realigning
    cfg = [];
    cfg.method = 'fiducial';
    cfg.target = fid;
    cfg.elec = elec;
    cfg.fiducial = {'NAS', 'LTR', 'RTR'};
    
    % Realign electrodes
    elec_realigned = ft_electroderealign(cfg);
    
    % Remove fiducials from the realigned structure
    elec_realigned.chanpos = elec_realigned.chanpos(~ismember(elec_realigned.label, cfg.fiducial), :);
    elec_realigned.chantype = elec_realigned.chantype(~ismember(elec_realigned.label, cfg.fiducial), :);
    elec_realigned.chanunit = elec_realigned.chanunit(~ismember(elec_realigned.label, cfg.fiducial), :);
    elec_realigned.elecpos = elec_realigned.elecpos(~ismember(elec_realigned.label, cfg.fiducial), :);
    elec_realigned.label = elec_realigned.label(~ismember(elec_realigned.label, cfg.fiducial));

    % % Visualize realigned electrodes
    clf;hold on;
    PlotMeshes(bmeshes);
    PlotMesh(subj_smesh,'facealpha',0.9,'facecolor',[.6 .6 .6], 'edgecolor','black', 'edgealpha', 0.4);
    PlotPoints(elec_realigned.elecpos,'b.',25);
    PlotPoints(elec_realigned.elecpos,'r.',15);
    rotate3d
    close all;

    % Save realigned electrodes in headmodel
    headmodel.elec_ft = elec_realigned;
    headmodel.elec_ft = ft_convert_units(headmodel.elec_ft, 'm'); %change units to m 
    headmodel.elec_ft.label(strcmp(headmodel.elec_ft.label, 'REF')) = {'FCz'};
    headmodel.refchannel = 'FCz';

    % Create ordered layout for hbf projection
    cfg = [];
    cfg.channel = elec_realigned.label;
    cfg.layout = 'ordered';
    elec_realigned2 = ft_prepare_layout(cfg, elec_realigned);
    elec_realigned2.elecpos = elec_realigned.elecpos;
    elecs.porig = elec_realigned2.elecpos;
    elecs.name = elec_realigned2.label';

    %% Make a 3-shell BEM model and calculate lead field using the hbf toolbox

    assert(strcmp(headmodel.elec_ft.unit, 'm'), 'Units of electrode positions in sensor array structure must be ''m'', you can use ft_convert_units to convert');
    
    headmodel.hbf_elecs = hbf_ProjectElectrodesToScalp(elecs.porig, bmeshes);
    ci = [1 1/80 1] *.33; %conductivities
    co = [1/80 1 0] *.33;
    D = hbf_BEMOperatorsPhi_LC(bmeshes);
    Tphi_full = hbf_TM_Phi_LC_ISA2(D, ci, co, 1);
    Tphi_elecs = hbf_InterpolateTfullToElectrodes(Tphi_full, bmeshes, headmodel.hbf_elecs);
    headmodel.smesh.nn = CalcNodeNormals(headmodel.smesh); % direction of dipoles normal to surface mesh
    LFMphi_dir = hbf_LFM_LC(bmeshes, Tphi_elecs, headmodel.smesh.p, headmodel.smesh.nn); % leadfield for normally oriented sources
    LFMphi_xyz = hbf_LFM_LC(bmeshes, Tphi_elecs, headmodel.smesh.p);
    
    headmodel.leadfield = LFMphi_dir;
    headmodel.label = elecs.name;
    
    clear('bmeshes3', 'ci', 'co', 'D', 'Tphi_full', 'Tphi_elecs', 'LFMphi_dir', 'LFMphi_xyz');

    % % Visualize the projected electrodes on the scalp
    % close all
    % hold on;
    % PlotMeshes(bmeshes, 'facealpha',0.4, 'view', [180 0]);
    % PlotMesh(subj_smesh,'facealpha',0.4,'facecolor',[.6 .6 .6], 'edgecolor','black', 'edgealpha', 0.4, 'view', [180 0]);
    % PlotPoints(headmodel.hbf_elecs.porig,'b.',35);
    % PlotPoints(headmodel.hbf_elecs.pproj,'r.',35);
    % rotate3d

    % % % Sanity check
    [LnrN]=prepareLFM(headmodel, 1, []); 
    sensitivity_profile=zeros(size(headmodel.leadfield,1),1);
    sensitivity_profile(ismember(headmodel.label, 'C3'))=1;
    PlotDataOnMesh(headmodel.smesh,sensitivity_profile'*LnrN,'colormap', jet ,'colorbar', 0, 'view', [-90 50]);

    % Save the headmodel
    fileName = sprintf('headmodel_REFTEP_%03d', subnum);
    fullFilePath = fullfile(savePath, fileName);
    save(fullFilePath, 'headmodel');
    fprintf('Saved headmodel for Subject %d\n', subnum);

end

fprintf('All subjects processed.\n');




    

    