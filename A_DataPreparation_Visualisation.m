%% Path setup (cross-OS: fullfile uses / or \ as appropriate)
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'PR_CW_mat');
if ~isfolder(dataDir)
    dataDir = scriptDir;
end

%% Colour convention for materials (reuse in report)
% PLA = blue, TPU = red, rubber = green
colPLA = [0 0.45 0.74];
colTPU = [0.85 0.33 0.10];
colRubber = [0.47 0.67 0.19];

%% A.1 - End-effector trajectory for cylinder PLA and hexagon PLA
% Dataset .mat files contain end_effector_poses (position [1:3], orientation [4:6]).
rawCyl = fullfile(dataDir, 'cylinder_papillarray_single.mat');
rawHex = fullfile(dataDir, 'hexagon_papillarray_single.mat');
if isfile(rawCyl) && isfile(rawHex)
    d1 = load(rawCyl);
    d2 = load(rawHex);
    pos1 = d1.end_effector_poses(:, 1:3);
    pos2 = d2.end_effector_poses(:, 1:3);
    figure('Name', 'A.1 Trajectories');
    plot3(pos1(:,1), pos1(:,2), pos1(:,3), 'b-', 'DisplayName', 'Cylinder PLA'); hold on;
    plot3(pos2(:,1), pos2(:,2), pos2(:,3), 'r-', 'DisplayName', 'Hexagon PLA');
    xlabel('X'); ylabel('Y'); zlabel('Z'); title('End-effector trajectory: Cylinder vs Hexagon PLA');
    legend; grid on; hold off;
    saveas(gcf, fullfile(dataDir, 'A1_trajectories.fig'));
else
    disp('A.1 skipped: cylinder_papillarray_single.mat and hexagon_papillarray_single.mat not found.');
end

%% A.2 - Segment data: find peaks of normal contact force, extract per-contact data
% ft_values: Nx6 (forces 1:3, torques 4:6). Normal force = Z (column 3).
% Saves extracted data to *_contact.mat; B-G load _contact.mat if present, else _single.mat.
trialFiles = {'cylinder_papillarray_single.mat', 'cylinder_TPU_papillarray_single.mat', 'cylinder_rubber_papillarray_single.mat', ...
    'oblong_papillarray_single.mat', 'oblong_TPU_papillarray_single.mat', 'oblong_rubber_papillarray_single.mat', ...
    'hexagon_papillarray_single.mat', 'hexagon_TPU_papillarray_single.mat', 'hexagon_rubber_papillarray_single.mat'};
for i = 1:length(trialFiles)
    f = fullfile(dataDir, trialFiles{i});
    if ~isfile(f), continue; end
    d = load(f);
    if ~isfield(d, 'ft_values'), continue; end
    ft = d.ft_values;
    normalForce = -ft(:, 3); %Z direction reversed
    [pks, locs] = findpeaks(normalForce, 'MinPeakHeight', max(normalForce)*0.85, 'MinPeakDistance', 50);
    fn_force = 'sensor_matrices_force'; if ~isfield(d, fn_force), fn_force = 'sensor_matricies_force'; end
    fn_disp = 'sensor_matrices_displacement'; if ~isfield(d, fn_disp), fn_disp = 'sensor_matricies_displacement'; end
    sensor_matrices_force = d.(fn_force)(locs, :);
    sensor_matrices_displacement = d.(fn_disp)(locs, :);
    figure('Name', 'A.2 Peaks');
    plot(1:length(normalForce), normalForce, 'b-'); hold on;
    plot(locs, pks, 'ro', 'MarkerSize', 8);
    xlabel('Time index'); ylabel('Normal force'); title(['Peaks: ' trialFiles{i}]);
    hold off;
    outName = strrep(trialFiles{i}, '.mat', '_contact.mat');
    save(fullfile(dataDir, outName), 'sensor_matrices_force', 'sensor_matrices_displacement', 'pks', 'locs');
end

%% A.3 - 3D scatter of P4 force for cylinder, oblong, hexagon (3 materials each)
% Middle papillae P4 = 4th papillae -> columns 10:12 (1-based). Use _contact.mat if present (after A.2).
P4_cols = (4-1)*3 + (1:3);

shapes = {'cylinder', 'oblong', 'hexagon'};
for s = 1:3
    shape = shapes{s};
    fPLA = fullfile(dataDir, [shape '_papillarray_single_contact.mat']);
    if ~isfile(fPLA), fPLA = fullfile(dataDir, [shape '_papillarray_single.mat']); end
    fTPU = fullfile(dataDir, [shape '_TPU_papillarray_single_contact.mat']);
    if ~isfile(fTPU), fTPU = fullfile(dataDir, [shape '_TPU_papillarray_single.mat']); end
    fRubber = fullfile(dataDir, [shape '_rubber_papillarray_single_contact.mat']);
    if ~isfile(fRubber), fRubber = fullfile(dataDir, [shape '_rubber_papillarray_single.mat']); end
    if ~isfile(fPLA), continue; end
    dPLA = load(fPLA);
    dTPU = load(fTPU);
    dRubber = load(fRubber);
    % Variable name from appendix: sensor_matricies_force (typo); fallback sensor_matrices_force
    fn = 'sensor_matricies_force';
    if ~isfield(dPLA, fn), fn = 'sensor_matrices_force'; end
    if ~isfield(dPLA, fn)
        fns = fieldnames(dPLA);
        for k = 1:length(fns), if contains(lower(fns{k}), 'force'), fn = fns{k}; break; end, end
    end
    F_PLA = dPLA.(fn)(:, P4_cols);
    F_TPU = dTPU.(fn)(:, P4_cols);
    F_Rubber = dRubber.(fn)(:, P4_cols);
    figure('Name', ['A.3 P4 force ' shape]);
    scatter3(F_PLA(:,1), F_PLA(:,2), F_PLA(:,3), 36, colPLA, 'filled', 'DisplayName', 'PLA'); hold on;
    scatter3(F_TPU(:,1), F_TPU(:,2), F_TPU(:,3), 36, colTPU, 'filled', 'DisplayName', 'TPU');
    scatter3(F_Rubber(:,1), F_Rubber(:,2), F_Rubber(:,3), 36, colRubber, 'filled', 'DisplayName', 'Rubber');
    xlabel('F_X'); ylabel('F_Y'); zlabel('F_Z');
    title(['P4 force - ' shape ' (PLA, TPU, Rubber)']);
    legend; grid on; hold off;
    saveas(gcf, fullfile(dataDir, ['A3_P4_force_' shape '.fig']));
end
