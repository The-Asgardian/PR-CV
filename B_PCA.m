scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'PR_CW_mat');
if ~isfolder(dataDir), dataDir = scriptDir; end
colPLA = [0 0.45 0.74];
colTPU = [0.85 0.33 0.10];
colRubber = [0.47 0.67 0.19];
P4_cols = (4-1)*3 + (1:3);  % columns 10:12 for P4 force

%% B.1 - Cylinder objects only (P4 force, 3D) - use _contact.mat if present (after A.2)
f = fullfile(dataDir, 'cylinder_papillarray_single_contact.mat'); if ~isfile(f), f = fullfile(dataDir, 'cylinder_papillarray_single.mat'); end
dPLA = load(f);
f = fullfile(dataDir, 'cylinder_TPU_papillarray_single_contact.mat'); if ~isfile(f), f = fullfile(dataDir, 'cylinder_TPU_papillarray_single.mat'); end
dTPU = load(f);
f = fullfile(dataDir, 'cylinder_rubber_papillarray_single_contact.mat'); if ~isfile(f), f = fullfile(dataDir, 'cylinder_rubber_papillarray_single.mat'); end
dRubber = load(f);
fn = getForceFn(dPLA);
X_cyl = [dPLA.(fn)(:, P4_cols); dTPU.(fn)(:, P4_cols); dRubber.(fn)(:, P4_cols)];
n1 = size(dPLA.(fn), 1); n2 = size(dTPU.(fn), 1); n3 = size(dRubber.(fn), 1);
labels_cyl = [ones(n1,1); 2*ones(n2,1); 3*ones(n3,1)];  % 1=PLA, 2=TPU, 3=Rubber

% Standardise and validate
X_cyl_std = zscoreLocal(X_cyl);
if isempty(X_cyl)
    error('X_cyl is empty. Check the data loading process.');
end
if size(X_cyl, 2) < 3
    error('X_cyl must have at least 3 columns for PCA.');
end

% PCA on standardised data (Centered false: already zscored)
[coeff_cyl, score_cyl, latent_cyl, ~, explained_cyl] = pcaLocal(X_cyl_std);
mu_cyl = mean(X_cyl_std, 1);

% B.1.a - Standardised 3D data with principal component vectors
figure('Name', 'B.1.a Cylinder P4 - 3D with PC vectors');
idx1 = labels_cyl == 1; idx2 = labels_cyl == 2; idx3 = labels_cyl == 3;
scatter3(X_cyl_std(idx1,1), X_cyl_std(idx1,2), X_cyl_std(idx1,3), 36, colPLA, 'filled'); hold on;
scatter3(X_cyl_std(idx2,1), X_cyl_std(idx2,2), X_cyl_std(idx2,3), 36, colTPU, 'filled');
scatter3(X_cyl_std(idx3,1), X_cyl_std(idx3,2), X_cyl_std(idx3,3), 36, colRubber, 'filled');
scale = 2;
for i = 1:3
    quiver3(mu_cyl(1), mu_cyl(2), mu_cyl(3), scale*coeff_cyl(1,i), scale*coeff_cyl(2,i), scale*coeff_cyl(3,i), 'k', 'LineWidth', 2);
end
xlabel('F_X (std)'); ylabel('F_Y (std)'); zlabel('F_Z (std)');
title('B.1.a Cylinder P4 force - standardised with PC directions');
legend('PLA', 'TPU', 'Rubber', 'Location', 'best'); grid on; hold off;

% B.1.b - Reduce to 2D and replot
figure('Name', 'B.1.b Cylinder P4 - 2D PCA');
scatter(score_cyl(idx1,1), score_cyl(idx1,2), 36, colPLA, 'filled'); hold on;
scatter(score_cyl(idx2,1), score_cyl(idx2,2), 36, colTPU, 'filled');
scatter(score_cyl(idx3,1), score_cyl(idx3,2), 36, colRubber, 'filled');
xlabel('PC1'); ylabel('PC2'); title('B.1.b Cylinder P4 - 2D PCA');
legend('PLA', 'TPU', 'Rubber'); grid on; hold off;

% B.1.c - 1D number lines per PC
figure('Name', 'B.1.c Cylinder P4 - 1D number lines');
for pc = 1:3
    subplot(3,1,pc);
    hold on;
    scatter(score_cyl(idx1,pc), zeros(n1,1), 36, colPLA, 'filled');
    scatter(score_cyl(idx2,pc), zeros(n2,1), 36, colTPU, 'filled');
    scatter(score_cyl(idx3,pc), zeros(n3,1), 36, colRubber, 'filled');
    xlabel(['PC' num2str(pc)]); yticks([]); title(['PC' num2str(pc) ' number line']);
    hold off;
end
sgtitle('B.1.c Cylinder P4 - distribution across PCs');

%% B.2 - Oblong objects only (same as B.1)
f = fullfile(dataDir, 'oblong_papillarray_single_contact.mat'); if ~isfile(f), f = fullfile(dataDir, 'oblong_papillarray_single.mat'); end
dPLA = load(f);
f = fullfile(dataDir, 'oblong_TPU_papillarray_single_contact.mat'); if ~isfile(f), f = fullfile(dataDir, 'oblong_TPU_papillarray_single.mat'); end
dTPU = load(f);
f = fullfile(dataDir, 'oblong_rubber_papillarray_single_contact.mat'); if ~isfile(f), f = fullfile(dataDir, 'oblong_rubber_papillarray_single.mat'); end
dRubber = load(f);
fn = getForceFn(dPLA);
X_obl = [dPLA.(fn)(:, P4_cols); dTPU.(fn)(:, P4_cols); dRubber.(fn)(:, P4_cols)];
n1 = size(dPLA.(fn), 1); n2 = size(dTPU.(fn), 1); n3 = size(dRubber.(fn), 1);
labels_obl = [ones(n1,1); 2*ones(n2,1); 3*ones(n3,1)];

X_obl_std = zscoreLocal(X_obl);
[coeff_obl, score_obl, latent_obl, ~, ~] = pcaLocal(X_obl_std);
mu_obl = mean(X_obl_std, 1);

figure('Name', 'B.2.a Oblong P4 - 3D with PC vectors');
idx1 = labels_obl == 1; idx2 = labels_obl == 2; idx3 = labels_obl == 3;
scatter3(X_obl_std(idx1,1), X_obl_std(idx1,2), X_obl_std(idx1,3), 36, colPLA, 'filled'); hold on;
scatter3(X_obl_std(idx2,1), X_obl_std(idx2,2), X_obl_std(idx2,3), 36, colTPU, 'filled');
scatter3(X_obl_std(idx3,1), X_obl_std(idx3,2), X_obl_std(idx3,3), 36, colRubber, 'filled');
for i = 1:3
    quiver3(mu_obl(1), mu_obl(2), mu_obl(3), 2*coeff_obl(1,i), 2*coeff_obl(2,i), 2*coeff_obl(3,i), 'k', 'LineWidth', 2);
end
xlabel('F_X (std)'); ylabel('F_Y (std)'); zlabel('F_Z (std)');
title('B.2.a Oblong P4 force - standardised with PC directions');
legend('PLA', 'TPU', 'Rubber'); grid on; hold off;

figure('Name', 'B.2.b Oblong P4 - 2D PCA');
scatter(score_obl(idx1,1), score_obl(idx1,2), 36, colPLA, 'filled'); hold on;
scatter(score_obl(idx2,1), score_obl(idx2,2), 36, colTPU, 'filled');
scatter(score_obl(idx3,1), score_obl(idx3,2), 36, colRubber, 'filled');
xlabel('PC1'); ylabel('PC2'); title('B.2.b Oblong P4 - 2D PCA');
legend('PLA', 'TPU', 'Rubber'); grid on; hold off;

figure('Name', 'B.2.c Oblong P4 - 1D number lines');
for pc = 1:3
    subplot(3,1,pc);
    hold on;
    scatter(score_obl(idx1,pc), zeros(n1,1), 36, colPLA, 'filled');
    scatter(score_obl(idx2,pc), zeros(n2,1), 36, colTPU, 'filled');
    scatter(score_obl(idx3,pc), zeros(n3,1), 36, colRubber, 'filled');
    xlabel(['PC' num2str(pc)]); yticks([]); title(['PC' num2str(pc) ' number line']);
    hold off;
end
sgtitle('B.2.c Oblong P4 - distribution across PCs');

%% B.3 - All 9 papillae force, all 9 objects
% Load all 9 files; build 27-D force per contact; PCA to 2D; one plot per shape; scree for cylinder and oblong
objNames = {'cylinder', 'oblong', 'hexagon'};
mats = {'_papillarray_single', '_TPU_papillarray_single', '_rubber_papillarray_single'};
allData = cell(3, 3);  % shape x material
for sh = 1:3
    for m = 1:3
        f = fullfile(dataDir, [objNames{sh} mats{m} '_contact.mat']);
        if ~isfile(f), f = fullfile(dataDir, [objNames{sh} mats{m} '.mat']); end
        d = load(f);
        fn = getForceFn(d);
        allData{sh, m} = d.(fn);  % N x 27
    end
end

% B.3.a - 2D PCA plot per object shape (cylinder, oblong, hexagon)
for sh = 1:3
    X = [allData{sh,1}; allData{sh,2}; allData{sh,3}];
    n1 = size(allData{sh,1},1); n2 = size(allData{sh,2},1); n3 = size(allData{sh,3},1);
    labels = [ones(n1,1); 2*ones(n2,1); 3*ones(n3,1)];
    X_std = zscoreLocal(X);
    [~, score_sh, ~] = pcaLocal(X_std, 'NumComponents', 2);
    idx1 = labels == 1; idx2 = labels == 2; idx3 = labels == 3;
    figure('Name', ['B.3.a ' objNames{sh} ' all papillae 2D PCA']);
    scatter(score_sh(idx1,1), score_sh(idx1,2), 36, colPLA, 'filled'); hold on;
    scatter(score_sh(idx2,1), score_sh(idx2,2), 36, colTPU, 'filled');
    scatter(score_sh(idx3,1), score_sh(idx3,2), 36, colRubber, 'filled');
    xlabel('PC1'); ylabel('PC2'); title(['B.3.a ' objNames{sh} ' - all 9 papillae force, 2D PCA']);
    legend('PLA', 'TPU', 'Rubber'); grid on; hold off;
end

% B.3.b - Scree plots for cylinder and oblong (all 9 papillae)
for sh = 1:2  % cylinder, oblong
    X = [allData{sh,1}; allData{sh,2}; allData{sh,3}];
    X_std = zscoreLocal(X);
    [~, ~, latent_sh] = pcaLocal(X_std);
    figure('Name', ['B.3.b Scree ' objNames{sh}]);
    bar(latent_sh); xlabel('Principal component'); ylabel('Variance (eigenvalue)');
    title(['B.3.b Scree plot - ' objNames{sh} ' (all 9 papillae)']);
end

%% Local helpers (no Statistics Toolbox required)
function X_std = zscoreLocal(X)
    % Column-wise standardisation: mean 0, std 1 (same as zscore).
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma == 0) = 1;  % avoid division by zero for constant columns
    X_std = (X - mu) ./ sigma;
end

function [coeff, score, latent, tsq, explained] = pcaLocal(X, varargin)
    % PCA via SVD (no Statistics Toolbox). Matches pca(..., 'Centered', false).
    % Optional: 'NumComponents', k to return first k components only.
    tsq = [];  % 4th output (Hotelling T-squared) not computed
    numComp = size(X, 2);
    for i = 1:2:length(varargin)
        if strcmpi(varargin{i}, 'NumComponents')
            numComp = min(varargin{i+1}, size(X, 2));
        end
    end
    [U, S, V] = svd(X, 'econ');
    n = size(X, 1);
    s = diag(S);
    latent = (s.^2) / (n - 1);  % variance per PC
    if numComp < length(latent)
        coeff = V(:, 1:numComp);
        score = X * coeff;
        latent = latent(1:numComp);
    else
        coeff = V;
        score = U * S;
    end
    explained = 100 * latent / sum(latent);
end

function fn = getForceFn(d)
    fn = 'sensor_matricies_force';
    if ~isfield(d, fn), fn = 'sensor_matrices_force'; end
    if ~isfield(d, fn)
        fns = fieldnames(d);
        for k = 1:length(fns)
            if contains(lower(fns{k}), 'force'), fn = fns{k}; return; end
        end
    end
end
