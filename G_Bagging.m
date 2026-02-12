%% G. Bagging - Displacement from all 9 papillae, PCA then bootstrap aggregation
% Run A_DataPreparation_Visualisation.m and B_PCA.m context (same PCA pipeline on displacement).
% Class = object type (9 classes: cylinder/oblong/hexagon x PLA/TPU/rubber).
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'PR_CW_mat');

objNames = {'cylinder', 'oblong', 'hexagon'};
mats = {'_papillarray_single', '_TPU_papillarray_single', '_rubber_papillarray_single'};

% Resolve displacement variable name (appendix typo: sensor_matricies_displacement)
d0 = load(fullfile(dataDir, [objNames{1} mats{1} '_contact.mat']));
fn_disp = 'sensor_matrices_displacement'; if ~isfield(d0, fn_disp), fn_disp = 'sensor_matricies_displacement'; end

% Load displacement data (27-D) for all 9 objects
allDisp = cell(3, 3);
classLabels = [];
for sh = 1:3
    for m = 1:3
        f = fullfile(dataDir, [objNames{sh} mats{m} '_contact.mat']);
        if ~isfile(f), f = fullfile(dataDir, [objNames{sh} mats{m} '.mat']); end
        d = load(f);
        if ~isfield(d, fn_disp), fn_disp = 'sensor_matricies_displacement'; end
        allDisp{sh, m} = d.(fn_disp);
        classLabels = [classLabels; (sh-1)*3 + m * ones(size(d.(fn_disp),1), 1)];
    end
end

X_full = [allDisp{1,1}; allDisp{1,2}; allDisp{1,3}; allDisp{2,1}; allDisp{2,2}; allDisp{2,3}; allDisp{3,1}; allDisp{3,2}; allDisp{3,3}];

% PCA on displacement (same as B: standardise then PCA to 2D for consistency with B.3)
X_std = (X_full - mean(X_full, 1)) ./ (std(X_full, 0, 1) + 1e-10);
[U, S, V] = svd(X_std, 'econ');
X_pca = X_std * V(:, 1:2);  % 2D PCA projection

% Train/test split (70% train, 30% test) stratified by class
rng(42);
n = size(X_pca, 1);
if exist('cvpartition', 'file') == 2
    cv = cvpartition(classLabels, 'HoldOut', 0.3);
    idxTrain = training(cv); idxTest = test(cv);
else
    idxTrain = false(n, 1); idxTest = false(n, 1);
    for c = 1:9
        ic = find(classLabels == c);
        nc = length(ic);
        perm = ic(randperm(nc));
        nTrain = max(1, round(0.7 * nc));
        idxTrain(perm(1:nTrain)) = true;
        idxTest(perm(nTrain+1:end)) = true;
    end
end
X_train = X_pca(idxTrain, :); Y_train = classLabels(idxTrain);
X_test = X_pca(idxTest, :); Y_test = classLabels(idxTest);

%% G.1.a Number of bags
numTrees = 50;  % Choose 50 bags: balance between variance reduction and compute; often 50-200 for small data.
fprintf('G.1.a Using %d bags (trees). Reason: sufficient for variance reduction with small dataset.\n', numTrees);

%% Train bagged ensemble
mdl = TreeBagger(numTrees, X_train, Y_train, 'Method', 'classification', 'OOBPredict', 'on');

%% G.1.b Visualise two of the generated decision trees
figure('Name', 'G.1.b Tree 1');
view(mdl.Trees{1}, 'Mode', 'graph');
title('G.1.b First decision tree');

figure('Name', 'G.1.b Tree 2');
view(mdl.Trees{2}, 'Mode', 'graph');
title('G.1.b Second decision tree');

%% G.1.c Test data: predictions and confusion matrix
Y_pred = str2double(predict(mdl, X_test));
% If predict returns cell array of strings
if iscell(Y_pred), Y_pred = cellfun(@str2double, Y_pred); end

figure('Name', 'G.1.c Confusion matrix');
confMat = confusionmat(Y_test, Y_pred);
confusionchart(confMat, {'Cyl PLA', 'Cyl TPU', 'Cyl Rub', 'Obl PLA', 'Obl TPU', 'Obl Rub', 'Hex PLA', 'Hex TPU', 'Hex Rub'});
title('G.1.c Confusion matrix (object type)');

acc = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('G.1.c Overall accuracy (test): %.2f%%\n', 100*acc);
