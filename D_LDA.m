%% D. Linear Discriminant Analysis - Oblong TPU vs Oblong Rubber (central papillae displacement)
% Run A_DataPreparation_Visualisation.m first.
% Uses built-in LDA (fitcdiscr) if Statistics TB available; else local implementation.
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'PR_CW_mat');
colTPU = [0.85 0.33 0.10];
colRubber = [0.47 0.67 0.19];
P4_cols = (4-1)*3 + (1:3);  % D_X, D_Y, D_Z for P4

%% D.1.a Load Oblong TPU and Oblong Rubber
fTPU = fullfile(dataDir, 'oblong_TPU_papillarray_single_contact.mat'); if ~isfile(fTPU), fTPU = fullfile(dataDir, 'oblong_TPU_papillarray_single.mat'); end
fRubber = fullfile(dataDir, 'oblong_rubber_papillarray_single_contact.mat'); if ~isfile(fRubber), fRubber = fullfile(dataDir, 'oblong_rubber_papillarray_single.mat'); end
dTPU = load(fTPU); dRubber = load(fRubber);
fn = 'sensor_matrices_displacement'; if ~isfield(dTPU, fn), fn = 'sensor_matricies_displacement'; end

X_TPU = dTPU.(fn)(:, P4_cols);
X_Rubber = dRubber.(fn)(:, P4_cols);
X = [X_TPU; X_Rubber];
Y = [ones(size(X_TPU,1),1); 2*ones(size(X_Rubber,1),1)];  % 1 = TPU, 2 = Rubber
n1 = size(X_TPU,1); n2 = size(X_Rubber,1);

%% D.1.b 3D scatter of tactile displacement (D_X, D_Y, D_Z)
figure('Name', 'D.1.b Oblong TPU vs Rubber displacement');
scatter3(X_TPU(:,1), X_TPU(:,2), X_TPU(:,3), 36, colTPU, 'filled'); hold on;
scatter3(X_Rubber(:,1), X_Rubber(:,2), X_Rubber(:,3), 36, colRubber, 'filled');
xlabel('D_X'); ylabel('D_Y'); zlabel('D_Z');
title('D.1.b Central papilla displacement: Oblong TPU vs Rubber');
legend('TPU', 'Rubber'); grid on; view(3); hold off;

%% LDA: use fitcdiscr if available, else local 2-class LDA
useFitcdiscr = exist('fitcdiscr', 'file') == 2;

%% D.1.c LDA on all 2D combinations (D_X,D_Y), (D_X,D_Z), (D_Y,D_Z)
pairs = [1 2; 1 3; 2 3];
pairNames = {'D_X, D_Y', 'D_X, D_Z', 'D_Y, D_Z'};
for p = 1:3
    idx = pairs(p,:);
    X2 = X(:, idx);
    if useFitcdiscr
        mdl = fitcdiscr(X2, Y);
        w = mdl.Coeffs(1,2).Linear;
        c = mdl.Coeffs(1,2).Const;
    else
        [w, c] = lda2class(X2(Y==1,:), X2(Y==2,:));
    end
    score2 = X2 * w;
    thresh = -c;  % boundary where score + c = 0
    figure('Name', ['D.1.c LDA ' pairNames{p}]);
    scatter(score2(Y==1), zeros(n1,1), 36, colTPU, 'filled'); hold on;
    scatter(score2(Y==2), zeros(n2,1), 36, colRubber, 'filled');
    xline(thresh, 'k--', 'LineWidth', 1.5);
    xlabel('LD score'); title(['D.1.c LDA on ' pairNames{p}]);
    legend('TPU', 'Rubber', 'Boundary'); grid on; hold off;
end

%% D.1.d LDA on 3D displacement
if useFitcdiscr
    mdl3 = fitcdiscr(X, Y);
    w = mdl3.Coeffs(1,2).Linear;
    c = mdl3.Coeffs(1,2).Const;
else
    [w, c] = lda2class(X_TPU, X_Rubber);
end
score1D = X * w;
Xc = X - mean(X, 1);
[~, ~, V] = svd(Xc, 'econ');
proj2 = Xc * V(:, 2);

% D.1.d.i 2D re-plot with LD and discrimination line
thresh = -c;
figure('Name', 'D.1.d.i LDA 3D reduced to 2D');
scatter(score1D(Y==1), proj2(Y==1), 36, colTPU, 'filled'); hold on;
scatter(score1D(Y==2), proj2(Y==2), 36, colRubber, 'filled');
xline(thresh, 'k-', 'LineWidth', 2);
xlabel('LD (discriminant direction)'); ylabel('PC2 (for display)');
title('D.1.d.i LDA 3D -> 2D with discrimination line');
legend('TPU', 'Rubber', 'Boundary'); grid on; hold off;

% D.1.d.ii 3D plot with discrimination plane: w'*x + c = 0 => z = -(w(1)*x + w(2)*y + c)/w(3)
xr = [min(X(:,1)), max(X(:,1))]; yr = [min(X(:,2)), max(X(:,2))];
[xx, yy] = meshgrid(linspace(xr(1), xr(2), 20), linspace(yr(1), yr(2), 20));
zz = -(w(1)*xx + w(2)*yy + c) / (w(3) + 1e-10);

figure('Name', 'D.1.d.ii LDA 3D with discrimination plane');
scatter3(X_TPU(:,1), X_TPU(:,2), X_TPU(:,3), 36, colTPU, 'filled'); hold on;
scatter3(X_Rubber(:,1), X_Rubber(:,2), X_Rubber(:,3), 36, colRubber, 'filled');
surf(xx, yy, zz, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'k');
xlabel('D_X'); ylabel('D_Y'); zlabel('D_Z');
title('D.1.d.ii 3D displacement with LDA discrimination plane');
legend('TPU', 'Rubber', 'Boundary plane'); view(3); grid on; hold off;

%% Local 2-class LDA (no Statistics Toolbox): w = Sw \ (m1-m2)', decision: w'*x + c > 0 => class 1
function [w, c] = lda2class(X1, X2)
    m1 = mean(X1, 1)';
    m2 = mean(X2, 1)';
    S1 = (X1 - m1')' * (X1 - m1');
    S2 = (X2 - m2')' * (X2 - m2');
    Sw = S1 + S2 + 1e-6 * eye(size(S1,1));
    w = Sw \ (m1 - m2);
    % Midpoint on the line: (m1+m2)/2; threshold so that 0.5*(w'*m1 + w'*m2) + c = 0
    c = -w' * (m1 + m2) / 2;
end
