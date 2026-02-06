clc; clear;
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'PR_CW_mat');

f = fullfile(dataDir, 'hexagon_papillarray_single_contact.mat');
dPLA = load(f);
f = fullfile(dataDir, 'hexagon_TPU_papillarray_single_contact.mat');
dTPU = load(f);
f = fullfile(dataDir, 'hexagon_rubber_papillarray_single_contact.mat');
dRubber = load(f);

fn_disp = 'sensor_matrices_displacement';

n1 = size(dPLA.(fn_disp), 1); 
n2 = size(dTPU.(fn_disp), 1); 
n3 = size(dRubber.(fn_disp), 1);

%Dx Vs Dz
X_disp = [dPLA.(fn_disp)(:, [13, 15]); ...
          dTPU.(fn_disp)(:, [13, 15]); ...
          dRubber.(fn_disp)(:, [13, 15])];

X_disp = (X_disp - mean(X_disp)) ./ std(X_disp);

labels = [ones(n1,1); 2*ones(n2,1); 3*ones(n3,1)];

%
% (a) 2D Scatter Plot (Dx / Dz)
%

figure('Name', '2D Displacement', 'Position', [200, 200, 700, 500]);
gscatter(X_disp(:,1), X_disp(:,2), labels, ['r', 'g', 'b'], 'os^');

xlabel('Displacement D_x (mm)');
ylabel('Displacement D_z (mm)');
title('2D Displacement (Cylinder Objects)');
legend({'PLA', 'TPU', 'Rubber'}, 'Location', 'northeast');
grid on;

%
% (b) Gaussian Mixture Model
%

gmModel = fitgmdist(X_disp, 3, ...
    'CovarianceType','full', ...
    'SharedCovariance',false, ...
    'RegularizationValue',5e-3, ...
    'Replicates',1, ...
    'Options', statset('MaxIter',2000));

pad = 0.13;

xmin = min(X_disp(:,1)); xmax = max(X_disp(:,1));
ymin = min(X_disp(:,2)); ymax = max(X_disp(:,2));

xpad = pad * (xmax - xmin);
ypad = pad * (ymax - ymin);

x_range = linspace(xmin - xpad, xmax + xpad, 180);
y_range = linspace(ymin - ypad, ymax + ypad, 180);
[X_grid, Y_grid] = meshgrid(x_range, y_range);

gmPDF = @(x,y) pdf(gmModel, [x, y]);
Z = reshape(gmPDF(X_grid(:), Y_grid(:)), size(X_grid));

figure('Position', [100, 100, 800, 600]);
hold on;

contour(X_grid, Y_grid, Z, 20, 'LineWidth', 1);
colormap(flipud(gray));

h = gscatter(X_disp(:,1), X_disp(:,2), labels, ['r', 'g', 'b'], 'os^', 6);

xlabel('Displacement D_x (mm)');
ylabel('Displacement D_z (mm)');
title('GMM 3-Component Fit with Probability Contours');
legend({'PLA', 'TPU', 'Rubber'}, 'Location', 'northeast');
grid on;
hold off;

%
% (c) Surf Plot
%

figure('Position',[120 80 900 680]);
surf(X_grid, Y_grid, Z, 'EdgeColor', 'none');

colormap(parula);
colorbar;
hold on;

z0 = min(Z(:));
contour3(X_grid, Y_grid, Z, 12, 'k', 'LineWidth', 0.8);

xlabel('Displacement D_x (mm)');
ylabel('Displacement D_z (mm)');
zlabel('p(D_x, D_z)');
title('3D Surface of 3-Component GMM PDF');

view(45, 30);
camproj perspective;
grid on;
box on;
lighting gouraud;
camlight headlight;

%
% (d) Hard Clusters
%

idx = cluster(gmModel, X_disp);

figure('Position',[120 100 820 620]); hold on;
[X_grid, Y_grid] = meshgrid(x_range, y_range);
Z = reshape(pdf(gmModel, [X_grid(:), Y_grid(:)]), size(X_grid));
contour(X_grid, Y_grid, Z, 16, 'LineWidth', 0.8);

gscatter(X_disp(:,1), X_disp(:,2), idx, 'rgb', 'o^s', 6);

xlabel('Displacement D_x (mm)');
ylabel('Displacement D_z (mm)');
title('Hard GMM Cluster Assignments (MAP)');
legend({'Density contours','Cluster 1','Cluster 2','Cluster 3'}, 'Location','best');
grid on; box on;
hold off;



