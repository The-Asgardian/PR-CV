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

X_disp = [dPLA.(fn_disp)(:, 13:15); ...
          dTPU.(fn_disp)(:, 13:15); ...
          dRubber.(fn_disp)(:, 13:15)];

X_scaled = (X_disp - mean(X_disp)) ./ std(X_disp);

labels = [ones(n1,1); 2*ones(n2,1); 3*ones(n3,1)];

figure('Position', [100, 100, 800, 600]);
scatter3(X_scaled(labels==1,1), X_scaled(labels==1,2), X_scaled(labels==1,3), 'r', 'filled'); hold on;
scatter3(X_scaled(labels==2,1), X_scaled(labels==2,2), X_scaled(labels==2,3), 'g', 'filled');
scatter3(X_scaled(labels==3,1), X_scaled(labels==3,2), X_scaled(labels==3,3), 'b', 'filled');

xlabel('D_x (mm)'); ylabel('D_y (mm)'); zlabel('D_z (mm)');
title('Central Papilla Displacement: Oblong Objects');
legend('PLA', 'TPU', 'Rubber'); grid on; view(3);

%
% (b) K-Means Clustering (Squared Euclidian Distance)
%

idx_euclid = kmeans(X_scaled, 3, 'Replicates', 5);

figure;
shapes = ['o', 's', '^'];
colors = ['r', 'g', 'b'];

for m = 1:3 % Real Materials
    for c = 1:3 % K-means Clusters
        mask = (labels == m) & (idx_euclid == c);
        if any(mask)
            scatter3(X_scaled(mask,1), X_scaled(mask,2), X_scaled(mask,3), ...
                30, colors(m), shapes(c), 'filled'); hold on;
        end
    end
end
title('K-means Clustering (Euclidean Distance)');
xlabel('D_x'); ylabel('D_y'); zlabel('D_z');
legend('Material: PLA', 'Material: TPU', 'Material: Rubber');
view(3); grid on;

%
% (c) K-Means Clustering (Manhattan Distance)
%

idx_city = kmeans(X_scaled, 3, 'Distance', 'cityblock', 'Replicates', 5);

figure;
for m = 1:3 
    for c = 1:3 
        mask = (labels == m) & (idx_city == c);
        if any(mask)
            scatter3(X_scaled(mask,1), X_scaled(mask,2), X_scaled(mask,3), ...
                30, colors(m), shapes(c), 'filled'); hold on;
        end
    end
end
title('K-means Clustering (Manhattan)');
view(3); grid on;

%
% Trial with Hierearchical Clustering
%

distMatrix = pdist(X_scaled, 'cosine');
Z = linkage(distMatrix, 'average');

idx_hierarchical = cluster(Z, 'maxclust', 3);

figure;
shapes = ['o', 's', '^'];
colors = ['r', 'g', 'b'];

for m = 1:3 % Real Materials
    for c = 1:3 % K-means Clusters
        mask = (labels == m) & (idx_hierarchical == c);
        if any(mask)
            scatter3(X_scaled(mask,1), X_scaled(mask,2), X_scaled(mask,3), ...
                30, colors(m), shapes(c), 'filled'); hold on;
        end
    end
end
title('Hierarchical Clustering (Cosine Distance, Average Linkage)');
xlabel('D_x'); ylabel('D_y'); zlabel('D_z');
legend('Material: PLA', 'Material: TPU', 'Material: Rubber');
view(3); grid on;


