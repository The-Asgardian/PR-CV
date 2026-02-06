clc; clear;
scriptDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(scriptDir, 'PR_CW_mat');

f = fullfile(dataDir, 'cylinder_papillarray_single_contact.mat');
dPLA = load(f);
f = fullfile(dataDir, 'cylinder_TPU_papillarray_single_contact.mat');
dTPU = load(f);
f = fullfile(dataDir, 'cylinder_rubber_papillarray_single_contact.mat');
dRubber = load(f);

n1 = size(dPLA.sensor_matrices_force, 1); 
n2 = size(dTPU.sensor_matrices_force, 1); 
n3 = size(dRubber.sensor_matrices_force, 1);

labels_cyl = [ones(n1,1); 2*ones(n2,1); 3*ones(n3,1)];

p5_cols = 13:15;

X_p5 = [dPLA.sensor_matrices_force(:, p5_cols); ...
        dTPU.sensor_matrices_force(:, p5_cols); ...
        dRubber.sensor_matrices_force(:, p5_cols)];

perplexities = [5, 30];
figure('Position', [100, 100, 1000, 400]);

for i = 1:length(perplexities)
    perp = perplexities(i);
    
    [Y, loss] = tsne(X_p5, 'Algorithm', 'exact', 'Perplexity', perp);
    
    subplot(1, 2, i);
    gscatter(Y(:,1), Y(:,2), labels_cyl, ...
             ['r', 'g', 'b'], ['o', 'x', 's']);
    
    title(sprintf('t-SNE (Perplexity: %d)\nLoss: %.4f', perp, loss));
    xlabel('t-SNE dimension 1');
    ylabel('t-SNE dimension 2');
    grid on;
    legend({'PLA', 'TPU', 'Rubber'}, 'Location', 'best');
end