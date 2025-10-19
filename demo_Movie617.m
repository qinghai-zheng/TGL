clear;clc;
addpath('Measurements','Datasets','Tools');

load("Movie617.mat");
fprintf('-------- Experiments on Movie617 -------- \n');

lambda_1_list = [0.001 0.01];
lambda_2_list = [1 0.01];

num_view = length(X);
gt = Y;
num_cls = length(unique(gt));

% results of each view
for idx_view = 1:num_view
    X_single_view = NormalizeData(X{idx_view});
    
    lambda_1 = lambda_1_list(idx_view);
    lambda_2 = lambda_2_list(idx_view);
    fprintf('=== lambda_1: %f, lambda_2: %f === \n',lambda_1, lambda_2);

    [W,~] = main_TGL(X_single_view, lambda_1, lambda_2);
    W_avg = (abs(W{1})+abs(W{1}') + abs(W{2})+abs(W{2}'))/4;
    gt_pre = SpectralClustering(W_avg,num_cls);

    % res: [ACC nmi Purity Fscore Precision Recall AR Entropy];
    res = Clustering8Measure(gt, gt_pre);
    
    fprintf('\t View %d -- ACC:%12.6f \t NMI:%12.6f \t Purity:%12.6f \t Fscore:%12.6f \n', [idx_view res(1) res(2) res(3) res(4)]);   
end




