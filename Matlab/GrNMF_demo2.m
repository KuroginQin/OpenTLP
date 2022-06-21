%Demonstration of GrNMF
clear;

%====================
data_name = 'Mesh-1';
num_nodes = 38; %Number of nodes (i.e., fixed node set in Level-1)
num_snaps = 445; %Number of snapshots (time steps)
max_wei = 2000; %Maximum edge weight
hid_dim = 16; %Latent representation (embedding) dimensionality
win_size = 10; %Window size
alpha=0.1;
theta=0.2;
max_iter = 1e3; %Maximum number of iterations
min_error = 1e-6; %Minimum relative error to determine convergence
init_op = 0; %0 for NNDSVD init. & 1 for random init.
num_run = 10; %Number of independent runs for random init.

%====================
data = load(['data/', data_name, '.mat']);
adj_seq_gbl = data.adj_seq;
clear data;

%====================
RMSE_seq = [];
MAE_seq = [];
for tau=win_size:num_snaps-1
    gnd = adj_seq_gbl{tau+1}; %Ground-truth for evaluation
    gnd = full(gnd);
    adj_seq = cell(win_size);
    idx = 1;
    for t=tau-win_size+1:tau
        adj_seq{idx} = adj_seq_gbl{t}/max_wei;
        idx = idx+1;
    end
    [adj_est] = GrNMF(adj_seq,hid_dim,win_size,alpha,theta,max_iter,min_error,init_op,num_run);
    adj_est = adj_est*max_wei;
    adj_est = full(adj_est);
    %==========
    %Refine the prediction result
    adj_est = (adj_est + adj_est')/2;
    for i=1:num_nodes
        adj_est(i, i) = 0;
    end
    epsilon = 0.01; %Threshold to set 0
    adj_est(adj_est<epsilon) = 0;
    %==========
    %Evaluate the quality of current prediction operation
    RMSE = get_RMSE(adj_est, gnd, num_nodes);
    MAE = get_MAE(adj_est, gnd, num_nodes);
    RMSE_seq = [RMSE_seq, RMSE];
    MAE_seq = [MAE_seq, MAE];
    fprintf('Snap#%d RMSE %.4f MAE %.4f\n', [tau, RMSE, MAE]);
    
end
RMSE_mean = mean(RMSE_seq);
RMSE_std = std(RMSE_seq);
MAE_mean = mean(MAE_seq);
MAE_std = std(MAE_seq);
