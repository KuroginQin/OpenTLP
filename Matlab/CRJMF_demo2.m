%Demonstration of CRJMF
clear;

%====================
data_name = 'Mesh-1';
num_nodes = 38; %Number of nodes (i.e., fixed node set in Level-1)
num_snaps = 445; %Number of snapshots (time steps)
max_wei = 2000; %Maximum edge weight
hid_dim = 16; %Latent representation (embedding) dimensionality
win_size = 10; %Window size
alpha = 1;
lambd = 1;
theta = 0.1;
max_iter = 1e3; %Maximum number of iterations
min_error = 1e-6; %Minimum relative error to determine convergence
num_run = 10; %Number of independent runs

%====================
data = load(['data/', data_name, '.mat']);
adj_seq_gbl = data.adj_seq;
att = data.att;
clear data;

%====================
RMSE_seq = [];
MAE_seq = [];
for tau=win_size:num_snaps-1
    gnd = adj_seq_gbl{tau+1}; %Ground-truth for evaluation
    gnd = full(gnd);
    %=====
    col = zeros(num_nodes, num_nodes); %Collapsed graph
    for t=tau-win_size+1:tau
       col = col + (theta^(tau-t))*(adj_seq_gbl{t}/max_wei);
    end
    %=====
    neigh = col;
    neigh(neigh>0)=1;
    prox = zeros(num_nodes, num_nodes); %Second-order proximity
    for r=1:num_nodes
        for c=1:num_nodes
            prox(r, c) = neigh(r, :)*neigh(c, :)'; %Common neighbor similarity
            if prox(r, c)>0
                prox(r, c) = prox(r, c)/(sum(neigh(r, :))*sum(neigh(c, :)));
            end
        end
    end
    adj_est = CRJMF(col,att,prox,hid_dim,alpha,lambd,min_error,max_iter,num_run);
    adj_est = adj_est*max_wei;
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
