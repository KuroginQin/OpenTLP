function [MAE] = get_MAE(adj_est,gnd,num_nodes)
%Function to compute the MAE (mean absolute error) metric
%adj_est: prediction result (i.e., estimated adjacency matrix)
%gnd: ground-truth
%num_nodes: number of nodes
%MAE: MAE metric

    %====================
    MAE = sum(sum(abs(adj_est - gnd)))/(num_nodes*num_nodes);

end