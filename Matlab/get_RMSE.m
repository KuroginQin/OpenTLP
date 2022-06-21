function [RMSE] = get_RMSE(adj_est,gnd,num_nodes)
%Function to compute the RMSE (root mean squared error) metric
%adj_est: prediction result (i.e., estimated adjacency matrix)
%gnd: ground-truth
%num_nodes: number of nodes
%RMSE: RMSE metric

    %====================
    f_norm = norm(adj_est - gnd, 'fro')^2;
    RMSE = sqrt(f_norm/(num_nodes*num_nodes));

end