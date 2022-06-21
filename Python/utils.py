import numpy as np
import scipy as sp
from sklearn import metrics

def get_adj_un(edges, num_nodes):
    '''
    Function to get the (unweighted) adjacency matrix according to the edge list
    :param edges: edge list
    :param node_num: number of nodes
    :param max_thres: threshold of the maximum edge weight
    :return: adj: adjacency matrix
    '''
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        adj[src, dst] = 1
        adj[dst, src] = 1
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj

def get_adj_wei(edges, num_nodes, max_thres):
    '''
    Function to get the (weighted) adjacency matrix according to the edge list
    :param edges: edge list
    :param node_num: number of nodes
    :param max_thres: threshold of the maximum edge weight
    :return: adj: adjacency matrix
    '''
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei>max_thres:
            wei = max_thres
        adj[src, dst] = wei
        adj[dst, src] = wei
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj

def get_gnn_sup(adj):
    '''
    Function to get GNN support (normalized adjacency matrix w/ self-connected edges)
    :param adj: original adjacency matrix
    :return: GNN support
    '''
    # ====================
    num_nodes, _ = adj.shape
    adj = adj + np.eye(num_nodes)
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]
    for j in range(num_nodes):
        sup[:, j] /= degs[j]

    return sup

def get_gnn_sup_d(adj):
    '''
    Function to get GNN support (normalized adjacency matrix w/ self-connected edges)
    :param adj: original adjacency matrix
    :return: GNN support
    '''
    # ====================
    num_nodes, _ = adj.shape
    adj = adj + np.eye(num_nodes)
    degs = np.sum(adj, axis=1)
    sup = adj # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]

    return sup

def sparse_to_tuple(sparse_mx):
    '''
    Function to transfer sparse matrix to tuple format
    :param sparse_mx: original sparse matrix
    :return: corresponding tuple format
    '''
    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx): # sp.sparse.isspmatrix_coo(mx)
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def gen_noise(m, n):
    '''
    Function to generative noises w/ uniform distribution
    :param m: #rows of noise matrix
    :param n: #columns of noise matrix
    :return: noise matrix
    '''
    # ====================
    return np.random.uniform(0, 1., size=[m, n])

def get_RMSE(adj_est, gnd, num_nodes):
    '''
    Function to get the RMSE (root mean square error) metric
    :param adj_est: prediction result (i.e., the estimated adjacency matrix)
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: RMSE metric
    '''
    # =====================
    f_norm = np.linalg.norm(gnd-adj_est, ord='fro')**2
    #f_norm = np.sum((gnd - adj_est)**2)
    RMSE = np.sqrt(f_norm/(num_nodes*num_nodes))

    return RMSE

def get_MAE(adj_est, gnd, num_nodes):
    '''
    Funciton to get the MAE (mean absolute error) metric
    :param adj_est: prediction result (i.e., the estimated adjacency matrix)
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MAE metric
    '''
    # ====================
    MAE = np.sum(np.abs(gnd-adj_est))/(num_nodes*num_nodes)

    return MAE

def get_EW_KL(adj_est, gnd, num_nodes):
    '''
    Function to get the EW-KL (edge-wise KL divergence) metric
    :param adj_est: prediction result (i.e., the estimated adjacency matrix)
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: edge-wise KL divergence metric
    '''
    # ====================
    epsilon = 1e-5
    adj_est_ = np.maximum(adj_est, epsilon)
    gnd_ = np.maximum(gnd, epsilon)
    # ==========
    sum_est = np.sum(adj_est_) # Sum of all elements of the prediction result
    q = adj_est_/sum_est # Normalized prediction result
    # ==========
    sum_gnd = np.sum(gnd_) # Sum of all elements of the ground-truth
    p = gnd_/sum_gnd # Normalized ground-truth
    # ==========
    edge_wise_KL = 0 # Edge-wise KL divergence

    for r in range(num_nodes):
        for c in range(num_nodes):
            edge_wise_KL += p[r, c]*np.log(p[r, c]/q[r, c])

    return edge_wise_KL

def get_MR(adj_est, gnd, num_nodes):
    '''
    Function to get the MR (mismatch rate) metric
    :param adj_est: prediction result (i.e., the estimated adjacency matrix)
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MR metric
    '''
    # ====================
    MR = 0
    for r in range(num_nodes):
        for c in range(num_nodes):
            if (adj_est[r, c]>0 and gnd[r, c]==0) or (adj_est[r, c]==0 and gnd[r, c]>0):
                MR += 1
    # ==========
    MR = MR/(num_nodes*num_nodes)

    return MR

def get_AUC(adj_est, gnd, num_nodes):
    '''
    Function to get the AUC metric (for the prediction of unweighted graphs)
    :param adj_est: prediction result (i.e., the estimated adjacency matrix)
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: AUC metric
    '''
    gnd_vec = np.reshape(gnd, [num_nodes*num_nodes])
    pred_vec = np.reshape(adj_est, [num_nodes*num_nodes])

    fpr, tpr, thresholds = metrics.roc_curve(gnd_vec, pred_vec)
    AUC = metrics.auc(fpr, tpr)

    return AUC