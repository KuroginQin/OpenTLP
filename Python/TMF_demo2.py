# Demonstration of TMF

from TMF.TMF import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================
data_name = 'Mesh-1'
num_nodes = 38 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 445 # Number of snapshots
max_thres = 2000 # Threshold for maximum edge weight
hid_dim = 16 # Dimensionality of latent space
theta = 0.1
alpha = 0.01
beta = 0.01

# ====================
edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

# ====================
epsilon = 1e-2 # Threshold of zero-refining
learn_rate = 1e-2
win_size = 10 # Window size of historical snapshots
num_epochs = 500 # Number of training epochs

# ====================
RMSE_list = []
MAE_list = []
for tau in range(win_size, num_snaps):
    edges = edge_seq[tau]
    gnd = get_adj_wei(edges, num_nodes, max_thres)
    # ==========
    adj_list = [] # List of historical adjacency matrices
    for t in range(tau-win_size, tau):
        edges = edge_seq[t]
        adj = get_adj_wei(edges, num_nodes, max_thres)
        adj = adj/max_thres
        adj_tnr = torch.FloatTensor(adj).to(device)
        adj_list.append(adj_tnr)
    TMF_model = TMF(num_nodes, hid_dim, win_size, num_epochs, alpha, beta, theta, learn_rate, device)
    adj_est = TMF_model.TMF_fun(adj_list)
    adj_est = adj_est*max_thres
    if torch.cuda.is_available():
        adj_est = adj_est.cpu().data.numpy()
    else:
        adj_est = adj_est.data.numpy()

    # ==========
    # Refine prediction result
    adj_est = (adj_est+adj_est.T)/2
    for r in range(num_nodes):
        adj_est[r, r] = 0
    for r in range(num_nodes):
        for c in range(num_nodes):
            if adj_est[r, c] <= epsilon:
                adj_est[r, c] = 0

    # ==========
    # Evaluate the quality of current prediction operation
    RMSE = get_RMSE(adj_est, gnd, num_nodes)
    MAE = get_MAE(adj_est, gnd, num_nodes)
    RMSE_list.append(RMSE)
    MAE_list.append(MAE)
    print('snap %d RMSE %f MAE %f' % (tau, RMSE, MAE))

# ====================
RMSE_mean = np.mean(RMSE_list)
RMSE_std = np.std(RMSE_list, ddof=1)
MAE_mean = np.mean(MAE_list)
MAE_std = np.std(MAE_list, ddof=1)
