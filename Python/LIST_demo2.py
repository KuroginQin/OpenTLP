# Demonstration of LIST

from LIST.LIST import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================
data_name = 'Mesh-1'
num_nodes = 38 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 445 # Number of snapshots
max_thres = 2000 # Threshold for maximum edge weight
hid_dim = 16 # Dimensionality of latent space
theta = 5
beta = 0.01
lambd = 0.1

# ====================
edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

# ====================
epsilon = 1e-2 # Threshold of zero-refining
learn_rate = 5e-3
win_size = 10 # Window size of historical snapshots
num_epochs = 500 # Number of training epochs
dec_list = get_dec_list(win_size, theta) # Get the list of decaying factors

# ====================
RMSE_list = []
MAE_list = []
for tau in range(win_size, num_snaps):
    edges = edge_seq[tau]
    gnd = get_adj_wei(edges, num_nodes, max_thres)
    # ==========
    adj_list = [] # List of historical adjacency matrices
    P_list = [] # List of regularization matrices
    for t in range(tau - win_size, tau):
        edges = edge_seq[t]
        adj = get_adj_wei(edges, num_nodes, max_thres)
        adj_tnr = torch.FloatTensor(adj).to(device)
        adj_list.append(adj_tnr)
        P = get_P(adj, num_nodes, lambd, B=100, device=device)
        P_list.append(P)
    LIST_model = LIST(num_nodes, hid_dim, win_size, dec_list, P_list, num_epochs, beta, learn_rate, device)
    adj_est = LIST_model.LIST_fun(adj_list)
    if torch.cuda.is_available():
        adj_est = adj_est.cpu().data.numpy()
    else:
        adj_est = adj_est.data.numpy()

    # ==========
    # Refine prediction result
    #adj_est = (adj_est+adj_est.T)/2
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