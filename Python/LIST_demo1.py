# Demonstration of LIST

from LIST.LIST import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================
data_name = 'Enron'
num_nodes = 184 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 26 # Number of snapshots
hid_dim = 64 # Dimensionality of latent space
theta = 5
beta = 0.1
lambd = 0.1

# ====================
edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

# ====================
learn_rate = 5e-4
win_size = 5 # Window size of historical snapshots
num_epochs = 200 # Number of training epochs
dec_list = get_dec_list(win_size, theta) # Get the list of decaying factors

# ====================
AUC_list = []
for tau in range(win_size, num_snaps):
    edges = edge_seq[tau]
    gnd = get_adj_un(edges, num_nodes)
    # ==========
    adj_list = [] # List of historical adjacency matrices
    P_list = [] # List of regularization matrices
    for t in range(tau - win_size, tau):
        edges = edge_seq[t]
        adj = get_adj_un(edges, num_nodes)
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

    # ==========
    # Evaluate the quality of current prediction operation
    AUC = get_AUC(adj_est, gnd, num_nodes)
    AUC_list.append(AUC)
    print('snap %d AUC %f' % (tau, AUC))

# ====================
AUC_mean = np.mean(AUC_list)
AUC_std = np.std(AUC_list, ddof=1)