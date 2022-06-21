# Demonstration of NetworkGAN

import torch
import torch.optim as optim
from NetworkGAN.modules import *
from NetworkGAN.loss import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================
data_name = 'Mesh-1'
num_nodes = 38 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 445 # Number of snapshots
max_thres = 2000 # Threshold for maximum edge weight
noise_dim = 32 # Dimension of noise input
struc_dims = [noise_dim, 32, 16] # Layer configuration of structural encoder
temp_dims = [2*num_nodes*struc_dims[-1], 1024] # Layer configuration of temporal encoder
dec_dims = [temp_dims[-1]+num_nodes*struc_dims[-1], num_nodes*num_nodes] # Layer configuration of decoder
disc_dims = [num_nodes*num_nodes, 512, 256, 64, 1] # Layer configuration of discriminator
win_size = 10 # Window size of historical snapshots
gamma = 30 # Hyper-parameter to adjust the contribution of the MSE loss

# ====================
edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
# List of TMF features
# [S, M_list] for each snapshot
TMF_feat_list = np.load('data/NetworkGAN_TMF_feats_%s.npy' % (data_name), allow_pickle=True)

# ====================
dropout_rate = 0.2 # Dropout rate
epsilon = 1e-2 # Threshold of zero-refining
num_pre_epochs = 10 # Number of pre-training epochs
num_epochs = 100 # Number of training epochs
num_val_snaps = 10 # Number of validation snapshots
num_test_snaps = 50 # Number of test snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

# ==========
none_list = [] # None of TMF features for the first L snapshots
for t in range(win_size):
    none_list.append(None)
none_list.extend(TMF_feat_list)
TMF_feat_list = none_list

# ====================
# Define the model
gen_net = NetworkGAN(struc_dims, temp_dims, dec_dims, dropout_rate).to(device) # Generator
disc_net = DiscNetDenseF(disc_dims, dropout_rate).to(device) # Discriminator
# ==========
# Define the optimizer
pre_gen_opt = optim.Adam(gen_net.parameters(), lr=1e-4, weight_decay=1e-5)
gen_opt = optim.Adam(gen_net.parameters(), lr=5e-4, weight_decay=1e-5)
disc_opt = optim.Adam(disc_net.parameters(), lr=5e-4, weight_decay=1e-5)

# ====================
# Pre-training
for epoch in range(num_pre_epochs):
    # ====================
    # Pre-train the model
    gen_net.train()
    disc_net.train()
    # ==========
    train_cnt = 0
    gen_loss_list = []
    # ==========
    for tau in range(win_size, num_train_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
        gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        for _ in range(1):
            adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
            pre_gen_loss = get_gen_loss_pre(adj_est, gnd_tnr)
            pre_gen_opt.zero_grad()
            pre_gen_loss.backward()
            pre_gen_opt.step()

        # ====================
        gen_loss_list.append(pre_gen_loss.item())
        #print('Pre Loss %f - %d' % (pre_gen_loss.item(), train_cnt))
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    gen_loss_mean = np.mean(gen_loss_list)
    print('#%d Pre-Train G-Loss %f' % (epoch, gen_loss_mean))

    # ====================
    # Validate the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    EW_KL_list = []
    MR_list = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        #adj_est = (adj_est+adj_est.T)/2
        for r in range(num_nodes):
            adj_est[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        RMSE = get_RMSE(adj_est, gnd, num_nodes)
        MAE = get_MAE(adj_est, gnd, num_nodes)
        EW_KL = get_EW_KL(adj_est, gnd, num_nodes)
        MR = get_MR(adj_est, gnd, num_nodes)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        EW_KL_list.append(EW_KL)
        MR_list.append(MR)
        # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)
    EW_KL_mean = np.mean(EW_KL_list)
    EW_KL_std = np.std(EW_KL_list, ddof=1)
    MR_mean = np.mean(MR_list)
    MR_std = np.std(MR_list, ddof=1)
    print('Val Pre Epoch %d RMSE %f %f MAE %f %f EW-KL %f %f MR %f %f'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, EW_KL_mean, EW_KL_std, MR_mean, MR_std))

    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    EW_KL_list = []
    MR_list = []
    for tau in range(num_snaps-num_test_snaps, num_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        #adj_est = (adj_est+adj_est.T)/2
        for r in range(num_nodes):
            adj_est[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        RMSE = get_RMSE(adj_est, gnd, num_nodes)
        MAE = get_MAE(adj_est, gnd, num_nodes)
        EW_KL = get_EW_KL(adj_est, gnd, num_nodes)
        MR = get_MR(adj_est, gnd, num_nodes)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        EW_KL_list.append(EW_KL)
        MR_list.append(MR)
        # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)
    EW_KL_mean = np.mean(EW_KL_list)
    EW_KL_std = np.std(EW_KL_list, ddof=1)
    MR_mean = np.mean(MR_list)
    MR_std = np.std(MR_list, ddof=1)
    print('Test Pre Epoch %d RMSE %f %f MAE %f %f EW-KL %f %f MR %f %f'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, EW_KL_mean, EW_KL_std, MR_mean, MR_std))

# ====================
# Formal optimization
for epoch in range(num_epochs):
    # ====================
    # Train the model
    gen_net.train()
    disc_net.train()
    # ==========
    train_cnt = 0
    disc_loss_list = []
    gen_loss_list = []
    for tau in range(win_size, num_train_snaps):
        # ====================
        sup_list = []  # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
        gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        for _ in range(1):
            # ==========
            # Train the discriminator
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
            disc_real, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
            disc_loss = get_disc_loss(disc_real, disc_fake) # Loss of the discriminative network
            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()
            # ==========
            # Train the generator
            noise = gen_noise(num_nodes, noise_dim)
            adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
            _, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
            gen_loss = get_gen_loss(adj_est, gnd_tnr, disc_fake, gamma)
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

        # ====================
        gen_loss_list.append(gen_loss.item())
        disc_loss_list.append(disc_loss.item())
        #print('Pre G-Loss %f D-Loss %f - %d' % (gen_loss.item(), disc_loss.item(), train_cnt))
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    gen_loss_mean = np.mean(gen_loss_list)
    disc_loss_mean = np.mean(disc_loss_list)
    print('#%d Train G-Loss %f D-Loss %f' % (epoch, gen_loss_mean, disc_loss_mean))

    # ====================
    # Validate the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    EW_KL_list = []
    MR_list = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        #adj_est = (adj_est+adj_est.T)/2
        for r in range(num_nodes):
            adj_est[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        RMSE = get_RMSE(adj_est, gnd, num_nodes)
        MAE = get_MAE(adj_est, gnd, num_nodes)
        EW_KL = get_EW_KL(adj_est, gnd, num_nodes)
        MR = get_MR(adj_est, gnd, num_nodes)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        EW_KL_list.append(EW_KL)
        MR_list.append(MR)
        # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)
    EW_KL_mean = np.mean(EW_KL_list)
    EW_KL_std = np.std(EW_KL_list, ddof=1)
    MR_mean = np.mean(MR_list)
    MR_std = np.std(MR_list, ddof=1)
    print('Val Epoch %d RMSE %f %f MAE %f %f EW_KL %f %f MR %f %f'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, EW_KL_mean, EW_KL_std, MR_mean, MR_std))

    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    EW_KL_list = []
    MR_list = []
    for tau in range(num_snaps-num_test_snaps, num_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres  # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        #adj_est = (adj_est+adj_est.T)/2
        for r in range(num_nodes):
            adj_est[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        RMSE = get_RMSE(adj_est, gnd, num_nodes)
        MAE = get_MAE(adj_est, gnd, num_nodes)
        EW_KL = get_EW_KL(adj_est, gnd, num_nodes)
        MR = get_MR(adj_est, gnd, num_nodes)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        EW_KL_list.append(EW_KL)
        MR_list.append(MR)
        # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)
    EW_KL_mean = np.mean(EW_KL_list)
    EW_KL_std = np.std(EW_KL_list, ddof=1)
    MR_mean = np.mean(MR_list)
    MR_std = np.std(MR_list, ddof=1)
    print('Test Epoch %d RMSE %f %f MAE %f %f EW_KL %f %f MR %f %f'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, EW_KL_mean, EW_KL_std, MR_mean, MR_std))
