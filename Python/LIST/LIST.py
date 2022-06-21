import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np

class LIST(Module):
    '''
    Class to define the LIST model
    '''

    def __init__(self, num_nodes, hid_dim, win_size, dec_list, P_list, num_epoch, beta, learn_rate, device):
        super(LIST, self).__init__()
        self.win_size = win_size # Window size
        self.enc = LIST_Enc(num_nodes, hid_dim, win_size, dec_list, P_list, num_epoch, beta, learn_rate, device)
        self.dec = LIST_Dec()

    def LIST_fun(self, adj_list):
        '''
        Function for one prediction operation
        :param adj_list: sequence of historical adjacency matrices (ground-truth for model optmization)
        :return: prediction result (w.r.t. next time step)
        '''
        self.enc.model_opt(adj_list) # Model optimization
        param_list, _ = self.enc() # Get the learned model paramters
        adj_est = self.dec(param_list, self.win_size+1) # Derive prediction result

        return adj_est

class LIST_Enc(Module):
    '''
    Class to define the encoder of LIST
    '''

    def __init__(self, num_nodes, hid_dim, win_size, dec_list, P_list, num_epoch, beta, learn_rate, device):
        super(LIST_Enc, self).__init__()
        # ====================
        self.device = device
        # ==========
        self.num_nodes = num_nodes # Number of nodes (level-1 w/ fixed node set)
        self.hid_dim = hid_dim # Dimensionality of latent space
        self.win_size = win_size # Window size (#historical snapshots)
        self.dec_list = dec_list # List of decaying factors
        self.P_list = P_list # List of regularization matrices
        self.num_epoch = num_epoch # Number of training epochs
        self.beta = beta
        self.learn_rate = learn_rate # Learning rate
        # ==========
        # Initialize model parameters (order=2)
        self.W_0 = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.num_nodes, self.hid_dim)))
        self.W_1 = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.num_nodes, self.hid_dim)))
        self.W_2 = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.num_nodes, self.hid_dim)))
        self.param = nn.ParameterList()
        self.param.append(self.W_0)
        self.param.append(self.W_1)
        self.param.append(self.W_2)
        self.param.to(self.device)
        # ==========
        # Define the optimizer
        self.opt = optim.Adam(self.param, lr=self.learn_rate)

    def forward(self):
        '''
        Rewrite forward function
        :return: list of reconstructed adjacency matrix
        '''
        adj_est_list = []
        for t in range(self.win_size):
            V = self.param[0] + self.param[1]*(t+1) + self.param[2]*(t+1)*(t+1)
            P = self.P_list[t]
            F = torch.mm(P, V)
            adj_est = torch.mm(F, F.t())
            adj_est_list.append(adj_est)

        return self.param, adj_est_list

    def get_loss(self, adj_list, adj_est_list, dec_list, beta, win_size):
        '''
        Function to get the training loss
        :param adj_list: sequence of historical adjacency matrix
        :param adj_est_list: sequence of estimated adjacency matrix
        :param dec_list: list of decay factors
        :param beta: hyper-parameter
        :param win_size: window size
        :return: loss function
        '''
        loss = 0.5*beta*torch.norm(self.param[0], p='fro')**2
        loss += 0.5*beta*torch.norm(self.param[1], p='fro')**2
        loss += 0.5*beta*torch.norm(self.param[2], p='fro')**2
        for t in range(win_size):
            dec_t = dec_list[t]
            adj = adj_list[t]
            adj_est = adj_est_list[t]
            loss += 0.5*dec_t*torch.norm(adj - adj_est, p='fro')**2

        return loss

    def model_opt(self, adj_list):
        '''
        Function to implement the model optimization
        :param adj_list: sequence of historical adjacency matrices (ground-truth)
        :return:
        '''
        # ====================
        for epoch in range(self.num_epoch):
            _, adj_est_list = self.forward()
            loss = self.get_loss(adj_list, adj_est_list, self.dec_list, self.beta, self.win_size)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            #print('-Epoch %d Loss %f' % (epoch, loss))

class LIST_Dec(Module):
    '''
    Class to define the decoder of LIST
    '''

    def __init__(self):
        super(LIST_Dec, self).__init__()

    def forward(self, param_list, pre_t):
        '''
        Rewrite forward function
        :param param_list: list of learned model parameters
        :param pre_t: time step of prediction result (e.g., win_size+1)
        :return: prediction result
        '''
        # ====================
        V = param_list[0] + param_list[1]*pre_t + param_list[2]*pre_t*pre_t
        adj_est = torch.mm(V, V.t())

        return adj_est

def get_P(adj, num_nodes, lambd, B, device):
    '''
    Function to get the P regularization matrix
    :param adj: adjacency matrix
    :param num_nodes: number of nodes
    :param lambd: hyper-parameter
    :param B: number of iterations
    :return:
    '''
    # ====================
    adj_norm = get_adj_norm(adj)
    adj_tnr = torch.FloatTensor(lambd*adj_norm).to(device)
    mul_res = torch.eye(num_nodes).to(device)
    sum_res = mul_res
    for _ in range(B-1):
        mul_res = torch.mm(mul_res, adj_tnr)
        sum_res = sum_res+mul_res
    sum_res = (1-lambd)*sum_res

    return sum_res

def get_adj_norm(adj):
    '''
    Function to get normalized adjacency matrix
    :param adj: original adjacency matrix
    :return: normalized adjacency matrix
    '''
    # ====================
    num_nodes, _ = adj.shape
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj # GNN support
    for i in range(num_nodes):
        if degs[i]>0:
            sup[i, :] /= degs[i]
    for j in range(num_nodes):
        if degs[j]>0:
            sup[:, j] /= degs[j]

    return sup

def get_dec_list(win_size, theta):
    '''
    Function to get the list of decaying factors
    :param win_size: window size (#historical snapshots)
    :param theta: hyper-parameter
    :return: list of decaying factors
    '''
    # ====================
    dec_list = []
    for t in range(win_size):
        dec_t = np.exp(-theta*(win_size-t)) # Current decaying factor
        dec_list.append(dec_t)

    return dec_list
