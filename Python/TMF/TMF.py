import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np

class TMF(Module):
    '''
    Class to define the TMF model
    '''

    def __init__(self, num_nodes, hid_dim, win_size, num_epoch, alpha, beta, theta, learn_rate, device):
        super(TMF, self).__init__()
        self.win_size = win_size # Window size
        self.enc = TMF_Enc(num_nodes, hid_dim, win_size, num_epoch, alpha, beta, theta, learn_rate, device)
        self.dec = TMF_Dec()

    def TMF_fun(self, adj_list):
        '''
        Function for one prediction operation
        :param adj_list: sequence of historical adjacency matrices (ground-truth for model optmization)
        :return: prediction result (w.r.t. next time step)
        '''
        self.enc.model_opt(adj_list) # Model optimization
        param_list, _ = self.enc() # Get the learned model paramters
        adj_est = self.dec(param_list, self.win_size+1) # Derive prediction result

        return adj_est

class TMF_Enc(Module):
    '''
    Class to define the encoder of TMF
    '''

    def __init__(self, num_nodes, hid_dim, win_size, num_epoch, alpha, beta, theta, learn_rate, device):
        super(TMF_Enc, self).__init__()
        # ====================
        self.device = device
        # ==========
        self.num_nodes = num_nodes # Number of nodes (level-1 w/ fixed node set)
        self.hid_dim = hid_dim # Dimensionality of latent space
        self.win_size = win_size # Window size (#historical snapshots)
        self.num_epoch = num_epoch # Number of training epochs
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.learn_rate = learn_rate # Learning rate
        # ==========
        self.dec_list = [] # List of decaying factor
        for t in range(self.win_size):
            dec_t = np.exp(-theta*(win_size-t)) # Current decaying factor
            self.dec_list.append(dec_t)
        # ====================
        # Initialize model parameters (order=2)
        self.W_0 = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.num_nodes, self.hid_dim)))
        self.W_1 = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.num_nodes, self.hid_dim)))
        self.W_2 = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.num_nodes, self.hid_dim)))
        self.U = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.num_nodes, self.hid_dim)))
        self.param = nn.ParameterList()
        self.param.append(self.W_0)
        self.param.append(self.W_1)
        self.param.append(self.W_2)
        self.param.append(self.U)
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
            U = self.param[3]
            adj_est = torch.mm(U, V.t())
            adj_est_list.append(adj_est)
        return self.param, adj_est_list

    def get_loss(self, adj_list, adj_est_list, dec_list, alpha, beta):
        '''
        Function to get training loss
        :param adj_list: sequence of historical adjacency matrix (ground-truth)
        :param adj_est_list: sequence of estimated adjacency matrix
        :param dec_list: list of decay factors
        :param alpha, beta: hyper-parameters
        :return: loss function
        '''
        win_size = len(adj_list) # Window size (#historical snapshots)
        loss = 0.5*alpha*torch.norm(self.param[3], p='fro')**2
        loss += 0.5*beta*torch.norm(self.param[0], p='fro')**2
        loss += 0.5*beta*torch.norm(self.param[1], p='fro')**2
        loss += 0.5*beta*torch.norm(self.param[2], p='fro')**2
        for t in range(win_size):
            dec_t = dec_list[t] # Current decaying factor
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
            loss = self.get_loss(adj_list, adj_est_list, self.dec_list, self.alpha, self.beta)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            #print('-Epoch %d Loss %f' % (epoch, loss))

class TMF_Dec(Module):
    '''
    Class to define the decoder of TMF
    '''

    def __init__(self):
        super(TMF_Dec, self).__init__()

    def forward(self, param_list, pre_t):
        '''
        Rewrite forward function
        :param param_list: list of learned model parameters
        :param pre_t: time step of prediction result (e.g., win_size+1)
        :return: prediction result
        '''
        # ====================
        V = param_list[0] + param_list[1]*pre_t + param_list[2]*pre_t*pre_t
        U = param_list[3]
        adj_est = torch.mm(U, V.t())

        return adj_est