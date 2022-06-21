import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_d2v_loss(adj_est, gnd, beta):
    '''
    Function to derive the loss of dyngraph2vec
    :param adj_est: prediction result (the estimated adjacency matrix)
    :param gnd: ground-truth (adjacency matrix of the next snapshot)
    :param beta: hyper-parameter
    :return: loss of dyngraph2vec
    '''
    # ====================
    P = torch.ones_like(gnd)
    P_beta = beta*P
    P = torch.where(gnd==0, P_beta, P)
    loss = torch.norm(torch.mul((adj_est - gnd), P), p='fro')**2

    return loss