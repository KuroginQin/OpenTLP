import torch

def get_STGSN_loss_wei(adj_est, gnd):
    '''
    Function to derive the loss of STGSN for weighted graphs (error minimization)
    :param adj_est: prediction result (the estimated adjacency matrix)
    :param gnd: ground-truth (adjacency matrix of the next snapshot)
    :return: loss of STGSN
    '''
    # ====================
    loss = torch.norm((adj_est - gnd), p='fro')**2

    return loss