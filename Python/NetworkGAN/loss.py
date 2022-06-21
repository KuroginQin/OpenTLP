import torch

def get_gen_loss_pre(adj_est, gnd):
    '''
    Function to define the pre-training loss function of generator
    :param adj_est: prediction result
    :param gnd: training ground-truth
    :return: loss of generator
    '''
    # ====================
    loss = torch.norm((adj_est - gnd), p='fro')**2

    return loss

def get_gen_loss(adj_est, gnd, disc_fake, gamma):
    '''
    Function to define loss of generator (in formal optimization)
    :param adj_est: prediction result
    :param gnd: training ground-truth
    :param disc_fake: output of discriminator w.r.t. the fake input
    :param gamma: hyper-parameter to adjust the contribution of MSE loss
    :return: loss of generator
    '''
    # ====================
    epsilon = 1e-5
    #loss = torch.mean(torch.log(1-disc_fake+epsilon))
    #loss = -torch.mean(torch.log(disc_fake+epsilon))
    loss = -torch.mean(disc_fake) # Loss of GAN
    loss += gamma*torch.norm((adj_est - gnd), p='fro')**2 # MSE loss of error minimization

    return loss

def get_disc_loss(disc_real, disc_fake):
    '''
    Function to define loss of discriminator
    :param disc_real: output of discriminator w.r.t. the real input
    :param disc_fake: output of discriminator w.r.t. the fake input
    :return: loss of discriminator
    '''
    # ====================
    epsilon = 1e-5
    #loss = -(torch.mean(torch.log(1-disc_fake+epsilon)) + torch.mean(torch.log(disc_real+epsilon)))
    #loss = torch.mean(torch.log(disc_fake+epsilon)) - torch.mean(torch.log(disc_real+epsilon))
    loss = torch.mean(disc_fake) - torch.mean(disc_real)

    return loss