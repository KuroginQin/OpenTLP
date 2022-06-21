import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_gen_loss_pre(adj_est, gnd):
    '''
    Function to define the pre-training loss function of generator
    :param adj_est: prediction result
    :param gnd: training ground-truth
    :return: loss of generator
    '''
    # ====================
    loss = torch.norm((adj_est - gnd), p='fro')**2 # Error reconstruct

    return loss

def get_gen_loss(adj_est, gnd, disc_fake, alpha):
    '''
    Function to define loss of generator (in formal optimization)
    :param adj_est: prediction result
    :param gnd: training ground-truth
    :param disc_fake: output of discriminator w.r.t. the fake input
    :param alpha: hyper-parameter to adjust the contribution of MSE loss
    :return: loss of generator
    '''
    # ====================
    loss = -torch.mean(disc_fake) # Loss of GAN
    loss += alpha*torch.norm((adj_est - gnd), p='fro')**2 # MSE loss of error minimization

    return loss

def get_disc_loss(disc_real, disc_fake):
    '''
    Function to define loss of discriminator
    :param disc_real: output of discriminator w.r.t. the real input
    :param disc_fake: output of discriminator w.r.t. the fake input
    :return: loss of discriminator
    '''
    # ====================
    loss = torch.mean(disc_fake) - torch.mean(disc_real)
    #loss = torch.mean(torch.log(disc_fake)) - torch.mean(torch.log(disc_real))

    return loss