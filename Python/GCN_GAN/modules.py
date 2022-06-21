import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import GNN

class GCN_GAN(nn.Module):
    '''
    Class to define GCN-GAN (the generator)
    '''
    def __init__(self, struc_dims, temp_dims, dec_dims, dropout_rate):
        super(GCN_GAN, self).__init__()
        # ====================
        self.struc_dims = struc_dims # Layer configuration of structural encoder
        self.temp_dims = temp_dims # Layer configuration of temporal encoder
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = GCN_GAN_Enc(self.struc_dims, self.temp_dims, self.dropout_rate)
        # Decoder
        self.dec = GCN_GAN_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, sup_list, noise_list):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN support
        :param noise_list: list of noise (feature) inputs
        :return: prediction result
        '''
        # ====================
        num_nodes, _ = noise_list[0].shape
        dyn_emb = self.enc(sup_list, noise_list)
        adj_est = self.dec(dyn_emb, num_nodes)

        return adj_est

class GCN_GAN_Enc(nn.Module):
    '''
    Class to define the encoder of GCN-GAN
    '''
    def __init__(self, struc_dims, temp_dims, dropout_rate):
        super(GCN_GAN_Enc, self).__init__()
        # ====================
        self.struc_dims = struc_dims # Layer configuration of structural encoder
        self.temp_dims = temp_dims # Layer configuration of temporal encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Structural encoder
        self.num_struc_layers = len(self.struc_dims)-1 # Number of GNN layers
        self.struc_enc = nn.ModuleList()
        for l in range(self.num_struc_layers):
            self.struc_enc.append(
                GNN(self.struc_dims[l], self.struc_dims[l+1], dropout_rate=self.dropout_rate))
        # ==========
        # Temporal encoder
        self.num_temp_layers = len(self.temp_dims)-1 # Number of LSTM layers
        self.temp_enc = nn.ModuleList()
        for l in range(self.num_temp_layers):
            self.temp_enc.append(
                nn.LSTM(input_size=self.temp_dims[l], hidden_size=self.temp_dims[l+1]))

    def forward(self, sup_list, noise_list):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (normalized adjacency matrices)
        :param noise_list: list of noise (feature) inputs
        :return: dynamic node embedding
        '''
        # ====================
        win_size = len(sup_list) # Window size of historical snapshots
        num_nodes, _ = noise_list[0].shape
        # Structural encoder
        struc_output_tnr = None
        for i in range(win_size):
            sup = sup_list[i] # GNN support
            noise = noise_list[i] # Noise (feature) input
            struc_input = noise
            struc_output = None
            for l in range(self.num_struc_layers):
                struc_output = self.struc_enc[l](struc_input, sup)
                struc_input = struc_output
            # Reshape the embedding output to a long row-wise vector
            struc_output = torch.reshape(struc_output, (1, num_nodes*self.struc_dims[-1]))
            if i == 0:
                struc_output_tnr = struc_output
            else:
                struc_output_tnr = torch.cat((struc_output_tnr, struc_output), dim=0)
        # ====================
        # Temporal encoder
        temp_input = torch.reshape(struc_output_tnr, (win_size, 1, self.temp_dims[0]))
        temp_output = None
        for l in range(self.num_temp_layers):
            temp_output, _ = self.temp_enc[l](temp_input)
            temp_input = temp_output
        dyn_emb = temp_output[-1, :] # Dynamic node embedding

        return dyn_emb

class GCN_GAN_Dec(nn.Module):
    '''
    Class to define the decoder of GCN-GAN
    '''
    def __init__(self, dec_dims, dropout_rate):
        super(GCN_GAN_Dec, self).__init__()
        # ====================
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.num_dec_layers = len(self.dec_dims)-1 # Number of FC layers
        self.dec = nn.ModuleList()
        self.dec_drop = nn.ModuleList()
        for l in range(self.num_dec_layers):
            self.dec.append(
                nn.Linear(in_features=self.dec_dims[l], out_features=self.dec_dims[l+1]))
        for l in range(self.num_dec_layers-1):
            self.dec_drop.append(nn.Dropout(p=self.dropout_rate))

    def forward(self, dyn_emb, num_nodes):
        '''
        Rewrite the forward function
        :param dyn_emb: dynamic embedding given by encoder
        :param num_nodes: number of nodes
        :return: prediction result
        '''
        # ====================
        dec_input = dyn_emb
        dec_output = None
        for l in range(self.num_dec_layers-1):
            dec_output = self.dec[l](dec_input)
            dec_output = self.dec_drop[l](dec_output)
            dec_output = torch.sigmoid(dec_output)
            dec_input = dec_output
        dec_output = self.dec[-1](dec_input)
        adj_est = torch.sigmoid(dec_output) # Prediction result
        # Reshape the prediction result to the matrix form
        adj_est = torch.reshape(adj_est, (num_nodes, num_nodes))

        return adj_est

class DiscNet(nn.Module):
    '''
    Class to define the (auxiliary) discriminator
    '''
    def __init__(self, disc_dims, dropout_rate):
        super(DiscNet, self).__init__()
        # ====================
        self.disc_dims = disc_dims # Layer configuration of discriminator
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.num_disc_layers = len(disc_dims)-1 # Number of FC layers
        self.disc = nn.ModuleList()
        self.disc_drop = nn.ModuleList()
        for l in range(self.num_disc_layers):
            self.disc.append(nn.Linear(in_features=self.disc_dims[l], out_features=self.disc_dims[l+1]))
        for l in range(self.num_disc_layers-1):
            self.disc_drop.append(nn.Dropout(p=self.dropout_rate))

    def forward(self, real, fake, num_nodes):
        '''
        Rewrite the forward function
        :param real: training ground-truth
        :param fake: prediction result
        :return: output w.r.t. the real & fake input
        '''
        # ====================
        # Reshape the input adjacency matrix to a long vector
        real_input = torch.reshape(real, (1, num_nodes*num_nodes))
        fake_input = torch.reshape(fake, (1, num_nodes*num_nodes))
        # ==========
        for l in range(self.num_disc_layers-1):
            # ==========
            FC_layer = self.disc[l]
            drop_layer = self.disc_drop[l]
            # ==========
            real_output = FC_layer(real_input)
            real_output = drop_layer(real_output)
            real_output = torch.relu(real_output)
            real_input = real_output
            # ==========
            fake_output = FC_layer(fake_input)
            fake_output = drop_layer(fake_output)
            fake_output = torch.relu(fake_output)
            fake_input = fake_output
        # ==========
        FC_layer = self.disc[-1]
        # ==========
        real_output = FC_layer(real_input)
        #real_output = torch.sigmoid(real_output)
        # ==========
        fake_output = FC_layer(fake_input)
        #fake_output = torch.sigmoid(fake_output)

        return real_output, fake_output