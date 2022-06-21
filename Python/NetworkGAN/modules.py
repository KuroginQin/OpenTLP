from .GNN import *

class NetworkGAN(nn.Module):
    '''
    Class to define NetworkGAN (the generator)
    '''
    def __init__(self, struc_dims, temp_dims, dec_dims, dropout_rate):
        super(NetworkGAN, self).__init__()
        # ====================
        self.struc_dims = struc_dims # Layer configuration of structural encoder
        self.temp_dims = temp_dims # Layer configuration of temporal encoder
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = NetworkGAN_Enc(self.struc_dims, self.temp_dims, self.dropout_rate)
        # Decoder
        self.dec = NetworkGAN_Dec(self.dec_dims)

    def forward(self, sup_list, feat_list, S, M_list):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN support
        :param feat_list: list of feature inputs
        :param S:
        :param M_list:
        :return: prediction result
        '''
        # ====================
        num_nodes, _ = feat_list[0].shape
        dyn_emb, s = self.enc(sup_list, feat_list, S, M_list)
        adj_est = self.dec(dyn_emb, s, num_nodes)

        return adj_est

class NetworkGAN_Enc(nn.Module):
    '''
    Class to define the encoder of NetworkGAN
    '''
    def __init__(self, struc_dims, temp_dims, dropout_rate):
        super(NetworkGAN_Enc, self).__init__()
        # ====================
        self.struc_dims = struc_dims # Layer configuration of structural encoder
        self.temp_dims = temp_dims # Layer configuration of temporal encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Structural encoder
        self.struc_enc = AttGNN(self.struc_dims[0], self.struc_dims[1], self.struc_dims[2], self.dropout_rate)
        # Temporal encoder
        self.att_layer = nn.Linear(in_features=self.struc_dims[-1], out_features=1) # Attention layer
        self.temp_enc = nn.LSTM(input_size=self.temp_dims[0], hidden_size=self.temp_dims[1])

    def forward(self, sup_list, feat_list, S, M_list):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN support
        :param feat_list: list of feature inputs
        :param S:
        :param M_list:
        :return: dynamic node embedding
        '''
        # ====================
        win_size = len(sup_list) # Window size of historical snapshots
        num_nodes, _ = feat_list[0].shape
        # Structural encoder
        struc_ouput_list = [] # List of output of structural encoder
        for t in range(win_size):
            feat = feat_list[t] # Feature input
            sup = sup_list[t] # GNN support
            struc_output = self.struc_enc(feat, sup)
            struc_ouput_list.append(struc_output)
        # ====================
        # Temporal encoder
        att = None
        for t in range(win_size):
            beta = self.att_layer(struc_ouput_list[t] + M_list[t])
            if t == 0:
                att = beta
            else:
                att = torch.cat((att, beta), dim=1)
        att = F.softmax(att, dim=1)
        c_list = []
        for t in range(win_size):
            C_t = torch.zeros_like(struc_ouput_list[-1]).to(device)
            for k in range(win_size):
                prob = torch.reshape(att[:, k], (num_nodes, 1))
                prob = prob.repeat(1, self.struc_dims[-1])
                C_t += torch.mul(prob, struc_ouput_list[k])
            c_t = torch.reshape(C_t, (1, num_nodes*self.struc_dims[-1]))
            c_list.append(c_t)
        s = torch.reshape(S, (1, num_nodes*self.struc_dims[-1]))
        # ==========
        temp_input_tnr = None # Input of temporal encoder
        for t in range(win_size):
            c_t = c_list[t]
            temp_input = torch.cat((c_t, s), dim=1)
            if t == 0:
                temp_input_tnr = temp_input
            else:
                temp_input_tnr = torch.cat((temp_input_tnr, temp_input), dim=0)
        temp_input_tnr = torch.reshape(temp_input_tnr, (win_size, 1, self.temp_dims[0]))
        temp_output, _ = self.temp_enc(temp_input_tnr)
        dyn_emb = temp_output[-1, :] # Dynamic node embedding

        return dyn_emb, s

class NetworkGAN_Dec(nn.Module):
    '''
    Class to define the decoder of NetworkGAN
    '''
    def __init__(self, dec_dims):
        super(NetworkGAN_Dec, self).__init__()
        # ====================
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dec = nn.Linear(in_features=self.dec_dims[0], out_features=self.dec_dims[1])

    def forward(self, dyn_emb, s, num_nodes):
        '''
        Rewrite the forward function
        :param dyn_emb: dynamic embedding given by encoder
        :param s:
        :param num_nodes: number of nodes
        :return: prediction result
        '''
        # ====================
        dec_input = torch.cat((dyn_emb, s), dim=1)
        dec_output = torch.sigmoid(self.dec(dec_input))
        adj_est = torch.reshape(dec_output, (num_nodes, num_nodes))

        return adj_est

class DiscNetDenseF(nn.Module):
    '''
    Class to define the (auxiliary) discriminator
    '''
    def __init__(self, disc_dims, dropout_rate):
        super(DiscNetDenseF, self).__init__()
        # ====================
        self.disc_dims = disc_dims # Layer configuration of discriminator
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.num_disc_layers = len(disc_dims)-1 # Number of FC layers
        self.disc = nn.ModuleList()
        for l in range(self.num_disc_layers):
            self.disc.append(nn.Linear(in_features=self.disc_dims[l], out_features=self.disc_dims[l+1]))
        self.disc_drop = nn.ModuleList()
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
