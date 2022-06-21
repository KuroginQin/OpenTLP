import torch
import torch.nn as nn

class DDNE(nn.Module):
    '''
    Class to define DDNE
    '''
    def __init__(self, end_dims, dec_dims, dropout_rate):
        super(DDNE, self).__init__()
        # ====================
        self.enc_dims = end_dims # Layer configuration of encoder
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = DDNE_Enc(self.enc_dims, self.dropout_rate)
        # Decoder
        self.dec = DDNE_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, adj_list):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: prediction result
        '''
        # ====================
        dyn_emb = self.enc(adj_list)
        adj_est = self.dec(dyn_emb)

        return adj_est, dyn_emb

class DDNE_Enc(nn.Module):
    '''
    Class to define the encoder of DDNE
    '''
    def __init__(self, enc_dims, dropout_rate):
        super(DDNE_Enc, self).__init__()
        # ====================
        self.enc_dims = enc_dims # Layer configuration of encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Define the encoder, i.e., multi-layer RNN
        self.num_enc_layers = len(self.enc_dims)-1 # Number of RNNs
        self.for_RNN_layer_list = nn.ModuleList() # Forward RNN encoder
        self.rev_RNN_layer_list = nn.ModuleList() # Reverse RNN encoder
        for l in range(self.num_enc_layers):
            self.for_RNN_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))
            self.rev_RNN_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))

    def forward(self, adj_list):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: dynamic node embedding
        '''
        # ====================
        win_size = len(adj_list) # Window size, i.e., #historical snapshots
        num_nodes, _ = adj_list[0].shape
        # ====================
        for_RNN_input = None
        rev_RNN_input = None
        for i in range(win_size):
            # ==========
            if i == 0:
                for_RNN_input = adj_list[i]
                rev_RNN_input = adj_list[win_size-1]
            else:
                for_RNN_input = torch.cat((for_RNN_input, adj_list[i]), dim=0)
                rev_RNN_input = torch.cat((rev_RNN_input, adj_list[win_size-1-i]), dim=0)
        for_RNN_input = torch.reshape(for_RNN_input, (win_size, int(num_nodes), self.enc_dims[0]))
        rev_RNN_input = torch.reshape(rev_RNN_input, (win_size, int(num_nodes), self.enc_dims[0]))
        # ==========
        for_RNN_output = None
        rev_RNN_output = None
        for l in range(self.num_enc_layers):
            # ==========
            for_RNN_output, _ = self.for_RNN_layer_list[l](for_RNN_input)
            for_RNN_input = for_RNN_output
            # ==========
            rev_RNN_output, _ = self.rev_RNN_layer_list[l](rev_RNN_input)
            rev_RNN_input = rev_RNN_output
        # ==========
        RNN_output = None
        for i in range(win_size):
            for_RNN_output_ = for_RNN_output[i, :, :]
            rev_RNN_output_ = rev_RNN_output[i, :, :]
            RNN_output_ = torch.cat((for_RNN_output_, rev_RNN_output_), dim=1)
            if i == 0:
                RNN_output = RNN_output_
            else:
                RNN_output = torch.cat((RNN_output, RNN_output_), dim=1)
        dyn_emb = RNN_output # Dynamic node embedding

        return dyn_emb

class DDNE_Dec(nn.Module):
    '''
    Class to define the decoder of DDNE
    '''
    def __init__(self, dec_dims, dropout_rate):
        super(DDNE_Dec, self).__init__()
        # ====================
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Decoder
        self.num_dec_layers = len(self.dec_dims)-1  # Number of FC layers
        self.dec = nn.ModuleList()
        for l in range(self.num_dec_layers):
            self.dec.append(
                nn.Linear(in_features=self.dec_dims[l], out_features=self.dec_dims[l+1]))
        self.dec_drop = nn.ModuleList()
        for l in range(self.num_dec_layers-1):
            self.dec_drop.append(nn.Dropout(p=self.dropout_rate))

    def forward(self, dyn_emb):
        '''
        Rewrite the forward function
        :param dyn_emb: dynamic embedding given by encoder
        :return: prediction result
        '''
        # ====================
        dec_input = dyn_emb
        dec_output = None
        for l in range(self.num_dec_layers-1):
            dec_output = self.dec[l](dec_input)
            dec_output = self.dec_drop[l](dec_output)
            dec_output = torch.relu(dec_output)
            dec_input = dec_output
        dec_output = self.dec[-1](dec_input)
        dec_output = torch.sigmoid(dec_output)
        adj_est = dec_output

        return adj_est