import torch
import torch.nn as nn

class E_LSTM_D(nn.Module):
    '''
    Class to define E-LSTM-D
    '''
    def __init__(self, struc_dims, temp_dims, dec_dims, dropout_rate):
        super(E_LSTM_D, self).__init__()
        # ====================
        self.struc_dims = struc_dims # Layer configuration of structural encoder
        self.temp_dims = temp_dims # Layer configuration of temporal encoder
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = E_LSTM_D_Enc(self.struc_dims, self.temp_dims, self.dropout_rate)
        # Decoder
        self.dec = E_LSTM_D_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, adj_list):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: prediction result
        '''
        # ====================
        dyn_emb = self.enc(adj_list)
        adj_est = self.dec(dyn_emb)

        return adj_est

class E_LSTM_D_Enc(nn.Module):
    '''
    Class to define the encoder of E-LSTM-D
    '''
    def __init__(self, struc_dims, temp_dims, dropout_rate):
        super(E_LSTM_D_Enc, self).__init__()
        # ====================
        self.struc_dims = struc_dims # Layer configuration of structural encoder
        self.temp_dims = temp_dims # Layer configuration of temporal encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Structural encoder
        self.num_struc_layers = len(self.struc_dims)-1 # Number of FC layers
        self.struc_enc = nn.ModuleList()
        for l in range(self.num_struc_layers):
            self.struc_enc.append(
                nn.Linear(in_features=self.struc_dims[l], out_features=self.struc_dims[l+1]))
        self.struc_drop = nn.ModuleList()
        for l in range(self.num_struc_layers):
            self.struc_drop.append(nn.Dropout(p=self.dropout_rate))
        # ==========
        # Temporal encoder
        self.num_temp_layers = len(self.temp_dims)-1 # Numer of LSTM layers
        self.temp_enc = nn.ModuleList()
        for l in range(self.num_temp_layers):
            self.temp_enc.append(
                nn.LSTM(input_size=self.temp_dims[l], hidden_size=self.temp_dims[l+1]))

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
        # Structural encoder
        temp_input_tnr = None
        for t in range(win_size):
            adj = adj_list[t]
            struc_input = adj
            struc_output = None
            for l in range(self.num_struc_layers):
                struc_output = self.struc_enc[l](struc_input)
                struc_output = self.struc_drop[l](struc_output)
                struc_output = torch.relu(struc_output)
                struc_input = struc_output
            if t == 0:
                temp_input_tnr = struc_output
            else:
                temp_input_tnr = torch.cat((temp_input_tnr, struc_output), dim=0)
        # ==========
        # Temporal encoder
        temp_input = torch.reshape(temp_input_tnr, (win_size, int(num_nodes), self.temp_dims[0]))
        temp_output = None
        for l in range(self.num_temp_layers):
            temp_output, _ = self.temp_enc[l](temp_input)
            temp_input = temp_output
        dyn_emb = temp_output[-1, :] # Dynamic node embedding (N*d)

        return dyn_emb

class E_LSTM_D_Dec(nn.Module):
    '''
    Class to define the decoder of E-LSTM-D
    '''
    def __init__(self, dec_dims, dropout_rate):
        super(E_LSTM_D_Dec, self).__init__()
        # ====================
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Decoder
        self.num_dec_layers = len(self.dec_dims)-1 # Number of FC layers
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