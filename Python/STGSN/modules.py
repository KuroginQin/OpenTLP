import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphNeuralNetwork
from .layers import Attention

class STGSN(nn.Module):
    '''
    Class to define DDNE
    '''
    def __init__(self, end_dims, dropout_rate):
        super(STGSN, self).__init__()
        # ====================
        self.enc_dims = end_dims # Layer configuration of encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = STGSN_Enc(self.enc_dims, self.dropout_rate)
        self.dec = STGSN_Dec(self.enc_dims[-1], self.dropout_rate)

    def forward(self, sup_list, feat_list, gbl_sup, gbl_feat, num_nodes):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (i.e., normalized adjacency matrices)
        :param feat_list: list of GNN feature inputs (i.e., node attributes)
        :param gbl_sup: global GNN support
        :param gbl_feat: global GNN feature input
        :param num_nodes: number of associated nodes
        :return: prediction result
        '''
        # ====================
        dyn_emb = self.enc(sup_list, feat_list, gbl_sup, gbl_feat, num_nodes)
        adj_est = self.dec(dyn_emb, num_nodes)

        return adj_est

class STGSN_Enc(nn.Module):
    '''
    Class to define the encoder of STGSN
    '''
    def __init__(self, enc_dims, dropout_rate):
        super(STGSN_Enc, self).__init__()
        # ====================
        self.enc_dims = enc_dims # Layer configuration of encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Structural encoder
        self.num_struc_layers = len(self.enc_dims)-1  # Number of GNN layers
        self.struc_enc = nn.ModuleList()
        for l in range(self.num_struc_layers):
            self.struc_enc.append(
                GraphNeuralNetwork(self.enc_dims[l], self.enc_dims[l+1], dropout_rate=self.dropout_rate))
        # ===========
        # Temporal encoder
        self.att = Attention(self.enc_dims[-1])

    def forward(self, sup_list, feat_list, gbl_sup, gbl_feat, num_nodes):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (i.e., normalized adjacency matrices)
        :param feat_list: list of GNN feature inputs (i.e., node attributes)
        :param gbl_sup: global GNN support
        :param gbl_feat: global GNN feature input
        :param num_nodes: number of associated nodes
        :return: dynamic node embedding
        '''
        # ====================
        win_size = len(sup_list) # Window size, i.e., #historical snapshots
        # ====================
        # Structural encoder
        ind_input_list = feat_list # List of attribute inputs w.r.t. historical snapshots
        gbl_input = gbl_feat
        ind_output_list = None # List of embedding outputs w.r.t. historical snapshots
        gbl_output = None
        for l in range(self.num_struc_layers):
            gbl_output = self.struc_enc[l](gbl_input, gbl_sup)
            gbl_input = gbl_output
            # ==========
            ind_output_list = []
            for i in range(win_size):
                ind_input = ind_input_list[i]
                ind_sup = sup_list[i]
                ind_output = self.struc_enc[l](ind_input, ind_sup)
                ind_output_list.append(ind_output)
            ind_input_list = ind_output_list
        gbl_emb = gbl_output
        ind_emb_list = ind_output_list
        # ==========
        # Temporal encoder
        agg_emb = self.att(ind_emb_list, gbl_emb, num_nodes)
        dyn_emb = torch.cat((agg_emb, gbl_emb), dim=1) # Dynamic node embedding

        return dyn_emb

class STGSN_Dec(nn.Module):
    '''
    Class to define the decoder of STGSN
    '''
    def __init__(self, emb_dim, dropout_rate):
        super(STGSN_Dec, self).__init__()
        # ====================
        self.emb_dim = emb_dim # Dimensionality of dynamic embedding
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.dec = nn.Linear(4*self.emb_dim, 1)

    def forward(self, dyn_emb, num_nodes):
        '''
        Rewrite the forward function
        :param dyn_emb: dynamic embedding given by encoder
        :param num_nodes: number of associated nodes
        :return: prediction result
        '''
        # ====================
        adj_est = None
        for i in range(num_nodes):
            cur_emb = dyn_emb[i, :]
            cur_emb = torch.reshape(cur_emb, (1, self.emb_dim*2))
            cur_emb = cur_emb.repeat(num_nodes, 1)
            cat_emb = torch.cat((cur_emb, dyn_emb), dim=1)
            col_est = torch.sigmoid(self.dec(cat_emb))
            if i == 0:
                adj_est = col_est
            else:
                adj_est = torch.cat((adj_est, col_est), dim=1)

        return adj_est
