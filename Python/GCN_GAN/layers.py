import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GNN(Module):
    '''
    Class to define the GNN layer
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GNN, self).__init__()
        # ====================
        self.input_dim = input_dim # Dimension of input features
        self.output_dim = output_dim # Dimension of output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize the model parameters via the Xavier algorithm
        self.agg_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(input_dim, output_dim))) # Aggregation weight matrix
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input
        :param sup: GNN support (normalized adjacency matrix)
        :return: aggregated feature output
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        feat_agg = torch.spmm(sup, feat) # Aggregated feature
        agg_output = torch.mm(feat_agg, self.param[0])
        agg_output = torch.relu(agg_output)
        agg_output = F.normalize(agg_output, dim=1, p=2) # l2-normalization
        agg_output = self.dropout_layer(agg_output)

        return agg_output