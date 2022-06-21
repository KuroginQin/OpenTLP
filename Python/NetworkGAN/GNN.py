import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttGNN(Module):
    '''
    Class to define the attentive GNN (i.e., structural encoder)
    '''
    def __init__(self, GAT_input_dim, GAT_output_dim, GCN_output_dim, dropout_rate):
        super(AttGNN, self).__init__()
        # ====================
        self.GAT_layer = GATDense(GAT_input_dim, GAT_output_dim, dropout_rate)
        self.GCN_layer = GCNDense(GAT_output_dim, GCN_output_dim, dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of GNN
        :param sup: GNN support (normalized adjacency matrix)
        :return: aggregated feature output
        '''
        # ====================
        GAT_feat = self.GAT_layer(feat, sup)
        GCN_feat = self.GCN_layer(GAT_feat, sup)

        return GCN_feat

class GCNDense(Module):
    '''
    Class to define the GCN layer (via dense matrix multiplication)
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GCNDense, self).__init__()
        # ====================
        self.input_dim = input_dim # Dimension of input features
        self.output_dim = output_dim # Dimension of output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize the model parameters
        self.agg_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.input_dim, self.output_dim))) # Aggregation weight matrix
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        # =========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of GCN
        :param sup: GCN support (normalized adjacency matrix)
        :return: aggregated feature output
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        num_nodes, _ = sup.shape
        sup = sup + torch.eye(num_nodes).to(device)
        feat_agg = torch.mm(sup, feat) # Aggregated feature
        agg_output = torch.relu(torch.mm(feat_agg, self.param[0]))
        agg_output = F.normalize(agg_output, dim=1, p=2)  # l2-normalization
        agg_output = self.dropout_layer(agg_output)

        return agg_output

class GATDense(Module):
    '''
    Class to define the GAT layer (via dense matrix multiplication)
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GATDense, self).__init__()
        # ====================
        self.input_dim = input_dim # Dimension of input features
        self.output_dim = output_dim # Dimension of output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize the model parameters
        self.map_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.input_dim, self.output_dim)))
        self.map_bias = Parameter(torch.zeros(1))
        self.U = Init.xavier_uniform_(Parameter(torch.FloatTensor(output_dim, 1)))
        self.V = Init.xavier_uniform_(Parameter(torch.FloatTensor(output_dim, 1)))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.map_wei)
        self.param.append(self.map_bias)
        self.param.append(self.U)
        self.param.append(self.V)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of GAT
        :param sup: GCN support (normalized adjacency matrix)
        :return: aggregated feature output
        '''
        # ====================
        num_nodes, _ = sup.shape
        sup = sup + torch.eye(num_nodes).to(device)
        # ==========
        feat_map = torch.matmul(feat, self.param[0])
        att = None
        for i in range(num_nodes):
            feat_cur = feat_map[i, :]
            feat_cur = torch.reshape(feat_cur, (1, -1))
            feat_cur = feat_cur.repeat(num_nodes, 1)
            att_cur = torch.tanh(torch.matmul(feat_cur, self.param[2]) + torch.matmul(feat_map, self.param[3]) + self.param[1])
            if i==0:
                att = att_cur
            else:
                att = torch.cat((att, att_cur), dim=1)
        aux = -100*torch.ones_like(sup).to(device)
        att = torch.where(sup>0, att, aux)
        att = F.softmax(att, dim=1)
        feat_out = torch.tanh(torch.matmul(att, feat_map))

        return feat_out
