import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphNeuralNetwork(Module):
    '''
    Class to define the GNN layer
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GraphNeuralNetwork, self).__init__()
        # ====================
        self.input_dim = input_dim # Dimension of input features
        self.output_dim = output_dim # Dimension of output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize model parameters
        self.agg_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.input_dim, self.input_dim)))
        self.cat_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2*self.input_dim, self.output_dim)))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        self.param.append(self.cat_wei)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of GNN layer
        :param sup: GNN support (normalized adjacency matrix)
        :return: aggregated feature output
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        feat_agg = torch.spmm(sup, feat) # Aggregated feature
        agg_output = torch.mm(feat_agg, self.param[0])
        cat_input = torch.cat((agg_output, feat), dim=1)
        cat_output = torch.relu(torch.mm(cat_input, self.param[1]))
        cat_output = F.normalize(cat_output, dim=1, p=2) # l2-normalization
        cat_output = self.dropout_layer(cat_output)

        return cat_output


class Attention(Module):
    '''
    Class to define the attention layer
    '''

    def __init__(self, emb_dim):
        super(Attention, self).__init__()
        # ====================
        self.emb_dim = emb_dim # Dimensionality of latent space
        # ====================
        # Initialize model parameters
        self.ind_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim)))
        self.total_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim)))
        self.a = Init.xavier_uniform_(Parameter(torch.FloatTensor(2*self.emb_dim, 1)))
        self.input_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim)))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.ind_wei)
        self.param.append(self.total_wei)
        self.param.append(self.a)
        self.param.append(self.input_wei)

    def forward(self, ind_emb_list, total_emb, num_nodes):
        '''
        Rewrite the forward function
        '''
        # ====================
        total_map = torch.mm(total_emb, self.param[1])
        win_size = len(ind_emb_list)
        prob = None
        for i in range(win_size):
            ind_emb = ind_emb_list[i]
            ind_map = torch.mm(ind_emb, self.param[0])
            cat_vec = torch.cat((ind_map, total_map), dim=1)
            e = F.leaky_relu(torch.mm(cat_vec, self.param[2]))
            if i==0:
                prob = e
            else:
                prob = torch.cat((prob, e), dim=1)
        prob = F.softmax(prob, dim=1)
        # ==========
        emb = None
        for i in range(win_size):
            input_emb = ind_emb_list[i]
            input_emb = torch.mm(input_emb, self.param[3])
            cur_prob = torch.reshape(prob[:, i], (num_nodes, 1))
            cur_prob = cur_prob.repeat(1, self.emb_dim)
            cur_emb = torch.mul(cur_prob, input_emb)
            if i==0:
                emb = cur_emb
            else:
                emb += cur_emb

        return emb
