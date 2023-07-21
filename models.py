import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import os

from utils import preprocess_graph, features_to_adj

seed = 1234

random.seed(seed)
os.environ["PYTHONSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)




def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.matmul(x.float(), self.weight)
        output = torch.sparse.mm(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(GCNModel, self).__init__()

        self.dropout = dropout 

        ## encoder
        self.gc1 = GraphConvolution(in_dim, hidden_dim[0])

        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim[1], hidden_dim[2])
            )  

        # decoder
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim[1], hidden_dim[0])
            )
        
        self.gc6 = GraphConvolution(hidden_dim[0], in_dim)

        self.dc = InnerProductDecoder(self.dropout, act=lambda x: x) ## adj

    def encode(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.01)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.classifier1(x)
        return x 
    
    
    def decode(self, x, adj):
        x = self.classifier2(x)
        x = F.leaky_relu(x, 0.01)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        return x
    
    def forward(self, x, adj):
        z = self.encode(x, adj)
        adj = self.dc(z) ## adj for decoder
        x = self.decode(z, adj)
        return x, z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


    
class VCDN(nn.Module):
    def __init__(self, num_view, hidden_dim, vcdn_dim, out_dim, dropout, view_weight, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.view_weight = view_weight
        self.device = device
        
        self.gc = GraphConvolution(pow(self.hidden_dim[0][-1], num_view), vcdn_dim)

        self.linear = nn.Sequential(
            nn.Linear(vcdn_dim, out_dim)
        )

    def forward(self, in_list): ## in_list: view list
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), self.view_weight[0]*in_list[1].unsqueeze(1)),(-1,pow(self.hidden_dim[0][-1],2),1))
        x = torch.reshape(torch.matmul(x, self.view_weight[1]*in_list[i].unsqueeze(1)),(-1,pow(self.hidden_dim[0][-1],3),1))
        vcdn_feat = torch.reshape(x, (-1,pow(self.hidden_dim[0][-1],num_view))).requires_grad_(True)
        output = self.gc(vcdn_feat, preprocess_graph(features_to_adj(vcdn_feat.detach().cpu())).to(self.device))
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.linear(output)
        
        return output


def init_model_dict(num_view, input_dim_list, hidden_dim, vcdn_dim, num_scfa, dropout, view_weight, device):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCNModel(input_dim_list[i], hidden_dim[i], dropout).to(device)
    model_dict["C"] = VCDN(num_view, hidden_dim, vcdn_dim, num_scfa, dropout, view_weight, device).to(device)
    return model_dict


def init_optim_dict(num_view, model_dict, lr_e, lr_c):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["E{:}".format(i+1)] = torch.optim.Adam(list(model_dict["E{:}".format(i+1)].parameters()), lr_e)
    optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr_c)
    return optim_dict