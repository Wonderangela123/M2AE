import numpy as np
import scipy.sparse as sp
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

cuda = True if torch.cuda.is_available() else False


def preprocess_data(data_folder, device):
    data_tr_list, data_te_list, scfa_train_set, scfa_test_set = load_data(data_folder)

    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_train_list.append(preprocess_graph(features_to_adj(data_tr_list[i])).to(device).requires_grad_(True)) # Some preprocessing (normalization)
        adj_test_list.append(preprocess_graph(features_to_adj(data_te_list[i])).to(device).requires_grad_(True))
        data_tr_list[i] = data_tr_list[i].to(device).requires_grad_(True)
        data_te_list[i] = data_te_list[i].to(device).requires_grad_(True)

    return adj_train_list, adj_test_list, data_tr_list, data_te_list, scfa_train_set, scfa_test_set 


def load_data(data_folder):
    print('loading data ...')

    ## metagenome
    features_name = np.genfromtxt(os.path.join(data_folder,'metag_tr.csv'), dtype=np.dtype(str)) 
    train_set = np.array(features_name[:, 1:], dtype=np.float32)
    features_name = np.genfromtxt(os.path.join(data_folder,'metag_te.csv'), dtype=np.dtype(str)) 
    test_set = np.array(features_name[:, 1:], dtype=np.float32)
    
    ## covariates
    covariates_name = np.genfromtxt(os.path.join(data_folder, 'covariates_tr.csv'), dtype=np.dtype(str)) 
    covariates_train_set = np.array(covariates_name[:, 1:], dtype=np.float32)
    covariates_name = np.genfromtxt(os.path.join(data_folder, 'covariates_te.csv'), dtype=np.dtype(str)) 
    covariates_test_set = np.array(covariates_name[:, 1:], dtype=np.float32)

    ## dietary
    dietary_name = np.genfromtxt(os.path.join(data_folder, 'dietary_tr.csv'), dtype=np.dtype(str)) 
    dietary_train_set = np.array(dietary_name[:, 1:], dtype=np.float32)
    dietary_name = np.genfromtxt(os.path.join(data_folder, 'dietary_te.csv'), dtype=np.dtype(str)) 
    dietary_test_set = np.array(dietary_name[:, 1:], dtype=np.float32)

    ## SCFA
    scfa_name = np.genfromtxt(os.path.join(data_folder,'scfa_tr.csv'), dtype=np.dtype(str)) 
    scfa_train_set = np.array(scfa_name[:, 1:], dtype=np.float32)
    scfa_name = np.genfromtxt(os.path.join(data_folder,'scfa_te.csv'), dtype=np.dtype(str)) 
    scfa_test_set = np.array(scfa_name[:, 1:], dtype=np.float32)

    ## column-wise normalization (feature normalization)
    train_set = normalize(train_set, norm='l2', axis=0)
    covariates_train_set = normalize(covariates_train_set, norm='l2', axis=0)
    dietary_train_set = normalize(dietary_train_set, norm='l2', axis=0)
    scfa_train_set = normalize(scfa_train_set, norm='l2', axis=0)

    test_set = normalize(test_set, norm='l2', axis=0)
    covariates_test_set = normalize(covariates_test_set, norm='l2', axis=0)
    dietary_test_set = normalize(dietary_test_set, norm='l2', axis=0)
    scfa_test_set = normalize(scfa_test_set, norm='l2', axis=0)

    ## combine three data views into one list
    data_tr_list=[]
    data_tr_list.append(torch.tensor(train_set)) # view 1 
    data_tr_list.append(torch.tensor(covariates_train_set)) # view 2
    data_tr_list.append(torch.tensor(dietary_train_set)) # view 3

    data_te_list=[]
    data_te_list.append(torch.tensor(test_set))
    data_te_list.append(torch.tensor(covariates_test_set))
    data_te_list.append(torch.tensor(dietary_test_set))
    
    scfa_train_set = torch.tensor(scfa_train_set)
    scfa_test_set = torch.tensor(scfa_test_set)

    return data_tr_list, data_te_list, scfa_train_set, scfa_test_set
    

## convert features tensor to adjacency scipy sparse matrix
def features_to_adj(data):  
    edge_per_node = 10
    adj_parameter_adaptive = cal_adj_mat_parameter(edge_per_node, data)
    adj = gen_adj_mat(data, adj_parameter_adaptive)
    adj = sp.csr_matrix(adj)
    return adj


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())

    
def gen_adj_mat(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    
    return adj
    
    
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    
def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)
    
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)