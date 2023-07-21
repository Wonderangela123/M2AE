import torch
import torch.nn.functional as F
import os
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

from utils import preprocess_data
from models import init_model_dict, init_optim_dict
from train_test import train, test



if __name__ == '__main__':
    ############
    # Parameters Section

    device = torch.device('cuda:0')

    num_view = 3
 
    # input layer size
    input_dim_list = [194, 17, 33]

    # the size of the new space learned by the model (number of the new features)
    num_scfa = 8

    # number of layers with nodes in each one
    hidden_dim = [[64, 32, 16],  ## view 1 
                  [16, 16, 16],  ## view 2
                  [32, 16, 16]]  ## view 3
    
    # number of nodes in VCDN
    vcdn_dim = 32 

    # the parameters for training the network
    lr_e = 1e-2
    lr_c = 1e-3
    epoch_num = 2500
    dropout = 0.1
    view_weight = [1, 1]  ## weight for view 2 and view 3

    # end of parameters section
    ############
 
    data_folder = os.path.join('US_MGS', '4')
    adj_train_list, adj_test_list, data_tr_list, data_te_list, scfa_train_set, scfa_test_set = preprocess_data(data_folder, device)
            
    model_dict = init_model_dict(num_view, input_dim_list, hidden_dim, vcdn_dim, num_scfa, dropout, view_weight, device)
    optim_dict = init_optim_dict(num_view, model_dict, lr_e, lr_c)

    for epoch in range(epoch_num):
        train(model_dict, optim_dict, data_tr_list, adj_train_list, scfa_train_set, device)
        if (epoch+1)%epoch_num==0:
            print(epoch+1)
            c = test(model_dict, data_te_list, adj_test_list, scfa_test_set)

    
    c_mae = []
    for i in range(8):  ## num_scfa
        r, p = pearsonr(c.detach().cpu().numpy()[:,i], scfa_test_set.detach().cpu().numpy()[:,i])
        r_s, p_s = spearmanr(c.detach().cpu().numpy()[:,i], scfa_test_set.detach().cpu().numpy()[:,i])
        rmse = math.sqrt(mean_squared_error(c.detach().cpu().numpy()[:,i], scfa_test_set.detach().cpu().numpy()[:,i]))
        mae = mean_absolute_error(c.detach().cpu().numpy()[:,i], scfa_test_set.detach().cpu().numpy()[:,i])
        c_mae.append(mae)

        # with open('prelim_results.txt', 'a') as f:
        #     print('For feature {}: Pearson correlation {} and P value {}'.format(i+1, r, p),
        #           'Spearman correlation {} and P value {}'.format(i+1, r_s, p_s),
        #           'RMSE {}'.format(rmse),
        #           'MAE {}'.format(mae), file=f) 
            
        print('For feature {}: Spearman correlation {} and P value {}'.format(i+1, r_s, p_s))

    c_mae = sum(c_mae)

    print("Sum of MAE: {:.4f}".format(c_mae))