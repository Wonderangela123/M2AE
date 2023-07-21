import torch.nn.functional as F


def train(model_dict, optim_dict, data_tr_list, adj_train_list, scfa_train_set, device):
    loss_dict = {}
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_tr_list)
    for i in range(num_view):
        optim_dict["E{:}".format(i+1)].zero_grad()
        x, _ = model_dict["E{:}".format(i+1)](data_tr_list[i], adj_train_list[i])
        ci_loss = F.l1_loss(x, data_tr_list[i])
        ci_loss.backward()
        optim_dict["E{:}".format(i+1)].step()
        loss_dict["E{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
   
    optim_dict["C"].zero_grad()
    ci_list = []
    for i in range(num_view):
        _, z = model_dict["E{:}".format(i+1)](data_tr_list[i], adj_train_list[i])
        ci_list.append(z)
    c = model_dict["C"](ci_list)   
    
    c_loss = []
    for i in range(8):  ## num_scfa
        loss = F.l1_loss(c[:,i], scfa_train_set.double()[:,i].to(device)) 
        c_loss.append(loss)

    c_loss = sum(c_loss)

    c_loss.backward()
    optim_dict["C"].step()
    loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict


def test(model_dict, data_te_list, adj_test_list, scfa_test_set):
    for m in model_dict:
        model_dict[m].eval()

    ci_list = []
    for i in range(len(data_te_list)):
        _, z = model_dict["E{:}".format(i+1)](data_te_list[i], adj_test_list[i])
        ci_list.append(z)  
    c = model_dict["C"](ci_list) 

    return c