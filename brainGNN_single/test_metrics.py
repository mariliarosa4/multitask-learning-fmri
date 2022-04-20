import os
import numpy as np
import argparse
import time
import copy

import torch
from torch.functional import Tensor
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.HCPDataset import HCPDataset
from torch_geometric.data import DataLoader

from net.braingnn_single import Network_single
from imports.functions import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, roc_auc_score
from torchsummary import summary
from torchviz import make_dot

from roda_tudo_single_classification import *
# from roda_tudo_single import *
import pandas as pd

target_vec = ['y0']

from torch.nn.functional import normalize
 
torch.manual_seed(123)

EPS = 1e-10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network_single(200, 0.5, 1).to(device)

indexes_test = np.loadtxt("/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_single/data/hcp_1200/indexes_test_balanced.txt",  dtype=float).astype(int)
path_data = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_single/data/hcp_1200'
path = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_single/results'
subdirectory = "Single_MMSE_Score_v3"


dataset = HCPDataset(path_data,subdirectory)
dataset.data['y0'] = dataset.data['y0'].squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0
dataset.data['y0'] = normalize(dataset.data.y0.to(torch.float32), p=10, dim = 0)


regression = True

list_mae = []
list_cc = []

list_mse = []

list_acc_y0 = []
list_auc_y0 = []

for i in range(5):
    model.load_state_dict(torch.load(os.path.join(path,subdirectory, "model", subdirectory+"_fold"+str(i)+".pth"),device))

    test_dataset = dataset[indexes_test]

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    test_loss_list, y_pred_list = test_loss_single(test_loader, model, target_vec, subdirectory+"fold"+str(i))
    np.save(os.path.join(path,subdirectory, "model", subdirectory  + 'fold'+str(i)+'_y_pred_list.npy'), y_pred_list)
    from torch_geometric.utils import to_networkx
    y_pred = np.round(np.concatenate(y_pred_list), 3)
    batch = next(iter(test_loader))
    # only keep the one edge attr as edge weights
    data = batch.to_data_list()[0]
    # data.edge_attr = data.edge_attr
    raw_networkx = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True, remove_self_loops=True)
    # raw_adj = nx.to_numpy_array(raw_networkx, weight='edge_attr') # plot to connectome
    print(batch)
    # %%
    param_dict = model.interpret(batch)

    # %%
    data_process = Data(edge_index=param_dict['l2_edge_index_dropped'], edge_attr=param_dict['l2_edge_attr_dropped'],
                        y=data.y).to('cpu')
    data_process.edge_attr = data_process.edge_attr
    G = to_networkx(data_process, edge_attrs=['edge_attr'], to_undirected=True, remove_self_loops=True)
    adj = nx.to_numpy_array(G, weight='edge_attr')  # plot to connectome
    print(adj.shape)
    coordinates = np.load(f'experimentos_finais/GNN/BrainGNN_single/data/AAL_coordinates.npy', allow_pickle=True)
    from nilearn import plotting
    # raw_view = plotting.view_connectome(raw_adj, coordinates, edge_threshold='10%') # .save_as_html('./outputs/raw')
    view = plotting.plot_connectome(adj, coordinates, edge_threshold='10%').save_as_html('experimentos_finais/GNN/BrainGNN_single/aqui.html')
    break    
    if regression:
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
        print(explained_variance_score(batch.y0, y_pred) )
        print( mean_absolute_error(batch.y0, y_pred) )
        print( mean_squared_error(batch.y0, y_pred) )
        print( r2_score(batch.y0, y_pred) )
        list_mae.append(mean_absolute_error(batch.y0, y_pred))
        list_mse.append(mean_squared_error(batch.y0, y_pred, squared=False))
        list_cc.append(r2_score(batch.y0, y_pred))
    else:
        confusion_matrix_df = pd.DataFrame(confusion_matrix(batch.y0, np.concatenate(y_pred_list)))
        print(confusion_matrix_df)

        list_acc_y0.append(accuracy_score(batch.y0, y_pred))
        list_auc_y0.append(roc_auc_score(batch.y0, y_pred))

if regression:
    print(f"Mean MAE +- std: {np.mean(list_mae):.3f} +- {np.std(list_mae):.3f}")
    print(f"Mean RMSE +- std : {np.mean(list_mse):.3f} +- {np.std(list_mse):.3f}")
    print(f"Mean R²/CC +- std : {np.mean(list_cc):.3f} +- {np.std(list_cc):.3f}")

else:
    print(f"Mean AUC +- std Gender: {np.mean(list_auc_y0):.3f} +- {np.std(list_auc_y0):.3f}")
    print(f"Mean ACC +- std Gender: {np.mean(list_auc_y0):.3f} +- {np.std(list_auc_y0):.3f}")
