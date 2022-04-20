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
subdirectory = "Single_Gender_v1"


dataset = HCPDataset(path_data,subdirectory)
dataset.data['y0'] = dataset.data['y0'].squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0
# dataset.data['y0'] = normalize(dataset.data.y0.to(torch.float32), p=10, dim = 0)


regression = False

list_mae = []
list_cc = []

list_mse = []

list_acc_y0 = []
list_auc_y0 = []

for i in range(5):
    model.load_state_dict(torch.load(os.path.join(path,subdirectory, "model", subdirectory+"_fold"+str(i)+".pth"),device))

    test_dataset = dataset[indexes_test]

    test_loader = DataLoader(test_dataset, batch_size=len(indexes_test), shuffle=False)

    test_loss_list, y_pred_list = test_loss_single(test_loader, model, target_vec, subdirectory+"fold"+str(i))
    np.save(os.path.join(path,subdirectory, "model", subdirectory  + 'fold'+str(i)+'_y_pred_list.npy'), y_pred_list)

    y_pred = np.round(np.concatenate(y_pred_list), 3)
    batch = next(iter(test_loader))

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
