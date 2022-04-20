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

from imports.functions import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torchsummary import summary
from torchviz import make_dot

from imports.validation import *

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, roc_auc_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/data/hcp_1200', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.0001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=2, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=200, help='feature dim')
parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=6, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='model/', help='path to save model')
opt = parser.parse_args()

path = opt.dataroot
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold

target_vec = ['y0', 'y1', 'y2']

from torch.nn.functional import normalize


torch.manual_seed(123)

EPS = 1e-10

indexes_test = np.loadtxt("/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/data/hcp_1200/indexes_test_balanced.txt",  dtype=float).astype(int)

print('load')
path_data = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/data/hcp_1200'

path = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/results'
# for root, subdirectories, files in os.walk(path):
    
#     for subdirectory in set(subdirectories):
# subdirectory = "HCP_MTL_PMAT24_A_CR_MMSE_Score_Gender_v1"
# from multi import Network, test_multi

subdirectory = "GINAttNet_v14"
# from multi_shared_class_weight import Network, test_multi
# from multi_shared_class_weight_hard import Network, test_multi
# from multi_shared import Network, test_multi
# from roda_tudo import Network, test_multi
# from multi_shared_copy import Network, test_multi
# from multi_shared_copy_2 import Network, test_multi
# from multi_shared_copy_3 import Network, test_multi
from net.GINAttNet import GINAttNet as Network
from GINAttNet_v1 import test_multi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(200, 0.5, 3).to(device)
batch_size = 32
print(subdirectory)
list_mae = [[],[],[]]
list_cc = [[],[],[]]

list_mse = [[],[],[]]

list_acc_y0 = []
list_auc_y0 = []

for i in range(5):
    model.load_state_dict(torch.load(os.path.join(path,subdirectory, "model", subdirectory+"_fold"+str(i)+".pth"),device))

    dataset = HCPDataset(path_data,subdirectory)
    dataset.data.x[dataset.data.x == float('inf')] = 0
    dataset.data['y0'] = dataset.data['y0'].squeeze() 
    dataset.data.y1 = dataset.data.y1.squeeze()
    dataset.data.y1 = normalize(dataset.data.y1.to(torch.float32), p=20, dim = 0)
    dataset.data.y2 = normalize(dataset.data.y2.to(torch.float32), p=20, dim = 0)

    test_dataset = dataset[indexes_test]
    print(len(indexes_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_pred_list, y_true = test_multi(test_loader, model, target_vec, subdirectory + 'fold'+str(i))
    import numpy as np
    a = np.array([y_pred_list, None])
    b = a.transpose()
    np.save(os.path.join(path,subdirectory, "model", subdirectory + 'fold'+str(i)+'_y_pred_list.npy'), b)

    regression = False
    for x in range(len(target_vec)):

        if x > 0: #y2 é regressao
            list_mae[x].append(mean_absolute_error(y_true[x], y_pred_list[0][x]))
            list_mse[x].append(mean_squared_error(y_true[x], y_pred_list[0][x], squared=False))
            list_cc[x].append(r2_score(y_true[x], y_pred_list[0][x]))

            # print( mean_absolute_error(batch['y%s'%x], y_pred_list[0][x]) )
            # print( mean_squared_error(batch['y%s'%x], y_pred_list[0][x], squared=False))

            # print( r2_score(batch['y%s'%x], y_pred_list[0][x]) )
        else:
            confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true[x], y_pred_list[0][x]))
            print(confusion_matrix_df)
            list_acc_y0.append(accuracy_score(y_true[x], y_pred_list[0][x]))
            list_auc_y0.append(roc_auc_score(y_true[x], y_pred_list[0][x]))
            print(roc_auc_score(y_true[x], y_pred_list[0][x]))
            print(classification_report(y_true[x], y_pred_list[0][x]))
        # import matplotlib.pyplot as plt

        # plt.figure(1)

        # import networkx as nx
        # import torch_geometric
        # g = torch_geometric.utils.to_networkx(batch[0], to_undirected=True)
        # nx.draw(g)
        # plt.show()

print(f"Mean MAE +- std PMAT: {np.mean(list_mae[1]):.3f} +- {np.std(list_mae[1]):.3f}")
print(f"Mean MAE +- std MMSE: {np.mean(list_mae[2]):.3f} +- {np.std(list_mae[2]):.3f}")

print(f"Mean RMSE +- std PMAT: {np.mean(list_mse[1]):.3f} +- {np.std(list_mse[1]):.3f}")
print(f"Mean RMSE +- std MMSE: {np.mean(list_mse[2]):.3f} +- {np.std(list_mse[2]):.3f}")

print(f"Mean R²/CC +- std PMAT: {np.mean(list_cc[1]):.3f} +- {np.std(list_cc[1]):.3f}")
print(f"Mean R²/CC +- std MMSE: {np.mean(list_cc[2]):.3f} +- {np.std(list_cc[2]):.3f}")

print(f"Mean AUC +- std Gender: {np.mean(list_auc_y0):.3f} +- {np.std(list_auc_y0):.3f}")
print(f"Mean ACC +- std Gender: {np.mean(list_acc_y0):.3f} +- {np.std(list_acc_y0):.3f}")
# print(f"Min, Max MAE over k: {np.min(mae_arr):.3f}, {np.max(mae_arr):.3f}")