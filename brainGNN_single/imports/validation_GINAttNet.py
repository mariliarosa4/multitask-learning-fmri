import os
from aioitertools import count
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

from net.braingnn import Network
from imports.functions import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torchsummary import summary
from torchviz import make_dot

device = 'cpu'
target_vec = ['y0','y1','y2']
EPS = 1e-10
path = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/scores'

############################### Define Other Loss Functions ########################################
def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

def isnan(x):
    """ Simple utility to see what is NaN """
    return x!=x

def myloss(output_vec, target_vec):
    """ Main Loss that is used for MulitTargets"""
    criterion = torch.nn.MSELoss()
 
    mse_part = 0
    masks = dict()
    loss1 = dict()
    for x in range(0,len(target_vec)):
        masks[x] = isnan(target_vec[x])

        if target_vec[x][~masks[x]].nelement() == 0:
            loss1[x] = torch.sqrt(torch.tensor(1e-20))
            continue
        else: # non nans
            mse_part += criterion(output_vec[x][~masks[x]],target_vec[x][~masks[x]])
            loss1[x] = torch.sqrt(criterion(output_vec[x][~masks[x]],target_vec[x][~masks[x]])+1e-16)
    
    loss = torch.sqrt(mse_part)
    mylist = [loss]
    for x in range(0, len(target_vec)):
        mylist.append(loss1[x]) 
    return mylist


###################### Network Training Function#####################################
def train(train_loader, epoch, model, optimizer, train_dataset, writer):
    device = 'cpu'
    model.train()
    s1_list = []
    s2_list = []
    step = 0
    target_vec = ['y0','y1','y2']
    loss_all = 0
    output_vec = []
    tar_vec = []

    for data in train_loader:
        print("training...")
        print(data)
        data = data.to(device)
        optimizer.zero_grad()
        # output, w1, w2, s1, s2, perm1, perm2  = model(data)
        output  = model(data)
        loss = myloss([output[:,x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])[0]

        # loss = myloss([output[x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])[0]
        # s1_list.append(s1.view(-1).detach().cpu().numpy())
        # s2_list.append(s2.view(-1).detach().cpu().numpy())


        step = step + 1

        loss.backward()
        try:
            node_saliency_map = []
            # input_grads = model.input.grad.view(1200000)

            # for n in range(input_grads.shape[0]): # nth node
            #     node_grads = input_grads[n]
            #     node_saliency = torch.norm(F.relu(node_grads)).item()
            #     node_saliency_map.append(node_saliency)
            # print(node_saliency_map)

            # plot_explanations(model, data)
        # except ValueError as e:
        except Exception as e:
            print(e)
            continue

        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        # s1_arr = np.hstack(s1_list)
        # s2_arr = np.hstack(s2_list)
    

    #return loss_all / len(train_dataset)
    return loss_all / len(train_dataset)

def nonmissingvales(loader, target_num):
    """ function that computes the amount of molecules that do have a specific target """
    count = 0
    for data in loader:
        count +=isnan(data['y%s'%target_num]).sum()
    return len(loader.dataset) - count

###################### Network Testing Function#####################################
# def test_acc(loader):
#     model.eval()
#     correct = 0
#     for data in loader:
#         data = data.to(device)
#         outputs= model(data)
#         print(outputs.max(dim=0))
#         pred = outputs.max(dim=1)[1]
#         pred_y1 = outputs.max(dim=1)[1]
#         ## isso ta errado
#         correct += pred.eq(data.y0).sum().item()
#         correct += pred_y1.eq(data.y1).sum().item()

#     return correct / len(loader.dataset)

def val_loss_f(loader,epoch, model):
    """ Main function to validate the model """
    model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    device = 'cpu'
    loss_all = 0
    loss1_all = dict()
    output_vec1 = []
    tar_vec1 = []
    for x in range(len(target_vec)):
        loss1_all[x]= 0
    
    c = 0
    for data in loader:
        print("validating...")
        print(data)
        data = data.to(device)
        # output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        output  = model(data)
        loss = myloss([output[:,x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])[0]

        #loss = myloss([output[x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])[0]
        loss_all += loss.item() * data.num_graphs
        c += data.y0.size(0)
    return loss_all/c

def test_loss(loader,model,target_vec):
    """ Main function to test the model """
    model.eval()
    cont = 0
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    device = 'cpu'
    loss_all = 0
    loss1_all = dict()
    output_vec1 = []
    tar_vec1 = []
    for x in range(len(target_vec)):
        loss1_all[x]= 0
        
    for data in loader:
        print("testing...")
        print(data)
        data = data.to(device)
        # output, w1, w2, s1, s2,  perm1, perm2  = model(data)
        output  = model(data)

        from numpy import save
        if not os.path.exists(path + "/" + str(cont)):
            os.makedirs(path + "/" + str(cont))
        # save(path + "/" + str(cont) + '/s1.npy',  s1.detach().numpy())
        # save(path + "/" + str(cont) + '/s2.npy',  s2.detach().numpy())
        # save(path + "/" + str(cont) + '/w1.npy',  w1.detach().numpy())
        # save(path + "/" + str(cont) + '/w2.npy',  w2.detach().numpy())
        # save(path + "/" + str(cont) + '/perm1.npy',  perm1.detach().numpy())
        # save(path + "/" + str(cont) + '/perm2.npy',  perm2.detach().numpy())
        cont += 1
        masks0 = isnan(data['y0']) 
        #output_vec1.append(output[0][~masks0])
        output_vec1.append(output[:,0][~masks0])

        tar_vec1.append(data['y0'][~masks0])
        losslist = myloss([output[:,x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])

        #losslist = myloss([output[x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])
        loss_all += losslist[0].item() * data.num_graphs
        for x in range(len(target_vec)):
            loss1_all[x] += losslist[x+1] * (data.num_graphs - isnan(data['y%s'%x]).sum().item())
    
    
    mylist = [loss_all / len(loader.dataset)]
    for x in range(len(target_vec)):
        mylist.append(loss1_all[x] /nonmissingvales(loader, x).float())
    return mylist

def clean_print( string_vec, loss_vec):
    """ Printing function that is used at the end of each CV """
    str1 = "Run : Total RMSE: %0.3f" %(loss_vec[0])
    for x in range(len(string_vec)):
        str1 += " | %s Loss %0.3f" %(string_vec[x], loss_vec[x+1])
    print(str1)

        
def overall_clean_print(string_vec, loss_vec, std_vec):
    """ Printing function that prints the aggregated results """
    str1 = "Overall RMSE on test: %0.3f +/- %0.2f" %(loss_vec[0], std_vec[0])
    for x in range(len(string_vec)):
        str1 += " | RMSE on %s: %0.3f +/- %0.2f " %(string_vec[x], loss_vec[x+1], std_vec[x+1])
    print(str1)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
