import os
from aioitertools import count
import numpy as np
import argparse
import time
import copy
import pandas as pd
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
from numpy import save
device = 'cpu'
target_vec = ['y0','y1','y2']
EPS = 1e-10

path_scores_test = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/scores/test'
path_scores_train = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/scores/train'
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
    print(mylist)
    return mylist

def myacc(model, loader, output_vec):
    model.eval()
    correct = 0
    mylistacc = [0,0,0]
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        for x in range(len(target_vec)):
            pred = torch.argmax(output[x], dim=1)
            # print("predito")
            # print(pred)
            # print("real")
            # print(data['y%s'%x])
            mylistacc[x] += pred.eq(data['y%s'%x]).sum().item()

    # print(mylistacc)

    for x in range(len(target_vec)):
        mylistacc[x] = mylistacc[x]/len(loader.dataset)
    return mylistacc

###################### Network Training Function#####################################
def train(train_loader, epoch, model, scheduler, train_dataset, optimizer, opt):
    print('train...........')
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    device = 'cpu'
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    target_vec = ['y0','y1','y2']

    cont = 0
    for data in train_loader:
        # print("training...")
        test_loss_vec = []
        loss_vec = dict()
        data = data.to(device)
        optimizer.zero_grad()
        
        output, w1, w2, s1, s2,  perm1, perm2  = model(data)
        
        # from numpy import save
        # if not os.path.exists(path_scores_train + "/" + str(cont)):
        #     os.makedirs(path_scores_train + "/" + str(cont))
        # save(path_scores_train + "/" + str(cont) + '/s1.npy',  s1.detach().numpy())
        # save(path_scores_train + "/" + str(cont) + '/s2.npy',  s2.detach().numpy())
        # save(path_scores_train + "/" + str(cont) + '/w1.npy',  w1.detach().numpy())
        # save(path_scores_train + "/" + str(cont) + '/w2.npy',  w2.detach().numpy())
        # save(path_scores_train + "/" + str(cont) + '/perm1.npy',  perm1.detach().numpy())
        # save(path_scores_train + "/" + str(cont) + '/perm2.npy',  perm2.detach().numpy())
        # cont += 1
        #output  = model(data)
        #loss = myloss([output[x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])[0]
        a = torch.amax(output[0], dim=1)
        aa = torch.amax(output[1], dim=1)
        aaa = torch.amax(output[2], dim=1)
        loss_func = torch.nn.BCEWithLogitsLoss()
        nl_loss = torch.nn.NLLLoss()
        loss_c = nl_loss(output[0], data.y0)
        loss_cc = nl_loss(output[1], data.y1)
        loss_ccc = loss_func(output[2][:,0].to(torch.float32), data.y2.to(torch.float32))
        # print(loss_c, loss_cc, loss_ccc)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        
        loss_consist = 0
        # for c in range(2):
        #     loss_consist += consist_loss(s1[data.y0 == c])
        # for c in range(4):
        #     loss_consist += consist_loss(s1[data.y1 == c])
        # for c in range(6):
        #     loss_consist += consist_loss(s1[data.y2 == c])
        # loss = opt.lamb0*(loss_c+loss_cc+loss_ccc) + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
        #            + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5 * loss_consist

        loss = loss_c+loss_cc+loss_ccc

        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())
        print(loss)

        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs

        # try:
        #     plot_explanations(model, data)
        # # except ValueError as e:
        # except Exception as e:
        #     print(e)
        #     continue
        optimizer.step()
        scheduler.step()
        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
    
    
    #return loss_all / len(train_dataset)
    return loss_all / len(train_dataset)

def train_single(train_loader, epoch, model, optimizer, train_dataset, writer, target_vec, opt):
    
    print('train...........')
    
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    for data in train_loader:
        print("training...")
        data = data.to(device)
        optimizer.zero_grad()
        
        output, w1, w2, s1, s2,  perm1, perm2  = model(data)
        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())
        #torch.tensor(torch.argmax(x, dim=1),  dtype=torch.float)
        # a = torch.amax(output, dim=1)
        output = output.to(torch.float32)
        data.y0 = data.y0.to(torch.float32)

        loss_c = F.mse_loss(output, data.y0)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y0 == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5 
                   
        # writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        # writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
        # writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
        # writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        # writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        # writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
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

def val_loss_f(loader,epoch, model, opt):
    """ Main function to validate the model """
    print('validation ...........')
    model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    device = 'cpu'
    loss = 0
    loss_all = 0
    # for x in range(len(target_vec)):
    #     loss1_all[x]= 0
    with torch.no_grad():
        for data in loader:
            # print("validating...")
            # print(data)
            data = data.to(device)
            output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
            
            a = torch.amax(output[0], dim=1)
            aa = torch.amax(output[1], dim=1)
            aaa = torch.amax(output[2], dim=1)

            print(data.y1)
            print(torch.argmax(output[1], dim=1))
            loss_func = torch.nn.BCEWithLogitsLoss()
            nl_loss = torch.nn.NLLLoss()
            loss_c = nl_loss(output[0], data.y0)
            loss_cc = nl_loss(output[1], data.y1)
            loss_ccc = loss_func(output[2][:,0].to(torch.float32), data.y2.to(torch.float32))
            print(loss_c, loss_cc, loss_ccc)

            loss_p1 = (torch.norm(w1, p=2)-1) ** 2
            loss_p2 = (torch.norm(w2, p=2)-1) ** 2
            loss_tpk1 = topk_loss(s1,opt.ratio)
            loss_tpk2 = topk_loss(s2,opt.ratio)
            # loss_consist = 0
            # for c in range(opt.nclass):
            #     loss_consist += consist_loss(s1[data.y0 == c])
            # loss = opt.lamb0*(loss_c+loss_cc+loss_ccc) + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
            #         + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5 
            loss = loss_c+loss_cc+loss_ccc
            print(loss)
            loss_all += loss.item() * data.num_graphs
        # c += data.y0.size(0)
    return loss_all/len(loader)


def val_loss_single(loader,epoch, model, target_vec, opt):
    """ Main function to validate the model """
    model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    device = 'cpu'
    loss_all = 0
    for data in loader:
        output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        # a = torch.amax(output, dim=1)
        output = output.to(torch.float32)
        data.y0 = data.y0.to(torch.float32)
        
        loss_c = F.mse_loss(output, data.y0)
        print(data.y0)
        print(output)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y0 == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

def test_loss_single(test_loader, model, target_vec, opt, name):
    """ Main function to validate the model """
    model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    device = 'cpu'
    loss_all = 0
    y_pred_list = []
    cont = 0
    for data in test_loader:
        output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        output = output.to(torch.float32)
        data.y0 = data.y0.to(torch.float32)
        # a = torch.argmax(output, dim=1)
        # y_pred_list.append(a.detach().numpy())
        y_pred_list.append(output.detach().numpy())
        
        if not os.path.exists(path_scores_test +'_'+ name +  "/" + str(cont)):
            os.makedirs(path_scores_test +'_'+ name +  "/" + str(cont))
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/s1.npy',  s1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/s2.npy',  s2.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/w1.npy',  w1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/w2.npy',  w2.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/perm1.npy',  perm1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/perm2.npy',  perm2.detach().numpy())
        cont += 1
        # loss_c = F.nll_loss(output, data.y0) 

        loss_c = F.mse_loss(output, data.y0) 

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y0 == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(test_loader.dataset), y_pred_list

def test_loss(test_loader, model, target_vec, opt, name):
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
    y_pred_list = []
    for data in test_loader:
        print("testing...")
        print(data)
        data = data.to(device)
        output, w1, w2, s1, s2,  perm1, perm2  = model(data)
        a = torch.argmax(output[0], dim=1).detach().numpy()
        aa = torch.argmax(output[1], dim=1).detach().numpy()
        aaa = output[2].detach().numpy()
        y_pred_list.append([a, aa, aaa])
        # y_pred_list.append(output) # for regression
        
        if not os.path.exists(path_scores_test +'_'+ name +  "/" + str(cont)):
            os.makedirs(path_scores_test +'_'+ name +  "/" + str(cont))
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/s1.npy',  s1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/s2.npy',  s2.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/w1.npy',  w1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/w2.npy',  w2.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/perm1.npy',  perm1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/perm2.npy',  perm2.detach().numpy())
        cont += 1
        
        # masks0 = isnan(data['y0']) 
        # output_vec1.append(output[0][~masks0])
        # tar_vec1.append(data['y0'][~masks0])
        # losslist = myloss([output[x] for x in range(len(target_vec))], [data['y%s'%x] for x in range(len(target_vec))])
        # loss_all += losslist[0].item() * data.num_graphs
        # for x in range(len(target_vec)):
        #     loss1_all[x] += losslist[x+1] * (data.num_graphs - isnan(data['y%s'%x]).sum().item())


        loss_c = F.nll_loss(output[0], data.y0)
        loss_cc = F.nll_loss(output[1], data.y1)
        loss_ccc = F.l1_loss(output[2].to(torch.float32), data.y2.to(torch.float32))

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)

        loss = opt.lamb0*(loss_c+loss_cc+loss_ccc) + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5
        
        loss_all += loss.item() * data.num_graphs
    
    # mylist = [loss_all / len(test_loader.dataset)]
    # for x in range(len(target_vec)):
    #     mylist.append(loss1_all[x] /nonmissingvales(test_loader, x).float())
    return loss_all/len(test_loader), y_pred_list 

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

def test_acc_single(loader, model):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        pred = torch.tensor(torch.argmax(output, dim=1),  dtype=torch.float)
        correct += pred.eq(data.y0).sum().item()
    print(correct)
    print(len(loader.dataset))
    return float(correct / (len(loader.dataset)))

def test_pearson_single(loader, model):
    from torchmetrics import PearsonCorrCoef
    model.eval()
    p = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        pearson = PearsonCorrCoef()
        p += pearson(output[:,0], data.y0)

    print(p)
    print(len(loader.dataset))
    return float(p / (len(loader.dataset)))
