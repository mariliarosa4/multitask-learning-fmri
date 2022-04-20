from pandas import array
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm
from numpy import save
from net.braingraphconv import MyNNConv
import os

##########################################################################################################################
class Networkkkk(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, k=8, R=200):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Networkkkk, self).__init__()
        print("no shared")
        # explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        self.indim = indim
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 512
        self.dim4 = 256
        self.dim5 = 8
        self.k = k
        self.R = R
        self.n_tasks = nclass

        self.n1 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1 * self.indim))
        self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
        self.pool1 = TopKPooling(self.dim1, ratio=1, multiplier=1, nonlinearity=torch.sigmoid)
        self.n2 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1))
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = TopKPooling(self.dim2, ratio=1, multiplier=1, nonlinearity=torch.sigmoid)
        
        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        # self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        # self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        # self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        # self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        # self.fc3 = torch.nn.Linear(self.dim3, nclass)

        self.heads = torch.nn.ModuleList()
        for _ in range(self.n_tasks):
            self.heads.append([
                nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1)), 
                MyNNConv(self.dim1, self.dim2, self.n2, normalize=False), 
                TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid),
                torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)])

    def forward(self, data):
        x, edge_index, batch, edge_attr, pos = data.x, data.edge_index, data.batch, data.edge_attr,data.pos
        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        perm1 = perm
        # os.makedirs(path + "/" + str(self.round))
        # save(path + "/" + str(self.round) + '/score1.npy', score1.detach().numpy())
        # save(path + "/" + str(self.round)  + '/perm.npy', perm.detach().numpy())
        # save(path + "/" + str(self.round) + '/pos.npy',  pos.detach().numpy())
        # print("salvando...")

        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        # with torch.enable_grad():
        #     self.final_conv_acts = self.conv2(x, edge_index, edge_attr, pos)
        # self.final_conv_acts.register_hook(self.activations_hook)

        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)
        perm2 = perm
        # save(path + "/" + str(self.round)  + '/pos_2.npy',  pos.detach().numpy())
        # save(path + "/" + str(self.round) + '/perm_2.npy', perm.detach().numpy())
        # save(path + "/" + str(self.round) + '/score2.npy', score2.detach().numpy())
        
        
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(F.log_softmax(self.fc2(x), dim=-1))
        

        outputs = []
        for head in self.heads:
            outputs.append(head(x)) 

        return outputs, self.pool1.weight, self.pool2.weight, torch.sigmoid(score1).view(x.size(0),-1), torch.sigmoid(score2).view(x.size(0),-1), perm1, perm2
        # return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def activations_hook(self, grad):
        self.final_conv_grads = grad