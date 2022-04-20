import sys
from time import time
import warnings

from sklearn.utils import shuffle
from imports import functions as func
# %%
from hypothesis import target
from pandas import array
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops, add_remaining_self_loops,softmax)
from torch_sparse import spspmm, coalesce
from numpy import save
from torch.nn import Parameter
import math
from torch.nn.functional import normalize
from pytorchtools import EarlyStopping
import copy
import sys
import inspect
from torch_geometric.typing import (OptTensor)
import h5py
from torch_scatter import scatter,scatter_add
import os
from torch_geometric.data import InMemoryDataset,Data, DataLoader
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
import deepdish as dd
from sklearn.preprocessing import LabelEncoder
from os import listdir
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from functools import partial
import deepdish as dd
from tensorboardX import SummaryWriter



# %%

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


class MyMessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),
    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.
    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """
    def __init__(self, aggr='add', flow='source_to_target', node_dim=0):
        super(MyMessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferred and assumed to be symmetric.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        dim = self.node_dim
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(dim)
                            if size[1 - idx] != tmp[1 - idx].size(dim):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if tmp is None:
                        message_args.append(tmp)
                    else:
                        if size[idx] is None:
                            size[idx] = tmp.size(dim)
                        if size[idx] != tmp.size(dim):
                            raise ValueError(__size_error_msg__)

                        tmp = torch.index_select(tmp, dim, edge_index[idx])
                        message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out = self.message(*message_args)
        # out = scatter_(self.aggr, out, edge_index[i], dim, dim_size=size[i])
        out = scatter_add(out, edge_index[i], dim, dim_size=size[i])
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

class MyNNConv(MyMessagePassing):
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=True,
                 **kwargs):
        super(MyNNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.nn = nn

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
#        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, pseudo= None, size=None):
        """"""
        edge_weight = edge_weight.squeeze()
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0))

        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        if torch.is_tensor(x):
            x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)
        else:
            x = (None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
                 None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1))
        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)

    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
        edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


##########################################################################################################################

class MTLnet(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, k=10, R=200):
        super(MTLnet, self).__init__()
        self.indim = indim
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 128
        self.dim4 = 64
        self.dim5 = 10
        self.k = k
        self.R = R
        self.n_tasks = nclass

        self.sharedlayer =nn.Sequential(
            nn.Linear(self.R, self.k, bias=False),
            nn.ReLU(), 
            nn.Linear(self.k, self.dim1 * self.indim),
            MyNNConv(self.indim, self.dim1, self.n1, normalize=False),
            TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid) )

        self.tower1 = nn.Sequential(
                        torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2),
                        torch.nn.BatchNorm1d(self.dim2),
                        torch.nn.Linear(self.dim2, self.dim2),
                        torch.nn.Linear(self.dim2, self.dim4),
                        torch.nn.BatchNorm1d(self.dim4),
                        torch.nn.Linear(self.dim4, 1)
                    )
     
        self.tower2 = nn.Sequential(
                        torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2),
                        torch.nn.BatchNorm1d(self.dim2),
                        torch.nn.Linear(self.dim2, self.dim2),
                        torch.nn.Linear(self.dim2, self.dim4),
                        torch.nn.BatchNorm1d(self.dim4),
                        torch.nn.Linear(self.dim4, 1)
                    )

    def forward(self, x):
        h_shared = self.sharedlayer(x)
        out1 = self.tower1(h_shared)
        out2 = self.tower2(h_shared)
        return out1, out2


import random
import torch
import math
##########################################################################################################################
class Network(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, k=50, R=200):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network, self).__init__()

        self.indim = indim
        self.dim1 = 64
        self.dim2 = 64
        self.dim3 = 64
        self.dim4 = 32
        self.dim5 = 12
        self.k = k
        self.R = R
        self.n_tasks = nclass
        

        self.n1 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1 * self.indim))
        self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
        self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.n2 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1))
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        
        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        # self.middle_linear = torch.nn.Linear(self.dim2, self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim4, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(self.dim4)
        # self.fc3 = torch.nn.Linear(self.dim3, nclass)

        self.heads = torch.nn.ModuleList()
        # for _ in range(self.n_tasks):
        self.heads.append(nn.Sequential(torch.nn.Linear(self.dim4, self.dim5), nn.Dropout(0.2), nn.ReLU(),torch.nn.Linear(self.dim5, 1)))
        self.heads.append(nn.Sequential(torch.nn.Linear(self.dim4, self.dim5), nn.Dropout(0.4), nn.ReLU(),torch.nn.Linear(self.dim5, 1)))
        self.heads.append(nn.Sequential(torch.nn.Linear(self.dim4, self.dim5), nn.Dropout(0.4), nn.ReLU(),torch.nn.Linear(self.dim5, 1)))

        


    def forward(self, data):
        x, edge_index, batch, edge_attr, pos = data.x, data.edge_index, data.batch, data.edge_attr,data.pos
        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        perm1 = perm

        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)
        perm2 = perm
        
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # for _ in range(random.randint(0, 3)):
        #     x = self.middle_linear(x).clamp(min=-1, max=1)

       
        x = self.bn2(F.relu(self.fc2(x)))
        outputs = []
        out1 = F.sigmoid(self.heads[0](x))
        # out1 = self.heads[0](x)
        out2 = self.heads[1](x)
        out3 = self.heads[2](x)
        outputs = [out1, out2, out3]

        # for head in self.heads:
        #     outputs.append(head(x)) 

        return outputs, self.pool1.weight, self.pool2.weight, torch.sigmoid(score1).view(x.size(0),-1), torch.sigmoid(score2).view(x.size(0),-1), perm1, perm2
        # return x



class HCPDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(HCPDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        print()
        print("vamo le os dados")
        print()
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

import multiprocessing

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

from functools import partial

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess        
def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    y1_list = []
    y2_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)
    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)
        
    if (len(res[0])>5):
        print("multitask")

        for j in range(len(res)):
            edge_att_list.append(res[j][0])
            edge_index_list.append(res[j][1]+j*res[j][6])
            att_list.append(res[j][2])
            y_list.append(res[j][3])
            y1_list.append(res[j][4])
            y2_list.append(res[j][5])
            batch.append([j]*res[j][6])
            pseudo.append(np.diag(np.ones(res[j][6])))
        y1_arr = np.stack(y1_list)
        y2_arr = np.stack(y2_list)
        y1_torch = torch.from_numpy(y1_arr).long()  # classification 1
        y2_torch = torch.from_numpy(y2_arr).long()  # classification 2
    else:
        for j in range(len(res)):
            edge_att_list.append(res[j][0])
            edge_index_list.append(res[j][1]+j*res[j][4])
            att_list.append(res[j][2])
            y_list.append(res[j][3])
            batch.append([j]*res[j][4])
            pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)

    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
  

    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()

    if (len(res[0])>5):
        data = Data(x=att_torch, edge_index=edge_index_torch, y0=y_torch, y1=y1_torch, y2=y2_torch, edge_attr=edge_att_torch, pos = pseudo_torch )
    else:
        data = Data(x=att_torch, edge_index=edge_index_torch, y0=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices

def read_sigle_data(data_dir,filename,use_gdc =False):
    temp = dd.io.load(osp.join(data_dir, filename))
    # read edge and edge attribute
    pcorr = np.abs(temp['pcorr'][()])

    num_nodes = pcorr.shape[0]
    G = from_numpy_matrix(pcorr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp['corr'][()]
    label1 = temp['y0'][()]
    att_torch = torch.from_numpy(att).float()
    y_torch1 = torch.from_numpy(np.array(label1)).long()  # classification

    if ('y1' in temp.keys()):
        label2 = temp['y1'][()]
        label3 = temp['y2'][()]
        y_torch2 = torch.from_numpy(np.array(label2)).long()  # classification
        y_torch3 = torch.from_numpy(np.array(label3)).long()  # classification

        data = Data(x=att_torch, edge_index=edge_index.long(), y0=y_torch1, y1=y_torch2, y2 = y_torch3, edge_attr=edge_att, A = adj)
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label1, label2 ,label3,num_nodes
    else:
        data = Data(x=att_torch, edge_index=edge_index.long(), y0=y_torch1, edge_attr=edge_att, A = adj)
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label1, num_nodes
def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y0 is not None:
        if data.y0.size(0) == batch.size(0):
            slices['y0'] = node_slice
        else:
            slices['y0'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if ('y1' in data):

        if data.y1 is not None:
            if data.y1.size(0) == batch.size(0):
                slices['y1'] = node_slice
            else:
                slices['y1'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
                
        if data.y2 is not None:
            if data.y2.size(0) == batch.size(0):
                slices['y2'] = node_slice
            else:
                slices['y2'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices

# %%
target_vec = ['y0','y1','y2']
def isnan(x):
    """ Simple utility to see what is NaN """
    return x!=x

def nonmissingvales(loader, target_num):
    """ function that computes the amount of molecules that do have a specific target """
    count = 0
    for data in loader:
        count +=isnan(data['y%s'%target_num]).sum()
    return len(loader.dataset) - count

def myloss(output_vec, data):
    """ Main Loss that is used for MulitTargets"""
    criterion = F.binary_cross_entropy
    
    regress =  F.mse_loss
    mse_part = 0
    masks = dict()
    loss1 = dict()

    for x in range(len(target_vec)):
        if x >0:
            l = regress(output_vec[x][:,0].view(-1, 1), data['y%s'%x].view(-1, 1))
        else:
            l = criterion(output_vec[x], data['y%s'%x].unsqueeze(1))
            # print(data['y%s'%x].unsqueeze(1))
            # print(output_vec[x])

        mse_part += l
        loss1[x] = torch.sqrt(l+1e-16)
    
    loss = torch.sqrt(mse_part)
    mylist = [loss]
    for x in range(0, len(target_vec)):
        mylist.append(loss1[x]) 
    return mylist

EPS = 1e-10
lamb0 = 1 # classification loss weight, nesse caso a loss total de todas as tasks
lamb1 = 0.1 # s1 unit regularization
lamb2 = 0.1 # s2 unit regularization
lamb3 = 0.1 # s1 entropy regularization
lamb4 = 0.2 # s2 entropy regularization
ratio = 0.5

def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res

def train_multi(train_loader, model, scheduler, optimizer):
    print('train...........')
    """ Main function to train the model """
    model.train()
    device = 'cpu'
    loss_all = 0
    output_vec = []
    tar_vec = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, w1, w2, s1, s2,  perm1, perm2  = model(data)
        loss_t = myloss(output, data)
        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,ratio)
        loss_tpk2 = topk_loss(s2,ratio)

        print(loss_t, loss_p1, loss_p2, loss_tpk1, loss_tpk2)
        lossP = loss_t[0]

        lossP.backward(retain_graph=True)
        loss = lamb0*lossP + lamb1 * loss_p1 + lamb2 * loss_p2 
                #    + lamb3 * loss_tpk1 + lamb4 *loss_tpk2

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def val_multi(loader, model, target_vec):
    """ Main function to validate the model """
    model.eval()
    device = 'cpu'
    loss_all = 0
    loss1_all = dict()
    y_pred_list = []
    tar_vec1 = []
    for x in range(len(target_vec)):
        loss1_all[x]= 0
    
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2,  perm1, perm2  = model(data)
        loss_t = myloss(output, data)
        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,ratio)
        loss_tpk2 = topk_loss(s2,ratio)
        a = np.round(output[0].detach().numpy())
        aa = output[1].detach().numpy()
        aaa = output[2].detach().numpy()

        y_pred_list.append((a, aa, aaa))
        tar_vec1.append((data.y0, data.y1, data.y2))

        # lossP = loss_t[1] * 0.8 + loss_t[2] * 0.1 + loss_t[3] * 0.1
        lossP = loss_t[0]
        loss = lamb0*lossP + lamb1 * loss_p1 + lamb2 * loss_p2 
                #    + lamb3 * loss_tpk1 + lamb4 *loss_tpk2

        loss_all += loss.item() * data.num_graphs
    
    return loss_all/len(loader.dataset), y_pred_list, tar_vec1

def test_multi(loader, model, target_vec, name):
    model.eval()
    device = 'cpu'
    loss_all = 0
    loss1_all = dict()
    output_vec1 = []
    y_true = []
    for x in range(len(target_vec)):
        loss1_all[x]= 0
    y_pred_list = []
    cont = 0    
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2,  perm1, perm2  = model(data)

        a = np.round(output[0].detach().numpy())
        aa = output[1].detach().numpy()
        aaa = output[2].detach().numpy()
        y_pred_list.append((a, aa, aaa))
        y_true.append((data.y0,data.y1, data.y2))

        if not os.path.exists(path_scores_test +'_'+ name +  "/" + str(cont)):
            os.makedirs(path_scores_test +'_'+ name +  "/" + str(cont))
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/s1.npy',  s1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/s2.npy',  s2.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/w1.npy',  w1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/w2.npy',  w2.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/perm1.npy',  perm1.detach().numpy())
        save(path_scores_test +'_'+ name + "/" + str(cont) + '/perm2.npy',  perm2.detach().numpy())
        cont += 1

    return y_pred_list,y_true[0]

# def binary_acc(y_pred, y_test):
#     y_pred_tag = torch.round(y_pred)

#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum/y_test.shape[0]
#     acc = torch.round(acc * 100)
    
#     return acc

warnings.filterwarnings("ignore")
root_folder = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/'
data_folder = os.path.join(root_folder, 'data/')
result_folder = os.path.join(root_folder, 'results')
dataset_folder = 'hcp_1200'
rst_data_file = 'hcp_200_netmaps_2D_v2.hdf5'
phenotype_csv = os.path.join(data_folder,dataset_folder,'behaviors_merge.csv')
     

dictVars = {
    'Gender':{ 'regression' : False,'n_outputs': 2 },
    'PMAT24_A_CR':{ 'regression' : True,'n_outputs': 1 },
    'MMSE_Score':{ 'regression' : True,'n_outputs': 1 },
    'PSQI_Score':{ 'regression' : True,'n_outputs': 1 },
    'NEOFAC_N':{ 'regression' : True,'n_outputs': 1 },
    }
path_scores_test = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_multitask/scores/test'

def main():
    name = 'hard_middle_SHARED_CWEIGHT_HCP_MTL_PMAT24_A_CR_MMSE_Score_FAM_HIST_v6'   
    if not os.path.exists(os.path.join(result_folder,name)):
        os.makedirs(os.path.join(result_folder, name, 'model'))
        
    RECREATE_DATA  = True
    if RECREATE_DATA:
        params = dict()
        params['seed'] = 123
        file_rst = h5py.File(os.path.join(data_folder,dataset_folder,rst_data_file), 'r')
        subject_IDs = list(file_rst.keys()) # Reader.get_ids()
        
        labels = func.get_subject_score(phenotype_csv, subject_IDs, score='FamHist_Fath_None', regression=False) # 2
        labels1 = func.get_subject_score(phenotype_csv, subject_IDs, score='PMAT24_A_CR', regression=True) # 40
        labels3 = func.get_subject_score(phenotype_csv, subject_IDs, score='MMSE_Score', regression=True) #6
        # Number of subjects and classes for binary classification
        num_subjects = len(subject_IDs)
        params['n_subjects'] = num_subjects

        # y_data = np.zeros([num_subjects, num_classes]) # n x 2
        y0 = np.zeros([num_subjects, 1]) # n x 1
        y1 = np.zeros([num_subjects, 1])
        y2 = np.zeros([num_subjects, 1])
        # Get class labels for all subjects

        for i in range(num_subjects):
            # y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
            y0[i] = int(labels[subject_IDs[i]])
            if labels1[subject_IDs[i]] != '':
                y1[i] = int(float(labels1[subject_IDs[i]]))
            else: 
                y1[i] = 0
            
            if labels3[subject_IDs[i]] != '':
                y2[i] = int(float(labels3[subject_IDs[i]]))
            else: 
                y2[i] = 0
            

        # Compute feature vectors (vectorised connectivity networks)
        fea_corr = func.get_networks(subject_IDs, file_rst, kind='cn_matrix1') #(1035, 200, 200)
        fea_pcorr = func.get_networks(subject_IDs,file_rst, kind='cn_matrix2') #(1035, 200, 200)
        if not os.path.exists(os.path.join(data_folder,dataset_folder,'raw')):
            os.makedirs(os.path.join(data_folder,dataset_folder,'raw'))
        for i, subject in enumerate(subject_IDs):
            dd.io.save(os.path.join(data_folder,dataset_folder,'raw',subject+'.h5'),{'corr':fea_corr[i],'pcorr':fea_pcorr[i],'y0':y0[i], 'y1':y1[i], 'y2':y2[i]})

    batchSize = 64
    opt_method = 'Adam'
    weightdecay = 0.0005
    lr = 0.01
    num_epoch = 100
    save_model = True
        
    dataset = HCPDataset(os.path.join(data_folder,dataset_folder),name)
    dataset.data['y0'] = dataset.data['y0'].squeeze().to(torch.float32)
    dataset.data.y1 = normalize(dataset.data.y1.squeeze().to(torch.float32), p=20, dim = 0) # quando é regressao
    dataset.data.y2 = normalize(dataset.data.y2.squeeze().to(torch.float32), p=20, dim = 0)# quando é regressao
    from sklearn.utils import compute_class_weight
    dataset.data.x[dataset.data.x == float('inf')] = 0
    global class_weights
    class_weights=compute_class_weight(class_weight = 'balanced',classes = np.unique(dataset.data['y0']), y = dataset.data['y0'].numpy())
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    print(class_weights) #([1.0000, 1.0000, 4.0000, 1.0000, 0.5714])

    indexes_train = np.loadtxt(os.path.join(data_folder,dataset_folder, "indexes_train_balanced.txt"), dtype=float).astype(int)

    train_dataset = dataset[indexes_train]

    from sklearn.model_selection import KFold
    kfold=KFold(n_splits=5,shuffle=True, random_state=42)
    results_per_fold = []
    for fold,(train_idx,test_idx) in enumerate(kfold.split(train_dataset)):
        writer = SummaryWriter(os.path.join(root_folder,'logy',str(name), str(fold)))
        model = Network(200,0.5,3).to('cpu')
    
        print(model, file=open(os.path.join(result_folder, name, 'model', str(name)+'.txt'), 'a'))

        if opt_method == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=weightdecay)
        elif opt_method == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr =lr, momentum = 0.9, weight_decay=weightdecay, nesterov = True)
        elif opt_method == 'RMSPROP':
            optimizer = torch.optim.RMSprop(model.parameters(), lr =lr, weight_decay=weightdecay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, threshold=1e-3,
                                                                            verbose=True)

        best_loss = 200
        print('------------fold no---------{}----------------------'.format(fold))

        train_loader = DataLoader(
                            dataset[train_idx], 
                            batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(
                            dataset[test_idx],
                            batch_size=batchSize, shuffle=True)

        early_stopping = EarlyStopping(patience=8, verbose=True)
        for epoch in range(1, num_epoch):
            for param_group in optimizer.param_groups:
                print("LR", param_group['lr'])
            train_loss = train_multi(train_loader, model, scheduler, optimizer)
            val_loss, outputs, target   = val_multi(val_loader, model, target_vec)
            from sklearn.metrics import mean_squared_error, accuracy_score
            print( "Acc: {:.3f}", accuracy_score(target[0][0],np.round(outputs[0][0])))
            
            print("MSE PMAT: {:.3f}", mean_squared_error(target[0][1],outputs[0][1]))
            print("MSE MMSE_Score: {:.3f}", mean_squared_error(target[0][2],outputs[0][2]))
            # mean_squared_error()
            print('epoch %i: normalized train loss %0.2f val loss %0.2f' %(epoch, train_loss, val_loss), end="\r")

            writer.add_scalars('Loss', {'train_loss': train_loss, 'val_loss': val_loss},  epoch)
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                'Val Loss: {:.7f}'.format(epoch, train_loss,val_loss))
            scheduler.step(val_loss)
            early_stopping(val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if val_loss < best_loss :
                print("saving best model")
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_model:
                    torch.save(best_model_wts, os.path.join(result_folder, name, 'model',str(name)+'_fold'+ str(fold)+'.pth'))

        results_per_fold.append([train_loss, val_loss])

if __name__ == "__main__":
    main()
