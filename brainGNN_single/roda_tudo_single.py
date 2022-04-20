import sys
from time import time
import warnings
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

EPS = 1e-10

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
    L = L.to('cpu')
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res
##########################################################################################################################

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
target_vec = ['y0']
EPS = 1e-10
lamb0 = 1 # classification loss weight, nesse caso a loss total de todas as tasks
lamb1 = 0 # s1 unit regularization
lamb2 = 0 # s2 unit regularization
lamb3 = 0.1 # s1 entropy regularization
lamb4 = 0.1 # s2 entropy regularization
lamb5 = 0
ratio = 0.5

def isnan(x):
    """ Simple utility to see what is NaN """
    return x!=x

def nonmissingvales(loader, target_num):
    """ function that computes the amount of molecules that do have a specific target """
    count = 0
    for data in loader:
        count +=isnan(data['y%s'%target_num]).sum()
    return len(loader.dataset) - count

def val_loss_single(loader,epoch, model, target_vec):
    """ Main function to validate the model """
    model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    device = 'cpu'
    loss_all = 0
    y_pred_list = []
    target_list = []
    for data in loader:
        output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        # a = torch.amax(output, dim=1)
        output = output.to(torch.float32)
        data.y0 = data.y0.to(torch.float32)
        
        loss_c = F.mse_loss(output, data.y0)


        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,ratio)
        loss_tpk2 = topk_loss(s2,ratio)
        print(output.detach().numpy())
        y_pred_list.append(output.detach().numpy())
        target_list.append(data.y0.detach().numpy())

        loss_consist = 0
        for c in range(1):
            loss_consist += consist_loss(s1[data.y0 == c])
        loss = lamb0*loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 \
                   + lamb3 * loss_tpk1 + lamb4 *loss_tpk2 + lamb5 * loss_consist
        # loss = loss_c

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset), y_pred_list, target_list

def test_pearson_single(loader, model):
    from torchmetrics import PearsonCorrCoef
    model.eval()
    p = 0
    for data in loader:
        data = data.to('cpu')
        output, w1, w2, s1, s2 ,perm1, perm2  = model(data)
        pearson = PearsonCorrCoef()
        p += pearson(output[:,0], data.y0)

    print(p)
    print(len(loader.dataset))
    return float(p / (len(loader.dataset)))

def test_loss_single(test_loader, model, target_vec, name):
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
        loss_tpk1 = topk_loss(s1,ratio)
        loss_tpk2 = topk_loss(s2,ratio)
        loss_consist = 0
        for c in range(1):
            loss_consist += consist_loss(s1[data.y0 == c])
        loss = lamb0*loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 \
                   + lamb3 * loss_tpk1 + lamb4 *loss_tpk2 + lamb5

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(test_loader.dataset), y_pred_list

def train_single(train_loader, epoch, model, optimizer, train_dataset, writer, target_vec):
    
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
        data = data.to('cpu')
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
        loss_tpk1 = topk_loss(s1,ratio)
        loss_tpk2 = topk_loss(s2,ratio)
        loss_consist = 0
        for c in range(1):
            loss_consist += consist_loss(s1[data.y0 == c])
        loss = lamb0*loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 \
                   +lamb3 * loss_tpk1 + lamb4 *loss_tpk2 + lamb5 * loss_consist

        # loss = loss_c          
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


warnings.filterwarnings("ignore")
root_folder = '/home/dbserver/Desktop/projetos/repositorios_mestrado/1_mestrado_final/experimentos_finais/GNN/BrainGNN_single/'
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
path_scores_test = './scores/test'

def main():
    name = 'Single_MMSE_Score_v3'   
    if not os.path.exists(os.path.join(result_folder,name)):
        os.makedirs(os.path.join(result_folder, name, 'model'))
        
    RECREATE_DATA  = True
    if RECREATE_DATA:
        params = dict()
        params['seed'] = 123
        file_rst = h5py.File(os.path.join(data_folder,dataset_folder,rst_data_file), 'r')
        subject_IDs = list(file_rst.keys()) # Reader.get_ids()
        
        labels = func.get_subject_score(phenotype_csv, subject_IDs, score='MMSE_Score', regression=True) # 2
        num_subjects = len(subject_IDs)
        params['n_subjects'] = num_subjects
        y0 = np.zeros([num_subjects, 1]) # n x 1
        for i in range(num_subjects):
            y0[i] = int(labels[subject_IDs[i]])


        # Compute feature vectors (vectorised connectivity networks)
        fea_corr = func.get_networks(subject_IDs, file_rst, kind='cn_matrix1') #(1035, 200, 200)
        fea_pcorr = func.get_networks(subject_IDs,file_rst, kind='cn_matrix2') #(1035, 200, 200)
        if not os.path.exists(os.path.join(data_folder,dataset_folder,'raw')):
            os.makedirs(os.path.join(data_folder,dataset_folder,'raw'))
        for i, subject in enumerate(subject_IDs):
            dd.io.save(os.path.join(data_folder,dataset_folder,'raw',subject+'.h5'),{'corr':fea_corr[i],'pcorr':fea_pcorr[i],'y0':y0[i]})

    batchSize = 64
    opt_method = 'SGD'
    weightdecay = 0.005
    lr = 0.001
    num_epoch = 50
    save_model = True
        
    dataset = HCPDataset(os.path.join(data_folder,dataset_folder),name)
    dataset.data.y0 = normalize(dataset.data.y0.squeeze().to(torch.float32), p=20, dim = 0)
    print(dataset.data.y0)
    # dataset.data.y1 = normalize(dataset.data.y1.squeeze().to(torch.float32), p=20, dim = 0) # quando é regressao
    # dataset.data.y2 = normalize(dataset.data.y2.squeeze().to(torch.float32), p=20, dim = 0)# quando é regressao
    dataset.data.x[dataset.data.x == float('inf')] = 0
    
    indexes_train = np.loadtxt(os.path.join(data_folder,dataset_folder, "indexes_train_balanced.txt"), dtype=float).astype(int)

    train_dataset = dataset[indexes_train]

    from net.braingnn_single import Network_single

    from sklearn.model_selection import KFold
    kfold=KFold(n_splits=5,shuffle=True)
    results_per_fold = []
    for fold,(train_idx,test_idx) in enumerate(kfold.split(train_dataset)):
        writer = SummaryWriter(os.path.join(root_folder,'logy',str(name), str(fold)))
        model = Network_single(200,0.5,1).to('cpu')
    
        print(model, file=open(os.path.join(result_folder, name, 'model', str(name)+'.txt'), 'a'))

        if opt_method == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=weightdecay)
        elif opt_method == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr =lr, momentum = 0.9, weight_decay=weightdecay, nesterov = True)
        elif opt_method == 'RMSPROP':
            optimizer = torch.optim.RMSprop(model.parameters(), lr =lr, weight_decay=weightdecay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1,
                                                                    verbose=True)

        best_loss = 200
        print('------------fold no---------{}----------------------'.format(fold))

        train_loader = DataLoader(
                            dataset[train_idx], 
                            batch_size=batchSize)
        val_loader = DataLoader(
                            dataset[test_idx],
                            batch_size=batchSize)

        early_stopping = EarlyStopping(patience=8, verbose=True)
        for epoch in range(1, num_epoch):
            for param_group in optimizer.param_groups:
                print("LR", param_group['lr'])
            tr_loss  = train_single(train_loader,epoch, model,optimizer, train_dataset, writer, target_vec)
            # tr_acc = test_acc_single(train_loader, model)
            # val_acc = test_acc_single(val_loader, model)
            val_loss, y_pred_list, target_list = val_loss_single(val_loader,epoch, model, target_vec)
            writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)

            print(target_list[0][:10], y_pred_list[0][:10])

            print('epoch %i: normalized train loss %0.2f val loss %0.2f' %(epoch, tr_loss, val_loss), end="\r")
            
            from sklearn.metrics import mean_squared_error, accuracy_score
            print("MSE: {:.3f}", mean_squared_error(target_list[0], y_pred_list[0]))

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

        results_per_fold.append([tr_loss, val_loss])

if __name__ == "__main__":
    main()