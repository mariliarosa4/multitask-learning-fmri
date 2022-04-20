
import pandas as pd
import numpy as np
import deepdish as dd
import csv
from sklearn.preprocessing import LabelEncoder
import os.path as osp
from os import listdir

import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
from sklearn.model_selection import KFold

def get_adjacency(cn_matrix, threshold=0.0):
    # mask = (cn_matrix > np.percentile(cn_matrix, threshold)).astype(np.uint8)
    abs_matrix = np.abs(cn_matrix)
    mask = (abs_matrix > threshold).astype(np.uint8)
    sparse_matrix = cn_matrix * mask
    nodes, neighbors = np.nonzero(mask)
    sparse_indices = {}
    for i, node in enumerate(nodes):
        #remove self-loops of indices dict
        if not neighbors[i] == node:
            if not node in sparse_indices: 
                sparse_indices[node] = [neighbors[i]]
            else:
                sparse_indices[node].append(neighbors[i])
    return sparse_matrix, mask

def get_subject_score(phenotype_csv,subject_list, score, regression):
    scores_dict = {}
    labelencoder = LabelEncoder()

    dfPhenotype = pd.read_csv(phenotype_csv)

    dfPhenotype[str(score+'_cat')] = labelencoder.fit_transform(dfPhenotype[score])
    
    print(score)
    print(dfPhenotype[score])
    print(dfPhenotype[str(score+'_cat')])
    
    for index, row in dfPhenotype.iterrows():
        if str(int(row['Subject'])) in subject_list:
            if regression:
                scores_dict[str(int(row['Subject']))] = row[str(score)]
            else:
                scores_dict[str(int(row['Subject']))] = row[str(score+'_cat')]
    #            else:
            # print('Nao tem esse subject')
    return scores_dict

from sklearn import preprocessing
# Load precomputed fMRI connectivity networks
def get_networks(subject_list, hdf5_data, kind,threshold=0.0, binary=False):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []

    for subject in subject_list:
        matrix = hdf5_data[subject]['visit1'][kind][:]
        # scaler = preprocessing.StandardScaler().fit(matrix)
        # matrix_ = scaler.transform(matrix)
        all_networks.append(matrix)        

    norm_networks = [mat for mat in all_networks]


    networks = np.stack(norm_networks)

    return networks

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


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

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

def train_val_test_split(kfold = 5, fold = 0):
    n_sub = 50
    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id

