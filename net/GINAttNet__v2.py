# Copyright (c) 2019, Firmenich SA (Fabio Capela)
# Copyright (c) 2019, Firmenich SA (Guillaume Godin)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Import functions used everywhere
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sequential as Seq, Linear
from torch_geometric.nn import GINConv, GATConv, global_add_pool
from torch_geometric.data import DataLoader, Batch

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GINAttNet(torch.nn.Module):
    def __init__(self, n_features, n_outputs, dim=95):
        super(GINAttNet, self).__init__()
        self.input = None
        ratio = 0.5
        self.dim1 = 64
        self.dim2 = 100
        self.dim5 = 12
        self.dim4 = 64
        self.k = 8
        self.R = dim
        # Preparation of the Graph Isomorphism Convolutional Layer
        nn1 = Seq(Linear(n_features, self.k), ReLU(), Linear(self.k, dim))
        nn2 = Seq(Linear(n_features, self.k), ReLU(), Linear(self.k, self.dim2))

        self.conv1 = GINConv(nn1, dim)
        self.pool1 = TopKPooling(dim, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

        # self.bn1 = torch.nn.BatchNorm1d(dim)
        # Preparation of the Attention Layer
        self.conv2 = GINConv(nn2,self.dim2)
        self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

        # Preparation of the Fully Connected Layer
        self.fc1 = Linear((self.dim1+self.dim2)*2, self.dim4)
        
        self.heads = torch.nn.ModuleList()
        # for _ in range(self.n_tasks):
        self.heads.append(Seq(torch.nn.Linear( self.dim4, self.dim5), torch.nn.Dropout(0.2), torch.nn.ReLU(),torch.nn.Linear(self.dim5, 1)))
        self.heads.append(Seq(torch.nn.Linear( self.dim4, self.dim5), torch.nn.Dropout(0.2), ReLU(),torch.nn.Linear(self.dim5, 1)))
        self.heads.append(Seq(torch.nn.Linear( self.dim4, self.dim5), torch.nn.Dropout(0.2), ReLU(),torch.nn.Linear(self.dim5, 1)))

        
    def forward(self, data):
        x, edge_index, batch, edge_attr, pos = data.x, data.edge_index, data.batch, data.edge_attr,data.pos
        x = self.conv1(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)
        perm2 = perm

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.fc1(x))
        # x = self.bn1(x)
        # Attention Layer
        # Fully Connected Layer
        # x = global_add_pool(x, batch)
        # x = F.relu(self.fc1(x))
        outputs = []
        out1 = F.sigmoid(self.heads[0](x))
        # out1 = self.heads[0](x)
        out2 = self.heads[1](x)
        out3 = self.heads[2](x)
        outputs = [out1, out2, out3]

        return outputs