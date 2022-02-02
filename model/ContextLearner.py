import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,hidden_dim):
        super(MLP, self).__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = len(hidden_dim) - 1
        self._linear = list()

        for i in range(1,self._num_layers+1):
            self._linear.append(
                    nn.Linear(self._hidden_dim[i - 1], self._hidden_dim[i]))
        self._linear = nn.Sequential(*self._linear)

    def forward(self,x):
        return self._linear(x)

class GConv(nn.Module):
    def __init__(self, in_hidden, out_hidden, num_adj_mats, order=2):
        super(GConv, self).__init__()
        self._in_hidden = in_hidden
        self._out_hidden = out_hidden
        self._num_adj_mats = num_adj_mats
        self._order = order
        self._linear = nn.Linear(self._in_hidden * (self._num_adj_mats * self._order + 1), self._out_hidden)

    def forward(self, x, adj_mats):
        '''
        :param x: input, shape = [B,N,C]
        :param adj_mats: tensor, shape = [N,N,num_adjs]
        :return:
        '''
        # adj_mats = normalize_adj_mats(adj_mats)
        out = [x]
        for i in range(self._num_adj_mats):
            _x = x
            for k in range(self._order):
                #torch.Size([256, 256]) torch.Size([32, 256, 2])
                _x = torch.matmul(adj_mats, _x) #(N,N) x (B,N,C) -> (1,N,N) x (B,N,C) -> (B,N,C)
                out += [_x]
        h = torch.cat(out, dim=-1)
        h = self._linear(h)
        return h