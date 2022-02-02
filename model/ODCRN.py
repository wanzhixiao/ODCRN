import torch
import torch.nn as nn

from model.ODRNCell import *
from model.ContextLearner import *

class ODCRN(nn.Module):

    def __init__(self,
                 num_nodes,
                 hidden_dim,
                 num_layers,
                 static_feat_hiddens,
                 dynamic_feat_hiddens,
                 max_diffusion_step):

        super(ODCRN, self).__init__()

        self.encoder = ODDCRNN(
                 num_nodes,
                 hidden_dim,
                 num_layers,
                 static_feat_hiddens,
                 dynamic_feat_hiddens,
                 max_diffusion_step)

        self.decoder= nn.Conv2d(hidden_dim, num_nodes, kernel_size=(1, 1), bias=True)

    def forward(self, od_flow, adj_mat, crowd_flow, node_embedding):
        '''
        :param od_matrix: B,T,N,N
        :param crowd_flow: B,T,N,F
        :param node_embedding: B,N,F
        :return:
        '''

        #1. hidden state
        init_state = self.encoder.init_hidden(od_flow.shape[0])
        #2 encoder
        out = self.encoder(od_flow, init_state, adj_mat, crowd_flow, node_embedding)
        #3. decoder
        out = out.unsqueeze(1)
        out = out.permute(0, 3, 2, 1)  # (B,1,N,hidden) -> (B,hidden,N,1)
        out = self.decoder(out)  # (B,hidden,N,1) -> (B,N,N,1)
        out = out.permute(0, 3, 2, 1)  # B, T, N, N

        return out


class ODDCRNN(nn.Module):
    '''
    RNN
    '''
    def __init__(self,
                 num_nodes,
                 hidden_dim,
                 num_layers,
                 static_feat_hiddens, #list
                 dynamic_feat_hiddens, #list
                 max_diffusion_step):


        super(ODDCRNN, self).__init__()

        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_layers = num_layers

        #保证flow网络个数等于rnn层数
        assert num_layers == len(dynamic_feat_hiddens)

        #保证上一层输出维度等于下一层输入维度
        for i in range(1,len(dynamic_feat_hiddens)):
            assert dynamic_feat_hiddens[i][0] == dynamic_feat_hiddens[i-1][-1]

        self.dcrnn_cells = nn.ModuleList()
        self.src_dynamic_context_learner = nn.ModuleList()
        self.dst_dynamic_context_learner = nn.ModuleList()
        self.static_context_learner = nn.ModuleList()

        for i in range(num_layers):
            self.src_dynamic_context_learner.append(
                GConv(dynamic_feat_hiddens[i][0],
                      dynamic_feat_hiddens[i][-1],
                      num_adj_mats=1,
                      order=2)
            )
            self.dst_dynamic_context_learner.append(
                GConv(dynamic_feat_hiddens[i][0],
                      dynamic_feat_hiddens[i][-1],
                      num_adj_mats=1,
                      order=2)
            )
            self.static_context_learner.append(
                MLP(static_feat_hiddens)
            )
            self.dcrnn_cells.append(
                ODRNCell(
                         num_nodes = num_nodes,
                         flow_dim=dynamic_feat_hiddens[i][-1],
                         node_embeddig_dim=static_feat_hiddens[-1],
                         hidden_dim=hidden_dim,
                         max_diffusion_step=max_diffusion_step)
            )

        self.use_dynamic_context_info = True
        self.use_static_context_info = True

    def forward(self, od_flow, init_state, graph, crowd_flow, node_embeddings):
        '''
        :param x: B,T,N,N
        :param init_state: (num_layers,B,N, hidden_dim)
        :param node_embeddings: (N,F)
        todo: adjacent matrix (forward, backward),  有向图 + GAT
        :return:
        '''

        # 校验维度
        # assert (od_flow.shape[2] == self.node_num) and (od_flow.shape[3] == self.input_dim)

        seq_length = od_flow.shape[1]
        output_hidden = []

        current_inputs_src = crowd_flow[...,0].unsqueeze(-1)
        current_inputs_dst = crowd_flow[...,1].unsqueeze(-1)

        # (N,F) -> (1,N,F)
        node_embeddings = node_embeddings.unsqueeze(0)
        node_embeddings = node_embeddings.repeat(od_flow.shape[0], 1, 1)

        for i in range(self.num_layers):
            state = init_state[i]
            inner_state = []
            flow_state_src = []
            flow_state_dst = []

            for t in range(seq_length):
                src_flow_input = current_inputs_src[:, t, ...]
                dst_flow_input = current_inputs_dst[:, t, ...]
                od_graph = od_flow[:,t,...]

                #获取节点信息
                if self.use_dynamic_context_info:

                    flow_hidden_src = self.src_dynamic_context_learner[i](src_flow_input, graph)
                    flow_hidden_dst = self.dst_dynamic_context_learner[i](dst_flow_input, graph)
                    flow_state_src.append(flow_hidden_src)
                    flow_state_dst.append(flow_hidden_dst)

                if self.use_static_context_info:
                    node_embedding = self.static_context_learner[i](node_embeddings)

                state = self.dcrnn_cells[i](od_graph, state, flow_hidden_src, flow_hidden_dst, node_embedding)

                # 记录每个时间步的hidden state
                inner_state.append(state)

            # 记录当前层的最后一个cell的state
            output_hidden.append(state)

            # 当前层的输出作为下一层的输入
            current_inputs_src = torch.stack(flow_state_src, dim=1)  # (B,T,N,F)
            current_inputs_dst = torch.stack(flow_state_dst, dim=1)  # (B,T,N,F)

        return output_hidden[-1]

    def init_hidden(self, batch_size):
        '''
        初始化每个时间步的hidden status
        :param batch_size:
        :return:
        '''
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


if __name__ == '__main__':

    hidden_dim = 128
    num_layers = 2
    static_feat_hiddens = [10,32,32]
    dynamic_feat_hiddens = [[1,32,32],[32,32,32]]
    max_diffusion_step = 2
    batch_size = 32
    num_nodes = 256
    timestep = 8

    od_graph = torch.randn(batch_size,timestep,num_nodes,num_nodes)
    state = torch.randn(2,batch_size,num_nodes,hidden_dim)
    crowd_flow = torch.randn(batch_size,timestep, num_nodes,2)
    node_embeddings = torch.randn(num_nodes,static_feat_hiddens[0])
    graph = torch.randn(num_nodes,num_nodes)

    model = ODCRN(
        num_nodes,
        hidden_dim,
        num_layers,
        static_feat_hiddens,
        dynamic_feat_hiddens,
        max_diffusion_step
    )

    out = model(od_graph, graph, crowd_flow, node_embeddings)
    print(out.shape)