from ODGCN import *

class ODRNCell(nn.Module):
    def __init__(self,
                 num_nodes,
                 flow_dim,
                 node_embeddig_dim,
                 hidden_dim,
                 max_diffusion_step):
        super(ODRNCell, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        input_dim = 0
        if flow_dim is not None:
            input_dim += flow_dim
        if node_embeddig_dim is not None:
            input_dim += node_embeddig_dim

        input_dim += hidden_dim
        self.gate = ODGCN(in_hidden = input_dim, out_hidden =  2 * hidden_dim, max_diffusion_step=max_diffusion_step)
        self.update = ODGCN(in_hidden = input_dim, out_hidden = hidden_dim, max_diffusion_step=max_diffusion_step)

    def forward(self, od_graph, state, flow_hidden_src, flow_hidden_dst, node_embedding):
        '''
        :param od_matrix: B,N,N
        :param state: B,N,F
        :param src_node_feat: B,N,2
        :param dest_node_feat: B,N,F
        :return:
        '''

        #上一时刻的hidden_state
        state = state.to(od_graph.device)
        value = torch.sigmoid(self.gate(od_graph, state, flow_hidden_src, flow_hidden_dst, node_embedding))

        # torch.split(tensor, split_size, dim=) split_size是切分后每块的大小，不是切分为多少块, dim是切分维度
        z, r = torch.split(value, self.hidden_dim, dim=-1)

        # candidate = torch.cat((node_feats, z * state), dim=-1)

        hc = torch.tanh(self.update(od_graph, z * state, flow_hidden_src, flow_hidden_dst,node_embedding))

        h = r * state + (1 - r) * hc

        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros((batch_size, self.num_nodes, self.hidden_dim))

if __name__ == '__main__':
    batch_size = 32
    num_nodes = 256
    hidden_size = 128
    node_dim = 10
    max_diffusion_step = 2

    od_graph = torch.randn(batch_size,num_nodes,num_nodes)
    state = torch.randn(batch_size,num_nodes,hidden_size)
    crowd_flow = torch.randn(batch_size,num_nodes,2)
    node_embedding = torch.randn(batch_size,num_nodes,node_dim)

    odgcn = ODGCN(hidden_size+node_dim+1,
                 hidden_size,
                 max_diffusion_step=2)
    out = odgcn(od_graph, state, crowd_flow, node_embedding)
    print(out.shape)

    odcrncell = ODRNCell(flow_dim=1,
            node_embeddig_dim=node_dim,
            hidden_dim=hidden_size,
            max_diffusion_step=max_diffusion_step)

    out = odcrncell(od_graph, state, crowd_flow, node_embedding)
    print(out.shape)