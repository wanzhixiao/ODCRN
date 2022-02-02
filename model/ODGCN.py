import torch.nn as nn
import torch

class ODGCN(nn.Module):
    def __init__(self,
                 in_hidden,
                 out_hidden,
                 max_diffusion_step):
        super(ODGCN, self).__init__()

        self._max_diffusion_step = max_diffusion_step
        self._linear = nn.Linear(2*in_hidden*(max_diffusion_step + 1), out_hidden)


    def forward(self,od_graph, state, flow_hidden_src, flow_hidden_dst, node_embedding):

        forward_graph = od_graph
        backward_graph = od_graph.permute(0,2,1)

        src_feats = torch.cat([flow_hidden_src,node_embedding,state],dim=-1)
        dst_feats = torch.cat([flow_hidden_dst,node_embedding,state],dim=-1)

        src_out = self._diffusion(forward_graph,src_feats)
        dst_out = self._diffusion(backward_graph,dst_feats)

        out = torch.cat([src_out,dst_out],dim=-1)
        out = self._linear(out)
        return out

    def _diffusion(self,graph,node_feats):

        x = node_feats
        x0 = x
        x1 = torch.bmm(graph, x0)
        x = self._concat(x, x1)

        for k in range(2, self._max_diffusion_step + 1):
            x2 = 2 * torch.bmm(graph, x1) - x0
            x = self._concat(x, x2)
            x1, x0 = x2, x1

        return x;

    @staticmethod
    def _concat(x, x_):
        return torch.cat([x, x_], dim=-1)