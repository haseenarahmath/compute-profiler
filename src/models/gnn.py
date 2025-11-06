import torch
import torch.nn as nn
from src.layers import GraphConv, PairNorm

class TinyDeepGCN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, n_layers=3, dropout=0.1,
                 norm_mode="PN-SI", norm_scale=1.0):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList([GraphConv(hid, hid) for _ in range(n_layers)])
        self.norm = PairNorm(norm_mode, norm_scale)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(hid, out_dim)

    def forward(self, x, adj):
        h = self.fc_in(x)
        for conv in self.convs:
            h = self.drop(h)
            h = conv(h, adj)
            h = self.norm(h)
            h = self.relu(h)
        return self.fc_out(h)


def generate_synthetic_graph(num_nodes=100, feat_dim=32, edge_prob=0.05):
    """Generate a small random adjacency + features for profiling."""
    x = torch.randn(num_nodes, feat_dim)
    adj = (torch.rand(num_nodes, num_nodes) < edge_prob).float()
    adj = (adj + adj.T) / 2  # symmetrize
    return x, adj
