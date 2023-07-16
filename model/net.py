from dataclasses import dataclass, field

import torch.cuda
from omegaconf import MISSING
from torch_geometric_temporal import EvolveGCNO, MTGNN
import torch.nn as nn


@dataclass
class MtgnnConfig:
    gcn_true: bool = True
    buildA_true: bool = MISSING
    gcn_depth: int = 2
    num_nodes: int = MISSING
    dropout: float = 0.3
    subgraph_size: int = 20
    node_dim: int = 40
    dilation_exponential: int = 2
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    seq_in_len: int = 168
    in_dim: int = MISSING
    seq_out_len: int = 1
    layers: int = 5
    propalpha: float = 0.05
    tanhalpha: int = 3
    layer_norm_affline: bool = True
    kernel_size: int = 7
    kernel_set: list = field(default_factory=lambda: [2, 3, 6, 7])
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, netconf):
        super(Net, self).__init__()
        self.netconf = netconf

        self.evolvegcn = EvolveGCNO(self.netconf['in_dim']).to(self.netconf['device'])

        self.mtgnn = MTGNN(self.netconf['gcn_true'], self.netconf['buildA_true'],
                           self.netconf['gcn_depth'],
                           self.netconf['num_nodes'], kernel_set=self.netconf['kernel_set'],
                           kernel_size=self.netconf['kernel_size'],
                           dropout=self.netconf['dropout'],
                           subgraph_size=self.netconf['subgraph_size'],
                           node_dim=self.netconf['node_dim'],
                           dilation_exponential=self.netconf['dilation_exponential'],
                           conv_channels=self.netconf['conv_channels'],
                           residual_channels=self.netconf['residual_channels'],
                           skip_channels=self.netconf['skip_channels'],
                           end_channels=self.netconf['end_channels'],
                           seq_length=self.netconf['seq_in_len'], in_dim=self.netconf['in_dim'],
                           out_dim=self.netconf['seq_out_len'],
                           layers=self.netconf['layers'], propalpha=self.netconf['propalpha'],
                           tanhalpha=self.netconf['tanhalpha'], layer_norm_affline=False).to(
            self.netconf['device'])

    def forward(self, x, edge_index):
        x_transform = self.evolvegcn(x, edge_index).transpose(1, -1)
        y = self.mtgnn(x_transform, edge_index)
        return y
