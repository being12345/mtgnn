from dataclasses import dataclass, field
from omegaconf import MISSING
from torch_geometric_temporal import EvolveGCNO, MTGNN
import torch.nn as nn


@dataclass
class MtgnnConfig:
    gcn_true: bool = True
    buildA_true: bool = MISSING
    gcn_depth: bool = 2
    num_nodes: int = 50
    dropout: int = 0.3
    subgraph_size: int = 20
    node_dim: int = 40
    dilation_exponential: int = 2
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    seq_in_len: int = 168
    in_dim: int = 7
    seq_out_len: int = 1
    layers: int = 5
    propalpha: int = 0.05
    tanhalpha: int = 3
    layer_norm_affline: bool = True
    kernel_size: int = 7
    kernel_set: list = field(default_factory=lambda: [2, 3, 6, 7])


class Net(nn.Module):
    def __init__(self, model_config, device):
        self.model_config = model_config
        self.evolvegcn = EvolveGCNO(self.model_config['in_dim']).to(device)

        self.mtgnn = MTGNN(self.model_config['gcn_true'], self.model_config['buildA_true'],
                                   self.model_config['gcn_depth'],
                                   self.model_config['num_nodes'], kernel_set=self.model_config['kernel_set'],
                                   kernel_size=self.model_config['kernel_size'],
                                   dropout=self.model_config['dropout'],
                                   subgraph_size=self.model_config['subgraph_size'],
                                   node_dim=self.model_config['node_dim'],
                                   dilation_exponential=self.model_config['dilation_exponential'],
                                   conv_channels=self.model_config['conv_channels'],
                                   residual_channels=self.model_config['residual_channels'],
                                   skip_channels=self.model_config['skip_channels'],
                                   end_channels=self.model_config['end_channels'],
                                   seq_length=self.model_config['seq_in_len'], in_dim=self.model_config['in_dim'],
                                   out_dim=self.model_config['seq_out_len'],
                                   layers=self.model_config['layers'], propalpha=self.model_config['propalpha'],
                                   tanhalpha=self.model_config['tanhalpha'], layer_norm_affline=False).to(device)

    def forward(self, x, edge_index):
        x_transform = self.evolvegcn(x, edge_index).transpose(1, -1)
        y = self.mtgnn(x_transform, edge_index)
        return y
