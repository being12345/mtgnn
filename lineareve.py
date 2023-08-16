from dataclasses import dataclass, field
from typing import List, Any

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torch import nn

from Data.generate_data import generate_data
from model.criterions import MapeCriterion, MaeCriterion, RrseCriterion
from model.log import Log, LogConfig
from model.regression_model import ModelConfig, ReModel
from model.net import Net, MtgnnConfig, LineveConf, LinearEvolve

defaults = [
    {"netconf": "lineveConf"},
    {"logconf": "logconf"},
    {"modelconf": "modelconf"}
]


@dataclass
class Conf:
    defaults: List[Any] = field(default_factory=lambda: defaults)


cs = ConfigStore.instance()
cs.store(name='conf', node=Conf)
cs.store(group="netconf", name="lineveConf", node=LineveConf(in_dim=6, out_dim=1))
cs.store(group="logconf", name="logconf", node=LogConfig)
cs.store(group="modelconf", name="modelconf", node=ModelConfig)


@hydra.main(version_base=None, config_name='conf')
def train(conf):
    train_loader, valid_loader, test_loader, edge_index = generate_data(168, 50, 8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = LinearEvolve(conf['netconf'])
    loss = nn.MSELoss(reduction='mean').to(device)
    train_criterion_set = [MapeCriterion(device), MaeCriterion(device), RrseCriterion(device)]
    valid_criterion_set = [MapeCriterion(device), MaeCriterion(device), RrseCriterion(device)]
    optim = torch.optim.Adam(net.parameters())
    log = Log(conf['logconf']['train_criterion_set'], conf['logconf']['valid_criterion_set'])

    model = ReModel(net, loss, train_criterion_set, valid_criterion_set, optim, log, conf['modelconf'], edge_index)
    model.trainer(train_loader, valid_loader)


if __name__ == '__main__':
    train()
