from dataclasses import dataclass, field
from typing import List, Any

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torch import nn

from Data.generate_data import generate_data
from model.criterions import MapeCriterion, MaeCriterion, RrseCriterion
from model.log import Log
from model.net import Net
from modle import MtgnnConfig, LogConfig, ModelConfig, Model

defaults = [
    {"netconf": "mtgnnconf"},
    {"logconf": "logconf"},
    {"modelconf": "modelconf"}
]


@dataclass
class Conf:
    defaults: List[Any] = field(default_factory=lambda: defaults)


cs = ConfigStore.instance()
cs.store(name='conf', node=Conf)
cs.store(group="netconf", name="mtgnnconf", node=MtgnnConfig(buildA_true=True, num_nodes=50, in_dim=6))
cs.store(group="logconf", name="logconf", node=LogConfig)
cs.store(group="modelconf", name="modelconf", node=ModelConfig)


@hydra.main(version_base=None, config_name='conf')
def main(conf):
    train_loader, valid_loader, test_loader, edge_index = generate_data(168, 50, 8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(conf.netconf)
    loss = nn.MSELoss(reduction='mean').to(device)
    train_criterion_set = [MapeCriterion(device), MaeCriterion(device), RrseCriterion(device)]
    valid_criterion_set = [MapeCriterion(device), MaeCriterion(device), RrseCriterion(device)]
    optim = torch.optim.Adam(net.parameters())
    log = Log(conf.logconf, conf.logconf)

    model = Model(net, loss, train_criterion_set, valid_criterion_set, optim, log, conf.modelconf, edge_index)
    model.trainer(train_loader, valid_loader)


if __name__ == '__main__':
    main()
