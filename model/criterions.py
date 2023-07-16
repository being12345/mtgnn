import torch
from torchmetrics import RelativeSquaredError, PearsonCorrCoef

"""
Flowing classes only used for service single step predict
"""


class Criterion:
    """
    use for other no gradient criterion

    """

    def __init__(self, device):  # TODO: update other criterion
        """
        all criterion init with `[]`
        """
        self.record = []
        self.device = device

    def _calculate(self, pred, y):
        raise NotImplementedError

    def cache(self, pred, y):
        self.record.append(self._calculate(pred, y).item())

    def calculate_mean(self):
        return sum(self.record) / len(self.record)


class MapeCriterion(Criterion):
    def __init__(self, device):  # TODO: update other criterion
        super().__init__(device)

    def _calculate(self, pred, y):
        with torch.no_grad():
            return torch.mean(torch.abs(y - pred) / torch.abs(y))


class MaeCriterion(Criterion):
    def __init__(self, device):  # TODO: update other criterion
        super().__init__(device)

    def _calculate(self, pred, y):
        with torch.no_grad():
            return torch.mean(torch.abs(pred - y))


class RrseCriterion(Criterion):
    def __init__(self, device):
        super().__init__(device)

    def _calculate(self, pred, y):
        with torch.no_grad():
            return RelativeSquaredError(num_outputs=len(y), squared=False).to(self.device)(torch.squeeze(pred).transpose(0, 1),
                                                                           torch.squeeze(y).transpose(0, 1))

