import torch
from torch.utils.data import Dataset, DataLoader, random_split


class Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y=None, single_step=True):
        self.single_step = single_step

        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            if self.single_step: return self.x[idx], self.y[idx]
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
