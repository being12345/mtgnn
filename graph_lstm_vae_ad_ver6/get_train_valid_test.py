import logging

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange, tqdm
from torch_geometric.nn import GCNConv, GATConv
import IPython
import matplotlib.pyplot as plt


def get_train_valid_test(X: pd.DataFrame):
    X.interpolate(inplace=True)
    X.bfill(inplace=True)
    data = X.values

    sequences = self.get_time_sequence(data, nodes_num, selected_indexes)

    indices = np.random.permutation(len(sequences))
    split_point = int(0.3 * len(sequences))
    train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=False)
