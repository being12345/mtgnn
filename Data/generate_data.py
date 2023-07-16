import pickle
from dataclasses import dataclass, MISSING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from Data.DataSet import Dataset


@dataclass
class DataConfig:
    seq_len: int = MISSING
    num_nodes: int = MISSING
    batch_size: int = MISSING


def get_and_normalize_data(path, is_dropna=False):
    """
    'PodLatency(s)' isn't normalized
    :param is_dropna: if False, use time interpolation
    :param path:
    :return:
    """
    df = pd.read_csv(path, header=[0, 1])

    metrics = df.drop(['TimeStamp', 'label'], axis=1)

    metrics.columns.names = ['pod', 'metric']
    tempm = metrics.swaplevel('metric', 'pod', axis=1).stack()
    latency = tempm['PodLatency(s)']
    tempm = (tempm - tempm.mean()) / (tempm.std())
    tempm['PodLatency(s)'] = latency
    metrics = tempm.unstack().swaplevel('metric', 'pod', axis=1).stack().unstack()

    if is_dropna:
        return metrics.drop_na()

    metrics.interpolate(inplace=True)
    metrics.bfill(inplace=True)
    return metrics


def get_time_sequence(data, sequence_len, nodes_num, selected_indexes=None):
    """
    get time series batch
    :param data: ndarray
        shape: (time_num, microservice)
    :param selected_indexes: int
        select total size
    :return: list
        shape: (total_size, sequence_length(known as timestep), nodes_num, node_dim)
    """
    if selected_indexes is None:
        sequences = [data[i:i + sequence_len].reshape(sequence_len, nodes_num, -1) for i in
                     range(data.shape[0] - sequence_len)]
    else:
        sequences = [data[i:i + sequence_len].reshape(sequence_len, nodes_num, -1) for i in
                     range(selected_indexes)]
    return sequences


def get_data(num_nodes, table_path='./DatasetUpdate/MMS.csv',
             edge_path='./DatasetUpdate/MMS_topology.pk'):
    """
    default window size is 30
    :return: x shape: (total_size, in_dim, num_nodes, seq_len))
    y shape: (batch_size, seq_len, num_nodes, 1)
    and edge_index(coo)
    """
    metrics = get_and_normalize_data(table_path)

    metrics.columns.names = ['instance', 'metrics']
    latency = metrics.stack('instance')['PodLatency(s)'].to_numpy().reshape((metrics.shape[0], -1))
    performance = metrics.drop(['PodLatency(s)'], axis=1, level=1).to_numpy()

    with open(edge_path, 'rb') as f:
        edge_index = pickle.load(f)

    # generate time series
    sequences_x = np.array(get_time_sequence(performance))
    sequences_y = np.array(get_time_sequence(latency))

    sequences_x = sequences_x.swapaxes(1, 3)

    return sequences_x, sequences_y, edge_index


def get_singel_step(seq_len, nodes_num, is_latency, table_path,
                    edge_path):
    """
    default window size is 30
    :return: x shape: (total_size, in_dim, num_nodes, seq_len))
    y shape: (batch_size, seq_len, num_nodes, 1)
    and edge_index(coo)
    """
    metrics = get_and_normalize_data(table_path)

    metrics.columns.names = ['instance', 'metrics']

    latency = metrics.stack('instance')['PodLatency(s)'].to_numpy().reshape((metrics.shape[0], -1))
    if not is_latency:
        metrics = metrics.drop(['PodLatency(s)'], axis=1, level=1)
    data = metrics.values

    edge = get_edge_index(edge_path)

    # generate time series
    sequences_x = np.array(get_time_sequence(data, seq_len, nodes_num))

    y = latency[seq_len - 1:]
    if not is_latency:
        y = y[:-1]
    sequences_y = np.expand_dims(y, (1, 3))

    # sequences_x = sequences_x.swapaxes(1, 3)

    return sequences_x, sequences_y, edge


def get_edge_index(edge_path):
    with open(edge_path, 'rb') as f:
        edge_index = pickle.load(f)
    edge = torch.Tensor(edge_index).type(torch.int64)
    return edge


def get_train_valid_test(data, batch_size, train_ratio, valid_ratio, test_ratio):
    train_set, valid_set, test_set = random_split(data, [train_ratio, valid_ratio, test_ratio])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, drop_last=True, shuffle=False)

    return train_loader, valid_loader, test_loader


def generate_data(seq_len, nodes_num, batch_size, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,
                  is_single_step=True, is_latency=False, table_path='../DatasetUpdate/MMS.csv',
                  edge_path='../DatasetUpdate/MMS_topology.pk'):
    if is_single_step:
        x, y, edge_index = get_singel_step(seq_len, nodes_num, is_latency, table_path, edge_path)
    else:
        x, y, edge_index = get_data()

    data = Dataset(x, y)
    train_loader, valid_loader, test_loader = get_train_valid_test(data, batch_size, train_ratio, valid_ratio,
                                                                   test_ratio)

    return train_loader, valid_loader, test_loader, edge_index
