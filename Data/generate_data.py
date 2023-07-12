import pickle

import numpy as np
import pandas as pd
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from Data.DataSet import Dataset


def get_and_normalize_data(path):
    df = pd.read_csv(path, header=[0, 1])

    metrics = df.dropna()

    metrics = metrics.drop(['TimeStamp', 'label'], axis=1)
    metrics.columns.names = ['pod', 'metric']
    tempm = metrics.swaplevel('metric', 'pod', axis=1).stack()
    tempm = (tempm - tempm.mean()) / (tempm.std())
    metrics = tempm.unstack().swaplevel('metric', 'pod', axis=1).stack().unstack()
    metrics = metrics.dropna()

    return metrics


def get_time_sequence(data, sequence_len=30, nodes_num=50, selected_indexes=None):
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
                     range(data.shape[0] - sequence_len + 1)]
    else:
        sequences = [data[i:i + sequence_len].reshape(sequence_len, nodes_num, -1) for i in
                     selected_indexes]
    return sequences


def get_data(nodes_num, table_path='./DatasetUpdate/MMS.csv',
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

    total_size, sequence_len, num_nodes, node_dim = sequences_x.shape

    sequences_x = sequences_x.swapaxes(1, 3)

    return sequences_x, sequences_y, edge_index


def get_train_valid_test(data, batch_size=32, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
    train_set, valid_set, test_set = random_split(data, [train_ratio, valid_ratio, test_ratio])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, drop_last=True, shuffle=False)

    return train_loader, valid_loader, test_loader


def generate_data(nodes_num):
    x, y, edge_index = get_data(nodes_num)
    data = Dataset(x, y)
    train_loader, valid_loader, test_loader = get_train_valid_test(data, nodes_num)

    return train_loader, valid_loader, test_loader, edge_index
