import pickle

import pandas as pd
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from Data.DataSet import Dataset
from graph_lstm_vae_ad_ver6 import GraphLSTM_VAE_AD


def get_and_normalize_data(path):
    # X.interpolate(inplace=True)
    # X.bfill(inplace=True)
    # data = X.values

    df = pd.read_csv(path, header=[0, 1])
    metrics = df.dropna()

    metrics = metrics.drop(['TimeStamp', 'label'], axis=1)
    metrics.columns.names = ['pod', 'metric']
    tempm = metrics.swaplevel('metric', 'pod', axis=1).stack()
    tempm = (tempm - tempm.mean()) / (tempm.std())
    metrics = tempm.unstack().swaplevel('metric', 'pod', axis=1).stack().unstack()
    metrics = metrics.dropna()

    return metrics


def get_data(nodes_num, table_path='./DatasetUpdate/MMS.csv',
             edge_path='./DatasetUpdate/MMS_topology.pk'):
    """
    default window size is 30
    :return: x, y, and edge_index(coo)
    """
    metrics = get_and_normalize_data(table_path)

    metrics.columns.names = ['instance', 'metrics']
    latency = metrics.stack('instance')['PodLatency(s)'].to_numpy().reshape((metrics.shape[0], -1))
    performance = metrics.drop(['PodLatency(s)'], axis=1, level=1).to_numpy()

    with open(edge_path, 'rb') as f:
        edge_index = pickle.load(f)

    # generate time series
    data_helper = GraphLSTM_VAE_AD()
    sequences_x = data_helper.get_time_sequence(performance, nodes_num)
    sequences_y = data_helper.get_time_sequence(latency, nodes_num)

    return sequences_x, sequences_y, edge_index


def get_train_valid_test(self, data, nodes_num: int, selected_indexes=None):
    train_set, valid_set, test_set = random_split(data, [0.6, 0.2, 0.2])

    train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=self.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, drop_last=True, shuffle=False)

    return train_loader, valid_loader, test_loader


def generate_data(nodes_num):
    x, y, edge_index = get_data(nodes_num)
    data = Dataset(x, y)
    data_helper = GraphLSTM_VAE_AD()
    train_loader, valid_loader, test_loader = data_helper.get_train_valid_test(data, 50)

    return train_loader, valid_loader, test_loader, edge_index
