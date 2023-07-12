import pickle

import pandas as pd

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


def get_data(table_path='./DatasetUpdate/MMS.csv',
             edge_path='./DatasetUpdate/MMS_topology.pk'):
    """
    :return: x, y, and edge_index(coo)
    """
    metrics = get_and_normalize_data(table_path)

    metrics.columns.names = ['instance', 'metrics']
    latency = metrics.stack('instance')['PodLatency(s)'].to_numpy().reshape((metrics.shape[0], -1))
    performance = metrics.drop(['PodLatency(s)'], axis=1, level=1).to_numpy()

    with open(edge_path, 'rb') as f:
        edge_index = pickle.load(f)

    # generate time series
    data_help = GraphLSTM_VAE_AD()
    sequences = data_help.get_time_sequence(data, nodes_num, selected_indexes)

    return performance, latency, edge_index


def generate_data():
    x, y, edge = get_data()
    data = Dataset(x, y)
    data_helper = GraphLSTM_VAE_AD()
    train_loader, valid_loader, test_loader = data_helper.get_train_valid_test(data, 50)

    return train_loader, valid_loader, test_loader, edge_index
