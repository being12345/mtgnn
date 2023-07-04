import pandas as pd


# TODO: get a whole table
def get_service_metric(df, i):
    df = df.drop(['TimeStamp', 'label'], axis=1)
    df.columns.names = ['service', 'metric']
    df = df.stack('service')
    data = df.loc[i]
    data = data.reset_index().drop('service', axis=1)
    latency = data['PodLatency(s)']
    data = data.drop('PodLatency(s)', axis=1)

    return data, latency


def get_and_normalize_data(path):
    df = pd.read_csv(path, header=[0, 1])
    metric = df.drop(['TimeStamp', 'label'], axis=1)
    metric.columns.names = ['pod', 'metric']
    tempm = metric.swaplevel('metric', 'pod', axis=1).stack()
    tempm = (tempm - tempm.mean()) / (tempm.std())
    metric = tempm.unstack().swaplevel('metric', 'pod', axis=1).stack().unstack()

    return metric
