from unittest import TestCase

from gcn.NodeDataSet import NodeDataSet
from util.data import get_data


class TestNodeDataSet(TestCase):
    def setUp(self):
        """
        TODO: using random number
        :return:
        """
        self.pred = 0
        self.node_num = 50
        self.node_dim = 6
        self.feature, self.latency, self.edge_index = get_data(self.pred, self.node_num, self.node_dim)

    def test_shape(self):
        """
        TODO: using random number
        test metrics_num, nodes_num, whether latency is right
        """
        self.assertEqual(6, self.feature.shape[2])  # test node_dim
        self.assertEqual(50, self.feature.shape[1])  # test node_num
        self.assertEqual(3216, self.feature.shape[0])  # test time_num

        self.assertEqual(-0.25043782773339, self.feature[0, 1, 0])  # test time_num

        self.assertEqual(-0.2650067609647671, self.latency[1])

    # 1. len 2. [] index
    def test_dataset(self):
        data = NodeDataSet(self.feature, self.latency)
        self.assertEqual(len(data), self.feature.shape[0])
        for x, y in data:
            print(x.shape)
            print(y.shape)



