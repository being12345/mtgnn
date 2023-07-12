from unittest import TestCase

from Data.generate_data import generate_data, get_data, get_and_normalize_data


class TestNodeDataSet(TestCase):
    def setUp(self):
        """
        TODO: using random number
        :return:
        """
        self.node_num = 50
        self.train_loader, self.valid_loader, self.test_loader, self.edge_index = generate_data(self.node_num)

    def test_data_value(self):
        """
        TODO: using random number
        test metrics_num, nodes_num, whether latency is right
        """
        df = get_and_normalize_data('./DatasetUpdate/MMS.csv')
        x, y, edge = get_data(50)
        print(x[0, :, 0, 0])
        print(y[0, ])
        print(df.iloc[0, 0:7])

    def test_shape(self):
        """
        TODO: using random number
        test metrics_num, nodes_num, whether latency is right
        """
        for x, y in self.train_loader:
            self.assertEqual((30, 50, 6), x.shape[1:])  # test node_dim
            self.assertEqual((30, 50, 1), y.shape[1:])  # test node_num

            assert x[0][0][1, 2] == df.iloc[0, 9]
            # assert y[0][0][1] == df.iloc[0, 11]

            break

        for x, y in self.valid_loader:
            self.assertEqual((30, 50, 6), x.shape[1:])  # test node_dim
            self.assertEqual((30, 50, 1), y.shape[1:])  # test node_num
            break

        for x, y in self.test_loader:
            self.assertEqual((30, 50, 6), x.shape[1:])  # test node_dim
            self.assertEqual((30, 50, 1), y.shape[1:])  # test node_num
            break
