from unittest import TestCase

from regression_model.log import LogConfig, Log


# 1. set up Logconfig 2. set record 3 3. loop 50 add scaler
class TestLog(TestCase):
    def setUp(self) -> None:
        log = LogConfig()

        self.Log = Log(log.train_criterion_set, log.valid_criterion_set)

    def test_record_train_info(self):
        for i in range(50):
            self.Log.epoch_train_info(i, 50, i ** 2, [i, i + 1, i ** 2])

    def test_record_valid_info(self):
        for i in range(50):
            self.Log.epoch_valid_info(i, 50, i ** 2, [i, i + 1, i ** 2])

