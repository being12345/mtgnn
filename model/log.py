from dataclasses import dataclass, field

from torch.utils.tensorboard import SummaryWriter


# set dataclass
@dataclass
class LogConfig:
    train_criterion_set: list = field(default_factory=lambda: ['MAPE', 'MAE', 'RRSE'])
    valid_criterion_set: list = field(default_factory=lambda: ['MAPE', 'MAE', 'RRSE'])


class Log:
    def __init__(self, train_config, valid_config):
        self.writer = SummaryWriter()
        self.train_config = train_config
        self.valid_config = valid_config

    def add_scaler_to_writer(self, description, metric, step):
        self.writer.add_scalar(description, metric, step)

    def step_train_info(self, index, result, step):
        self.writer.add_scalar(f'train_{self.train_config[index]}', result, step)

    def step_valid_info(self, index, result, step):
        self.writer.add_scalar(f'valid_{self.valid_config[index]}', result, step)

    def epoch_train_info(self, epoch, n_epochs, loss, criterion_record):
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] train_loss = {loss:.5f},")

        for i, info in enumerate(criterion_record):
            print(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] {self.train_config[i]} = {info:.5f},")
            self.writer.add_scalar(f'train_{self.train_config[i]}', info, epoch)

    def epoch_valid_info(self, epoch, n_epochs, loss, criterion_record):
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] valid_loss = {loss:.5f},")

        for i, info in enumerate(criterion_record):
            print(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] {self.valid_config[i]} = {info:.5f},")
            self.writer.add_scalar(f'valid_{self.valid_config[i]}', info, epoch)
