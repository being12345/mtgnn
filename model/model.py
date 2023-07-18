import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tqdm.auto import tqdm
import torch


@dataclass
class ModelConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs: int = 30  # Number of epochs.
    learning_rate: float = 1e-3
    early_stop: int = 400  # If model has not improved for this many consecutive epochs, stop training.
    save_path: Any = f'./models/model_{datetime.now()}.ckpt'  # Your model will be saved here.


class Model:
    def __init__(self, net, loss, train_criterion_set: list, valid_criterion_set: list, optim, log, train_config,
                 edge_index=None):
        self.net = net
        self.loss = loss
        self.train_criterion_set = train_criterion_set
        self.valid_criterion_set = valid_criterion_set
        self.optim = optim
        self.log = log
        self.train_config = train_config
        self.edge_index = edge_index.to(train_config['device'])
        self._set_model_path()

    def trainer(self, train_loader, valid_loader):
        n_epochs, best_loss, early_stop_count = self.train_config['n_epochs'], math.inf, 0

        for epoch in range(n_epochs):
            self.net.train()  # Set your model to train mode.

            train_loss_record = []  # set record to class
            train_criterion_record = []  # set record to class

            for step, (x, y) in enumerate(tqdm(train_loader)):
                x, y = x.to(self.train_config['device']), y.to(self.train_config['device'])  # Move your data to device.
                train_loss, pred = self.train_optim(x, y)

                train_loss_record.append(train_loss.detach().item())  # add to log

                for i, criterion in enumerate(self.train_criterion_set):
                    result = criterion.cache(pred, y)

            mean_train_loss = sum(train_loss_record) / len(train_loss_record)
            
            self.log.add_scaler_to_writer('Loss/train', mean_train_loss, epoch)

            for criterion in self.train_criterion_set:
                train_criterion_record.append(criterion.calculate_mean())

            self.log.epoch_train_info(epoch, n_epochs, mean_train_loss, train_criterion_record)

            self.net.eval()  # Set your model to evaluation mode.

            valid_loss_record = []
            valid_criterion_record = []  # set record to class

            for x, y in tqdm(valid_loader):
                x, y = x.to(self.train_config['device']), y.to(self.train_config['device'])

                valid_loss, pred = self.get_valid_loss(x, y)

                valid_loss_record.append(valid_loss)

                for i, criterion in enumerate(self.valid_criterion_set):
                    result = criterion.cache(pred, y)


            mean_valid_loss = sum(valid_loss_record) / len(valid_loss_record)
            self.log.add_scaler_to_writer('Loss/valid', mean_valid_loss, epoch)

            for criterion in self.valid_criterion_set:
                valid_criterion_record.append(criterion.calculate_mean())

            self.log.epoch_valid_info(epoch, n_epochs, mean_valid_loss, valid_criterion_record)

            if mean_valid_loss < best_loss:
                best_loss = self._save_model(mean_valid_loss)
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.train_config['early_stop']:
                return self._early_stop(best_loss)

    def _early_stop(self, best_loss):
        print('\nModel is not improving, so we halt the training session.')
        print(f'\nbest loss is {best_loss}')
        return

    def _save_model(self, mean_valid_loss):
        best_loss = mean_valid_loss
        torch.save(self.net.state_dict(), self.train_config['save_path'])  # Save your best model
        print('Saving model with loss {:.3f}... -> best'.format(best_loss))
        return best_loss

    def get_valid_loss(self, x, y):
        with torch.no_grad():
            if self.edge_index is None:
                pred = self.net(x)
            else:
                pred = self.net(x, self.edge_index)

            valid_loss = self.loss(pred, y)
        return valid_loss, pred

    def train_optim(self, x, y):
        self.optim.zero_grad()  # Set gradient to zero.

        if self.edge_index is None:
            pred = self.net(x)
        else:
            pred = self.net(x, self.edge_index)

        train_loss = self.loss(pred, y)
        train_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        self.optim.step()  # Update parameters.
        return train_loss, pred

    def _set_model_path(self):
        if not os.path.isdir('./models'):
            os.mkdir('./models')  # Create directory of saving models.  # TODO: in init
