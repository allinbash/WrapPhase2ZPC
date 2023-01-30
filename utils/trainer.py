import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device, logs_dir):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(os.path.join(logs_dir, 'vision{}'.format(time.strftime('%y%m%d%H%M%S', time.localtime()))))
        self.global_step = 0
        self.writer.add_graph(self.model, torch.rand([1, 1, 512, 512]).to(self.device))

    def train(self, dataloader):
        data_train, data_vali = dataloader
        size_data = len(data_train.dataset)
        size_batch = len(data_train)
        loss_vali = 0

        for batch, (X, y) in enumerate(data_train, start=1):
            self.model.train()
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            pred = pred.view(y.shape)
            loss_train = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()

            if batch % (size_batch // 20) == 0 or batch == size_batch:
                loss_train = loss_train.item()
                loss_vali = self.validate(data_vali)
                if batch == size_batch:
                    current = size_data
                else:
                    current = batch * len(pred)
                self.logger(loss_train, loss_vali, current, size_data)
                self.global_step += 1

        return loss_vali

    def validate(self, dataloader):
        size_batchs = len(dataloader)
        self.model.eval()
        loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                pred = pred.view(y.shape)
                loss += self.loss_fn(pred, y).item()

        loss /= size_batchs

        return loss

    def logger(self, loss_train, loss_vali, current, total):
        log_str = '[{}] loss_train: {:>7f} | loss_vali: {:>7f}  [{:d}/{:d}]'.format(
            time.strftime('%H:%M:%S', time.localtime()),
            loss_train, loss_vali, current, total)
        self.writer.add_scalars('Loss', {'train': loss_train, 'vali': loss_vali}, self.global_step)
        print(log_str)


if __name__ == '__main__':
    pass
