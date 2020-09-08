import numpy as np
from networks.LSTMetallica import LSTMetallica
from dataset.thrash_dataset import ThrashDataset
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from model.dt_config import Config


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


class DeepThrashModel(object):
    def __init__(self, opt):
        torch.autograd.set_detect_anomaly(True)
        self.opt = opt
        self.device = torch.device('cuda') if self.opt.gpu else torch.device('cpu')
        self.network = None
        self.optimizer = None

    def run(self):
        character_mode = self.opt.character_mode
        maxlen = self.opt.maxlen
        num_units = self.opt.num_units
        model_prefix = self.opt.model_prefix

        if character_mode:
            if maxlen is None:
                maxlen = 1024
            if num_units is None:
                num_units = 32
            step = 2 * 17

        else:
            if maxlen is None:
                maxlen = 128
            if num_units is None:
                num_units = 512
            step = 8

        if character_mode:
            num_char_pred = maxlen * 3 / 2
        else:
            num_char_pred = 17 * 30

        num_layers = 2

        if character_mode:
            prefix = 'char'
        else:
            prefix = 'word'

        criterion = nn.CrossEntropyLoss()

        result_directory = 'result_%s_%s_%d_%d_units/' % (prefix, model_prefix, maxlen, num_units)
        filepath_model = '%sbest_model.hdf' % result_directory
        description_model = '%s, %d layers, %d units, %d maxlen, %d steps' % (prefix, num_layers, num_units, maxlen, step)

        if not os.path.exists(result_directory):
            os.mkdir(result_directory)

        with open(result_directory + description_model, 'w') as f_description:
            pass

        dataset = ThrashDataset(character_mode=character_mode, maxlen=maxlen, step=step)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        self.network = LSTMetallica(self.opt.num_layers, dataset.get_total_chars(), self.opt.num_units).to(self.device)
        self.optimizer = Adam(params=self.network.parameters())

        batch_size = 128
        loss_history = []
        pt_x = [1, 29, 30, 40, 100, 100, 200, 300, 400]
        nb_epochs = [np.sum(pt_x[:i + 1]) for i in range(len(pt_x))]

        loss = None
        for epoch in range(30):
            loader = DataLoader(dataset, batch_size=128, shuffle=True)

            for i, data in enumerate(loader):
                X = data[:, :, :-1]
                # print(X.shape)
                # X = torch.reshape(X, (X.shape[2], -1, X.shape[0])).to(self.device, dtype=torch.float)
                X = X.to(self.device, dtype=torch.float)

                y = data[:, :, -1]
                # y = torch.reshape(y, (y.shape[1], y.shape[0])).to(self.device, dtype=torch.long)
                y = y.to(self.device, dtype=torch.long)

                Y = self.network(X)

                self.optimizer.zero_grad()
                loss = criterion(Y, y)
                loss.backward()
                self.optimizer.step()

            print(loss)


if __name__ == '__main__':
    config = Config(False, maxlen=256, num_units=256)
    model = DeepThrashModel(config)
    model.run()
