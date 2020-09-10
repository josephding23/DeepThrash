import numpy as np
from networks.LSTMetallica import LSTMetallica
from dataset.thrash_dataset import ThrashDataset
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from model.dt_config import Config
import random
import sys


def sample(a, temperature=1.0):
    # sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    # print(a)
    try:
        result = np.argmax(np.random.multinomial(1, a, 1))
        # print(sum(a))
        return result
    except:
        print(a, sum(a))
        return None


class DeepThrashModel(object):
    def __init__(self, opt):
        torch.autograd.set_detect_anomaly(True)
        self.opt = opt
        self.device = torch.device('cuda') if self.opt.gpu else torch.device('cpu')
        self.network = None
        self.optimizer = None

        self.character_mode = self.opt.character_mode
        self.maxlen = self.opt.maxlen
        self.num_units = self.opt.num_units
        self.model_prefix = self.opt.model_prefix

        if self.character_mode:
            if self.maxlen is None:
                self.maxlen = 1024
            if self.num_units is None:
                self.num_units = 32
            self.step = 2 * 17

        else:
            if self.maxlen is None:
                self.maxlen = 128
            if self.num_units is None:
                self.num_units = 512
            self.step = 8

        if self.character_mode:
            self.num_char_pred = maxlen * 3 / 2
        else:
            self.num_char_pred = 17 * 30

        self.num_layers = 2

        if self.character_mode:
            self.prefix = 'char'
        else:
            self.prefix = 'word'

    def predict_model_batch(self):

        print('Starting predictions...')
        dataset = ThrashDataset(character_mode=self.opt.character_mode, maxlen=maxlen, step=self.step)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

        y_hat = torch.empty(data_loader.batch_size, 1).to(self.device)

        with torch.no_grad():
            for X_batch in data_loader:
                y_hat_batch = self.network(X_batch)
                y_hat = torch.cat([y_hat, y_hat_batch])

        y_hat = torch.flatten(
            y_hat[data_loader.batch_size:, :]).cpu().numpy()  # y_hat[batchsize:] is to remove first empty 'section'
        print('Predictions complete...')
        return y_hat

    def predict_model_single(self, x):

        # print('Starting predictions...')
        # y_hat = torch.empty(1, 1).to(self.device)
        y_hat = self.network(x)
        # print(y_hat.shape, self.network(x).shape)
        # y_hat = torch.cat([y_hat, self.network(x)])

        y_hat = torch.flatten(
            y_hat).detach().cpu().numpy()  # y_hat[batchsize:] is to remove first empty 'section'
        # print('Predictions complete...')
        # print(y_hat.shape)
        return y_hat

    def run(self):
        prefix = self.prefix
        model_prefix = self.model_prefix
        maxlen = self.maxlen
        num_units = self.num_units
        num_layers = self.num_layers
        step = self.step
        character_mode = self.character_mode
        num_char_pred = self.num_char_pred

        result_directory = '../static/models/result_%s_%s_%d_%d_units/' % (prefix, model_prefix, maxlen, num_units)
        filepath_model = '%sbest_model.hdf' % result_directory
        description_model = '%s, %d layers, %d units, %d maxlen, %d steps' % (prefix, num_layers,
                                                                              num_units, maxlen, step)

        if not os.path.exists(result_directory):
            os.mkdir(result_directory)

        criterion = nn.MSELoss()

        dataset = ThrashDataset(character_mode=character_mode, maxlen=maxlen, step=step)

        self.network = LSTMetallica(self.opt.num_layers, dataset.get_total_chars(), self.opt.num_units).to(self.device)
        self.optimizer = Adam(params=self.network.parameters())

        batch_size = 128
        loss_history = []
        pt_x = [1, 29, 30, 40, 100, 100, 200, 300, 400]
        nb_epochs = [np.sum(pt_x[:i + 1]) for i in range(len(pt_x))]
        print(nb_epochs)

        text = dataset.get_text()
        # not random seed, but the same seed for all.
        start_index = random.randint(0, len(text) - maxlen - 1)

        for (iteration, nb_epoch) in zip(pt_x, nb_epochs):
            model_path = '%smodel_after_%d.hdf' % (result_directory, nb_epoch)
            if not os.path.exists(model_path):
                self.network.train()

                for epoch in range(nb_epoch):
                    loader = DataLoader(dataset, batch_size=128, shuffle=True)
                    for i, data in enumerate(loader):
                        X = data[:, :-1, :]
                        X = X.to(self.device, dtype=torch.float)

                        y = data[:, -1, :]
                        y = y.to(self.device, dtype=torch.float)

                        Y = self.network(X).squeeze()
                        self.optimizer.zero_grad()
                        loss = criterion(Y, y.squeeze())
                        loss.backward()

                        self.optimizer.step()

                        loss_history.append(loss.item())

                    print(f'Epoch {epoch}, Loss: {loss_history[-1]}')
                torch.save(self.network.state_dict(), model_path)

            else:
                self.network.train()

                for epoch in range(nb_epoch):
                    loader = DataLoader(dataset, batch_size=128, shuffle=True)
                    for i, data in enumerate(loader):
                        X = data[:, :-1, :]
                        X = X.to(self.device, dtype=torch.float)

                        y = data[:, -1, :]
                        y = y.to(self.device, dtype=torch.float)

                        Y = self.network(X).squeeze()
                        self.optimizer.zero_grad()
                        loss = criterion(Y, y.squeeze())
                        loss.backward()

                        self.optimizer.step()

                        loss_history.append(loss.item())

                    print(f'Epoch {epoch}, Loss: {loss_history[-1]}')
                torch.save(self.network.state_dict(), model_path)
                # self.network.load_state_dict(torch.load(model_path))

            self.network.eval()
            for diversity in [0.9, 1.0, 1.2]:
                with open(('%sresult_%s_iter_%02d_diversity_%4.2f.txt' %
                           (result_directory, prefix, iteration, diversity)), 'w') as f_write:

                    print()
                    print('----- diversity:', diversity)
                    f_write.write('diversity:%4.2f\n' % diversity)
                    if character_mode:
                        generated = ''
                    else:
                        generated = []

                    sentence = text[start_index: start_index + maxlen]
                    seed_sentence = text[start_index: start_index + maxlen]

                    if character_mode:
                        generated += sentence
                    else:
                        generated = generated + sentence

                    print('----- Generating with seed:')

                    if character_mode:
                        print(sentence)
                        sys.stdout.write(generated)
                    else:
                        print(' '.join(sentence))

                    for i in range(num_char_pred):
                        # if generated.endswith('_END_'):
                        # 	break
                        x = np.zeros((1, maxlen, dataset.num_chars))

                        for t, char in enumerate(sentence):
                            x[0, t, dataset.char_indices[char]] = 1.

                        preds = self.predict_model_single(torch.from_numpy(x).to(self.device, dtype=torch.float))
                        # preds = preds.detach().squeeze().cpu().numpy()
                        # print(preds)
                        # print(preds.shape)
                        next_index = sample(preds, diversity)
                        if next_index is None:
                            return
                        next_char = dataset.indices_char[next_index]

                        if character_mode:
                            generated += next_char
                            sentence = sentence[1:] + next_char
                        else:
                            generated.append(next_char)
                            sentence = sentence[1:]
                            sentence.append(next_char)

                        if character_mode:
                            sys.stdout.write(next_char)
                        # else:
                        # 	for ch in next_char:
                        # 		sys.stdout.write(ch)

                        sys.stdout.flush()

                    if character_mode:
                        f_write.write(seed_sentence + '\n')
                        f_write.write(generated)
                    else:
                        f_write.write(' '.join(seed_sentence) + '\n')
                        f_write.write(' '.join(generated))

            np.save('%sloss_%s.npy' % (result_directory, prefix), loss_history)


if __name__ == '__main__':
    for maxlen in [256]:
        for num_units in [128, 512]:
            config = Config(False, maxlen=maxlen, num_units=num_units)
            model = DeepThrashModel(config)
            model.run()
