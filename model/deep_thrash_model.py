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

        result_directory = 'result_%s_%s_%d_%d_units/' % (prefix, model_prefix, maxlen, num_units)
        filepath_model = '%sbest_model.hdf' % result_directory
        description_model = '%s, %d layers, %d units, %d maxlen, %d steps' % (prefix, num_layers, num_units, maxlen, step)

        if not os.path.exists(result_directory):
            os.mkdir(result_directory)

        criterion = nn.CrossEntropyLoss()

        dataset = ThrashDataset(character_mode=character_mode, maxlen=maxlen, step=step)

        self.network = LSTMetallica(self.opt.num_layers, dataset.get_total_chars(), self.opt.num_units).to(self.device)
        self.optimizer = Adam(params=self.network.parameters())

        batch_size = 128
        loss_history = []
        pt_x = [1, 29, 30, 40, 100, 100, 200, 300, 400]
        nb_epochs = [np.sum(pt_x[:i + 1]) for i in range(len(pt_x))]
        print(nb_epochs)

        text = dataset.get_text()
        start_index = random.randint(0, len(text) - maxlen - 1)

        for (iteration, nb_epoch) in zip(pt_x, nb_epochs):
            for epoch in range(nb_epoch):
                loader = DataLoader(dataset, batch_size=128, shuffle=True)

                for i, data in enumerate(loader):
                    X = data[:, :-1, :]
                    X = X.to(self.device, dtype=torch.float)

                    y = data[:, -1, :]
                    y = y.to(self.device, dtype=torch.long)

                    Y = self.network(X).squeeze()

                    self.optimizer.zero_grad()

                    loss = criterion(Y, y.squeeze())
                    loss.backward()

                    self.optimizer.step()

                    loss_history.append(loss.item())

                print(f'Epoch {epoch}, Loss: {loss_history[-1]}')

            torch.save(self.network.state_dict(), '%smodel_after_%d.hdf'%(result_directory, nb_epoch))

            for diversity in [0.9, 1.0, 1.2]:
                with open(('%sresult_%s_iter_%02d_diversity_%4.2f.txt' % (
                result_directory, prefix, iteration, diversity)), 'w') as f_write:

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
                        print(x.shape)

                        for t, char in enumerate(sentence):
                            x[0, t, dataset.char_indices[char]] = 1.

                        preds = self.network(x)
                        next_index = sample(preds, diversity)
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
