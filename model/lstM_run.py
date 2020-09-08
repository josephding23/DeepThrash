import numpy as np
from networks.LSTMetallica import LSTMetallica
from dataset.thrash_dataset import ThrashDataset
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from torch.optim import lr_scheduler, Adam


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def run(is_charactor=False, maxlen=None, num_units=None, model_prefix=''):
    character_mode = is_charactor

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

    model = LSTMetallica(num_layers, num_units)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters())

    result_directory = 'result_%s_%s_%d_%d_units/' % (prefix, model_prefix, maxlen, num_units)
    filepath_model = '%sbest_model.hdf' % result_directory
    description_model = '%s, %d layers, %d units, %d maxlen, %d steps' % (prefix, num_layers, num_units, maxlen, step)

    if not os.path.exists(result_directory):
        os.mkdir(result_directory)

    with open(result_directory + description_model, 'w') as f_description:
        pass

    dataset = ThrashDataset(character_mode, maxlen, step)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    batch_size = 128
    loss_history = []
    pt_x = [1, 29, 30, 40, 100, 100, 200, 300, 400]
    nb_epochs = [np.sum(pt_x[:i + 1]) for i in range(len(pt_x))]

    for epoch in range(30):
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        result = model.forward()
