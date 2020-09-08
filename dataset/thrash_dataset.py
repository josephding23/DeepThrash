import torch.utils.data as data
import numpy as np


class ThrashDataset(data.Dataset):
    def __init__(self,  character_mode, maxlen, step, band='Metallica'):
        data_path = '../static/metallica/metallica_drums_text.txt'
        self.text = open(data_path).read()

        print(f'corpus length: {len(self.text)}')

        if character_mode:
            self.chars = set(self.text)
        else:
            self.chord_seq = self.text.split(' ')
            self.chars = set(self.chord_seq)
            self.text = self.chord_seq

        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.num_chars = len(self.char_indices)
        print('total chars:', self.num_chars)

        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - maxlen, step):
            sentences.append(self.text[i: i + maxlen])
            next_chars.append(self.text[i + maxlen])
        print('nb sequences:', len(sentences))
        print('Vectorization...')

        X = np.zeros((len(sentences), self.num_chars, maxlen), dtype=np.bool)
        y = np.zeros((len(sentences), self.num_chars), dtype=np.bool)

        print(X.shape, y.shape)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, self.char_indices[char], t] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        _y = y.reshape((y.shape[0], y.shape[1], 1))
        self.data = np.concatenate((X, _y), axis=2)
        '''
        X = np.zeros((len(sentences), maxlen, self.num_chars), dtype=np.bool)
        y = np.zeros((len(sentences), self.num_chars), dtype=np.bool)

        print(X.shape, y.shape)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        _y = y.reshape((y.shape[0], 1, y.shape[1]))
        self.data = np.concatenate((X, _y), axis=1)
        '''

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item, :, :]

    def get_total_chars(self):
        return self.num_chars

    def get_data(self):
        return self.data


if __name__ == '__main__':
    thrash_dataset = ThrashDataset(False, maxlen=256, step=8)

    print(len(thrash_dataset))