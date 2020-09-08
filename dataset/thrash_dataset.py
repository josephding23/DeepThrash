import torch.utils.data as data
import numpy as np


class ThrashDataset(data.Dataset):
    def __init__(self,  character_mode, maxlen, step, band='Metallica'):
        data_path = '../data/metallica/metallica_drums_self.text.txt'
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
        X = np.zeros((len(sentences), maxlen, self.num_chars), dtype=np.bool)
        y = np.zeros((len(sentences), self.num_chars), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
            
    def __len__(self):
        return len(self.text)