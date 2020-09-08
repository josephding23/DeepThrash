import numpy as np


def old_function():
    maxlen = 256
    num_units = 128
    character_mode = False

    if character_mode:
        if maxlen == None:
            maxlen = 1024
        if num_units == None:
            num_units = 32
        step = 2 * 17  # step to create training data for truncated-BPTT
    else:  # word mode
        if maxlen == None:
            maxlen = 128  #
        if num_units == None:
            num_units = 512
        step = 8

    if character_mode:
        num_char_pred = maxlen * 3 / 2
    else:
        num_char_pred = 17 * 30

    num_layers = 2
    #
    if character_mode:
        prefix = 'char'
    else:
        prefix = 'word'

    path = '../static/metallica/metallica_drums_text.txt'  # Corpus file
    text = open(path).read()
    print('corpus length:', len(text))

    if character_mode:
        chars = set(text)
    else:
        chord_seq = text.split(' ')
        chars = set(chord_seq)
        text = chord_seq

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    num_chars = len(char_indices)
    print('total chars:', num_chars)

    # cut the text in semi-redundant sequences of maxlen characters

    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool)
    y = np.zeros((len(sentences), num_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print(X.shape, y.shape)


if __name__ == '__main__':
    old_function()