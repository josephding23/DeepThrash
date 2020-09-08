
class Config(object):
    def __init__(self, is_charactor, maxlen, num_units=None, model_prefix=''):
        self.character_mode = is_charactor
        self.maxlen = maxlen
        self.num_units = num_units
        self.model_prefix = model_prefix

        self.gpu = True

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
            self.num_char_pred = self.maxlen * 3 / 2
        else:
            self.num_char_pred = 17 * 30

        self.num_layers = 2

        if self.character_mode:
            self.prefix = 'char'
        else:
            self.prefix = 'word'