import torch.nn as nn


class LSTMetallica(nn.Module):
    def __init__(self, num_layers, num_units):
        super(LSTMetallica, self).__init__()
        self.net = nn.Sequential()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.net.add_module(name='first_LSTM',
                                    module=nn.LSTM(input_size=1, hidden_size=num_units))
            else:
                self.net.add_module(name='LSTM',
                                    module=nn.LSTM(input_size=num_units, hidden_size=num_units))
            self.net.add_module('dropout', nn.Dropout(0.2))

        self.net.add_module('dense', nn.Linear(in_features=50, out_features=1))
        self.net.add_module('softmax', nn.Softmax())

    def forward(self, in_tensor):
        return self.net(in_tensor)

