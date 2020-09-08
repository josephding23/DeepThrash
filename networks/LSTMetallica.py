import torch.nn as nn
import torch


class LSTMetallica(nn.Module):
    def __init__(self, num_layers, num_chars, num_units):
        super(LSTMetallica, self).__init__()
        self.num_units = num_units

        self.lstm = nn.Sequential()
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(in_features=num_units, out_features=num_chars)
        self.softmax = nn.Softmax()

        '''
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.lstm.add_module(name='first_LSTM',
                                     module=nn.LSTM(input_size=num_units, hidden_size=num_units))
            else:
                self.lstm.add_module(name='LSTM',
                                     module=nn.LSTM(input_size=num_units, hidden_size=num_units))
        '''
        self.lstm1 = nn.LSTM(input_size=num_units, hidden_size=num_units)
        self.lstm2 = nn.LSTM(input_size=num_units, hidden_size=num_units)

    def forward(self, in_tensor):
        x = in_tensor
        # print(x.shape)
        batch_size = x.size(1)
        h0 = torch.zeros([1, batch_size, self.num_units]).to(torch.device('cuda'), dtype=torch.float)
        c0 = torch.zeros([1, batch_size, self.num_units]).to(torch.device('cuda'), dtype=torch.float)

        x, (h1, c1) = self.lstm1(x, (h0, c0))

        # print(x.shape)

        x, (h0, c0) = self.lstm2(x, (h1, c1))
        # print(x.shape)

        x = self.dropout(x)
        # print(x.shape)

        x = self.dense(x)
        # print(x.shape)

        x = self.softmax(x)

        return x

