import torch.nn as nn
import torch


class LSTMetallica(nn.Module):
    def __init__(self, num_layers, num_chars, num_units):
        super(LSTMetallica, self).__init__()
        self.num_layers = num_layers
        self.num_chars = num_chars
        self.num_units = num_units

        '''
        self.lstm1 = nn.LSTM(input_size=num_chars, hidden_size=num_units, num_layers=1, dropout=0.2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=num_units, hidden_size=num_units, num_layers=1, dropout=0.2, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=num_units, hidden_size=num_units, num_layers=1, dropout=0.2, batch_first=True)
        '''

        self.lstm = nn.LSTM(input_size=num_chars, hidden_size=num_units, num_layers=num_layers, dropout=0.2, batch_first=True)

        self.dense = nn.Linear(in_features=num_units, out_features=num_chars)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_tensor):
        x = in_tensor
        batch_size = x.size(0)
        h0 = torch.randn(self.num_layers, batch_size, self.num_units).to(torch.device('cuda'), dtype=torch.float)
        c0 = torch.randn(self.num_layers, batch_size, self.num_units).to(torch.device('cuda'), dtype=torch.float)

        x, _ = self.lstm(x, (h0, c0))
        # print(x.shape)

        x = x[:, -1, :]
        x = self.dense(x)
        # print(x.shape)

        x = self.softmax(x)
        # print(x.shape)

        return x

