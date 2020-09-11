import torch.nn as nn
import torch


class LSTMetallica(nn.Module):
    def __init__(self, num_layers, num_chars, num_units):
        super(LSTMetallica, self).__init__()
        self.num_units = num_units

        self.lstm1 = nn.LSTM(input_size=num_chars, hidden_size=num_units, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=num_units, hidden_size=num_units, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=num_units, hidden_size=num_units, num_layers=1, batch_first=True)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        self.dense = nn.Linear(in_features=num_units, out_features=num_chars)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_tensor):
        x = in_tensor
        # print(x.shape)
        batch_size = x.size(0)
        h0 = torch.randn(1, batch_size, self.num_units).to(torch.device('cuda'), dtype=torch.float)
        c0 = torch.randn(1, batch_size, self.num_units).to(torch.device('cuda'), dtype=torch.float)

        # h0 = torch.zeros([1, batch_size, self.num_units]).to(torch.device('cuda'), dtype=torch.float)
        # c0 = torch.zeros([1, batch_size, self.num_units]).to(torch.device('cuda'), dtype=torch.float)

        x, (h1, c1) = self.lstm1(x, (h0, c0))
        x = self.dropout1(x)
        # print(x.shape)

        x, (h2, c2) = self.lstm2(x, (h1, c1))
        x = self.dropout2(x)
        # print(x.shape)

        x, (h3, c3) = self.lstm3(x, (h2, c2))
        x = self.dropout3(x)
        # print(x.shape)

        x = x[:, -1, :]
        x = self.dense(x)
        # print(x.shape)

        x = self.softmax(x)
        # print(x.shape)

        return x

