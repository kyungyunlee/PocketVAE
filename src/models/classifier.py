import torch
import torch.nn as nn
import torch.nn.functional as F


class GenreLSTM(nn.Module):
    def __init__(self, cfg):
        super(GenreLSTM, self).__init__()
        self.cfg = cfg
        hidden_dim = self.cfg.model.hidden_dim
        self.in_fc = nn.Sequential(
            nn.Linear(self.cfg.n_drum_classes * 3, hidden_dim),
            nn.Dropout(self.cfg.model.dropout))

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=self.cfg.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.cfg.model.dropout
        )

        self.out_fc = nn.Sequential(
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(hidden_dim, len(self.cfg.genre_list)))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        out = self.in_fc(inputs)
        _, (hidden, _) = self.lstm(out)
        hidden = hidden.view(
            self.cfg.model.num_layers,
            2,
            batch_size,
            self.cfg.model.hidden_dim)
        hidden = hidden[-1, 0, :, :]
        out = self.out_fc(hidden)
        return out
