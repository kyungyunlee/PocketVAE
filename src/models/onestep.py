import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.hidden_dim = self.cfg.model.hidden_dim
        self.in_fc = nn.Linear(self.cfg.n_drum_classes * 4, self.hidden_dim)
        self.gru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=self.cfg.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.cfg.model.dropout
        )

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.cfg.model.dropout)
        )

        self.mean_fc = nn.Linear(self.hidden_dim, self.cfg.model.z_dim)
        self.var_fc = nn.Linear(self.hidden_dim, self.cfg.model.z_dim)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        out = self.in_fc(inputs)
        _, out = self.gru(out)
        out = out.view(
            self.cfg.model.num_layers,
            2,
            batch_size,
            self.hidden_dim)
        # Take the last layer output
        out = out[-1]
        out = out.transpose(0, 1).contiguous()
        out = out.view(batch_size, -1)
        out = self.out_fc(out)
        z, mu, var = self.sample_mean_var(out)
        return z, mu, var

    def sample_mean_var(self, out):
        mu = self.mean_fc(out)
        var = self.var_fc(out)
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, var


class Decoder(nn.Module):
    def __init__(self, cfg, device):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.device = device
        hidden_dim = self.cfg.model.hidden_dim

        self.in_fc = nn.Linear(self.cfg.model.z_dim, hidden_dim)
        self.gru1 = nn.GRUCell(self.cfg.n_drum_classes * 4, hidden_dim)
        self.dropout = nn.Dropout(self.cfg.model.dropout)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.cfg.model.dropout)
        )

        self.note_fc = nn.Linear(hidden_dim, self.cfg.n_drum_classes)
        self.vel_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.cfg.n_drum_classes),
            nn.Sigmoid())
        self.mt_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.cfg.n_drum_classes),
            nn.Tanh())

    def forward(self, z, skel, target, training, teacher_forcing_ratio):
        batch_size = z.size(0)

        init = self.in_fc(z)
        init = torch.tanh(init)

        hx = [init, None]

        note_outs, vel_outs, mt_outs = [], [], []
        out = torch.zeros(
            (batch_size,
             self.cfg.n_drum_classes *
             3)).to(
            self.device)

        for i in range(self.cfg.seq_len):
            curr_skel = skel[:, i, :]
            out = torch.cat((out, curr_skel), dim=-1)
            hx[0] = self.gru1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[0] = self.dropout(hx[0])
            hx[1] = self.gru2(hx[0], hx[1])

            out = self.out_fc(hx[1])
            note_out = self.note_fc(out)
            vel_out = self.vel_fc(out)
            mt_out = self.mt_fc(out)

            note_outs.append(note_out)
            vel_outs.append(vel_out)
            mt_outs.append(mt_out)

            if training:
                p = torch.rand(1).item()
                if p < teacher_forcing_ratio:
                    out = target[:, i, :]
                else:
                    note_out, vel_out, mt_out = self._sampling(
                        note_out, vel_out, mt_out)
                    out = torch.cat((note_out, vel_out, mt_out), dim=-1)
            else:
                note_out, vel_out, mt_out = self._sampling(
                    note_out, vel_out, mt_out)
                out = torch.cat((note_out, vel_out, mt_out), dim=-1)
        return torch.stack(note_outs, 1), torch.stack(
            vel_outs, 1), torch.stack(mt_outs, 1)

    def _sampling(self, note_out, vel_out, mt_out):
        note_out = torch.sigmoid(note_out)
        # WITHOUT RESTRICITON

        rand = torch.rand(note_out.size()).to(self.device)
        prob = note_out - rand
        note_tmp = torch.zeros_like(prob).to(self.device)
        note_tmp[prob > 0] = 1

        # note_tmp = torch.zeros_like(note_out).to(self.device)
        # note_tmp[note_out >= 0.5] = 1

        vel_out = note_tmp * vel_out
        mt_out = note_tmp * mt_out

        return note_tmp, vel_out, mt_out


class onestepVAE(nn.Module):
    def __init__(self, cfg, device):
        super(onestepVAE, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg, device)

    def forward(self, orig, skel, training, teacher_forcing_ratio):
        encoder_input = torch.cat((orig, skel), dim=-1)
        # encoder_input = skel
        z, mu, var = self.encoder(encoder_input)
        note_recon, vel_recon, mt_recon = self.decoder(
            z, skel, orig, training, teacher_forcing_ratio)
        return note_recon, vel_recon, mt_recon, z, mu, var
