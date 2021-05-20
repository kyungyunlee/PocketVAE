import torch
import torch.nn as nn


class Skel_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Skel_Encoder, self).__init__()
        self.cfg = cfg
        hidden_dim = self.cfg.model.skel_hidden_dim

        self.in_fc = nn.Linear(self.cfg.n_drum_classes, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            self.cfg.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.cfg.model.dropout
        )

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.cfg.model.skel_encode_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.cfg.model.dropout)
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        out = self.in_fc(inputs)
        _, out = self.gru(out)

        out = out.transpose(0, 1).contiguous()
        out = out.view(batch_size, -1)
        out = self.out_fc(out)
        return out


class Note_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Note_Encoder, self).__init__()
        self.cfg = cfg

        self.cond_dim = self.cfg.genre_cond_dim + \
             self.cfg.model.skel_encode_dim
        hidden_dim = self.cfg.model.note_hidden_dim

        self.hidden_fc = nn.Linear(self.cond_dim, hidden_dim)

        self.in_fc = nn.Linear(self.cfg.n_drum_classes, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            self.cfg.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.cfg.model.dropout
        )

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.cfg.model.dropout)
        )

        self.mean_fc = nn.Linear(hidden_dim, self.cfg.model.note_z_dim)
        self.var_fc = nn.Linear(hidden_dim, self.cfg.model.note_z_dim)

    def forward(self, inputs, skel_encoded, cond):
        batch_size = inputs.size(0)

        if self.cfg.use_genre_cond :
            hidden = torch.cat((skel_encoded, cond), dim=-1)
        else:
            hidden = skel_encoded

        hidden = self.hidden_fc(hidden)
        hidden = torch.unsqueeze(hidden, 0)
        # hidden = torch.cat((hidden, hidden, hidden, hidden), dim=0)
        hidden = torch.cat((hidden, hidden), dim=0)

        out = self.in_fc(inputs)
        _, out = self.gru(out, hidden)
        out = out.view(
            self.cfg.model.num_layers,
            2,
            batch_size,
            self.cfg.model.note_hidden_dim)
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


class Note_Decoder(nn.Module):
    def __init__(self, cfg, device):
        super(Note_Decoder, self).__init__()
        self.cfg = cfg
        self.device = device

        cond_dim = self.cfg.genre_cond_dim + \
                self.cfg.model.skel_encode_dim
        hidden_dim = self.cfg.model.note_hidden_dim

        self.hidden_fc = nn.Linear(
            self.cfg.model.note_z_dim + cond_dim, hidden_dim)
        self.gru1 = nn.GRUCell(self.cfg.n_drum_classes * 2, hidden_dim)
        self.dropout = nn.Dropout(self.cfg.model.dropout)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_fc = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(hidden_dim, self.cfg.n_drum_classes)
        )

    def forward(self, z, skel_encoded, skel, note_target,
                cond, training, teacher_forcing_ratio):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)

        z = torch.cat((z, skel_encoded), dim=-1)
        if self.cfg.use_genre_cond :
            z = torch.cat((z, cond), dim=-1)

        init = self.hidden_fc(z)
        init = torch.tanh(init)

        hx = [init, None]

        note_outs = []
        out = torch.zeros(
            (batch_size, self.cfg.n_drum_classes)).to(
            self.device)

        for i in range(self.cfg.seq_len):
            curr_skel = skel[:, i, :]
            out = torch.cat((out, curr_skel), dim=-1)
            hx[0] = self.gru1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[0] = self.dropout(hx[0])
            hx[1] = self.gru2(hx[0], hx[1])

            note_out = self.out_fc(hx[1])

            note_outs.append(note_out)

            if training:
                p = torch.rand(1).item()
                if p < teacher_forcing_ratio:
                    out = note_target[:, i, :]
                else:
                    out = self._sampling(note_out)
            else:
                out = self._sampling(note_out)
        return torch.stack(note_outs, 1)

    def _sampling(self, note_out):
        note_out = torch.sigmoid(note_out)

        rand = torch.rand(note_out.size()).to(self.device)
        prob = note_out - rand
        note_tmp = torch.zeros_like(prob).to(self.device)
        note_tmp[prob > 0] = 1

        return note_tmp


class Note_VAE(nn.Module):
    def __init__(self, cfg, device):
        super(Note_VAE, self).__init__()
        self.cfg = cfg
        self.encoder = Note_Encoder(cfg)
        self.decoder = Note_Decoder(cfg, device)

    def forward(self, note, skel, skel_encoded, cond,
                training, teacher_forcing_ratio):
        # encoder_inputs = torch.cat((note, skel), dim=-1)
        encoder_inputs = note
        z, mu, var = self.encoder(encoder_inputs, skel_encoded, cond)
        recon = self.decoder(
            z,
            skel_encoded,
            skel,
            note,
            cond,
            training,
            teacher_forcing_ratio)
        return recon, z, mu, var


class Vel_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Vel_Encoder, self).__init__()
        self.cfg = cfg
        self.hidden_dim = self.cfg.model.vel_mt_hidden_dim

        self.hidden_fc = nn.Linear(
            self.cfg.genre_cond_dim +
            self.cfg.model.skel_encode_dim,
            self.hidden_dim)

        self.in_fc = nn.Sequential(
            nn.Linear(
                self.cfg.n_drum_classes *
                2 +
                self.cfg.vel_cond_dim,
                self.hidden_dim),
            nn.Dropout(self.cfg.model.dropout))

        self.gru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            self.cfg.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.cfg.model.dropout
        )

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.cfg.model.dropout)
        )

        self.mean_fc = nn.Linear(self.hidden_dim, self.cfg.model.vel_mt_z_dim)
        self.var_fc = nn.Linear(self.hidden_dim, self.cfg.model.vel_mt_z_dim)

    def forward(self, inputs, vel_cond, note_encoded, genre_cond):
        batch_size = inputs.size(0)
        if self.cfg.use_genre_cond:
            other_cond = torch.cat((note_encoded, genre_cond), dim=-1)
        else:
            other_cond = note_encoded
        # skel and/or genre condition
        hidden = self.hidden_fc(other_cond)
        hidden = torch.unsqueeze(hidden, 0)
        # hidden = torch.cat((hidden, hidden, hidden, hidden), dim=0)
        hidden = torch.cat((hidden, hidden), dim=0)

        if self.cfg.use_vel_cond:
            inputs = torch.cat((inputs, vel_cond), dim=-1)

        out = self.in_fc(inputs)
        _, out = self.gru(out, hidden)
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


class Vel_Decoder(nn.Module):
    def __init__(self, cfg, device):
        super(Vel_Decoder, self).__init__()
        self.cfg = cfg
        self.device = device

        hidden_dim = self.cfg.model.vel_mt_hidden_dim

        self.in_fc = nn.Sequential(
            nn.Linear(
                self.cfg.model.vel_mt_z_dim +
                self.cfg.genre_cond_dim +
                self.cfg.model.skel_encode_dim,
                hidden_dim),
            nn.Dropout(self.cfg.model.dropout))

        # self.gru1 = nn.GRUCell(self.cfg.n_drum_classes + self.cfg.model.skel_encode_dim + self.cfg.vel_cond_dim, hidden_dim)
        self.gru1 = nn.GRUCell(
            self.cfg.n_drum_classes *
            2 +
            self.cfg.vel_cond_dim,
            hidden_dim)
        self.dropout = nn.Dropout(self.cfg.model.dropout)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)

        self.out_fc = nn.Sequential(
            # nn.Tanh(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(hidden_dim, self.cfg.n_drum_classes),
            nn.Sigmoid()
        )

        # self.class_fc = nn.Linear(1, 128)

    def forward(self, z, note_encoded, vel_target, note, vel_cond,
                genre_cond, training, teacher_forcing_ratio):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)

        z = torch.cat((z, note_encoded), dim=-1)

        if self.cfg.use_genre_cond:
            z = torch.cat((z, genre_cond), dim=-1)

        init = self.in_fc(z)
        init = torch.tanh(init)

        hx = [init, None]

        outs = []
        out = torch.zeros(
            (batch_size, self.cfg.n_drum_classes)).to(
            self.device)

        for i in range(self.cfg.seq_len):
            curr_note = note[:, i, :]
            if self.cfg.use_vel_cond:
                curr_cond = vel_cond[:, i, :]
                out = torch.cat((out, curr_note, curr_cond), dim=-1)
            else:
                out = torch.cat((out, curr_note), dim=-1)
            hx[0] = self.gru1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[0] = self.dropout(hx[0])
            hx[1] = self.gru2(hx[0], hx[1])

            vel_out = self.out_fc(hx[1])

            outs.append(vel_out)

            if training:
                p = torch.rand(1).item()
                if p < teacher_forcing_ratio:
                    out = vel_target[:, i, :]
                else:
                    out = vel_out
            else:
                out = vel_out
        return torch.stack(outs, 1)


class Vel_VAE(nn.Module):
    def __init__(self, cfg, device):
        super(Vel_VAE, self).__init__()
        self.cfg = cfg
        self.encoder = Vel_Encoder(cfg)
        self.decoder = Vel_Decoder(cfg, device)

    def forward(self, vel, note, note_encoded, vel_cond,
                genre_cond, training, teacher_forcing_ratio):
        encoder_inputs = torch.cat((vel, note), dim=-1)
        # encoder_inputs = vel
        z, mu, var = self.encoder(
            encoder_inputs, vel_cond, note_encoded, genre_cond)
        recon = self.decoder(
            z,
            note_encoded,
            vel,
            note,
            vel_cond,
            genre_cond,
            training,
            teacher_forcing_ratio)
        return recon, z, mu, var


class Time_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Time_Encoder, self).__init__()
        self.cfg = cfg
        self.hidden_dim = self.cfg.model.vel_mt_hidden_dim

        self.hidden_fc = nn.Linear(
            self.cfg.genre_cond_dim +
            self.cfg.model.skel_encode_dim,
            self.hidden_dim)

        self.in_fc = nn.Sequential(
            nn.Linear(
                self.cfg.n_drum_classes *
                2 +
                self.cfg.time_cond_dim,
                self.hidden_dim),
            nn.Dropout(self.cfg.model.dropout))

        self.gru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            self.cfg.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.cfg.model.dropout
        )

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.cfg.model.dropout)
        )

        self.mean_fc = nn.Linear(self.hidden_dim, self.cfg.model.vel_mt_z_dim)
        self.var_fc = nn.Linear(self.hidden_dim, self.cfg.model.vel_mt_z_dim)

    def forward(self, inputs, time_cond, note_encoded, genre_cond):
        batch_size = inputs.size(0)

        if self.cfg.use_genre_cond:
            other_cond = torch.cat((note_encoded, genre_cond), dim=-1)
        else:
            other_cond = note_encoded

        hidden = self.hidden_fc(other_cond)
        hidden = torch.unsqueeze(hidden, 0)
        # hidden = torch.cat((hidden, hidden, hidden, hidden), dim=0)
        hidden = torch.cat((hidden, hidden), dim=0)

        if self.cfg.use_time_cond:
            inputs = torch.cat((inputs, time_cond), dim=-1)

        out = self.in_fc(inputs)
        _, out = self.gru(out, hidden)
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


class Time_Decoder(nn.Module):
    def __init__(self, cfg, device):
        super(Time_Decoder, self).__init__()
        self.cfg = cfg
        self.device = device

        hidden_dim = self.cfg.model.vel_mt_hidden_dim

        self.in_fc = nn.Sequential(
            nn.Linear(
                self.cfg.model.vel_mt_z_dim +
                self.cfg.genre_cond_dim +
                self.cfg.model.skel_encode_dim,
                hidden_dim),
            nn.Dropout(self.cfg.model.dropout))

        self.gru1 = nn.GRUCell(
            self.cfg.n_drum_classes *
            2 +
            self.cfg.time_cond_dim,
            hidden_dim)
        # self.gru1 = nn.GRUCell(self.cfg.n_drum_classes + self.cfg.model.skel_encode_dim + self.cfg.time_cond_dim, hidden_dim)
        self.dropout = nn.Dropout(self.cfg.model.dropout)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_fc = nn.Sequential(
            # nn.Tanh(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(hidden_dim, self.cfg.n_drum_classes),
            nn.Tanh()
        )

    def forward(self, z, note_encoded, mt_target, note, mt_cond,
                genre_cond, training, teacher_forcing_ratio):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)

        z = torch.cat((z, note_encoded), dim=-1)

        if self.cfg.use_genre_cond:
            z = torch.cat((z, genre_cond), dim=-1)

        init = self.in_fc(z)
        init = torch.tanh(init)

        hx = [init, None]

        outs = []
        out = torch.zeros(
            (batch_size, self.cfg.n_drum_classes)).to(
            self.device)

        for i in range(self.cfg.seq_len):
            curr_note = note[:, i, :]
            if self.cfg.use_time_cond:
                curr_cond = mt_cond[:, i, :]
                out = torch.cat((out, curr_note, curr_cond), dim=-1)
            else:
                out = torch.cat((out, curr_note), dim=-1)
            hx[0] = self.gru1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[0] = self.dropout(hx[0])
            hx[1] = self.gru2(hx[0], hx[1])

            mt_out = self.out_fc(hx[1])

            outs.append(mt_out)

            if training:
                p = torch.rand(1).item()
                if p < teacher_forcing_ratio:
                    out = mt_target[:, i, :]
                else:
                    out = mt_out
            else:
                out = mt_out
        return torch.stack(outs, 1)


class Time_VAE(nn.Module):
    def __init__(self, cfg, device):
        super(Time_VAE, self).__init__()
        self.cfg = cfg
        self.encoder = Time_Encoder(cfg)
        self.decoder = Time_Decoder(cfg, device)

    def forward(self, mt, note, note_encoded, time_cond,
                genre_cond, training, teacher_forcing_ratio):
        encoder_inputs = torch.cat((mt, note), dim=-1)
        # encoder_inputs = mt
        z, mu, var = self.encoder(
            encoder_inputs, time_cond, note_encoded, genre_cond)
        recon = self.decoder(
            z,
            note_encoded,
            mt,
            note,
            time_cond,
            genre_cond,
            training,
            teacher_forcing_ratio)
        return recon, z, mu, var


class DrumRefinementModel(nn.Module):
    def __init__(self, skel_model, note_model, vel_model, time_model):
        super(DrumRefinementModel, self).__init__()
        self.skel_model = skel_model
        self.note_model = note_model
        self.vel_model = vel_model
        self.time_model = time_model


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg

        self.hidden_dim = self.cfg.model.gan_hidden_dim

        self.in_fc = nn.Linear(self.cfg.n_drum_classes * 3, self.hidden_dim)

        self.gru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            self.cfg.model.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.cfg.model.dropout
        )

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.cfg.model.dropout)
        )

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

        return out
