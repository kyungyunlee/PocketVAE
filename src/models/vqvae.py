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
        hidden_dim = self.cfg.model.note_hidden_dim

        self.hidden_fc = nn.Linear(self.cfg.model.skel_encode_dim, hidden_dim)

        self.in_fc = nn.Sequential(
            nn.Linear(self.cfg.n_drum_classes, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(self.cfg.model.dropout)
        )

        self.gru1 = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True)
        self.aux1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(self.cfg.model.dropout)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.cfg.model.dropout))

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.cfg.model.dropout))

        self.conv3 = nn.Conv1d(
            hidden_dim,
            self.cfg.model.code_dim,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs, skel_encoded):
        # out = torch.cat((inputs,skel), dim=-1)

        hidden = self.hidden_fc(skel_encoded)
        hidden = torch.unsqueeze(hidden, 0)
        hidden = torch.cat((hidden, hidden), dim=0)

        out = self.in_fc(inputs)
        out, _ = self.gru1(out, hidden)
        out = out.transpose(1, 2)
        out = self.aux1(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class Note_Decoder(nn.Module):
    def __init__(self, cfg, device):
        super(Note_Decoder, self).__init__()
        self.cfg = cfg
        self.device = device
        hidden_dim = self.cfg.model.note_hidden_dim

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose1d(
                self.cfg.model.code_dim,
                hidden_dim,
                kernel_size=4,
                stride=2,
                output_padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.cfg.model.dropout),
            nn.ConvTranspose1d(
                hidden_dim,
                hidden_dim // 2,
                kernel_size=4,
                stride=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.cfg.model.dropout)
        )

        self.hidden_fc = nn.Linear(self.cfg.model.skel_encode_dim, hidden_dim)

        self.gru1 = nn.GRUCell(
            hidden_dim //
            2 +
            self.cfg.n_drum_classes *
            2,
            hidden_dim)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(self.cfg.model.dropout)
        self.dropout2 = nn.Dropout(self.cfg.model.dropout)
        self.note_fc = nn.Linear(hidden_dim, self.cfg.n_drum_classes)

    def forward(self, quantized_z, note_target, skel,
                skel_encoded, training, teacher_forcing_ratio):
        batch_size = quantized_z.size(0)
        inputs = self.conv_transpose(quantized_z)
        inputs = inputs.transpose(1, 2)

        hidden = self.hidden_fc(skel_encoded)

        hx = [hidden, None]

        note_outs = []
        out = torch.zeros(
            (batch_size, self.cfg.n_drum_classes)).to(
            self.device)
        for i in range(self.cfg.seq_len):
            curr_skel = skel[:, i, :]
            # out = torch.cat([out, curr_skel], dim=-1)
            curr_input = inputs[:, i, :]
            out = torch.cat([out, curr_input, curr_skel], dim=-1)
            hx[0] = self.gru1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[0] = self.dropout1(hx[0])
            hx[1] = self.gru2(hx[0], hx[1])
            hx[1] = self.dropout2(hx[1])

            note_out = self.note_fc(hx[1])
            note_outs.append(note_out)

            if training:
                p = torch.rand(1).item()
                if p < teacher_forcing_ratio:
                    out = note_target[:, i, :]
                else:
                    note_out = self._sampling(note_out)
                    out = note_out
            else:
                note_out = self._sampling(note_out)
                out = note_out

        return torch.stack(note_outs, 1)

    def _sampling(self, note_out):
        note_out = torch.sigmoid(note_out)
        # WITHOUT RESTRICITON

        rand = torch.rand(note_out.size()).to(self.device)
        prob = note_out - rand
        note_tmp = torch.zeros_like(prob).to(self.device)
        note_tmp[prob > 0] = 1

        return note_tmp


class Vector_Quantizer(nn.Module):
    def __init__(self, cfg, decay=0.99, eps=1e-5):
        super(Vector_Quantizer, self).__init__()
        self.cfg = cfg
        self.decay = decay
        self.eps = eps

        self.embed = nn.Embedding(
            self.cfg.model.n_codes,
            self.cfg.model.code_dim)
        self.embed.weight.data.normal_()

        self.register_buffer(
            "_ema_cluster_size", torch.zeros(
                self.cfg.model.n_codes))
        self._ema_w = nn.Parameter(
            torch.Tensor(
                self.cfg.model.n_codes,
                self.cfg.model.code_dim))
        self._ema_w.data.normal_()

    def forward(self, inputs, training):
        batch_size = inputs.size(0)
        # (batch_size, code_length, code_dim)
        inputs = inputs.transpose(1, 2).contiguous()
        input_shape = inputs.shape  # (batch_size, code_dim, code_length)
        # batch
        # (batch, 8) ( 8, 128) = (batch, 128)
        # (batch * code_length, code_dim) = (batch*code_length, 8)
        flat_inputs = inputs.view(-1, self.cfg.model.code_dim)
        # print (self.embed.weight.shape) # (256, 8)
        distances = (
            flat_inputs.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.embed.weight.t())
            + self.embed.weight.pow(2).sum(1, keepdim=False)
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.cfg.model.n_codes, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self.embed.weight).view(input_shape)
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + (
                1 - self.decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.eps) /
                (n + self.cfg.model.n_codes * self.eps) * n
            )

            dw = torch.matmul(encodings.t(), flat_inputs)
            self._ema_w = nn.Parameter(
                self._ema_w * self.decay + (1 - self.decay) * dw)

            self.embed.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # make quantized values
        codebook_loss = (quantized.detach() - inputs).pow(2).mean()
        cmt_loss = (quantized - inputs.detach()).pow(2).mean()
        loss = codebook_loss + cmt_loss * 0.2
        quantized = inputs + (quantized - inputs).detach()
        # quantized = quantized.view(input_shape)
        quantized = quantized.transpose(1, 2)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))
        encoding_indices = encoding_indices.reshape(batch_size, -1, 1)
        return quantized, loss, cmt_loss, encoding_indices, perplexity


class Note_VQVAE(nn.Module):
    def __init__(self, cfg, device):
        super(Note_VQVAE, self).__init__()
        self.encoder = Note_Encoder(cfg)
        self.vector_quantizer = Vector_Quantizer(cfg)
        self.decoder = Note_Decoder(cfg, device)

    def forward(self, note, skel, skel_encoded,
                training, teacher_forcing_ratio):
        note_z = self.encoder(note, skel_encoded)
        quantized_note_z, diff, cmt_loss, encoding_indices, perplexity = self.vector_quantizer(
            note_z, training)
        note_recon = self.decoder(
            quantized_note_z,
            note,
            skel,
            skel_encoded,
            training,
            teacher_forcing_ratio)
        return note_recon, quantized_note_z, diff, cmt_loss, encoding_indices, perplexity


class Vel_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Vel_Encoder, self).__init__()
        self.cfg = cfg
        self.hidden_dim = self.cfg.model.vel_hidden_dim

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

        self.mean_fc = nn.Linear(self.hidden_dim, self.cfg.model.vel_z_dim)
        self.var_fc = nn.Linear(self.hidden_dim, self.cfg.model.vel_z_dim)

    def forward(self, inputs, note_encoded, vel_cond, genre_cond):
        batch_size = inputs.size(0)

        hidden = note_encoded
        if self.cfg.use_genre_cond:
            hidden = torch.cat((hidden, genre_cond), dim=-1)

        hidden = self.hidden_fc(hidden)
        hidden = torch.unsqueeze(hidden, 0)
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

        hidden_dim = self.cfg.model.vel_hidden_dim

        self.in_fc = nn.Sequential(
            nn.Linear(
                self.cfg.model.vel_z_dim +
                self.cfg.genre_cond_dim +
                self.cfg.model.skel_encode_dim,
                hidden_dim),
            nn.Dropout(self.cfg.model.dropout))

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
            encoder_inputs, note_encoded, vel_cond, genre_cond)
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
        self.hidden_dim = self.cfg.model.mt_hidden_dim

        self.hidden_fc = nn.Sequential(
            nn.Linear(
                self.cfg.genre_cond_dim +
                self.cfg.model.skel_encode_dim,
                self.hidden_dim),
            nn.Dropout(0.5))

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

        self.mean_fc = nn.Linear(self.hidden_dim, self.cfg.model.mt_z_dim)
        self.var_fc = nn.Linear(self.hidden_dim, self.cfg.model.mt_z_dim)

    def forward(self, inputs, note_encoded, time_cond, genre_cond):
        batch_size = inputs.size(0)

        hidden = note_encoded
        if self.cfg.use_genre_cond:
            hidden = torch.cat((hidden, genre_cond), dim=-1)

        hidden = self.hidden_fc(hidden)
        hidden = torch.unsqueeze(hidden, 0)
        hidden = torch.cat((hidden, hidden), dim=0)
        # hidden = None

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

        hidden_dim = self.cfg.model.mt_hidden_dim

        self.in_fc = nn.Sequential(
            nn.Linear(
                self.cfg.model.mt_z_dim +
                self.cfg.genre_cond_dim +
                self.cfg.model.skel_encode_dim,
                hidden_dim),
            nn.Dropout(self.cfg.model.dropout))

        self.gru1 = nn.GRUCell(
            self.cfg.n_drum_classes *
            2 +
            self.cfg.time_cond_dim,
            hidden_dim)
        self.dropout = nn.Dropout(self.cfg.model.dropout)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_fc = nn.Sequential(
            # nn.Tanh(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(hidden_dim, self.cfg.n_drum_classes),
            nn.Tanh()
        )

    def forward(self, z, note_encoded, mt_target, note, time_cond,
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
                curr_cond = time_cond[:, i, :]
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
            encoder_inputs, note_encoded, time_cond, genre_cond)

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


class Prior(nn.Module):
    def __init__(self, cfg, device):
        super(Prior, self).__init__()

        self.cfg = cfg
        self.device = device

        self.hidden_dim = self.cfg.model.hidden_dim

        self.hidden_fc = nn.Linear(
            self.cfg.model.skel_encode_dim +
            self.cfg.genre_cond_dim,
            self.hidden_dim)
        self.embedding = nn.Embedding(self.cfg.model.n_codes, self.hidden_dim)

        self.gru1 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.gru2 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.out_fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.cfg.model.n_codes))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, note_encoded,  genre_cond,
                temp, training, teacher_forcing_ratio):
        batch_size = note_encoded.size(0)

        hidden = note_encoded

        if self.cfg.use_genre_cond:
            hidden = torch.cat((hidden, genre_cond), dim=-1)

        init = self.hidden_fc(hidden)
        hx = [init, None]

        out = torch.zeros((batch_size,)).long().to(self.device)

        code_outs = []
        pred_outs = []
        for i in range(self.cfg.model.code_length):
            out = self.embedding(out)
            # out = torch.cat((out, genre, tempo), dim=-1)
            hx[0] = self.gru1(out, hx[0])

            if i == 0:
                hx[1] = hx[0]
            hx[0] = self.dropout(hx[0])
            hx[1] = self.gru2(hx[0], hx[1])

            out = self.out_fc(hx[1])
            code_outs.append(out)
            out = torch.softmax(out, dim=-1)

            if training:
                p = torch.rand(1).item()
                if p < teacher_forcing_ratio:
                    out = inputs[:, i]
                else:
                    out = self._sampling(out, temp)
            else:
                out = self._sampling(out, temp)

            pred_outs.append(out)
        return torch.stack(code_outs, 1), torch.stack(pred_outs, 1)

    def _sampling(self, out, temp=0.2):
        # out = self.softmax(out)

        # rand = torch.rand(out.size()).to(self.device)
        # prob = out - rand
        # tmp = torch.zeros_like(prob).to(self.device)
        # tmp[prob > 0] = prob [prob>0]
        # out = torch.argmax(tmp, dim=-1)
        # out = torch.argmax(out, dim=-1)
        out = out / temp
        m = torch.distributions.Categorical(out)
        predicted_id = m.sample()

        return predicted_id
