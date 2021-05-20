import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
# import logging
import hydra
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
import wandb

from src.models import vae
from datasets.data_utils import DRUM_CLASSES, NOTE_DENSITY_CLASSES, VEL_CLASSES, MT_CLASSES, load_data, matrix2midi
from src.dataloader import DrumDataset
from src.utils import binarize, kl_loss, compute_metrical_kl, tensor2np, batch2array, frange_cycle_linear

# LOGGER = logging.getLogger(__name__)


class Experiment:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # Check if preprocessed data is there
        p = os.path.abspath(
            os.path.join(
                hydra.utils.to_absolute_path(""),
                ".."))
        self.data_pickle_path = os.path.join(
            p,
            "datasets",
            self.cfg.preprocess_dir,
            f"all_data_{cfg.split_number}.pkl")
        if not os.path.exists(self.data_pickle_path):
            print(
                f"Data pickle {self.data_pickle_path} not found. You need to preprocess data.")
            print("Run python prepare_data.py in datasets folder.")
            # LOGGER.info(f"Data pickle {self.data_pickle_path} not found. You need to preprocess data.")
            # LOGGER.info("Run python prepare_data.py in datasets folder.")
            exit()

        # Generated output directory
        self.gen_output_dir = os.path.join(
            p, "datasets", self.cfg.generation_dir, self.cfg.model_name)
        if not os.path.exists(self.gen_output_dir):
            os.makedirs(self.gen_output_dir, exist_ok=True)

        # Add more configs
        OmegaConf.set_struct(self.cfg, True)
        with open_dict(self.cfg):
            self.cfg.n_drum_classes = len(DRUM_CLASSES)
            self.cfg.seq_len = self.cfg.resolution * \
                self.cfg.bar_length * self.cfg.beats_per_bar
            # Check if there is any condition
            if self.cfg.use_genre_cond:
                self.cfg.genre_cond_dim = len(self.cfg.genre_list)
            if self.cfg.use_vel_cond:
                self.cfg.vel_cond_dim = len(VEL_CLASSES)
            if self.cfg.use_time_cond:
                self.cfg.time_cond_dim = len(MT_CLASSES)

        # Print to check options
        print(f"Genre cond: {self.cfg.use_genre_cond}")
        print(f"Vel cond: {self.cfg.use_vel_cond}")
        print(f"Time cond: {self.cfg.use_time_cond}")

        print(f"Split index: {self.cfg.split_number}")

        skel_model = vae.Skel_Encoder(cfg).to(device)
        note_model = vae.Note_VAE(cfg, device).to(device)
        vel_model = vae.Vel_VAE(cfg, device).to(device)
        time_model = vae.Time_VAE(cfg, device).to(device)
        self.all_model = vae.DrumRefinementModel(
            skel_model, note_model, vel_model, time_model)

        self.kl_weight = 0
        self.vel_mt_kl_weight = 0
        self.vel_mt_weight = 0
        self.teacher_forcing_ratio = 0

    def train(self):

        # Load data
        train_data, valid_data, _ = load_data(self.data_pickle_path)
        print(len(train_data), len(valid_data))
        # LOGGER.info(f"Number of train, valid data: {len(train_data)}, {len(valid_data)} ")

        only_vel_data = [td['vel'] for td in train_data]
        only_vel_data = np.array(only_vel_data) * 127 // 4

        only_vel_data = only_vel_data.reshape(-1).astype(int)
        occurences = Counter(only_vel_data)
        occurences = {
            k: v for k,
            v in sorted(
                occurences.items(),
                key=lambda item: -
                item[1])}


        train_dataset, valid_dataset = DrumDataset(
            train_data), DrumDataset(valid_data)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=True)

        # optimizer
        # self.optimizer = torch.optim.Adam(
        #     list(self.all_model.note_model.parameters()) + list(self.all_model.vel_model.parameters()) + list(self.all_model.time_model.parameters()) , lr=self.cfg.lr, betas=(0.5, 0.999)
        # )
        self.optimizer = torch.optim.Adam(
            self.all_model.parameters(), lr=self.cfg.lr, betas=(0.5, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=10,
            factor=self.cfg.lr_decay_rate,
            min_lr=1e-7,
            verbose=True,
        )

        # Logging to wandb
        wandb.init(
            project=self.cfg.wandb_project_name,
            entity=self.cfg.wandb_entity)
        wandb.watch(self.all_model)
        wandb.run.name = self.cfg.model_name
        wandb.run.save()

        # exponential weighting
        # np.logspace(0,1, num=20, base=2, dtype="float") - 1
        self.vel_mt_weight = 1
        tf_weight = np.logspace(1, 0.5, num=10, base=2, dtype="float") - 1

        # if self.cfg.use_cyclic_weight :
        vel_kl_weight = frange_cycle_linear(
            self.cfg.num_epochs, stop=0.2, n_cycle=5, ratio=0.9)
        mt_kl_weight = frange_cycle_linear(
            self.cfg.num_epochs, stop=0.2, n_cycle=5, ratio=0.9)
        note_kl_weight = frange_cycle_linear(
            self.cfg.num_epochs, stop=0.3, n_cycle=5, ratio=0.9)
        # else :
        # kl_weight = np.logspace(0, 0.01, num=100, base=2, dtype="float") - 1
        # vel_mt_kl_weight = np.logspace(0, 0.01, num=50, base=2, dtype="float") - 1

        for epoch in range(self.cfg.num_epochs):

            # self._validate(valid_dataloader,epoch, save_model=True, write_loss=True)

            self.all_model.train(True)

            # Update weights
            # if epoch >=30 and epoch < 130 :
            # self.vel_mt_weight = 1 # vel_mt_weight[epoch-3]
            # self.vel_mt_kl_weight = vel_mt_kl_weight[epoch-30]

            if epoch < 10:
                self.teacher_forcing_ratio = tf_weight[epoch]

            # if self.cfg.use_cyclic_weight:
            #      self.kl_weight = kl_weight[epoch]
            # else :
            self.vel_kl_weight = vel_kl_weight[epoch]
            self.mt_kl_weight = mt_kl_weight[epoch]
            self.note_kl_weight = note_kl_weight[epoch]
            # if epoch < 50 :
            # self.kl_weight = kl_weight[epoch]

            for i, data in enumerate(train_dataloader):
                skel = data["skel"].float().to(self.device)
                note = data["note"].float().to(self.device)
                vel = data["vel"].float().to(self.device)
                mt = data["mt"].float().to(self.device)
                genre_idx = data["genre"].to(self.device)
                genre = torch.nn.functional.one_hot(
                    genre_idx, num_classes=len(self.cfg.genre_list))
                # n_inst = data["n_inst"].to(self.device)

                # n_inst = torch.nn.functional.one_hot(n_inst, num_classes=self.cfg.n_drum_classes)

                # Conditions

                genre_cond = None
                vel_cond = None
                time_cond = None
                # inst_cond = None
                if self.cfg.use_genre_cond:
                    genre_cond = genre.float()

                if self.cfg.use_vel_cond:
                    vel_contour = data["vel_contour"].to(self.device)
                    vel_cond = vel_contour.float()

                if self.cfg.use_time_cond:
                    time_contour = data["time_contour"].to(self.device)
                    time_cond = time_contour.float()

                # if self.cfg.use_inst_cond :
                #     inst_cond = n_inst.float()

                # Augmentation
                if self.cfg.random_mask:
                    skel = self._random_mask(skel)

                if self.cfg.reduce_skel:
                    skel = self._reduce_skel(skel)

                ### SKEL #####
                skel_encoded = self.all_model.skel_model(skel)
                note_encoded = self.all_model.skel_model(note)
                ##### NOTE ######
                note_recon, note_z, note_mu, note_var = self.all_model.note_model(
                    note, skel, skel_encoded, genre_cond, training=True, teacher_forcing_ratio=self.teacher_forcing_ratio)

                # Pos weight
                deco_note = note - skel
                deco_note[deco_note < 0] = 0
                deco_note = deco_note * 1.5
                deco_note = None

                note_recon_loss = F.binary_cross_entropy_with_logits(
                    note_recon, note, pos_weight=deco_note)
                note_kl_loss = kl_loss(note_mu, note_var)
                loss = note_recon_loss + note_kl_loss * self.note_kl_weight

                # binarize the predicted note for velocity and microtime
                # conditioning
                binary_note_recon = binarize(note_recon, self.device)

                ###### VEL #######
                # vel_recon, vel_z, vel_mu, vel_var = self.all_model.vel_model(vel, binary_note_recon, vel_cond, genre_cond, training=True, teacher_forcing_ratio=self.teacher_forcing_ratio)
                vel_recon, vel_z, vel_mu, vel_var = self.all_model.vel_model(
                    vel, note, note_encoded, vel_cond, genre_cond, training=True, teacher_forcing_ratio=self.teacher_forcing_ratio)

                # vel_recon_loss = note_constrained_loss(vel_recon, vel, note)
                vel_recon_loss = F.mse_loss(vel_recon, vel)
                vel_kl_loss = kl_loss(vel_mu, vel_var)
                loss += (vel_recon_loss + self.vel_kl_weight * vel_kl_loss)

                ###### TIME #######
                # mt_recon, mt_z, mt_mu, mt_var = self.all_model.time_model(mt, binary_note_recon, time_cond, genre_cond, training=True, teacher_forcing_ratio=self.teacher_forcing_ratio)
                mt_recon, mt_z, mt_mu, mt_var = self.all_model.time_model(
                    mt, note, note_encoded, time_cond, genre_cond, training=True, teacher_forcing_ratio=self.teacher_forcing_ratio)
                # mt_recon_loss = note_constrained_loss(mt_recon, mt, note)
                mt_recon_loss = F.mse_loss(mt_recon, mt)
                mt_kl_loss = kl_loss(mt_mu, mt_var)
                loss += (mt_recon_loss + self.mt_kl_weight * mt_kl_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log losses to stdout and to wandb
                if i % 100 == 0:
                    print(f"Epoch {epoch}, step {i} /{len(train_dataloader)}")
                    print("note recon loss", note_recon_loss.detach().item())
                    print("vel recon loss", vel_recon_loss.detach().item())
                    print("mt recon loss", mt_recon_loss.detach().item())
                    print("note kl loss", note_kl_loss.detach().item())

                    wandb.log({
                        "Train Note Loss": note_recon_loss.detach().item(),
                        "Train Vel Loss": vel_recon_loss.detach().item(),
                        "Train MT Loss": mt_recon_loss.detach().item(),
                        "Train Note KL Loss": note_kl_loss.detach().item(),
                        "Train Vel KL Loss": vel_kl_loss.detach().item(),
                        "Train MT KL Loss": mt_kl_loss.detach().item(),
                        "Note KL weight": self.note_kl_weight,
                        "Vel KL weight": self.vel_kl_weight,
                        "MT KL weight": self.mt_kl_weight,
                    })

            # Validation
            self._validate(
                valid_dataloader,
                epoch,
                save_model=True,
                write_loss=True)

            if epoch % 10 == 0:
                print("GENERATING...")
                self.generate(
                    valid_dataloader,
                    epoch,
                    reconstruct=True,
                    to_midi=True)

    def run(self, dataloader, return_results):
        self.all_model.train(False)

        skels = []
        note_outs, vel_outs, mt_outs = [], [], []
        note_orig, vel_orig, mt_orig = [], [], []
        note_z_outs, vel_z_outs, mt_z_outs = [], [], []
        genre_idxs = []
        tempos = []
        note_loss, vel_loss, mt_loss = 0, 0, 0
        for i, data in enumerate(dataloader):
            skel = data["skel"].float().to(self.device)
            note = data["note"].float().to(self.device)
            vel = data["vel"].float().to(self.device)
            mt = data["mt"].float().to(self.device)
            genre_idx = data["genre"].to(self.device)
            genre = torch.nn.functional.one_hot(
                genre_idx, num_classes=len(self.cfg.genre_list))
            tempo = data["tempo"].to(self.device)
            n_inst = data["n_inst"].to(self.device)
            n_inst = torch.nn.functional.one_hot(
                n_inst, num_classes=self.cfg.n_drum_classes)

            # Conditions
            genre_cond = None
            vel_cond = None
            time_cond = None
            if self.cfg.use_genre_cond:
                genre_cond = genre.float()

            if self.cfg.use_vel_cond:
                vel_contour = data["vel_contour"].to(self.device)
                vel_cond = vel_contour.float()

            if self.cfg.use_time_cond:
                time_contour = data["time_contour"].to(self.device)
                time_cond = time_contour.float()

            skel_encoded = self.all_model.skel_model(skel)
            note_encoded = self.all_model.skel_model(note)

            ##### NOTE ######
            note_recon, note_z, note_mu, note_var = self.all_model.note_model(
                note, skel, skel_encoded, genre_cond, training=False, teacher_forcing_ratio=None)

            note_recon_loss = F.binary_cross_entropy_with_logits(
                note_recon, note, pos_weight=None)
            note_loss += note_recon_loss.detach().item()

            # binarize the predicted note for velocity and microtime conditioning
            # sigmoid + sampling
            binary_note_recon = binarize(note_recon, self.device)

            ###### VEL #######
            vel_recon, vel_z, vel_mu, vel_var = self.all_model.vel_model(
                vel, note, note_encoded, vel_cond, genre_cond, training=False, teacher_forcing_ratio=None)

            vel_recon_loss = F.mse_loss(vel_recon, vel)
            vel_loss += vel_recon_loss.detach().item()

            ###### TIME #######
            mt_recon, mt_z, mt_mu, mt_var = self.all_model.time_model(
                mt, note, note_encoded, time_cond, genre_cond, training=False, teacher_forcing_ratio=None)
            # mt_recon_loss = note_constrained_loss(mt_recon, mt, note)
            mt_recon_loss = F.mse_loss(mt_recon, mt)
            mt_loss += mt_recon_loss.detach().item()

            note_outs.append(tensor2np(binary_note_recon))
            vel_outs.append(tensor2np(vel_recon))
            mt_outs.append(tensor2np(mt_recon))

            note_z_outs.append(tensor2np(note_z))
            vel_z_outs.append(tensor2np(vel_z))
            mt_z_outs.append(tensor2np(mt_z))

            skels.append(tensor2np(skel))
            note_orig.append(tensor2np(note))
            vel_orig.append(tensor2np(vel))
            mt_orig.append(tensor2np(mt))

            tempos.append(tensor2np(tempo))
            genre_idxs.append(tensor2np(genre_idx))

        # reshape
        note_outs = batch2array(note_outs)
        note_orig = batch2array(note_orig)
        skels = batch2array(skels)
        vel_outs = batch2array(vel_outs)
        vel_orig = batch2array(vel_orig)
        mt_outs = batch2array(mt_outs)
        mt_orig = batch2array(mt_orig)

        # compute F1 for note classification
        note_outs_tmp = note_outs.reshape(-1, note_outs.shape[-1])
        note_orig_tmp = note_orig.reshape(-1, note_orig.shape[-1])
        cls_score = classification_report(
            note_outs_tmp, note_orig_tmp, output_dict=True)
        f1 = cls_score["micro avg"]["f1-score"]
        acc_score = accuracy_score(note_outs_tmp, note_orig_tmp)
        print("ACC", acc_score)

        # Compute kl
        vel_outs_tmp = vel_outs * 127
        vel_orig_tmp = vel_orig * 127
        vel_kl_div = compute_metrical_kl(
            vel_outs_tmp, vel_orig_tmp, note_outs, note_orig)

        ticks_per_subdivision = 120
        range1 = 2
        range2 = ticks_per_subdivision

        mt_outs_tmp = (range2 * (mt_outs + 1)) / \
            range1 - ticks_per_subdivision // 2
        mt_orig_tmp = (range2 * (mt_orig + 1)) / \
            range1 - ticks_per_subdivision // 2

        mt_kl_div = compute_metrical_kl(
            mt_outs_tmp, mt_orig_tmp, note_outs, note_orig)

        # vel_kl_div = compute_metrical_kl(vel_outs, vel_orig, note_outs, note_orig)
        # mt_kl_div = compute_metrical_kl(mt_outs, mt_orig, note_outs, note_orig)

        dlen = len(dataloader)
        all_losses = [note_loss / dlen, vel_loss / dlen, mt_loss / dlen]
        if return_results:
            all_orig_outs = [note_orig, vel_orig, mt_orig]
            all_recon_outs = [note_outs, vel_outs, mt_outs]
            all_z_outs = [note_z_outs, vel_z_outs, mt_z_outs]

            genre_idxs = np.array(genre_idxs)
            genre_idxs = genre_idxs.reshape(-1)

            tempos = np.array(tempos)
            tempos = tempos.reshape(-1)

            return all_orig_outs, all_recon_outs, skels, all_z_outs, all_losses, f1, vel_kl_div, mt_kl_div, tempos, genre_idxs
        else:
            return all_losses, f1, vel_kl_div, mt_kl_div

    def _validate(self, dataloader, epoch, save_model, write_loss):

        all_losses, f1, vel_kl_div, mt_kl_div = self.run(
            dataloader, return_results=False)
        note_loss, vel_loss, mt_loss = all_losses

        # if epoch < 30 :
        # self.scheduler.step(note_loss)
        # else :
        self.scheduler.step(note_loss + vel_loss + mt_loss)

        # Write loss
        if write_loss:
            print(f"Eval loss: {note_loss}, {vel_loss}, {mt_loss}")
            print(f"Eval f1: {f1}")
            print(f"Eval vel kl: {vel_kl_div}")
            print(f"Eval mt kl: {mt_kl_div}")
            wandb.log({
                "Valid Note Loss": note_loss,
                "Valid Vel Loss": vel_loss,
                "Valid MT Loss": mt_loss,
                "Valid F1": f1,
                "Valid Vel kl div": vel_kl_div,
                "Valid MT kl div": mt_kl_div
            })

        # directory for saving weights
        model_savepath = os.path.abspath(
            os.path.join(
                hydra.utils.to_absolute_path(""),
                "..",
                self.cfg.model_savepath,
                self.cfg.model_name))
        if not os.path.exists(model_savepath):
            os.makedirs(model_savepath)

        # Save model
        if save_model:
            savename = os.path.join(
                model_savepath,
                f"{self.cfg.model_name}_{epoch}.pt")
            torch.save(
                self.all_model,
                savename
            )
            print(f"Model at epoch {epoch} saved as {savename}")

        return note_loss, vel_loss, mt_loss

    def _reduce_skel(self, skel):
        r = random.random()
        if r < 0.2:
            new_skels_tmp = torch.sum(skel, dim=-1)
            new_skels_tmp[new_skels_tmp > 1] = 1
            new_skels = torch.zeros_like(skel)
            new_skels[:, :, 2] = new_skels_tmp
            skel = new_skels
        return skel

    def _random_mask(self, skel):
        # Randomly select which sample from the batch to mask
        batch_size = skel.size(0)
        r = torch.rand((batch_size,))
        idx = torch.where(r < 0.2)
        idx = idx[0]
        # Keep only first kick
        tmp = torch.zeros_like(skel[:, :, 0])
        tmp[:, ::4] = 1
        skel[idx, :, 0] = skel[idx, :, 0] * tmp[idx]

        r = torch.rand((batch_size,))
        idx = torch.where(r < 0.2)
        idx = idx[0]
        # keep only first snare
        skel[idx, :, 1] = skel[idx, :, 1] * tmp[idx]

        r = torch.rand((batch_size,))
        idx = torch.where(r < 0.2)
        idx = idx[0]

        # Mask random segment (8 beats)
        b = torch.randint(32 - 8, (1,))
        skel[idx, b:b + 8, :] = 0

        return skel

    def get_test_data(self, batch_size, shuffle=False):
        _, _, test_data = load_data(self.data_pickle_path)
        print(len(test_data))
        # Dataloader
        test_dataset = DrumDataset(test_data)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True)

        return test_data, test_dataloader

    def generate(self, dataloader, epoch, reconstruct,
                 fname="recon", n_samples=1, to_midi=False):
        """
        reconstruct : True -> reconstruct the input, False -> decode from randomly sampled z
        """
        self.all_model.train(False)

        if reconstruct:
            all_orig, all_recon_outs, skels, _, _, _, _, _, tempos, genre_idxs = self.run(
                dataloader, return_results=True)
            note_orig, vel_orig, mt_orig = all_orig
            note_outs, vel_outs, mt_outs = all_recon_outs
        else:
            skels = []
            note_orig, vel_orig, mt_orig = [], [], []
            note_outs, vel_outs, mt_outs = [], [], []
            tempos = []
            genre_idxs = []
            for i, data in enumerate(dataloader):
                skel = data["skel"].float().to(self.device)
                note = data["note"].float().to(self.device)
                vel = data["vel"].float().to(self.device)
                mt = data["mt"].float().to(self.device)
                genre_idx = data["genre"].to(self.device)
                genre = torch.nn.functional.one_hot(
                    genre_idx, num_classes=len(
                        self.cfg.genre_list)).to(
                    self.device)
                tempo = data["tempo"].to(self.device)

                # Conditions
                genre_cond = None
                vel_cond = None
                time_cond = None
                if self.cfg.use_genre_cond:

                    genre_cond = genre.float()

                if self.cfg.use_vel_cond:
                    vel_contour = data["vel_contour"].to(self.device)
                    vel_cond = vel_contour.float()

                if self.cfg.use_time_cond:
                    time_contour = data["time_contour"].to(self.device)
                    time_cond = time_contour.float()

                skel_encoded = self.all_model.skel_model(skel)
                note_encoded = self.all_model.skel_model(note)

                # sample n samples
                sample_notes = []
                sample_vels = []
                sample_mts = []
                for n in range(n_samples):
                    note_z = torch.randn(
                        (skel.size(0), self.cfg.model.note_z_dim)).to(
                        self.device)
                    vel_z = torch.randn(
                        (skel.size(0), self.cfg.model.vel_mt_z_dim)).to(
                        self.device)
                    mt_z = torch.randn(
                        (skel.size(0), self.cfg.model.vel_mt_z_dim)).to(
                        self.device)
                    # vel_z, _, _ = self.all_model.vel_model.encoder(vel, vel_cond, note_encoded, genre_cond)
                    # mt_z , _, _= self.all_model.time_model.encoder(mt, time_cond, note_encoded, genre_cond)

                    # decode

                    note_gen = self.all_model.note_model.decoder(
                        note_z,
                        skel_encoded,
                        skel,
                        None,
                        genre_cond,
                        training=False,
                        teacher_forcing_ratio=None)
                    binary_note_gen = binarize(note_gen, self.device)

                    note_encoded = self.all_model.skel_model(binary_note_gen)

                    vel_gen = self.all_model.vel_model.decoder(
                        vel_z,
                        note_encoded,
                        None,
                        binary_note_gen,
                        vel_cond,
                        genre_cond,
                        training=False,
                        teacher_forcing_ratio=None)
                    mt_gen = self.all_model.time_model.decoder(
                        mt_z,
                        note_encoded,
                        None,
                        binary_note_gen,
                        time_cond,
                        genre_cond,
                        training=False,
                        teacher_forcing_ratio=None)

                    sample_notes.append(tensor2np(binary_note_gen))
                    sample_vels.append(tensor2np(vel_gen))
                    sample_mts.append(tensor2np(mt_gen))

                sample_notes = np.transpose(
                    np.array(sample_notes), (1, 0, 2, 3))
                sample_vels = np.transpose(np.array(sample_vels), (1, 0, 2, 3))
                sample_mts = np.transpose(np.array(sample_mts), (1, 0, 2, 3))
                # sample_notes = sample_notes.view((-1, n_samples, self.cfg.seq_len, self.cfg.n_drum_classes))
                # sample_vels = sample_vels.view((-1, n_samples, self.cfg.seq_len, self.cfg.n_drum_classes))
                # sample_mts = sample_mts.view((-1, n_samples, self.cfg.seq_len, self.cfg.n_drum_classes))
                note_outs.append(sample_notes)
                vel_outs.append(sample_vels)
                mt_outs.append(sample_mts)

                skels.append(tensor2np(skel))
                note_orig.append(tensor2np(note))
                vel_orig.append(tensor2np(vel))
                mt_orig.append(tensor2np(mt))

                tempos.append(tensor2np(tempo))
                genre_idxs.append(tensor2np(genre_idx))

            # (76, 64, 2, 32, 7)
            note_outs = np.array(note_outs)
            note_outs = note_outs.reshape(
                (-1, n_samples, self.cfg.seq_len, self.cfg.n_drum_classes))

            vel_outs = np.array(vel_outs)
            vel_outs = vel_outs.reshape(
                (-1, n_samples, self.cfg.seq_len, self.cfg.n_drum_classes))

            mt_outs = np.array(mt_outs)
            mt_outs = mt_outs.reshape(
                (-1, n_samples, self.cfg.seq_len, self.cfg.n_drum_classes))

            skels = batch2array(skels)
            note_orig = batch2array(note_orig)
            vel_orig = batch2array(vel_orig)
            mt_orig = batch2array(mt_orig)

            all_recon_outs = [note_outs, vel_outs, mt_outs]

            genre_idxs = np.array(genre_idxs)
            genre_idxs = genre_idxs.reshape(-1)

            tempos = np.array(tempos)
            tempos = tempos.reshape(-1)

        if to_midi:
            rand_idx = np.random.randint(0, note_outs.shape[0], size=20)

            selected_note_recons = note_outs[rand_idx]
            selected_vel_recons = vel_outs[rand_idx]
            selected_mt_recons = mt_outs[rand_idx]
            selected_note_orig = note_orig[rand_idx]
            selected_vel_orig = vel_orig[rand_idx]
            selected_mt_orig = mt_orig[rand_idx]

            selected_skel = skels[rand_idx]

            for idx in range(20):
                curr_note_recon = selected_note_recons[idx]
                curr_vel_recon = selected_vel_recons[idx]
                curr_mt_recon = selected_mt_recons[idx]
                curr_note_orig = selected_note_orig[idx]
                curr_vel_orig = selected_vel_orig[idx]
                curr_mt_orig = selected_mt_orig[idx]
                curr_skel = selected_skel[idx]
                curr_tempo = int(tempos[rand_idx][idx])
                curr_genre = int(genre_idxs[rand_idx][idx])

                if n_samples > 1:
                    for n in range(n_samples):
                        # Save gen
                        filename = os.path.join(
                            self.gen_output_dir,
                            f"{fname}_epoch{epoch}_{idx}_gen{n}_{self.cfg.genre_list[curr_genre]}.mid")
                        matrix2midi(
                            curr_note_recon[n],
                            curr_vel_recon[n],
                            curr_mt_recon[n],
                            curr_tempo,
                            filename,
                            self.cfg.resolution,
                            only_note=False)
                else:
                    # Save gen
                    filename = os.path.join(
                        self.gen_output_dir,
                        f"{fname}_epoch{epoch}_{idx}_gen_{self.cfg.genre_list[curr_genre]}.mid")
                    matrix2midi(
                        curr_note_recon,
                        curr_vel_recon,
                        curr_mt_recon,
                        curr_tempo,
                        filename,
                        self.cfg.resolution,
                        only_note=False)

                # Save orig
                filename = os.path.join(
                    self.gen_output_dir,
                    f"{fname}_epoch{epoch}_{idx}_orig_{self.cfg.genre_list[curr_genre]}.mid")
                matrix2midi(
                    curr_note_orig,
                    curr_vel_orig,
                    curr_mt_orig,
                    curr_tempo,
                    filename,
                    self.cfg.resolution,
                    only_note=False)

                # Save skel
                filename = os.path.join(
                    self.gen_output_dir,
                    f"{fname}_epoch{epoch}_{idx}_skel_{self.cfg.genre_list[curr_genre]}.mid")
                matrix2midi(
                    curr_skel,
                    None,
                    None,
                    curr_tempo,
                    filename,
                    self.cfg.resolution,
                    only_note=True)

        return all_recon_outs, genre_idxs

    def genre_transfer(self, dataloader, new_genre_number):
        self.all_model.train(False)

        outputs = [[[], [], []], [[], [], []]]
        output_genres = [[], []]
        for i, data in enumerate(dataloader):

            skel = data["skel"].float().to(self.device)
            note = data["note"].float().to(self.device)
            vel = data["vel"].float().to(self.device)
            mt = data["mt"].float().to(self.device)
            orig_genre_idx = data["genre"].to(self.device)
            orig_genre = torch.nn.functional.one_hot(
                orig_genre_idx, num_classes=len(
                    self.cfg.genre_list)).to(
                self.device)
            tempo = data["tempo"].to(self.device)

            # New genre
            new_genre_idx = torch.ones_like(orig_genre_idx) * new_genre_number
            new_genre = torch.nn.functional.one_hot(
                new_genre_idx, num_classes=len(
                    self.cfg.genre_list)).to(
                self.device)

            # Conditions
            orig_genre_cond, new_genre_cond = None, None
            vel_cond = None
            time_cond = None
            if self.cfg.use_genre_cond:

                orig_genre_cond = orig_genre.float()

                new_genre_cond = new_genre.float()

            if self.cfg.use_vel_cond:
                vel_contour = data["vel_contour"].to(self.device)
                vel_cond = vel_contour.float()

            if self.cfg.use_time_cond:
                time_contour = data["time_contour"].to(self.device)
                time_cond = time_contour.float()

            skel_encoded = self.all_model.skel_model(skel)
            note_encoded = self.all_model.skel_model(note)

            for gcond in range(2):
                if gcond == 0:
                    genre_cond = orig_genre_cond
                    genre_list = orig_genre_idx
                else:
                    genre_cond = new_genre_cond
                    genre_list = new_genre_idx

                note_z = torch.randn(
                    (skel.size(0), self.cfg.model.note_z_dim)).to(
                    self.device)
                # vel_z = torch.randn((skel.size(0),self.cfg.model.vel_mt_z_dim)).to(self.device)
                # mt_z = torch.randn((skel.size(0),self.cfg.model.vel_mt_z_dim)).to(self.device)
                vel_z, _, _ = self.all_model.vel_model.encoder(
                    vel, vel_cond, note_encoded, genre_cond)
                mt_z, _, _ = self.all_model.time_model.encoder(
                    mt, time_cond, note_encoded, genre_cond)

                # decode
                note_gen = self.all_model.note_model.decoder(
                    note_z,
                    skel_encoded,
                    skel,
                    None,
                    genre_cond,
                    training=False,
                    teacher_forcing_ratio=None)
                binary_note_gen = binarize(note_gen, self.device)

                note_encoded = self.all_model.skel_model(binary_note_gen)

                vel_gen = self.all_model.vel_model.decoder(
                    vel_z,
                    note_encoded,
                    None,
                    binary_note_gen,
                    vel_cond,
                    genre_cond,
                    training=False,
                    teacher_forcing_ratio=None)
                mt_gen = self.all_model.time_model.decoder(
                    mt_z,
                    note_encoded,
                    None,
                    binary_note_gen,
                    time_cond,
                    genre_cond,
                    training=False,
                    teacher_forcing_ratio=None)

                outputs[gcond][0].append(tensor2np(binary_note_gen))
                outputs[gcond][1].append(tensor2np(vel_gen))
                outputs[gcond][2].append(tensor2np(mt_gen))
                output_genres[gcond].append(tensor2np(genre_list))

        # Return output
        original_outputs = np.array(outputs[0])
        transfer_outputs = np.array(outputs[1])
        original_genres = np.array(output_genres[0])
        transfer_genres = np.array(output_genres[1])
        outputs = []
        print(original_outputs.shape)
        original_outputs = original_outputs.reshape(
            3, -1, self.cfg.seq_len, self.cfg.n_drum_classes)
        transfer_outputs = transfer_outputs.reshape(
            3, -1, self.cfg.seq_len, self.cfg.n_drum_classes)
        original_genres = original_genres.reshape(-1)
        transfer_genres = transfer_genres.reshape(-1)
        # 3, n, 32, 7

        # # Save midi file
        for i in range(500, 520):

            curr_orig = original_outputs[:, i]
            curr_orig_genre = int(original_genres[i])
            curr_orig_genre_name = self.cfg.genre_list[curr_orig_genre]

            curr_tf = transfer_outputs[:, i]
            curr_tf_genre = int(transfer_genres[i])
            curr_tf_genre_name = self.cfg.genre_list[curr_tf_genre]

            orig_fname = os.path.join(
                self.gen_output_dir,
                f"tf_orig_{i}_{curr_orig_genre_name}.mid")
            tf_fname = os.path.join(
                self.gen_output_dir,
                f"tf_{i}_{curr_orig_genre_name}2{curr_tf_genre_name}.mid")

            matrix2midi(
                curr_orig[0],
                curr_orig[1],
                curr_orig[2],
                90,
                orig_fname,
                self.cfg.resolution,
                only_note=False)
            matrix2midi(
                curr_tf[0],
                curr_tf[1],
                curr_tf[2],
                90,
                tf_fname,
                self.cfg.resolution,
                only_note=False)

        return original_outputs, transfer_outputs, original_genres, transfer_genres
