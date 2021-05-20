import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import classification_report
# import logging
import hydra
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
import wandb

from src.models import vqvae
from datasets.data_utils import DRUM_CLASSES, NOTE_DENSITY_CLASSES, VEL_CLASSES, MT_CLASSES, load_data, matrix2midi
from src.dataloader import DrumDataset
from src.utils import binarize, kl_loss, compute_metrical_kl, tensor2np, batch2array

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
            if self.cfg.use_note_cond:
                self.cfg.note_cond_dim = len(NOTE_DENSITY_CLASSES)
            if self.cfg.use_vel_cond:
                self.cfg.vel_cond_dim = len(VEL_CLASSES)
            if self.cfg.use_time_cond:
                self.cfg.time_cond_tim = len(MT_CLASSES)

        # Print to check options
        print(f"Genre cond: {self.cfg.use_genre_cond}")
        print(f"Note cond: {self.cfg.use_note_cond}")
        print(f"Vel cond: {self.cfg.use_vel_cond}")
        print(f"Time cond: {self.cfg.use_time_cond}")

        print(f"Split index: {self.cfg.split_number}")

        self.model = vqvae.Prior(cfg, device).to(device)
        # self.note_model = vqvae.Note_VQVAE(cfg, device).to(device)
        self.vqvae_model = torch.load(os.path.join(
            p, self.cfg.model.vqvae_model_path)).to(device)
        self.vqvae_model.train(True)
        for vq in self.vqvae_model.parameters():
            vq.requires_grad = False

        self.teacher_forcing_ratio = 0

    def train(self):
        # Load data
        train_data, valid_data, _ = load_data(self.data_pickle_path)
        print(len(train_data), len(valid_data))
        # LOGGER.info(f"Number of train, valid data: {len(train_data)}, {len(valid_data)} ")

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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, betas=(0.5, 0.999)
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
        wandb.watch(self.model)
        wandb.run.name = self.cfg.model_name
        wandb.run.save()

        # exponential weighting
        tf_weight = np.logspace(1, 0.2, num=50, base=2, dtype="float") - 1

        for epoch in range(self.cfg.num_epochs):

            self.model.train(True)

            # self._validate(valid_dataloader,epoch, save_model=True, write_loss=True)

            if epoch < 30:
                self.teacher_forcing_ratio = tf_weight[epoch]

            for i, data in enumerate(train_dataloader):
                skel = data["skel"].float().to(self.device)
                note = data["note"].float().to(self.device)
                vel = data["vel"].float().to(self.device)
                mt = data["mt"].float().to(self.device)
                genre_idx = data["genre"].to(self.device)
                genre = torch.nn.functional.one_hot(
                    genre_idx, num_classes=len(self.cfg.genre_list))

                # Conditions
                genre_cond = None
                note_cond = None
                vel_cond = None
                time_cond = None

                ### SKEL #####
                skel_encoded = self.vqvae_model.skel_model(skel)
                # note_encoded = self.all_model.skel_model(note)

                ##### NOTE ######
                _, note_quantized_z, _, _, codebook_indices, _ = self.vqvae_model.note_model(
                    note, skel, skel_encoded, training=False, teacher_forcing_ratio=None)

                codebook_indices = codebook_indices.view(
                    self.cfg.batch_size, -1)

                if self.cfg.use_genre_cond:
                    genre_cond = genre.float()

                if self.cfg.use_note_cond:
                    note_density_idx = data["note_density_idx"].to(self.device)
                    note_density = torch.nn.functional.one_hot(
                        note_density_idx, num_classes=self.cfg.note_cond_dim).float()
                    note_cond = note_density

                out, sampled_out = self.model(codebook_indices, skel_encoded, note_cond, genre_cond,
                                              temp=0.5, training=True, teacher_forcing_ratio=self.teacher_forcing_ratio)
                out_reshaped = out.reshape(-1, self.cfg.model.n_codes)
                codebook_indices = codebook_indices.reshape(-1)

                loss = F.cross_entropy(out_reshaped, codebook_indices)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log losses to stdout and to wandb
                if i % 100 == 0:
                    print(f"Epoch {epoch}, step {i} /{len(train_dataloader)}")
                    print("loss", loss.detach().item())

                    wandb.log({
                        "Train Loss": loss.detach().item(),
                    })

            # Validation
            self._validate(
                valid_dataloader,
                epoch,
                save_model=True,
                write_loss=True)

            if epoch % 5 == 0:
                print("GENERATING...")
                self.generate(valid_dataloader, epoch, to_midi=True)

    def run(self, dataloader, return_results):
        self.model.train(False)

        pred_outs = []
        true_outs = []
        total_loss = 0
        correct = 0
        total_cnt = 0
        for i, data in enumerate(dataloader):
            skel = data["skel"].float().to(self.device)
            note = data["note"].float().to(self.device)
            vel = data["vel"].float().to(self.device)
            mt = data["mt"].float().to(self.device)
            genre_idx = data["genre"].to(self.device)
            genre = torch.nn.functional.one_hot(
                genre_idx, num_classes=len(self.cfg.genre_list))
            tempo = data["tempo"].to(self.device)

            # Conditions
            genre_cond = None
            note_cond = None
            vel_cond = None
            time_cond = None

            skel_encoded = self.vqvae_model.skel_model(skel)

            _, note_quantized_z, _, _, codebook_indices, _ = self.vqvae_model.note_model(
                note, skel, skel_encoded, training=False, teacher_forcing_ratio=None)

            codebook_indices = codebook_indices.view(self.cfg.batch_size, -1)

            if self.cfg.use_genre_cond:
                genre_cond = genre.float()

            if self.cfg.use_note_cond:
                note_density_idx = data["note_density_idx"].to(self.device)
                note_density = torch.nn.functional.one_hot(
                    note_density_idx, num_classes=self.cfg.note_cond_dim).float()
                note_cond = note_density

            out, sampled_out = self.model(codebook_indices, skel_encoded, note_cond, genre_cond,
                                          temp=0.5, training=True, teacher_forcing_ratio=self.teacher_forcing_ratio)
            out_reshaped = out.reshape(-1, self.cfg.model.n_codes)
            codebook_indices = codebook_indices.reshape(-1)

            loss = F.cross_entropy(out_reshaped, codebook_indices)

            total_loss += loss.detach()

            # ACC
            _, predicted = torch.max(out_reshaped.data, 1)
            total_cnt += codebook_indices.size(0)
            correct += (predicted == codebook_indices).sum().item()

            pred_outs.append(tensor2np(sampled_out))
            true_outs.append(tensor2np(codebook_indices))

        pred_outs = batch2array(pred_outs)
        true_outs = batch2array(true_outs)

        dlen = len(dataloader)

        if return_results:
            return total_loss / dlen, correct / total_cnt, pred_outs, true_outs

        else:
            return total_loss / dlen, correct / total_cnt

    def _validate(self, dataloader, epoch, save_model, write_loss):

        loss, acc = self.run(dataloader, return_results=False)

        self.scheduler.step(loss)

        # Write loss
        if write_loss:
            print(f"Eval loss: {loss}")
            print(f"Eval acc: {acc}")

            wandb.log({
                "Valid Loss": loss,
                "Valid Acc": acc,
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
                self.model,
                savename
            )
            print(f"Model at epoch {epoch} saved as {savename}")

        return loss, acc

    def get_test_data(self, batch_size):
        _, _, test_data = load_data(self.data_pickle_path)
        print(len(test_data))
        # Dataloader
        test_dataset = DrumDataset(test_data)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True)

        return test_data, test_dataloader

    def generate(self, dataloader, epoch, to_midi=False):

        self.model.train(False)

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
                genre_idx, num_classes=len(self.cfg.genre_list))
            tempo = data["tempo"].to(self.device)

            # Conditions
            genre_cond = None
            note_cond = None
            vel_cond = None
            time_cond = None
            if self.cfg.use_genre_cond:
                genre_cond = genre.float()

            if self.cfg.use_note_cond:
                note_density_idx = data["note_density_idx"].to(self.device)
                note_density = torch.nn.functional.one_hot(
                    note_density_idx, num_classes=self.cfg.note_cond_dim).float()
                note_cond = note_density

            if self.cfg.use_vel_cond:
                vel_contour = data["vel_contour"].to(self.device)
                vel_cond = vel_contour.float()

            if self.cfg.use_time_cond:
                time_contour = data["time_contour"].to(self.device)
                time_cond = time_contour.float()

            skel_encoded = self.vqvae_model.skel_model(skel)

            # Generate code sequence
            out, sampled_out = self.model(
                None, skel_encoded, note_cond, genre_cond, temp=0.5, training=False, teacher_forcing_ratio=None)
            out_reshaped = out.reshape(-1, self.cfg.model.n_codes)

            # Retrieve the code vectors from the code indices
            # (32, 16)
            w = self.vqvae_model.note_model.vector_quantizer.embed.weight

            sampled_out_idx = sampled_out.view(-1)
            quantized_note_z = w[sampled_out_idx]
            quantized_note_z = quantized_note_z.view(
                -1, self.cfg.model.code_length, self.cfg.model.code_dim)
            quantized_note_z = quantized_note_z.transpose(1, 2)

            # Generate note with the generated code sequence
            note_gen = self.vqvae_model.note_model.decoder(
                quantized_note_z,
                note,
                skel,
                skel_encoded,
                training=False,
                teacher_forcing_ratio=None)

            note_gen = binarize(note_gen, self.device)

            note_encoded = self.vqvae_model.skel_model(note)

            # Encode velocity and mt style from reference track
            vel_input = torch.cat((vel, note), dim=-1)
            mt_input = torch.cat((mt, note), dim=-1)
            vel_z, _, _ = self.vqvae_model.vel_model.encoder(
                vel_input, note_encoded, vel_cond, genre_cond)
            mt_z, _, _ = self.vqvae_model.time_model.encoder(
                mt_input, note_encoded, time_cond, genre_cond)

            # Generate vel, mt
            vel_gen = self.vqvae_model.vel_model.decoder(
                vel_z,
                note_encoded,
                None,
                note_gen,
                vel_cond,
                genre_cond,
                training=False,
                teacher_forcing_ratio=None)
            mt_gen = self.vqvae_model.time_model.decoder(
                mt_z,
                note_encoded,
                None,
                note_gen,
                time_cond,
                genre_cond,
                training=False,
                teacher_forcing_ratio=None)

            note_outs.append(tensor2np(note_gen))
            vel_outs.append(tensor2np(vel_gen))
            mt_outs.append(tensor2np(mt_gen))

            skels.append(tensor2np(skel))
            note_orig.append(tensor2np(note))
            vel_orig.append(tensor2np(vel))
            mt_orig.append(tensor2np(mt))

            tempos.append(tensor2np(tempo))
            genre_idxs.append(tensor2np(genre_idx))

        note_outs = batch2array(note_outs)
        note_orig = batch2array(note_orig)
        skels = batch2array(skels)

        vel_outs = batch2array(vel_outs)
        vel_orig = batch2array(vel_orig)

        mt_outs = batch2array(mt_outs)
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

                # Save gen
                filename = os.path.join(
                    self.gen_output_dir,
                    f"epoch{epoch}_{idx}_gen_{self.cfg.genre_list[curr_genre]}.mid")
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
                    f"epoch{epoch}_{idx}_orig_{self.cfg.genre_list[curr_genre]}.mid")
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
                    f"epoch{epoch}_{idx}_skel_{self.cfg.genre_list[curr_genre]}.mid")
                matrix2midi(
                    curr_skel,
                    None,
                    None,
                    curr_tempo,
                    filename,
                    self.cfg.resolution,
                    only_note=True)

        return all_recon_outs, genre_idxs
