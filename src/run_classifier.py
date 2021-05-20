import os
import torch
import torch.nn.functional as F
import numpy as np
import random
# import logging
import hydra
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
import wandb

from src.models import classifier
from datasets.data_utils import DRUM_CLASSES, NOTE_DENSITY_CLASSES, VEL_CLASSES, load_data
from src.dataloader import DrumDataset
from src.utils import binarize, kl_loss, note_constrained_loss, batch2array

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

        # Add more configs
        OmegaConf.set_struct(self.cfg, True)
        with open_dict(self.cfg):
            self.cfg.n_drum_classes = len(DRUM_CLASSES)
            self.cfg.seq_len = self.cfg.resolution * \
                self.cfg.bar_length * self.cfg.beats_per_bar

        print(f"Split index: {self.cfg.split_number}")

        self.model = classifier.GenreLSTM(cfg).to(device)

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
            project=self.cfg.model.wandb_project_name,
            entity=self.cfg.wandb_entity)
        wandb.watch(self.model)
        wandb.run.name = self.cfg.model_name
        wandb.run.save()

        for epoch in range(self.cfg.num_epochs):

            self.model.train(True)

            for i, data in enumerate(train_dataloader):

                note = data["note"].float().to(self.device)
                vel = data["vel"].float().to(self.device)
                mt = data["mt"].float().to(self.device)
                genre_idx = data["genre"].to(self.device)

                pred = self.model(torch.cat((note, vel, mt), dim=-1))
                loss = F.cross_entropy(pred, genre_idx)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log losses to stdout and to wandb
                if i % 100 == 0:
                    print(f"Epoch {epoch}, step {i} /{len(train_dataloader)}")
                    print("Train loss", loss.detach().item())

                    wandb.log({
                        "Train Loss": loss.detach().item(),
                    })

            # Validation
            self._validate(
                valid_dataloader,
                epoch,
                save_model=True,
                write_loss=True)

    def run(self, dataloader, return_results):
        self.model.train(False)

        pred_probs = []
        genre_label = []
        total_loss = 0
        correct = 0
        cnt = 0
        for i, data in enumerate(dataloader):
            note = data["note"].float().to(self.device)
            vel = data["vel"].float().to(self.device)
            mt = data["mt"].float().to(self.device)
            genre_idx = data["genre"].to(self.device)

            pred = self.model(torch.cat((note, vel, mt), dim=-1))
            loss = F.cross_entropy(pred, genre_idx)
            total_loss += loss.detach().item()

            pred_genre = torch.argmax(pred, dim=-1)
            correct += (pred_genre == genre_idx).float().sum().detach().item()
            cnt += note.size(0)

            if return_results:
                pred = torch.softmax(pred, dim=-1)
                pred_probs.append(pred.detach().cpu().numpy())
                genre_label.append(genre_idx.detach().cpu().numpy())

        if return_results:
            pred_probs = batch2array(pred_probs)
            genre_label = np.array(genre_label)
            genre_label = genre_label.reshape(-1)
            return total_loss / len(dataloader), correct / \
                cnt, pred_probs, genre_label
        return total_loss / len(dataloader), correct / cnt

    def _validate(self, dataloader, epoch, save_model, write_loss):

        loss, acc = self.run(dataloader, return_results=False)

        self.scheduler.step(loss)

        # Write loss
        if write_loss:
            print(f"Eval loss: {loss}")
            print(f"Eval acc: {acc}")
            wandb.log({
                "Valid Loss": loss,
                "Valid Acc": acc
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
