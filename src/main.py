"""
Main training/evaluation
"""

import os
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

import src.run_vae
import src.run_vqvae
import src.run_prior
import src.run_onestep
import src.run_classifier


@hydra.main(config_path="config", config_name="config")
def main_func(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cfg.model.model_type == "vae":
        exp = src.run_vae.Experiment(cfg, device)
        exp.train()

    elif cfg.model.model_type == "vqvae":
        exp = src.run_vqvae.Experiment(cfg, device)
        exp.train()

    elif cfg.model.model_type == "prior":
        exp = src.run_prior.Experiment(cfg, device)
        exp.train()

    elif cfg.model.model_type == "onestep":
        exp = src.run_onestep.Experiment(cfg, device)
        exp.train()

    elif cfg.model.model_type == "classifier":
        exp = src.run_classifier.Experiment(cfg, device)
        exp.train()


if __name__ == "__main__":
    main_func()
