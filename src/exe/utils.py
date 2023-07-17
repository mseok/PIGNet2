import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric as pyg
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBIO import Select
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from rdkit import Chem
from scipy.spatial import distance_matrix
from scipy.stats import kendalltau, linregress
from sklearn.metrics import r2_score
from torch.nn import Module, init
from torch.optim import Optimizer

PathLike = Union[str, os.PathLike]


def cuda_visible_devices(
    num_gpus: int,
    max_num_gpus: int = 16,
) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    idle_gpus = []

    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]
            proc = subprocess.run(cmd, capture_output=True, text=True)

            if "No devices were found" in proc.stdout:
                break

            if "No running" in proc.stdout:
                idle_gpus.append(i)

            if len(idle_gpus) >= num_gpus:
                break

        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)

        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

    else:
        idle_gpus = ""

    return idle_gpus


def seed(seed: int):
    """Manually seed the RNGs and turn off nondeterministic algorithms.
    Currently, calling this function as is does not guarantee reproducibility."""
    pyg.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The current code doesn't allow the following line:
    # torch.use_deterministic_algorithms(True)


def merge_configs(
    saved_config: DictConfig,
    new_config: DictConfig,
) -> DictConfig:
    with open_dict(saved_config):
        if hasattr(saved_config, "data"):
            del saved_config.data
        merged_config = OmegaConf.merge(saved_config, new_config)
    return merged_config


def initialize_state(
    device: torch.device,
    checkpoint: Optional[Dict[str, Any]] = None,
    config: Optional[DictConfig] = None,
    in_features: int = -1,
) -> Tuple[Module, int, DictConfig]:
    """Initialize the model. Load their states if given.

    Returns:
        model: torch.nn.Module
        epoch: int
            The epoch when the state was saved.
            0 if `checkpoint` not given.
    """
    # Check arguments.
    assert checkpoint or (checkpoint is None and config)

    # Load the model.
    if checkpoint:
        model = instantiate(config.model, config, checkpoint["in_features"])
        model.load_state_dict(checkpoint["model_state_dict"])
    # Initialize a fresh model.
    else:
        model = instantiate(config.model, config, in_features)
        for param in model.parameters():
            if param.dim() == 1:
                continue
            init.xavier_normal_(param)

    model.to(device)

    epoch = 0 if not checkpoint else checkpoint["epoch"]
    return model, epoch


def save_state(
    save_path: PathLike,
    epoch: int,
    model: Module,
    optimizer: Optimizer,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config,
            "in_features": model.in_features,
        },
        save_path,
    )


def get_losses(model: Module) -> Dict[str, float]:
    """Gather each task's loss from `model.losses` into a dict.

    Reurns:
        losses[task] -> loss_value
    """
    # model.losses[loss_type][task] -> List[float]

    losses = defaultdict(float)
    tasks_done = model.losses["energy"].keys()

    for task in tasks_done:
        losses[task] += np.mean(model.losses["energy"][task])

    # Combine vdW-radius losses
    losses["dvdw"] = np.mean(list(model.losses["dvdw"].values()))

    return losses


def get_stats(
    model: Module,
    task: str,
) -> Tuple[float, float]:
    """Get statistics using `model.prediction` and `model.labels`.

    Reurns:
        r: float
        r2: float
    """
    # model.predictions[task][key] -> [vdw, hbond, ml, hydro]
    # model.labels[task][key] -> label: float

    keys = model.predictions[task].keys()
    pred = [sum(model.predictions[task][key]) for key in keys]
    true = [model.labels[task][key] for key in keys]

    slop, intercept, r, p_value, std_err = linregress(true, pred)
    r2 = r2_score(true, pred)

    # kendalltau
    true, pred = zip(*sorted(zip(true, pred)))
    _, true_order = zip(*sorted(zip(true, [i + 1 for i in range(len(keys))])))
    _, pred_order = zip(*sorted(zip(pred, [i + 1 for i in range(len(keys))])))
    tau, _ = kendalltau(true_order, pred_order)

    return r, r2, tau


def write_predictions(
    model: Module,
    config: DictConfig,
    train: bool,
) -> None:
    # model.predictions[task][key] -> [vdw, hbond, ml, hydro]
    # model.labels[task][key] -> label: float

    if train:
        prefix = "train_"
    else:
        prefix = "test_"

    for task in model.predictions:
        keys = model.predictions[task].keys()
        pred = model.predictions[task]
        true = model.labels.get(task, {})

        with open(config.data[task][prefix + "result_path"], "w") as f:
            for key in sorted(keys):
                f.write(f"{key}\t{true.get(key, 0.0):.3f}")
                f.write(f"\t{sum(pred[key]):.3f}")
                for energy in pred[key]:
                    f.write(f"\t{energy:.3f}")
                f.write("\n")

    if train:
        try:
            with open("learnable_parameters.txt", "w") as f:
                f.write(f"hydrophobic_coeff: {model.hydrophobic_coeff.item()}\n")
                f.write(f"hbond_coeff: {model.hbond_coeff.item()}\n")
                f.write(f"rotor_coeff: {model.rotor_coeff.item()}\n")
                if config.model.get("include_ionic", False):
                    f.write(f"ionic_coeff: {model.ionic_coeff.item()}\n")
        except AttributeError:  # GNN
            pass


def initialize_logger(
    log_file: Optional[PathLike] = None,
    log_file_level: int = logging.NOTSET,
    rotate: bool = False,
    mode: str = "w",
) -> logging.Logger:
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set logging to stdout.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file:
        if rotate:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10
            )
        else:
            file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def get_log_line(
    tasks: Iterable[str],
    losses: Dict[str, float] = None,
    title: bool = False,
) -> str:
    """Get the header line or loss-values line to be printed.

    Usage:
        get_log_line(tasks, title=True) -> header line
        get_log_line(tasks, losses) -> loss-values line
    """
    if title:
        values = ["epoch"]
        values += ["train_l_" + task for task in tasks]
        values += ["train_l_dvdw"]
        values += ["test_l_" + task for task in tasks]
        values += ["test_l_dvdw"]
        values += ["train_r", "test_r", "train_tau", "test_tau", "time"]
    else:
        values = [f"{losses[task]:.3f}" for task in tasks]
        values += [f"{losses['dvdw']:.3f}"]
    return "\t".join(values)
