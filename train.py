"""
train.py  —  Entry point for training the PatchTST ECG classifier.

Usage:
    python train.py
"""

import random
import numpy as np
import torch

from src.data.dataset import get_dataloaders
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.utils.config import cfg


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ECG PatchTST + XAI  |  Device: {device}")
    print(f"  Sampling rate : {cfg.SAMPLING_RATE} Hz")
    print(f"  Patch len     : {cfg.PATCH_LEN}  |  Stride: {cfg.STRIDE}")
    print(f"  Classes       : {cfg.NUM_CLASSES} SCP codes + {cfg.NUM_SUPERCLASSES} superclasses")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    loaders, all_codes = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        num_workers=4,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = PatchTST()
    n_params = PatchTST.num_parameters(model)
    print(f"  Model parameters: {n_params:,}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, loaders, device=device)
    trainer.fit()

    print("\nTraining complete.")
    print(f"  Best val AUC  : {trainer.best_auc:.4f}")
    print(f"  Checkpoint    : {cfg.MODEL_DIR / 'best_model.pt'}")
    print(f"  History log   : {cfg.LOG_DIR / 'history.json'}")


if __name__ == "__main__":
    main()
