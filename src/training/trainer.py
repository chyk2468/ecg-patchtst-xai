import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.training.metrics import compute_all_metrics
from src.utils.config import cfg


# ── Loss ───────────────────────────────────────────────────────────────────

class MultiLabelLoss(nn.Module):
    """
    Combined loss:
      - BCE on 71-class SCP codes       (primary)
      - BCE on 7-class superclasses     (auxiliary, lower weight)
    """

    def __init__(self, aux_weight: float = 0.3):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.aux_weight = aux_weight

    def forward(self, class_logits, super_logits,
                class_targets, super_targets):
        primary   = self.bce(class_logits, class_targets)
        auxiliary = self.bce(super_logits, super_targets)
        return primary + self.aux_weight * auxiliary


# ── Trainer ────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, loaders, device=None):
        self.model   = model
        self.loaders = loaders
        self.device  = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.criterion  = MultiLabelLoss()
        self.optimizer  = AdamW(model.parameters(),
                                lr=cfg.LR,
                                weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler  = CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
        )

        cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.best_auc       = 0.0
        self.patience_count = 0
        self.history        = {"train": [], "val": []}

    # ── One epoch ─────────────────────────────────────────────────────────

    def _run_epoch(self, split: str) -> dict:
        is_train = (split == "train")
        self.model.train() if is_train else self.model.eval()

        loader     = self.loaders[split]
        total_loss = 0.0
        all_class_true, all_class_score = [], []
        all_super_true, all_super_score = [], []

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc=f"  {split:5s}", leave=False)
            for signals, class_labels, super_labels, _ in pbar:
                signals      = signals.to(self.device)
                class_labels = class_labels.to(self.device)
                super_labels = super_labels.to(self.device)

                class_logits, super_logits, _ = self.model(signals)
                loss = self.criterion(class_logits, super_logits,
                                      class_labels, super_labels)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()

                # Collect for metrics
                all_class_true.append(class_labels.cpu().numpy())
                all_class_score.append(
                    torch.sigmoid(class_logits).cpu().detach().numpy()
                )
                all_super_true.append(super_labels.cpu().numpy())
                all_super_score.append(
                    torch.sigmoid(super_logits).cpu().detach().numpy()
                )

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        y_class_true  = np.vstack(all_class_true)
        y_class_score = np.vstack(all_class_score)
        y_super_true  = np.vstack(all_super_true)
        y_super_score = np.vstack(all_super_score)

        class_metrics = compute_all_metrics(y_class_true, y_class_score)
        super_metrics = compute_all_metrics(y_super_true, y_super_score)

        return {
            "loss":        total_loss / len(loader),
            "class_auc":   class_metrics["auc"],
            "class_f1":    class_metrics["f1"],
            "class_map":   class_metrics["map"],
            "super_auc":   super_metrics["auc"],
            "super_f1":    super_metrics["f1"],
        }

    # ── Full training loop ─────────────────────────────────────────────────

    def fit(self):
        print(f"\n{'='*60}")
        print(f"  Training on {self.device}  |  {cfg.EPOCHS} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, cfg.EPOCHS + 1):
            t0 = time.time()

            train_metrics = self._run_epoch("train")
            val_metrics   = self._run_epoch("val")

            self.scheduler.step()

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            elapsed = time.time() - t0
            print(
                f"Ep {epoch:03d}/{cfg.EPOCHS} | "
                f"loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"class_AUC {train_metrics['class_auc']:.4f}/{val_metrics['class_auc']:.4f} | "
                f"super_AUC {val_metrics['super_auc']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # ── Checkpoint ────────────────────────────────────────────────
            val_auc = val_metrics["class_auc"]
            if val_auc > self.best_auc:
                self.best_auc   = val_auc
                self.patience_count = 0
                self._save_checkpoint(epoch, val_metrics)
                print(f"  ✓ New best AUC: {val_auc:.4f}  → checkpoint saved")
            else:
                self.patience_count += 1
                if self.patience_count >= cfg.EARLY_STOPPING_PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        # Save training history
        log_path = cfg.LOG_DIR / "history.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nHistory saved → {log_path}")

    def _save_checkpoint(self, epoch: int, metrics: dict):
        ckpt = {
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "metrics":     metrics,
            "best_auc":    self.best_auc,
        }
        path = cfg.MODEL_DIR / "best_model.pt"
        torch.save(ckpt, path)

    def load_best(self):
        path = cfg.MODEL_DIR / "best_model.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (epoch {ckpt['epoch']}, "
              f"AUC={ckpt['best_auc']:.4f})")
        return ckpt
