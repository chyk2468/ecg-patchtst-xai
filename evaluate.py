"""
evaluate.py  —  Run evaluation on the test set and generate XAI plots.

Usage:
    python evaluate.py
"""

import torch
import numpy as np
from tqdm import tqdm

from src.data.dataset import get_dataloaders
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.training.metrics import compute_all_metrics, per_class_report
from src.utils.config import cfg
from src.utils.label_utils import decode_scp_labels, decode_superclass_labels
from src.xai.attention_viz import (
    attn_to_signal_importance,
    plot_attention_ecg,
    plot_lead_importance,
)
from src.xai.gradcam import GradCAM, plot_gradcam


# ── Helpers ────────────────────────────────────────────────────────────────

def threshold_predictions(logits: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
    return (logits >= threshold).astype(int)


# ── Evaluation loop ────────────────────────────────────────────────────────

def evaluate_test_set(model, loader, device, all_codes):
    model.eval()
    all_class_true, all_class_score = [], []
    all_super_true, all_super_score = [], []

    with torch.no_grad():
        for signals, class_labels, super_labels, _ in tqdm(loader,
                                                            desc="  Evaluating"):
            signals = signals.to(device)

            class_logits, super_logits, _ = model(signals)

            all_class_true.append(class_labels.numpy())
            all_class_score.append(
                torch.sigmoid(class_logits).cpu().numpy()
            )
            all_super_true.append(super_labels.numpy())
            all_super_score.append(
                torch.sigmoid(super_logits).cpu().numpy()
            )

    y_class_true  = np.vstack(all_class_true)
    y_class_score = np.vstack(all_class_score)
    y_super_true  = np.vstack(all_super_true)
    y_super_score = np.vstack(all_super_score)

    print("\n── SCP Code Metrics (71-class) ──────────────────────")
    m = compute_all_metrics(y_class_true, y_class_score)
    for k, v in m.items():
        print(f"  {k.upper():8s}: {v:.4f}")

    print("\n── Superclass Metrics (7-class) ─────────────────────")
    ms = compute_all_metrics(y_super_true, y_super_score)
    for k, v in ms.items():
        print(f"  {k.upper():8s}: {v:.4f}")

    print("\n── Per-Superclass Report ────────────────────────────")
    print(per_class_report(y_super_true, y_super_score, cfg.SUPERCLASSES))

    return y_class_score, y_super_score


# ── XAI on N sample records ────────────────────────────────────────────────

def run_xai(model, loader, device, all_codes, n_samples: int = 10):
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    gradcam = GradCAM(model)

    processed = 0
    for signals, class_labels, super_labels, record_ids in loader:
        for i in range(signals.size(0)):
            if processed >= n_samples:
                break

            signal_1    = signals[i:i+1].to(device)      # (1, 12, seq_len)
            record_id   = int(record_ids[i])

            # ── Forward ───────────────────────────────────────────────
            with torch.no_grad():
                class_logits, super_logits, attn_weights = model(signal_1)

            class_probs = torch.sigmoid(class_logits)[0].cpu().numpy()
            super_probs = torch.sigmoid(super_logits)[0].cpu().numpy()

            pred_class_idx = np.where(class_probs >= 0.5)[0].tolist()
            pred_super_idx = np.where(super_probs >= 0.5)[0].tolist()

            pred_classes = decode_scp_labels(
                (class_probs >= 0.5).astype(float), all_codes
            )
            pred_supers  = decode_superclass_labels(
                (super_probs >= 0.5).astype(float)
            )

            print(f"\n  Record #{record_id}")
            print(f"    Superclasses : {pred_supers}")
            print(f"    SCP codes    : {pred_classes}")

            signal_np = signal_1[0].cpu().numpy()   # (12, seq_len)

            # ── Attention viz ──────────────────────────────────────────
            importance = attn_to_signal_importance(attn_weights)
            importance = importance[0]                # (12, seq_len)

            plot_attention_ecg(signal_np, importance,
                               record_id, all_codes,
                               pred_classes, pred_supers,
                               save_dir=cfg.PLOT_DIR)

            plot_lead_importance(importance, record_id,
                                 save_dir=cfg.PLOT_DIR)

            # ── GradCAM on top predicted class ─────────────────────────
            if pred_class_idx:
                top_class = pred_class_idx[
                    int(np.argmax(class_probs[pred_class_idx]))
                ]
                cam = gradcam.compute(signal_1, top_class, head="class")
                plot_gradcam(signal_np, cam, record_id,
                             all_codes[top_class],
                             save_dir=cfg.PLOT_DIR)

            processed += 1
        if processed >= n_samples:
            break

    gradcam.remove_hooks()
    print(f"\n  XAI plots saved → {cfg.PLOT_DIR}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ECG Evaluation + XAI  |  Device: {device}")
    print(f"{'='*60}\n")

    loaders, all_codes = get_dataloaders(batch_size=cfg.BATCH_SIZE)

    # Load model
    model   = PatchTST()
    trainer = Trainer(model, loaders, device=device)
    trainer.load_best()
    model.to(device)

    # Evaluate
    evaluate_test_set(model, loaders["test"], device, all_codes)

    # XAI on 10 test samples
    print("\n── Generating XAI Plots ──────────────────────────────")
    run_xai(model, loaders["test"], device, all_codes, n_samples=10)


if __name__ == "__main__":
    main()
