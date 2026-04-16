"""
predict.py  —  Run prediction on a single ECG record.

Usage:
    python predict.py --record records100/00000/00001_lr
    python predict.py --record records100/00000/00005_lr --xai
"""

import argparse
import torch
import numpy as np

from src.data.loader import load_signal, load_metadata, get_record_path
from src.data.preprocessing import preprocess_signal
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.data.dataset import get_dataloaders
from src.utils.config import cfg
from src.utils.label_utils import decode_scp_labels, decode_superclass_labels
from src.xai.attention_viz import attn_to_signal_importance, plot_attention_ecg, plot_lead_importance
from src.xai.gradcam import GradCAM, plot_gradcam


def predict(record_path: str, show_xai: bool = False):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    # ── Load metadata & model ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ECG Predict  |  Device: {device}")
    print(f"  Record: {record_path}")
    print(f"{'='*55}\n")

    loaders, all_codes = get_dataloaders(batch_size=1)

    model   = PatchTST()
    trainer = Trainer(model, loaders, device=device)
    trainer.load_best()
    model.to(device)
    model.eval()

    # ── Load & preprocess signal ───────────────────────────────────────────
    print("Loading signal...")
    signal_np = load_signal(record_path)                    # (seq_len, 12)
    signal_np = preprocess_signal(signal_np, cfg.SEQ_LEN)  # (12, seq_len)
    signal    = torch.from_numpy(signal_np).unsqueeze(0).to(device)  # (1, 12, seq_len)

    # ── Inference ──────────────────────────────────────────────────────────
    print("Running inference...\n")
    with torch.no_grad():
        class_logits, super_logits, attn_weights = model(signal)

    class_probs = torch.sigmoid(class_logits)[0].cpu().numpy()  # (71,)
    super_probs = torch.sigmoid(super_logits)[0].cpu().numpy()  # (7,)

    # ── Decode predictions ─────────────────────────────────────────────────
    pred_classes = decode_scp_labels((class_probs >= 0.5).astype(float), all_codes)
    pred_supers  = decode_superclass_labels((super_probs >= 0.5).astype(float))

    # ── Print results ──────────────────────────────────────────────────────
    print("─" * 45)
    print(f"  SUPERCLASSES DETECTED : {pred_supers if pred_supers else ['None']}")
    print(f"  SCP CODES DETECTED    : {pred_classes if pred_classes else ['None']}")
    print("─" * 45)

    print("\n  Superclass Probabilities:")
    for i, sc in enumerate(cfg.SUPERCLASSES):
        bar = "█" * int(super_probs[i] * 20)
        print(f"    {sc:8s} {super_probs[i]:.4f}  |{bar:<20}|")

    print("\n  Top 10 SCP Code Probabilities:")
    top10_idx = np.argsort(class_probs)[::-1][:10]
    for idx in top10_idx:
        bar = "█" * int(class_probs[idx] * 20)
        print(f"    {all_codes[idx]:10s} {class_probs[idx]:.4f}  |{bar:<20}|")

    # ── XAI ───────────────────────────────────────────────────────────────
    if show_xai:
        print("\n  Generating XAI plots...")
        cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)

        record_id = record_path.split("/")[-1]

        # Attention
        importance = attn_to_signal_importance(attn_weights)[0]  # (12, seq_len)
        plot_attention_ecg(signal_np, importance, record_id,
                           all_codes, pred_classes, pred_supers,
                           save_dir=cfg.PLOT_DIR)
        plot_lead_importance(importance, record_id, save_dir=cfg.PLOT_DIR)

        # GradCAM
        pred_class_idx = np.where(class_probs >= 0.5)[0].tolist()
        if pred_class_idx:
            gradcam  = GradCAM(model)
            top_class = pred_class_idx[int(np.argmax(class_probs[pred_class_idx]))]
            cam = gradcam.compute(signal, top_class, head="class")
            plot_gradcam(signal_np, cam, record_id,
                         all_codes[top_class], save_dir=cfg.PLOT_DIR)
            gradcam.remove_hooks()

        print(f"\n  XAI plots saved → {cfg.PLOT_DIR}")

    print("\nDone! ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ECG superclass and SCP codes")
    parser.add_argument("--record", type=str, required=True,
                        help="Record path without extension, e.g. records100/00000/00001_lr")
    parser.add_argument("--xai", action="store_true",
                        help="Generate XAI attention and GradCAM plots")
    args = parser.parse_args()

    predict(args.record, show_xai=args.xai)
