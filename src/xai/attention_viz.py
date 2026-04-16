"""
Attention-based XAI for PatchTST.

Converts patch-level attention weights back to the original time axis
and visualizes which temporal regions drove each classification decision.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from src.utils.config import cfg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ── Attention → time axis ──────────────────────────────────────────────────

def attn_to_signal_importance(attn_weights: torch.Tensor,
                               seq_len:      int = cfg.SEQ_LEN,
                               patch_len:    int = cfg.PATCH_LEN,
                               stride:       int = cfg.STRIDE) -> np.ndarray:
    """
    Convert patch-level attention weights to per-timestep importance scores.

    attn_weights : (B*C, N, N)  last encoder attention layer
    Returns      : (B, C, seq_len)  importance score per timestep
    """
    # Mean over heads (already done in nn.MultiheadAttention output),
    # then mean over query dimension → (B*C, N)
    patch_importance = attn_weights.mean(dim=1).cpu().detach().numpy()

    BC, N = patch_importance.shape
    B = BC // cfg.NUM_LEADS
    C = cfg.NUM_LEADS

    # Reshape: (B, C, N)
    patch_importance = patch_importance.reshape(B, C, N)

    # Scatter patch scores back to the time axis
    importance = np.zeros((B, C, seq_len), dtype=np.float32)
    counts     = np.zeros((B, C, seq_len), dtype=np.float32)

    for p in range(N):
        start = p * stride
        end   = start + patch_len
        if end > seq_len:
            end = seq_len
        importance[:, :, start:end] += patch_importance[:, :, p:p+1]
        counts[:, :, start:end]     += 1.0

    counts = np.where(counts == 0, 1, counts)
    importance = importance / counts

    # Normalize per lead to [0, 1]
    mn  = importance.min(axis=-1, keepdims=True)
    mx  = importance.max(axis=-1, keepdims=True)
    rng = np.where((mx - mn) == 0, 1, mx - mn)
    return (importance - mn) / rng


# ── Plot: one record ───────────────────────────────────────────────────────

def plot_attention_ecg(signal:      np.ndarray,
                       importance:  np.ndarray,
                       record_id:   int,
                       class_names: list,
                       pred_classes: list,
                       pred_supers:  list,
                       save_dir=None):
    """
    Plot all 12 leads with attention heatmap overlay.

    signal      : (12, seq_len)   preprocessed signal
    importance  : (12, seq_len)   per-timestep importance scores
    pred_classes: list of predicted SCP code strings
    pred_supers : list of predicted superclass strings
    """
    fig, axes = plt.subplots(12, 1, figsize=(18, 20), sharex=True)
    fig.suptitle(
        f"Record #{record_id}  |  "
        f"Superclasses: {', '.join(pred_supers)}  |  "
        f"SCP codes: {', '.join(pred_classes[:5])}{'...' if len(pred_classes) > 5 else ''}",
        fontsize=12, fontweight="bold", y=0.99
    )

    t = np.arange(signal.shape[1])
    cmap = cm.get_cmap("Reds")

    for i, ax in enumerate(axes):
        sig  = signal[i]
        imp  = importance[i]

        # Background colour: attention heatmap
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j+1],
                       color=cmap(float(imp[j])),
                       alpha=0.4, linewidth=0)

        ax.plot(t, sig, color="steelblue", linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], fontsize=8, rotation=0, labelpad=20)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Timestep (samples)", fontsize=9)

    # Colour bar
    sm  = plt.cm.ScalarMappable(cmap=cmap,
                                 norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=axes, location="right",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Attention Importance", fontsize=9)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"attention_{record_id}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ── Plot: per-lead importance bar chart ───────────────────────────────────

def plot_lead_importance(importance: np.ndarray,
                         record_id:  int,
                         save_dir=None):
    """
    Bar chart showing mean attention per lead.
    importance: (12, seq_len)
    """
    mean_imp = importance.mean(axis=-1)   # (12,)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.RdYlGn(1 - mean_imp / mean_imp.max())
    ax.barh(LEAD_NAMES[::-1], mean_imp[::-1], color=colors[::-1])
    ax.set_xlabel("Mean Attention Importance")
    ax.set_title(f"Lead Importance — Record #{record_id}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"lead_importance_{record_id}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
