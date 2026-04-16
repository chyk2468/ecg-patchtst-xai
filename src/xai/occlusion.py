"""
src/xai/occlusion.py

Occlusion Sensitivity for PatchTST ECG classification.

Systematically masks (occludes) patches of the ECG signal and measures
how much the prediction score drops — patches that cause a large drop
are considered most important.

This is one of the most intuitive and faithful XAI methods because
it directly measures prediction sensitivity to input regions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from src.utils.config import cfg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ── Occlusion Sensitivity ──────────────────────────────────────────────────

class OcclusionSensitivity:
    """
    Occlusion Sensitivity for PatchTST.

    Algorithm:
        1. Get baseline prediction score for target class
        2. For each patch window:
             - Replace that window with zeros (or mean)
             - Re-run forward pass
             - Score drop = baseline_score - occluded_score
        3. Map score drops back to the time axis
        4. High drop = important region

    Usage:
        occ = OcclusionSensitivity(model)
        importance = occ.compute(signal, target_class_idx)
    """

    def __init__(self, model, patch_len: int = cfg.PATCH_LEN,
                 stride: int = cfg.STRIDE):
        self.model     = model
        self.device    = next(model.parameters()).device
        self.patch_len = patch_len
        self.stride    = stride

    def compute(self, signal:           torch.Tensor,
                target_class_idx:       int,
                head:                   str   = "class",
                occlusion_value:        float = 0.0,
                per_lead:               bool  = False) -> np.ndarray:
        """
        Compute occlusion sensitivity map.

        signal           : (1, 12, seq_len)
        target_class_idx : index of target class
        head             : 'class' or 'super'
        occlusion_value  : value to fill occluded region (0 = flat line)
        per_lead         : if True, occlude one lead at a time too

        Returns: (12, seq_len) importance map normalized to [0, 1]
        """
        self.model.eval()
        signal    = signal.to(self.device)
        seq_len   = signal.shape[-1]
        n_leads   = signal.shape[1]

        # ── Baseline score ─────────────────────────────────────────────────
        with torch.no_grad():
            c_logits, s_logits, _ = self.model(signal)
            logits   = c_logits if head == "class" else s_logits
            baseline = torch.sigmoid(logits)[0, target_class_idx].item()

        # ── Patch positions ────────────────────────────────────────────────
        starts = list(range(0, seq_len - self.patch_len + 1, self.stride))

        # Result maps
        score_drops = np.zeros((n_leads, seq_len), dtype=np.float32)
        counts      = np.zeros((n_leads, seq_len), dtype=np.float32)

        # ── Occlude each patch across all leads simultaneously ─────────────
        for start in tqdm(starts, desc="  Occlusion", leave=False):
            end      = min(start + self.patch_len, seq_len)
            occluded = signal.clone()
            occluded[:, :, start:end] = occlusion_value

            with torch.no_grad():
                c_log, s_log, _ = self.model(occluded)
                log  = c_log if head == "class" else s_log
                score = torch.sigmoid(log)[0, target_class_idx].item()

            drop = max(0.0, baseline - score)  # only care about drops
            score_drops[:, start:end] += drop
            counts[:, start:end]      += 1.0

        counts = np.where(counts == 0, 1, counts)
        importance = score_drops / counts

        # ── Optional per-lead occlusion ────────────────────────────────────
        if per_lead:
            lead_importance = np.zeros(n_leads, dtype=np.float32)
            for lead_idx in range(n_leads):
                occluded              = signal.clone()
                occluded[:, lead_idx, :] = occlusion_value
                with torch.no_grad():
                    c_log, s_log, _ = self.model(occluded)
                    log   = c_log if head == "class" else s_log
                    score = torch.sigmoid(log)[0, target_class_idx].item()
                lead_importance[lead_idx] = max(0.0, baseline - score)
            # Scale importance per lead by its global importance weight
            for lead_idx in range(n_leads):
                if lead_importance.max() > 0:
                    importance[lead_idx] *= (
                        lead_importance[lead_idx] / lead_importance.max()
                    )

        # Normalize per lead to [0, 1]
        mn  = importance.min(axis=-1, keepdims=True)
        mx  = importance.max(axis=-1, keepdims=True)
        rng = np.where((mx - mn) == 0, 1, mx - mn)
        return (importance - mn) / rng

    def compute_lead_sensitivity(self, signal:         torch.Tensor,
                                  target_class_idx:     int,
                                  head:                 str = "class") -> np.ndarray:
        """
        Compute sensitivity of prediction to each lead individually.
        Returns: (12,) array of score drops per lead
        """
        self.model.eval()
        signal = signal.to(self.device)

        with torch.no_grad():
            c_log, s_log, _ = self.model(signal)
            log      = c_log if head == "class" else s_log
            baseline = torch.sigmoid(log)[0, target_class_idx].item()

        drops = np.zeros(cfg.NUM_LEADS, dtype=np.float32)
        for lead_idx in range(cfg.NUM_LEADS):
            occ = signal.clone()
            occ[:, lead_idx, :] = 0.0
            with torch.no_grad():
                c_log, s_log, _ = self.model(occ)
                log   = c_log if head == "class" else s_log
                score = torch.sigmoid(log)[0, target_class_idx].item()
            drops[lead_idx] = max(0.0, baseline - score)

        return drops


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_occlusion(signal:     np.ndarray,
                   occ_map:    np.ndarray,
                   record_id,
                   class_name: str,
                   save_dir=None):
    """
    Plot occlusion sensitivity overlaid on ECG.
    signal  : (12, seq_len)
    occ_map : (12, seq_len)
    """
    fig, axes = plt.subplots(12, 1, figsize=(18, 20), sharex=True)
    fig.patch.set_facecolor("#0a0e17")
    fig.suptitle(
        f"Occlusion Sensitivity — Record #{record_id}  |  Class: {class_name}",
        color="#e2e8f0", fontsize=13, fontweight="bold", y=0.99
    )

    cmap = cm.get_cmap("YlOrRd")
    t    = np.arange(signal.shape[1])

    for i, ax in enumerate(axes):
        ax.set_facecolor("#0a0e17")
        sig = signal[i]
        occ = occ_map[i]

        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j+1],
                       color=cmap(float(occ[j])),
                       alpha=0.55, linewidth=0)

        ax.plot(t, sig, color="#f0abfc", linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], color="#94a3b8",
                      fontsize=8, rotation=0, labelpad=22)
        ax.tick_params(colors="#475569", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#1f2937")

    axes[-1].set_xlabel("Timestep (samples)", color="#94a3b8", fontsize=9)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, location="right",
                        fraction=0.012, pad=0.01)
    cbar.set_label("Score Drop (Importance)", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.subplots_adjust(hspace=0.05)

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"occlusion_{record_id}_{class_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_occlusion_lead_sensitivity(drops:      np.ndarray,
                                    record_id,
                                    class_name: str,
                                    save_dir=None):
    """
    Bar chart of per-lead score drops.
    drops: (12,) array
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#0a0e17")

    norm   = drops / (drops.max() + 1e-8)
    colors = cm.YlOrRd(norm)
    ax.barh(LEAD_NAMES[::-1], drops[::-1],
            color=colors[::-1], edgecolor="#1f2937", height=0.6)

    ax.set_xlabel("Score Drop When Lead Occluded", color="#94a3b8", fontsize=9)
    ax.set_title(f"Lead Sensitivity — {class_name}",
                 color="#e2e8f0", fontsize=11, pad=12)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"occ_leads_{record_id}_{class_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
