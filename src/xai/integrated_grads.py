"""
src/xai/integrated_grads.py

Integrated Gradients (IG) for PatchTST ECG classification.

IG is a rigorous XAI method that satisfies two fundamental axioms:
  - Sensitivity: if input differs from baseline and changes output → non-zero attribution
  - Implementation Invariance: attributions are identical for functionally equivalent networks

Reference: Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.utils.config import cfg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ── Integrated Gradients ───────────────────────────────────────────────────

class IntegratedGradients:
    """
    Integrated Gradients for PatchTST.

    Computes the integral of gradients along a straight path from
    a baseline (zero signal) to the actual input.

    IG(x)_i = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x')) / ∂x_i  dα

    Where:
        x   = actual input signal
        x'  = baseline (zeros = electrically silent ECG)
        F   = model output for target class
        α   = interpolation coefficient
        i   = timestep index

    Usage:
        ig = IntegratedGradients(model)
        attributions = ig.compute(signal, target_class_idx)
    """

    def __init__(self, model):
        self.model  = model
        self.device = next(model.parameters()).device

    def compute(self, signal:           torch.Tensor,
                target_class_idx:       int,
                head:                   str   = "class",
                n_steps:                int   = 50,
                baseline:               torch.Tensor = None) -> np.ndarray:
        """
        Compute Integrated Gradients.

        signal           : (1, 12, seq_len)
        target_class_idx : index of target class
        head             : 'class' or 'super'
        n_steps          : number of Riemann approximation steps (50 is standard)
        baseline         : (1, 12, seq_len) baseline input (default: zeros)

        Returns: (12, seq_len) attribution map (absolute values, normalized)
        """
        signal = signal.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(signal)
        else:
            baseline = baseline.to(self.device)

        # Build interpolated inputs: x' + α*(x - x') for α in [0,1]
        alphas      = torch.linspace(0, 1, n_steps, device=self.device)
        delta       = signal - baseline                  # (1, 12, seq_len)
        interp      = baseline + alphas[:, None, None, None] * delta
        # interp shape: (n_steps, 1, 12, seq_len) → merge to (n_steps, 12, seq_len)
        interp      = interp.squeeze(1)                  # (n_steps, 12, seq_len)
        interp      = interp.requires_grad_(True)

        # Forward pass on all interpolated inputs at once
        with torch.enable_grad():
            interp = interp.detach().requires_grad_(True)
            class_logits, super_logits, _ = self.model(interp)
            logits = class_logits if head == "class" else super_logits
            scores = torch.sigmoid(logits)[:, target_class_idx]  # (n_steps,)

            # Backprop to get gradients at all interpolation points
            gradients = torch.autograd.grad(scores.sum(), interp)[0]
            gradients = gradients.detach()   # (n_steps, 12, seq_len)

        # Riemann approximation of the integral (trapezoidal rule)
        avg_grads = (gradients[:-1] + gradients[1:]) / 2.0
        avg_grads = avg_grads.mean(dim=0)              # (12, seq_len)

        # Element-wise product with (input - baseline)
        delta_np      = delta[0].cpu().detach().numpy()     # (12, seq_len)
        avg_grads_np  = avg_grads.cpu().numpy()
        attributions  = delta_np * avg_grads_np             # (12, seq_len)

        # Use absolute attributions
        attributions = np.abs(attributions)

        # Normalize per lead to [0, 1]
        mn  = attributions.min(axis=-1, keepdims=True)
        mx  = attributions.max(axis=-1, keepdims=True)
        rng = np.where((mx - mn) == 0, 1, mx - mn)
        return (attributions - mn) / rng

    def completeness_check(self, signal:         torch.Tensor,
                            attributions:         np.ndarray,
                            target_class_idx:     int,
                            head:                 str = "class") -> dict:
        """
        Verify the completeness axiom:
        Sum of attributions ≈ F(input) - F(baseline)

        Returns dict with actual delta, attribution sum, and error %.
        """
        signal   = signal.to(self.device)
        baseline = torch.zeros_like(signal)

        with torch.no_grad():
            c_in, s_in, _ = self.model(signal)
            c_bl, s_bl, _ = self.model(baseline)

            logits_in = c_in if head == "class" else s_in
            logits_bl = c_bl if head == "class" else s_bl

            score_in = torch.sigmoid(logits_in)[0, target_class_idx].item()
            score_bl = torch.sigmoid(logits_bl)[0, target_class_idx].item()

        actual_delta  = score_in - score_bl
        attr_sum      = attributions.sum()
        error_pct     = abs(actual_delta - attr_sum) / (abs(actual_delta) + 1e-8) * 100

        return {
            "f_input":    score_in,
            "f_baseline": score_bl,
            "delta":      actual_delta,
            "attr_sum":   attr_sum,
            "error_%":    error_pct,
        }


# ── Plot IG ────────────────────────────────────────────────────────────────

def plot_integrated_gradients(signal:       np.ndarray,
                               attributions: np.ndarray,
                               record_id,
                               class_name:   str,
                               save_dir=None):
    """
    Plot Integrated Gradients overlaid on ECG signal.
    signal       : (12, seq_len)
    attributions : (12, seq_len)
    """
    fig, axes = plt.subplots(12, 1, figsize=(18, 20), sharex=True)
    fig.patch.set_facecolor("#0a0e17")
    fig.suptitle(
        f"Integrated Gradients — Record #{record_id}  |  Class: {class_name}",
        color="#e2e8f0", fontsize=13, fontweight="bold", y=0.99
    )

    cmap = cm.get_cmap("coolwarm")
    t    = np.arange(signal.shape[1])

    for i, ax in enumerate(axes):
        ax.set_facecolor("#0a0e17")
        sig  = signal[i]
        attr = attributions[i]

        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j+1],
                       color=cmap(float(attr[j])),
                       alpha=0.5, linewidth=0)

        ax.plot(t, sig, color="#a3e635", linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], color="#94a3b8",
                      fontsize=8, rotation=0, labelpad=22)
        ax.tick_params(colors="#475569", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#1f2937")

    axes[-1].set_xlabel("Timestep (samples)", color="#94a3b8", fontsize=9)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, location="right",
                        fraction=0.012, pad=0.01)
    cbar.set_label("IG Attribution", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.subplots_adjust(hspace=0.05)

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"ig_{record_id}_{class_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_ig_summary(attributions: np.ndarray,
                    record_id,
                    class_name: str,
                    save_dir=None):
    """
    Bar chart: mean IG attribution per lead.
    attributions: (12, seq_len)
    """
    mean_attr = attributions.mean(axis=-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#0a0e17")

    colors = cm.coolwarm(mean_attr / mean_attr.max())
    ax.barh(LEAD_NAMES[::-1], mean_attr[::-1],
            color=colors[::-1], edgecolor="#1f2937", height=0.6)

    ax.set_xlabel("Mean IG Attribution", color="#94a3b8", fontsize=9)
    ax.set_title(f"IG Lead Importance — {class_name}",
                 color="#e2e8f0", fontsize=11, pad=12)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"ig_leads_{record_id}_{class_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
