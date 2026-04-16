"""
src/xai/shap_explainer.py

SHAP (SHapley Additive exPlanations) for PatchTST ECG classification.
Uses GradientExplainer which works well with deep learning models.
"""

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.utils.config import cfg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ── SHAP Wrapper ───────────────────────────────────────────────────────────

class ECGShapExplainer:
    """
    SHAP GradientExplainer for PatchTST.

    Computes SHAP values for each timestep × lead combination,
    showing the contribution of each input feature to the prediction.

    Usage:
        explainer = ECGShapExplainer(model, background_data)
        shap_vals = explainer.compute(signal_tensor, target_class_idx)
    """

    def __init__(self, model, background: torch.Tensor):
        """
        model      : trained PatchTST
        background : (N, 12, seq_len) background samples for SHAP baseline
                     Use 50-100 random training samples
        """
        self.model      = model
        self.device     = next(model.parameters()).device
        self.background = background.to(self.device)
        self._explainer = None

    def _build_explainer(self, target_head: str = "class"):
        """
        Build SHAP GradientExplainer with a wrapper that returns
        only the target head's output.
        """
        model = self.model

        class HeadWrapper(torch.nn.Module):
            def __init__(self, head):
                super().__init__()
                self.head = head  # "class" or "super"

            def forward(self, x):
                c_logits, s_logits, _ = model(x)
                return torch.sigmoid(c_logits) if self.head == "class" \
                       else torch.sigmoid(s_logits)

        wrapper = HeadWrapper(target_head)
        wrapper.eval()
        self._explainer = shap.GradientExplainer(wrapper, self.background)

    def compute(self, signal: torch.Tensor,
                target_class_idx: int,
                head: str = "class",
                absolute: bool = True) -> np.ndarray:
        """
        Compute SHAP values for one record.

        Returns: (12, seq_len) SHAP importance map
        """
        if self._explainer is None:
            self._build_explainer(head)
            self._explainer_head = head
        elif self._explainer_head != head:
            self._build_explainer(head)
            self._explainer_head = head

        signal = signal.to(self.device)
        shap_values = self._explainer.shap_values(signal)

        if isinstance(shap_values, list):
            vals = shap_values[target_class_idx][0]   # (12, seq_len)
        else:
            vals = shap_values[0, :, :, target_class_idx]  # (12, seq_len)

        if absolute:
            vals = np.abs(vals)
            # Normalize to [0, 1]
            mx = np.abs(vals).max() + 1e-9
            return vals / mx
        else:
            # Signed values, normalized to [-1, 1]
            mx = np.abs(vals).max() + 1e-9
            return vals / mx


# ── Creative SHAP Plots ───────────────────────────────────────────────────

def make_shap_radial_plot(shap_map: np.ndarray, title: str):
    """
    Creative Radial (Radar) plot for SHAP Lead Importance.
    shap_map: (12, seq_len)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate mean importance per lead
    importance = np.mean(np.abs(shap_map), axis=-1)
    
    # Radar chart setup
    num_leads = len(LEAD_NAMES)
    angles    = np.linspace(0, 2 * np.pi, num_leads, endpoint=False).tolist()
    angles   += angles[:1] # Close the circle
    
    values  = importance.tolist()
    values += values[:1] # Close the circle
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#050505")
    ax.set_facecolor((0.0, 0.898, 1.0, 0.05))
    
    # Draw the outline and fill
    ax.plot(angles, values, color="#00e5ff", linewidth=2, linestyle='solid')
    ax.fill(angles, values, color="#00e5ff", alpha=0.25)
    
    # Fix axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LEAD_NAMES, color="#00e5ff", fontfamily='Orbitron', fontsize=9)
    
    # Style the grid
    ax.yaxis.grid(True, color="white", alpha=0.1, linestyle='--')
    ax.xaxis.grid(True, color="white", alpha=0.1)
    ax.set_yticklabels([]) # Hide radial labels for cleaner look
    
    # Title
    ax.set_title(title, color="#f0f0f0", fontfamily='Orbitron', fontsize=11, pad=20)
    
    # Add a glowing center
    ax.plot(0, 0, marker='o', markersize=10, color="#ff0055", label="Target")
    
    plt.tight_layout()
    return fig


# ── Build background dataset ───────────────────────────────────────────────

def build_background(dataloader, n_samples: int = 100) -> torch.Tensor:
    """
    Collect n_samples signals from a dataloader to use as SHAP background.
    Returns: (n_samples, 12, seq_len)
    """
    samples = []
    for signals, _, _, _ in dataloader:
        samples.append(signals)
        if sum(s.size(0) for s in samples) >= n_samples:
            break
    bg = torch.cat(samples, dim=0)[:n_samples]
    return bg


# ── Plot SHAP ──────────────────────────────────────────────────────────────

def plot_shap(signal:     np.ndarray,
              shap_map:   np.ndarray,
              record_id,
              class_name: str,
              save_dir=None):
    """
    Plot SHAP values overlaid on ECG.
    signal   : (12, seq_len)
    shap_map : (12, seq_len)
    """
    fig, axes = plt.subplots(12, 1, figsize=(18, 20), sharex=True)
    fig.patch.set_facecolor("#0a0e17")
    fig.suptitle(
        f"SHAP Explanation — Record #{record_id}  |  Class: {class_name}",
        color="#e2e8f0", fontsize=13, fontweight="bold", y=0.99
    )

    cmap = cm.get_cmap("RdYlGn")
    t    = np.arange(signal.shape[1])

    for i, ax in enumerate(axes):
        ax.set_facecolor("#0a0e17")
        sig = signal[i]
        sv  = shap_map[i]

        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j+1],
                       color=cmap(float(sv[j])),
                       alpha=0.5, linewidth=0)

        ax.plot(t, sig, color="#00e5ff", linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], color="#94a3b8",
                      fontsize=8, rotation=0, labelpad=22)
        ax.tick_params(colors="#475569", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#1f2937")

    axes[-1].set_xlabel("Timestep (samples)", color="#94a3b8", fontsize=9)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, location="right",
                        fraction=0.012, pad=0.01)
    cbar.set_label("SHAP |Value|", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.subplots_adjust(hspace=0.05)

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"shap_{record_id}_{class_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_shap_summary(shap_map: np.ndarray,
                      record_id,
                      class_name: str,
                      save_dir=None):
    """
    Bar chart: mean |SHAP| per lead — shows which lead matters most.
    shap_map: (12, seq_len)
    """
    mean_shap = shap_map.mean(axis=-1)   # (12,)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#0a0e17")

    colors = cm.RdYlGn(mean_shap / mean_shap.max())
    ax.barh(LEAD_NAMES[::-1], mean_shap[::-1],
            color=colors[::-1], edgecolor="#1f2937", height=0.6)

    ax.set_xlabel("Mean |SHAP Value|", color="#94a3b8", fontsize=9)
    ax.set_title(f"SHAP Lead Importance — {class_name}",
                 color="#e2e8f0", fontsize=11, pad=12)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"shap_leads_{record_id}_{class_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
