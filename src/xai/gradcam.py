"""
GradCAM for PatchTST ECG Classification.

Computes gradient-weighted activation maps to highlight which
time regions and leads are most important for a given prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.config import cfg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ── GradCAM hook handler ───────────────────────────────────────────────────

class GradCAM:
    """
    GradCAM for a single target class on PatchTST.

    Hooks into the last transformer block's output to capture
    activations and gradients, then maps back to signal timeline.

    Usage:
        gradcam = GradCAM(model)
        heatmap = gradcam.compute(signal_tensor, target_class_idx)
        gradcam.remove_hooks()
    """

    def __init__(self, model):
        self.model      = model
        self._activations = None
        self._gradients   = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        # Target the last transformer block's output
        last_block = self.model.encoder.layers[-1]

        def fwd_hook(module, input, output):
            # output is (x, attn_weights) tuple
            self._activations = output[0].detach()   # (B*C, N, d_model)

        def bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()  # (B*C, N, d_model)

        self._hooks.append(last_block.register_forward_hook(fwd_hook))
        self._hooks.append(last_block.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def compute(self, signal: torch.Tensor,
                target_class_idx: int,
                head: str = "class") -> np.ndarray:
        """
        Compute GradCAM heatmap for a single sample.

        signal           : (1, 12, seq_len)  single record
        target_class_idx : index into 71-class or 7-class output
        head             : 'class' or 'super'

        Returns: (12, seq_len)  normalized importance map
        """
        self.model.eval()
        signal = signal.requires_grad_(True)

        class_logits, super_logits, _ = self.model(signal)

        # Select the target head
        logits = class_logits if head == "class" else super_logits
        score  = logits[0, target_class_idx]

        self.model.zero_grad()
        score.backward()

        # activations: (B*C, N, d_model) → take batch 0 → (C, N, d_model)
        C   = cfg.NUM_LEADS
        act = self._activations[:C]          # (C, N, d_model)
        grd = self._gradients[:C]            # (C, N, d_model)

        # Weight activations by mean gradient (global avg pooling over d_model)
        weights = grd.mean(dim=-1, keepdim=True)   # (C, N, 1)
        cam     = (weights * act).sum(dim=-1)       # (C, N)
        cam     = F.relu(torch.from_numpy(
            cam.cpu().numpy()
        )).numpy()                                   # (C, N)

        # Upsample from N patches → seq_len timesteps
        heatmap = self._patches_to_time(cam,
                                        cfg.SEQ_LEN,
                                        cfg.PATCH_LEN,
                                        cfg.STRIDE)   # (C, seq_len)

        # Normalize to [0, 1] per lead
        mn  = heatmap.min(axis=-1, keepdims=True)
        mx  = heatmap.max(axis=-1, keepdims=True)
        rng = np.where((mx - mn) == 0, 1, mx - mn)
        return (heatmap - mn) / rng

    @staticmethod
    def _patches_to_time(cam: np.ndarray, seq_len: int,
                         patch_len: int, stride: int) -> np.ndarray:
        """Average patch scores back to time axis. cam: (C, N)"""
        C, N = cam.shape
        result = np.zeros((C, seq_len), dtype=np.float32)
        counts = np.zeros((C, seq_len), dtype=np.float32)
        for p in range(N):
            start = p * stride
            end   = min(start + patch_len, seq_len)
            result[:, start:end] += cam[:, p:p+1]
            counts[:, start:end] += 1.0
        counts = np.where(counts == 0, 1, counts)
        return result / counts


# ── Plot GradCAM ───────────────────────────────────────────────────────────

def plot_gradcam(signal:     np.ndarray,
                 heatmap:    np.ndarray,
                 record_id:  int,
                 class_name: str,
                 save_dir=None):
    """
    Overlay GradCAM heatmap on ECG signal.
    signal  : (12, seq_len)
    heatmap : (12, seq_len)
    """
    fig, axes = plt.subplots(12, 1, figsize=(18, 20), sharex=True)
    fig.suptitle(
        f"GradCAM — Record #{record_id}  |  Target: {class_name}",
        fontsize=13, fontweight="bold"
    )

    import matplotlib.cm as cm
    cmap = cm.get_cmap("hot")
    t    = np.arange(signal.shape[1])

    for i, ax in enumerate(axes):
        sig = signal[i]
        imp = heatmap[i]

        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j+1],
                       color=cmap(float(imp[j])),
                       alpha=0.5, linewidth=0)

        ax.plot(t, sig, color="navy", linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], fontsize=8, rotation=0, labelpad=20)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Timestep (samples)", fontsize=9)

    sm   = plt.cm.ScalarMappable(cmap=cmap,
                                  norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=axes, location="right",
                        fraction=0.015, pad=0.01)
    cbar.set_label("GradCAM Score", fontsize=9)
    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        path = Path(save_dir) / f"gradcam_{record_id}_{class_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
