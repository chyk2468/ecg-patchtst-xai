import numpy as np


# ── Normalization ──────────────────────────────────────────────────────────

def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Per-lead Z-score normalization.
    signal: (seq_len, num_leads)
    Returns normalized signal of same shape.
    """
    mean = signal.mean(axis=0, keepdims=True)   # (1, 12)
    std  = signal.std(axis=0,  keepdims=True)   # (1, 12)
    std  = np.where(std == 0, 1.0, std)         # avoid div by zero
    return (signal - mean) / std


def minmax_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Per-lead min-max normalization → [0, 1].
    signal: (seq_len, num_leads)
    """
    mn  = signal.min(axis=0, keepdims=True)
    mx  = signal.max(axis=0, keepdims=True)
    rng = np.where((mx - mn) == 0, 1.0, mx - mn)
    return (signal - mn) / rng


# ── Clipping / artefact removal ────────────────────────────────────────────

def clip_outliers(signal: np.ndarray, n_std: float = 5.0) -> np.ndarray:
    """
    Clip per-lead values beyond ±n_std standard deviations.
    Helps remove baseline wander spikes.
    """
    mean = signal.mean(axis=0)
    std  = signal.std(axis=0)
    low  = mean - n_std * std
    high = mean + n_std * std
    return np.clip(signal, low, high)


# ── Length enforcement ─────────────────────────────────────────────────────

def pad_or_crop(signal: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ensure signal has exactly target_len timesteps.
    Pads with zeros (end) or crops (center) as needed.
    signal: (seq_len, num_leads)
    """
    current_len = signal.shape[0]
    if current_len == target_len:
        return signal
    elif current_len > target_len:
        start = (current_len - target_len) // 2
        return signal[start: start + target_len]
    else:
        pad = np.zeros((target_len - current_len, signal.shape[1]), dtype=signal.dtype)
        return np.concatenate([signal, pad], axis=0)


# ── Full pipeline ──────────────────────────────────────────────────────────

def preprocess_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
    """
    Full preprocessing pipeline:
        1. Clip outliers
        2. Z-score normalize per lead
        3. Pad / crop to target_len
        4. Transpose to (num_leads, seq_len)  ← model expects channel-first

    Returns: (num_leads, target_len)  float32
    """
    signal = clip_outliers(signal)
    signal = z_score_normalize(signal)
    signal = pad_or_crop(signal, target_len)
    return signal.T.astype(np.float32)         # (12, 1000)
