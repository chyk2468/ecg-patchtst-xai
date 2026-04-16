import math
import torch
import torch.nn as nn
from einops import rearrange

from src.utils.config import cfg


# ── Patch Embedding ────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits each lead's time series into overlapping patches and projects
    each patch to d_model dimensions.

    Channel-Independent design: each lead is processed independently,
    then features are aggregated before the classification head.

    Input  : (B, C, L)  where C=12 leads, L=seq_len
    Output : (B*C, N, d_model)  where N = number of patches
    """

    def __init__(self, patch_len: int, stride: int, d_model: int,
                 seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.projection = nn.Linear(patch_len, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """x: (B, C, L)"""
        B, C, L = x.shape
        # Unfold into patches: (B, C, N, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Merge batch and channel dims for shared projection
        x = rearrange(x, 'b c n p -> (b c) n p')
        x = self.projection(x)            # (B*C, N, d_model)
        return self.dropout(x), B, C


# ── Positional Encoding ────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(num_patches, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, N, d_model)"""
        N = x.size(1)
        pos = torch.arange(N, device=x.device)
        return x + self.pe(pos)


# ── Transformer Encoder Block ──────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn     = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout,
                                              batch_first=True)
        self.ff       = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # Self-attention + residual
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward + residual
        x = self.norm2(x + self.ff(x))
        return x, attn_weights


# ── PatchTST Encoder ───────────────────────────────────────────────────────

class PatchTSTEncoder(nn.Module):
    """
    Stack of transformer blocks applied to patch sequences.
    Also stores the last layer's attention weights for XAI.
    """

    def __init__(self, patch_len: int, stride: int, seq_len: int,
                 d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            patch_len, stride, d_model, seq_len, dropout
        )
        self.pos_enc = LearnablePositionalEncoding(
            self.patch_embed.num_patches, d_model
        )
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Store last attention for XAI
        self._last_attn_weights = None

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, L)
        Returns
          out           : (B, C, d_model)   mean-pooled patch representations
          attn_weights  : (B*C, N, N)        last layer attention map
        """
        x, B, C = self.patch_embed(x)    # (B*C, N, d_model)
        x = self.pos_enc(x)

        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x)   # keep last layer's weights

        x = self.norm(x)
        self._last_attn_weights = attn_weights  # store for XAI

        # Mean pool over patches → (B*C, d_model)
        x = x.mean(dim=1)
        # Restore batch dimension → (B, C, d_model)
        x = rearrange(x, '(b c) d -> b c d', b=B, c=C)
        return x, attn_weights

    @property
    def last_attn_weights(self):
        return self._last_attn_weights


from src.models.heads import ClassificationHead

# ── Full PatchTST Model ────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    PatchTST for multi-label ECG classification.

    Outputs:
        class_logits : (B, 71)   raw logits for SCP codes
        super_logits : (B, 7)    raw logits for superclasses (auxiliary head)
        attn_weights : (B*C, N, N)
    """

    def __init__(self,
                 num_classes:      int = cfg.NUM_CLASSES,
                 num_superclasses: int = cfg.NUM_SUPERCLASSES,
                 num_leads:        int = cfg.NUM_LEADS,
                 seq_len:          int = cfg.SEQ_LEN,
                 patch_len:        int = cfg.PATCH_LEN,
                 stride:           int = cfg.STRIDE,
                 d_model:          int = cfg.D_MODEL,
                 n_heads:          int = cfg.N_HEADS,
                 n_layers:         int = cfg.N_LAYERS,
                 d_ff:             int = cfg.D_FF,
                 dropout:          float = cfg.DROPOUT):
        super().__init__()

        self.encoder = PatchTSTEncoder(
            patch_len, stride, seq_len,
            d_model, n_heads, n_layers, d_ff, dropout
        )

        self.head = ClassificationHead(
            num_leads=num_leads,
            d_model=d_model,
            num_classes=num_classes,
            num_superclasses=num_superclasses,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor):
        """x: (B, 12, seq_len)"""
        features, attn_weights = self.encoder(x)   # (B, 12, d_model)
        class_logits, super_logits = self.head(features)

        return class_logits, super_logits, attn_weights

    def get_attention_maps(self):
        """Return last forward pass attention weights for XAI."""
        return self.encoder.last_attn_weights

    @staticmethod
    def num_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
