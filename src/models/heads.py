import torch
import torch.nn as nn
from src.utils.config import cfg

class ClassificationHead(nn.Module):
    """
    Aggregation and classification heads for the PatchTST model.
    Maps encoded lead-wise features to SCP class and superclass logits.
    """
    def __init__(self, 
                 num_leads: int = cfg.NUM_LEADS,
                 d_model: int = cfg.D_MODEL,
                 num_classes: int = cfg.NUM_CLASSES,
                 num_superclasses: int = cfg.NUM_SUPERCLASSES,
                 dropout: float = cfg.DROPOUT):
        super().__init__()
        
        # Aggregate across leads → single feature vector
        feature_dim = num_leads * d_model

        self.aggregator = nn.Sequential(
            nn.Flatten(start_dim=1),          # (B, 12*d_model)
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Primary head: 71 SCP classes
        self.class_head = nn.Linear(d_model * 2, num_classes)

        # Auxiliary head: 7 superclasses
        self.super_head = nn.Linear(d_model * 2, num_superclasses)

    def forward(self, features: torch.Tensor):
        """
        features: (B, num_leads, d_model) from encoder
        """
        x = self.aggregator(features)        # (B, d_model*2)
        class_logits = self.class_head(x)    # (B, 71)
        super_logits = self.super_head(x)    # (B, 7)
        
        return class_logits, super_logits
