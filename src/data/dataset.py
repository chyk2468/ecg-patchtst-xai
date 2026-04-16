import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.loader import (
    load_metadata,
    load_signal,
    encode_record,
    get_record_path,
    get_split,
)
from src.data.preprocessing import preprocess_signal
from src.utils.config import cfg


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG records.

    Each item returns:
        signal      : Tensor (12, seq_len)   float32
        class_label : Tensor (71,)           float32  multi-hot SCP codes
        super_label : Tensor (7,)            float32  multi-hot superclasses
        record_id   : int                    for XAI tracing
    """

    def __init__(self, split: str = "train"):
        """
        split: 'train' | 'val' | 'test'
        """
        self.split = split

        # Load metadata once
        self.df, self.all_codes, self.code_to_super = load_metadata()
        self.df = get_split(self.df, split)
        self.records = self.df.index.tolist()

        print(f"[PTBXLDataset] {split:5s} → {len(self.records):,} records | "
              f"{len(self.all_codes)} SCP codes | {cfg.SAMPLING_RATE} Hz")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_id = self.records[idx]
        row = self.df.loc[record_id]

        # ── Signal ──────────────────────────────────────────────────────
        record_path = get_record_path(row)
        signal_np   = load_signal(record_path)                # (seq_len, 12)
        signal_np   = preprocess_signal(signal_np, cfg.SEQ_LEN)  # (12, seq_len)
        signal      = torch.from_numpy(signal_np)             # (12, 1000)

        # ── Labels ──────────────────────────────────────────────────────
        class_label, super_label = encode_record(
            row, self.all_codes, self.code_to_super
        )
        class_label = torch.from_numpy(class_label)   # (71,)
        super_label = torch.from_numpy(super_label)   # (7,)

        return signal, class_label, super_label, record_id

    @property
    def num_classes(self):
        return len(self.all_codes)

    @property
    def all_scp_codes(self):
        return self.all_codes


# ── DataLoader factory ─────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = cfg.BATCH_SIZE,
                    num_workers: int = 4) -> dict:
    """
    Returns dict with 'train', 'val', 'test' DataLoaders.
    """
    datasets = {
        split: PTBXLDataset(split)
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders, datasets["train"].all_scp_codes
