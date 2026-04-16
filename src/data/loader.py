from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.utils.config import cfg
from src.utils.label_utils import (
    build_code_to_superclass,
    encode_scp_labels,
    derive_superclass_labels,
    get_all_scp_codes,
    load_scp_statements,
)


# ── Signal loading ─────────────────────────────────────────────────────────

def load_signal(record_path: str) -> np.ndarray:
    """
    Load a single ECG record using wfdb.
    Returns ndarray of shape (seq_len, num_leads), dtype float32.
    record_path: path WITHOUT extension, e.g. 'records100/00000/00001_lr'
    """
    full_path = cfg.DATA_ROOT / record_path
    record = wfdb.rdrecord(str(full_path))
    signal = record.p_signal.astype(np.float32)   # (seq_len, 12)
    return signal


# ── Metadata loading ───────────────────────────────────────────────────────

def load_metadata() -> tuple[pd.DataFrame, list, dict]:
    """
    Load ptbxl_database.csv and scp_statements.csv.

    Returns
    -------
    df              : full metadata dataframe
    all_codes       : sorted list of 71 SCP codes
    code_to_super   : {scp_code: superclass} mapping dict
    """
    df = pd.read_csv(cfg.CSV_PATH, index_col="ecg_id")

    scp_df = load_scp_statements()
    all_codes       = get_all_scp_codes(df)
    code_to_super   = build_code_to_superclass(scp_df)

    return df, all_codes, code_to_super


# ── Per-record label encoding ──────────────────────────────────────────────

def encode_record(row: pd.Series,
                  all_codes: list,
                  code_to_super: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a metadata row, return:
        class_label  : (71,)  multi-hot SCP codes
        super_label  : (7,)   multi-hot superclasses
    """
    scp_str = str(row["scp_codes"])
    class_label = encode_scp_labels(scp_str, all_codes)
    super_label = derive_superclass_labels(class_label, all_codes, code_to_super)
    return class_label, super_label


# ── Split helpers ──────────────────────────────────────────────────────────

def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Returns rows for a given split using PTB-XL's strat_fold column.
      split: 'train' | 'val' | 'test'
    """
    if split == "train":
        return df[df["strat_fold"].isin(cfg.TRAIN_FOLDS)]
    elif split == "val":
        return df[df["strat_fold"] == cfg.VAL_FOLD]
    elif split == "test":
        return df[df["strat_fold"] == cfg.TEST_FOLD]
    else:
        raise ValueError(f"Unknown split: {split}")


# ── Record path resolver ───────────────────────────────────────────────────

def get_record_path(row: pd.Series) -> str:
    """Return the correct filename column based on configured sampling rate."""
    if cfg.SAMPLING_RATE == 100:
        return row["filename_lr"]
    return row["filename_hr"]
