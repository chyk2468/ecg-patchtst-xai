import ast
import numpy as np
import pandas as pd

from src.utils.config import cfg


# ── Load & build mappings ──────────────────────────────────────────────────

def load_scp_statements() -> pd.DataFrame:
    """Load scp_statements.csv indexed by SCP code."""
    return pd.read_csv(cfg.SCP_PATH, index_col=0)


def build_code_to_superclass(scp_df: pd.DataFrame) -> dict:
    """
    Returns {scp_code: superclass_string} for every code in scp_statements.
    Priority: diagnostic_class > rhythm > form
    """
    mapping = {}
    for code, row in scp_df.iterrows():
        dc = row.get("diagnostic_class", None)
        if pd.notna(dc) and str(dc).strip():
            mapping[code] = str(dc).strip()
        elif row.get("rhythm", 0) == 1:
            mapping[code] = "RHYTHM"
        elif row.get("form", 0) == 1:
            mapping[code] = "FORM"
    return mapping


def get_all_scp_codes(df: pd.DataFrame) -> list:
    codes = set()
    for s in df["scp_codes"]:
        if isinstance(s, dict):
            codes.update(s.keys())
        elif isinstance(s, str):
            codes.update(ast.literal_eval(s).keys())
    return sorted(codes)


# ── Encoding helpers ───────────────────────────────────────────────────────

def encode_scp_labels(scp_codes_str, all_codes: list,
                      threshold: float = cfg.LABEL_THRESHOLD) -> np.ndarray:
    if isinstance(scp_codes_str, str):
        codes_dict = ast.literal_eval(scp_codes_str)
    else:
        codes_dict = scp_codes_str
    label = np.zeros(len(all_codes), dtype=np.float32)
    for i, code in enumerate(all_codes):
        if code in codes_dict and codes_dict[code] >= threshold:
            label[i] = 1.0
    return label


def derive_superclass_labels(class_label: np.ndarray,
                              all_codes: list,
                              code_to_superclass: dict) -> np.ndarray:
    """
    Map 71-class multi-hot → 7-superclass multi-hot.
    A superclass fires if ANY of its child SCP codes fires.
    """
    super_label = np.zeros(len(cfg.SUPERCLASSES), dtype=np.float32)
    for i, code in enumerate(all_codes):
        if class_label[i] == 1.0:
            sc = code_to_superclass.get(code)
            if sc in cfg.SUPERCLASSES:
                super_label[cfg.SUPERCLASSES.index(sc)] = 1.0
    return super_label


# ── Pretty print helpers ───────────────────────────────────────────────────

def decode_scp_labels(label_vec: np.ndarray, all_codes: list) -> list:
    """Return list of active SCP code strings from a multi-hot vector."""
    return [all_codes[i] for i, v in enumerate(label_vec) if v == 1.0]


def decode_superclass_labels(super_vec: np.ndarray) -> list:
    """Return list of active superclass strings from a multi-hot vector."""
    return [cfg.SUPERCLASSES[i] for i, v in enumerate(super_vec) if v == 1.0]
