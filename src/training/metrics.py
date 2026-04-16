import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    classification_report,
)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray,
                average: str = "macro") -> float:
    """
    Macro-averaged ROC-AUC for multi-label classification.
    Skips classes with only one label present (undefined AUC).
    """
    try:
        return roc_auc_score(y_true, y_score, average=average,
                             multi_class="ovr")
    except ValueError:
        # Some classes may have no positive samples in a batch
        valid = y_true.sum(axis=0) > 0
        return roc_auc_score(y_true[:, valid], y_score[:, valid],
                             average=average)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray,
               threshold: float = 0.5, average: str = "macro") -> float:
    """Macro F1 for multi-label classification after threshold binarization."""
    y_pred_bin = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_bin, average=average, zero_division=0)


def compute_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Average Precision (mAP) across all classes."""
    try:
        return average_precision_score(y_true, y_score, average="macro")
    except ValueError:
        return 0.0


def compute_all_metrics(y_true: np.ndarray,
                        y_score: np.ndarray,
                        threshold: float = 0.5) -> dict:
    """
    Compute AUC, F1, mAP in one call.
    y_true  : (N, C)  binary ground truth
    y_score : (N, C)  sigmoid probabilities
    """
    return {
        "auc":  compute_auc(y_true, y_score),
        "f1":   compute_f1(y_true, y_score, threshold),
        "map":  compute_map(y_true, y_score),
    }


def per_class_report(y_true: np.ndarray, y_score: np.ndarray,
                     class_names: list, threshold: float = 0.5) -> str:
    """Sklearn classification report for each class."""
    y_pred = (y_score >= threshold).astype(int)
    return classification_report(y_true, y_pred,
                                 target_names=class_names,
                                 zero_division=0)
