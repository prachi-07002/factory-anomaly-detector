"""Evaluation helpers: precision / recall / F1 and per-fault recall."""

import numpy as np


def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(y_true, y_pred):
    c = confusion(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, **c}


def per_fault_recall(y_true, y_pred, fault_labels):
    """Recall for each fault type (fraction of that fault's rows flagged)."""
    out = {}
    for fault in sorted(set(fault_labels)):
        if fault == "":
            continue
        mask = (fault_labels == fault) & (y_true == 1)
        if mask.sum():
            out[fault] = float(y_pred[mask].mean())
    return out
