"""Evaluation helpers: metrics that matter under heavy class imbalance."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from .config import CLASS_CODES


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def compute_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    preds = probs.argmax(1)
    mel = CLASS_CODES.index("mel")
    mel_mask = targets == mel
    return {
        "accuracy": float((preds == targets).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        # Missing a melanoma is the costly error, so track its recall explicitly.
        "melanoma_recall": float((preds[mel_mask] == mel).mean()) if mel_mask.any() else float("nan"),
        "confusion_matrix": confusion_matrix(targets, preds, labels=range(len(CLASS_CODES))).tolist(),
        "report": classification_report(
            targets, preds, labels=range(len(CLASS_CODES)), target_names=CLASS_CODES,
            zero_division=0, output_dict=True,
        ),
    }
