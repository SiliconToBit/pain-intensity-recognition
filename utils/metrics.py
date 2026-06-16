"""Evaluation metrics for pain intensity recognition.

Metrics include:
    - Weighted / Macro F1-score
    - Per-class recall and AUROC
    - Confusion matrix
    - Cohen's Kappa
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(all_labels, all_preds, all_probs, num_classes=5):
    """Compute comprehensive evaluation metrics.

    Returns:
        dict with all metrics
    """
    metrics = {}

    # Weighted F1-score
    metrics["weighted_f1"] = f1_score(all_labels, all_preds, average="weighted")

    # Macro F1-score
    metrics["macro_f1"] = f1_score(all_labels, all_preds, average="macro")

    # Per-class recall
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    metrics["per_class_recall"] = per_class_recall.tolist()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    # Cohen's Kappa
    metrics["cohens_kappa"] = cohen_kappa_score(all_labels, all_preds)

    # Multi-class AUROC
    if num_classes == 2:
        # Binary: use positive class probability directly
        try:
            metrics["auroc_weighted"] = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            metrics["auroc_weighted"] = 0.0
    else:
        labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        try:
            metrics["auroc_weighted"] = roc_auc_score(
                labels_bin, all_probs, multi_class="ovr", average="weighted"
            )
        except ValueError:
            metrics["auroc_weighted"] = 0.0

    # Per-class AUROC
    per_class_auc = []
    if num_classes == 2:
        # Binary: one AUC score for the positive class
        try:
            per_class_auc.append(roc_auc_score(1 - all_labels, all_probs[:, 0]))  # Class 0 AUC
            per_class_auc.append(roc_auc_score(all_labels, all_probs[:, 1]))       # Class 1 AUC
        except ValueError:
            per_class_auc = [0.0, 0.0]
    else:
        labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        for i in range(num_classes):
            try:
                auc = roc_auc_score(labels_bin[:, i], all_probs[:, i])
                per_class_auc.append(auc)
            except ValueError:
                per_class_auc.append(0.0)
    metrics["per_class_auc"] = per_class_auc

    return metrics


def print_metrics(metrics, num_classes=5):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print("Comprehensive Evaluation Metrics")
    print(f"{'='*60}")

    print(f"\n  Weighted F1-score:  {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1-score:     {metrics['macro_f1']:.4f}")
    print(f"  Cohen's Kappa:      {metrics['cohens_kappa']:.4f}")
    print(f"  AUROC (weighted):   {metrics['auroc_weighted']:.4f}")

    print(f"\n  Per-class Recall:")
    for i, r in enumerate(metrics["per_class_recall"]):
        print(f"    Class {i}: {r:.4f}")

    print(f"\n  Per-class AUC:")
    for i, a in enumerate(metrics["per_class_auc"]):
        print(f"    Class {i}: {a:.4f}")

    cm = np.array(metrics["confusion_matrix"])
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    # Print header
    header = "        " + "  ".join(f"{i:>5}" for i in range(num_classes))
    print(header)
    for i in range(num_classes):
        row = "  ".join(f"{cm[i, j]:>5}" for j in range(num_classes))
        print(f"    {i}   {row}")

    print(f"\n{'='*60}")
