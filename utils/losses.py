"""Loss functions for pain intensity recognition.

Supported losses:
    - CrossEntropyLoss (standard, with optional class weights)
    - CornLoss (ordinal regression, respects ordered pain levels)
    - FocalLoss (focuses on hard examples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CornLoss(nn.Module):
    """Conditional Ordinal Regression for Neural networks (Corn) Loss.

    Converts a K-class ordinal classification problem into K-1 binary
    classification tasks: "is label > k?" for k = 0, 1, ..., K-2.

    This respects the natural ordering of pain levels (0 < 1 < 2 < 3 < 4),
    penalizing larger ordinal errors more heavily than adjacent-class confusion.
    """

    def __init__(self, num_classes, class_weights=None):
        """
        Args:
            num_classes: number of ordinal classes (e.g., 5 for pain levels 0-4)
            class_weights: optional tensor of shape (num_classes,) with per-class
                           weights (applied per-task via cumulative distribution)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # K-1 binary tasks
        if class_weights is not None:
            # Convert per-class weights to per-task weights
            # Task k asks "label > k?", so its weight is based on the cumulative
            # proportion of classes > k vs <= k
            task_weights = []
            for k in range(self.num_tasks):
                w_gt = class_weights[k+1:].mean()   # classes > k
                w_le = class_weights[:k+1].mean()    # classes <= k
                task_weights.append((w_gt + w_le) / 2)
            self.task_weights = torch.stack(task_weights)
        else:
            self.task_weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K-1) — raw logits for each binary task
            targets: (B,) — integer class labels in [0, K-1]

        Returns:
            scalar loss
        """
        # Build binary targets: (B, K-1)
        # target[k] = 1 if true_label > k else 0
        binary_targets = torch.stack(
            [(targets > k).float() for k in range(self.num_tasks)], dim=1
        ).to(logits.device)

        # BCEWithLogitsLoss per task
        losses = []
        for k in range(self.num_tasks):
            loss_k = F.binary_cross_entropy_with_logits(
                logits[:, k], binary_targets[:, k], reduction="none"
            )
            if self.task_weights is not None:
                loss_k = loss_k * self.task_weights[k].to(logits.device)
            losses.append(loss_k)

        return torch.stack(losses, dim=1).sum(dim=1).mean()


def corn_logits_to_probs(logits):
    """Convert Corn K-1 logits to K-class probability distribution.

    Uses the cumulative chain rule:
        P(y=0)   = 1 - σ(z_0)
        P(y=k)   = σ(z_{k-1}) - σ(z_k)   for 0 < k < K-1
        P(y=K-1) = σ(z_{K-2})
    """
    probs_k_minus_1 = torch.sigmoid(logits)  # (B, K-1)
    # Build K-class probabilities
    probs = []
    K_minus_1 = logits.shape[1]
    for k in range(K_minus_1 + 1):
        if k == 0:
            p = 1 - probs_k_minus_1[:, 0]
        elif k == K_minus_1:
            p = probs_k_minus_1[:, k - 1]
        else:
            p = probs_k_minus_1[:, k - 1] - probs_k_minus_1[:, k]
        probs.append(p)
    return torch.stack(probs, dim=1)  # (B, K)


def corn_logits_to_preds(logits):
    """Convert Corn K-1 logits to class predictions via argmax of derived probs."""
    probs = corn_logits_to_probs(logits)
    return probs.argmax(dim=1)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples, forcing the model to focus on hard cases
    (e.g., adjacent pain level confusion).
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        """
        Args:
            alpha: optional (num_classes,) class weights tensor
            gamma: focusing parameter (higher = more focus on hard examples)
            label_smoothing: label smoothing factor (0 = no smoothing)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K) — raw logits
            targets: (B,) — integer class labels

        Returns:
            scalar loss
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)  # predicted probability of true class
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def build_loss(config, class_weights=None):
    """Build loss function based on config.

    Args:
        config: Config object
        class_weights: optional (num_classes,) tensor for CE/Focal loss weighting

    Returns:
        nn.Module loss function, and a bool `corn_mode` flag
    """
    smoothing = getattr(config, "label_smoothing", 0.0)

    if config.loss_type == "corn":
        return CornLoss(config.num_classes, class_weights), True
    elif config.loss_type == "focal":
        alpha = class_weights if config.focal_alpha is None else config.focal_alpha
        return FocalLoss(alpha=alpha, gamma=config.focal_gamma, label_smoothing=smoothing), False
    else:  # "ce" or default
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing), False
