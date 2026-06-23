"""Loss functions for pain intensity recognition.

Supported losses:
    - CrossEntropyLoss (standard, with optional class weights)
    - CornLoss (ordinal regression, conditional formulation)
    - CoralLoss (rank-consistent ordinal regression, CORAL framework)
    - FocalLoss (focuses on hard examples)
    - WeightedOrdinalCrossEntropy (CE + class weights + ordinal distance penalty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Shared helpers for K-1 ordinal outputs ─────────────────────────────────

def _build_binary_targets(targets, num_tasks, device):
    """Convert class labels to K-1 binary targets: target[k] = 1 if y > k else 0."""
    return torch.stack(
        [(targets > k).float() for k in range(num_tasks)], dim=1
    ).to(device)


def _k_minus_1_to_probs(logits):
    """Convert K-1 sigmoid outputs to K-class probabilities via chain rule.

        P(y=0)   = 1 - σ(z_0)
        P(y=k)   = σ(z_{k-1}) - σ(z_k)   for 0 < k < K-1
        P(y=K-1) = σ(z_{K-2})
    """
    s = torch.sigmoid(logits)  # (B, K-1)
    K_minus_1 = logits.shape[1]
    probs = []
    for k in range(K_minus_1 + 1):
        if k == 0:
            p = 1 - s[:, 0]
        elif k == K_minus_1:
            p = s[:, k - 1]
        else:
            p = s[:, k - 1] - s[:, k]
        probs.append(p)
    return torch.stack(probs, dim=1)  # (B, K)


def _k_minus_1_to_preds(logits):
    """Convert K-1 logits to class predictions."""
    return _k_minus_1_to_probs(logits).argmax(dim=1)


# ─── Corn Loss ──────────────────────────────────────────────────────────────

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
        binary_targets = _build_binary_targets(targets, self.num_tasks, logits.device)

        losses = []
        for k in range(self.num_tasks):
            loss_k = F.binary_cross_entropy_with_logits(
                logits[:, k], binary_targets[:, k], reduction="none"
            )
            if self.task_weights is not None:
                loss_k = loss_k * self.task_weights[k].to(logits.device)
            losses.append(loss_k)

        return torch.stack(losses, dim=1).sum(dim=1).mean()


# ─── Coral Loss ─────────────────────────────────────────────────────────────

class CoralLoss(nn.Module):
    """COnsistent RAnk Logits (CORAL) ordinal regression loss.

    Based on: "Rank-consistent Ordinal Regression for Neural Networks"
    (Cao et al., 2019).

    Like CornLoss, converts K-class ordinal classification into K-1 binary
    tasks: "is label > k?" for k = 0, ..., K-2.  Additionally applies a
    rank-consistency penalty that discourages non-monotonic predicted
    probabilities (σ(z_0) ≥ σ(z_1) ≥ ... ≥ σ(z_{K-2}) should hold).

    Output: K-1 logits → same model head as CornLoss.
    """

    def __init__(self, num_classes, class_weights=None, consistency_weight=0.05):
        """
        Args:
            num_classes: number of ordinal classes (e.g., 5 for pain 0-4)
            class_weights: optional (num_classes,) per-class weights
            consistency_weight: λ for the rank-consistency penalty (0 = off,
                                higher = stronger enforcement of σ(z_k) ≥ σ(z_{k+1}))
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1
        self.consistency_weight = consistency_weight

        if class_weights is not None:
            task_weights = []
            for k in range(self.num_tasks):
                w_gt = class_weights[k + 1:].mean()
                w_le = class_weights[:k + 1].mean()
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
            scalar loss = BCE + λ * consistency_penalty
        """
        binary_targets = _build_binary_targets(targets, self.num_tasks, logits.device)

        # BCE loss per task
        losses = []
        for k in range(self.num_tasks):
            loss_k = F.binary_cross_entropy_with_logits(
                logits[:, k], binary_targets[:, k], reduction="none"
            )
            if self.task_weights is not None:
                loss_k = loss_k * self.task_weights[k].to(logits.device)
            losses.append(loss_k)
        bce = torch.stack(losses, dim=1).sum(dim=1).mean()

        # Rank-consistency penalty: σ(z_k) should be ≥ σ(z_{k+1})
        # Penalize violations where σ(z_k) < σ(z_{k+1})
        if self.consistency_weight > 0 and self.num_tasks > 1:
            s = torch.sigmoid(logits)  # (B, K-1)
            violations = F.relu(s[:, 1:] - s[:, :-1])  # (B, K-2), >0 when decreasing
            consistency = violations.mean()
        else:
            consistency = 0.0

        return bce + self.consistency_weight * consistency


# ─── Weighted Ordinal Cross-Entropy ─────────────────────────────────────────

class WeightedOrdinalCrossEntropy(nn.Module):
    """Cross-entropy with class weights and ordinal distance penalty.

    Loss = class_weight[y] * CE(logits, y) + λ * Σ_j |j - y| * softmax(logits)[j]

    The ordinal penalty pushes probability mass towards the true class,
    penalizing distant misclassifications more heavily than adjacent
    confusion.  Addresses both class imbalance (via weights) and the
    natural ordering of pain levels (via ordinal penalty).

    Output: K logits (standard classification head).
    """

    def __init__(self, num_classes, class_weights=None, ordinal_lambda=0.1,
                 label_smoothing=0.0):
        """
        Args:
            num_classes: number of ordinal classes
            class_weights: optional (num_classes,) tensor for imbalance
            ordinal_lambda: λ for the ordinal distance penalty (higher =
                            stronger ordinal enforcement; 0 = plain CE)
            label_smoothing: label smoothing factor (0 = no smoothing)
        """
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_lambda = ordinal_lambda
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Precompute ordinal distance matrix: D[i][j] = |i - j|
        distances = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                distances[i, j] = abs(i - j)
        self.register_buffer('distances', distances)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K) — raw logits
            targets: (B,) — integer class labels

        Returns:
            scalar loss
        """
        # CE term with class weights and label smoothing
        ce = F.cross_entropy(
            logits, targets, reduction="none",
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # Ordinal distance penalty: E[|pred - true|] under predicted distribution
        if self.ordinal_lambda > 0:
            probs = logits.softmax(dim=1)  # (B, K)
            ordinal_idx = self.distances[targets]  # (B, K)
            ordinal_penalty = (probs * ordinal_idx).sum(dim=1)  # (B,)
        else:
            ordinal_penalty = 0.0

        return ce.mean() + self.ordinal_lambda * ordinal_penalty.mean()


# ─── Backward-compatible aliases ────────────────────────────────────────────

def corn_logits_to_probs(logits):
    """Alias for _k_minus_1_to_probs (kept for backward compatibility)."""
    return _k_minus_1_to_probs(logits)


def corn_logits_to_preds(logits):
    """Alias for _k_minus_1_to_preds (kept for backward compatibility)."""
    return _k_minus_1_to_preds(logits)


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
        class_weights: optional (num_classes,) tensor for CE/Focal/Ordinal loss weighting

    Returns:
        nn.Module loss function, and a bool `ordinal_mode` flag
        (ordinal_mode=True means the model outputs K-1 logits, used by corn/coral)
    """
    smoothing = getattr(config, "label_smoothing", 0.0)

    if config.loss_type == "corn":
        return CornLoss(config.num_classes, class_weights), True
    elif config.loss_type == "coral":
        cw = getattr(config, "coral_consistency_weight", 0.05)
        return CoralLoss(config.num_classes, class_weights, consistency_weight=cw), True
    elif config.loss_type == "weighted_ordinal":
        ol = getattr(config, "ordinal_lambda", 0.1)
        return WeightedOrdinalCrossEntropy(
            config.num_classes, class_weights, ordinal_lambda=ol,
            label_smoothing=smoothing,
        ), False
    elif config.loss_type == "focal":
        alpha = class_weights if config.focal_alpha is None else config.focal_alpha
        return FocalLoss(alpha=alpha, gamma=config.focal_gamma, label_smoothing=smoothing), False
    else:  # "ce" or default
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing), False
