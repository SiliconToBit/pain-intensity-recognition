"""Training and evaluation for ResNet-18 pain intensity recognition.

Pipeline:
    1. Load LOSO splits from pkl
    2. For each fold: generate frame windows → train model → collect predictions
    3. After all folds: compute comprehensive metrics on aggregated predictions

Metrics:
    - Weighted F1-score
    - Macro F1-score
    - Per-class recall
    - Confusion matrix
    - Cohen's Kappa
    - Multi-class AUROC
"""

import os
import re
import json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from model import PainRecognitionModel
from utils.dataset import FrameSequenceDataset, get_train_transforms, get_test_transforms, undersample_windows, compute_class_weights
from utils.checkpoint import save_checkpoint, load_checkpoint, save_progress, load_progress

# GPU optimization
torch.backends.cudnn.benchmark = True  # 加速固定输入尺寸的卷积


# ─── Data Loading ────────────────────────────────────────────────────────────

FRAME_PATTERN = re.compile(r"RGB-(\d+)-(\d+)-(\d+)-(\d+)\.\w+")


def parse_frame_timestamp(filename):
    m = FRAME_PATTERN.match(os.path.basename(filename))
    if m:
        h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600000 + mi * 60000 + s * 1000 + ms
    return 0


def scan_dataset(config):
    """Scan directory structure to build sweep list.

    Directory structure:
        preprocessed_dir/
        ├── Sub1 Daniel Simonsen/
        │   ├── Annotated_data_Sub01_Trial01/
        │   │   ├── Sub01_Trial01_Sweep01_Label0/rgb/*.jpg
        │   │   ├── Sub01_Trial01_Sweep01_Label3/rgb/*.jpg
        │   │   └── ...
        │   └── Annotated_data_Sub01_Trial02/
        └── ...

    Returns:
        list of dicts: [{subject, subject_id, sweep_id, trial, label, frame_paths}, ...]
    """
    base = config.preprocessed_dir
    sweeps = []

    for sub_name in sorted(os.listdir(base)):
        sub_path = os.path.join(base, sub_name)
        if not os.path.isdir(sub_path) or sub_name.startswith("."):
            continue

        # Extract subject ID: "Sub1 Daniel Simonsen" → "Sub01"
        sub_num = int(sub_name.split()[0].replace("Sub", ""))
        subject_id = f"Sub{sub_num:02d}"

        for trial_name in sorted(os.listdir(sub_path)):
            trial_path = os.path.join(sub_path, trial_name)
            if not os.path.isdir(trial_path):
                continue

            for sweep_name in sorted(os.listdir(trial_path)):
                sweep_path = os.path.join(trial_path, sweep_name)
                if not os.path.isdir(sweep_path):
                    continue

                # Extract label from dir name: "Sub01_Trial01_Sweep01_Label0" → 0
                label = None
                for part in sweep_name.split("_"):
                    if part.startswith("Label"):
                        label = int(part.replace("Label", ""))
                        break
                if label is None:
                    continue

                # Get frame paths
                rgb_dir = os.path.join(sweep_path, "rgb")
                if not os.path.isdir(rgb_dir):
                    continue
                frame_paths = sorted([
                    os.path.join(rgb_dir, f)
                    for f in os.listdir(rgb_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ], key=lambda x: parse_frame_timestamp(os.path.basename(x)))

                if len(frame_paths) == 0:
                    continue

                # Extract sweep_id: "Sub01_Trial01_Sweep01" from dir name
                sweep_id = "_".join(sweep_name.split("_")[:3])
                trial = trial_name.split("/")[-1]

                sweeps.append({
                    "subject": sub_name,
                    "subject_id": subject_id,
                    "sweep_id": sweep_id,
                    "trial": trial,
                    "label": label,
                    "frame_paths": frame_paths,
                })

    return sweeps


def remap_to_binary(sweeps):
    """Remap 5-class labels to binary: 0=no-pain, 1=pain.

    Label 0 (无痛) → 0
    Label 1-4 (有痛) → 1
    """
    for s in sweeps:
        s["label"] = 0 if s["label"] == 0 else 1
    return sweeps


def build_loso_folds(sweeps):
    """Build Leave-One-Subject-Out folds from sweep list.

    Returns:
        dict: {fold_name: {"test_subject": str, "train_sweeps": [...], "test_sweeps": [...]}}
    """
    # Group sweeps by subject
    subject_sweeps = {}
    for s in sweeps:
        subj = s["subject_id"]
        if subj not in subject_sweeps:
            subject_sweeps[subj] = []
        subject_sweeps[subj].append(s)

    folds = {}
    for test_subj in sorted(subject_sweeps.keys()):
        fold_name = f"LOSO_{test_subj}"
        train_sweeps = []
        test_sweeps = subject_sweeps[test_subj]

        for subj, sws in subject_sweeps.items():
            if subj != test_subj:
                train_sweeps.extend(sws)

        folds[fold_name] = {
            "test_subject": test_subj,
            "train_sweeps": train_sweeps,
            "test_sweeps": test_sweeps,
        }

    return folds


def generate_windows(sweeps, window_size=5, slide_step=2):
    """Generate overlapping frame windows from a list of sweeps."""
    windows = []
    for sweep in sweeps:
        frames = sweep["frame_paths"]
        label = sweep["label"]
        subject_id = sweep["subject_id"]
        sweep_id = sweep["sweep_id"]

        if len(frames) < window_size:
            continue

        n_windows = (len(frames) - window_size) // slide_step + 1
        for i in range(n_windows):
            start = i * slide_step
            window_frames = frames[start:start + window_size]
            sample_id = f"{subject_id}_{sweep_id}_Win{window_size}_{i:03d}"
            windows.append({
                "sample_id": sample_id,
                "subject_id": subject_id,
                "sweep_id": sweep_id,
                "frame_paths": window_frames,
                "label": label,
            })
    return windows


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """Train one epoch."""
    model.train()
    total_loss = 0
    preds, labels = [], []

    for sequences, targets in tqdm(dataloader, desc="Training", leave=False):
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(sequences)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), preds, labels


def evaluate(model, dataloader, criterion, device):
    """Evaluate model, return loss, predictions, labels, and probabilities."""
    model.eval()
    total_loss = 0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = sequences.to(device)
            targets = targets.to(device)

            with autocast():
                logits = model(sequences)
                loss = criterion(logits, targets)

            total_loss += loss.item()
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), np.array(preds), np.array(labels), np.array(probs)


# ─── Metrics ─────────────────────────────────────────────────────────────────

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


# ─── Main Training Loop ─────────────────────────────────────────────────────

def train_and_evaluate(config, resume=False):
    """Full LOSO cross-validation training and evaluation."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"Batch size: {config.batch_size} | Workers: {config.num_workers}")
        print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # Scan dataset directory
    print("Scanning dataset...")
    all_sweeps = scan_dataset(config)
    print(f"Found {len(all_sweeps)} sweeps across {len(set(s['subject_id'] for s in all_sweeps))} subjects")

    # Binary mode: remap labels 0→0, 1-4→1
    if config.binary_mode:
        all_sweeps = remap_to_binary(all_sweeps)
        print("Binary mode: remapped labels to 0 (no-pain) / 1 (pain)")

    # Build LOSO folds
    loso_folds = build_loso_folds(all_sweeps)

    all_fold_names = sorted(loso_folds.keys())
    if config.num_folds and config.num_folds > 0:
        fold_names = all_fold_names[:min(config.num_folds, len(all_fold_names))]
    else:
        fold_names = all_fold_names
    num_folds = len(fold_names)

    # Load progress if resuming
    completed_folds = []
    if resume:
        completed_folds, _ = load_progress(config, "train")
        print(f"Resuming training. {len(completed_folds)} folds already completed.")

    print(f"Using {num_folds} folds for LOSO cross-validation")

    # Collect all predictions across folds
    all_preds = []
    all_labels = []
    all_probs = []

    for fold_idx, fold_name in enumerate(fold_names):
        # Skip completed folds
        if resume and fold_name in completed_folds:
            print(f"\n⏭️  Skipping {fold_name} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"Fold: {fold_name} ({fold_idx + 1}/{num_folds})")
        print(f"{'='*60}")

        fold_data = loso_folds[fold_name]
        train_sweeps = fold_data["train_sweeps"]
        test_sweeps = fold_data["test_sweeps"]
        test_subject = fold_data["test_subject"]
        print(f"Test subject: {test_subject}")
        print(f"Train sweeps: {len(train_sweeps)}, Test sweeps: {len(test_sweeps)}")

        # Generate frame windows
        train_windows = generate_windows(train_sweeps, window_size=config.sequence_length, slide_step=config.slide_step)
        test_windows = generate_windows(test_sweeps, window_size=config.sequence_length, slide_step=config.slide_step)
        print(f"Train windows: {len(train_windows)}, Test windows: {len(test_windows)}")

        # Print class distribution
        train_dist = Counter(w["label"] for w in train_windows)
        test_dist = Counter(w["label"] for w in test_windows)
        print(f"  Train class distribution: {dict(sorted(train_dist.items()))}")
        print(f"  Test  class distribution: {dict(sorted(test_dist.items()))}")

        if len(train_windows) == 0 or len(test_windows) == 0:
            print(f"Skipping {fold_name}: no windows generated")
            continue

        # Compute class weights from ORIGINAL distribution (before any undersampling)
        if config.class_weight != "none":
            class_weights = compute_class_weights(
                train_windows, num_classes=config.num_classes, mode=config.class_weight,
            ).to(device)
            print(f"  Class weights ({config.class_weight}): {class_weights.tolist()}")
        else:
            class_weights = None

        # Undersample training data
        if config.undersample:
            train_windows = undersample_windows(train_windows, num_classes=config.num_classes)
            print(f"Undersampled train windows: {len(train_windows)}")

        # Create datasets and loaders
        train_dataset = FrameSequenceDataset(train_windows, transform=get_train_transforms())
        test_dataset = FrameSequenceDataset(test_windows, transform=get_test_transforms())
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
        )

        # Create model
        model = PainRecognitionModel(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            pretrained_source=config.pretrained_source,
            weights_path=config.vggface2_weights_path,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_num_layers=config.lstm_num_layers,
            dropout=config.dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scaler = GradScaler()

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        start_epoch = 0

        # Load checkpoint if resuming
        if resume:
            ckpt, start_epoch = load_checkpoint(config, fold_idx, device)
            if ckpt is not None:
                model.load_state_dict(ckpt["model_state_dict"])
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                patience_counter = ckpt.get("patience_counter", 0)
                print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

        # ── Phase 1: Train classifier only (backbone frozen) ──
        print(f"\n  Phase 1: Training classifier (backbone frozen)")
        model.freeze_backbone()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.phase1_lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )

        phase1_start = max(0, start_epoch)
        phase1_end = config.phase1_epochs

        for epoch in range(phase1_start, phase1_end):
            train_loss, train_preds, train_labels = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
            )
            val_loss, val_preds, val_labels, val_probs = evaluate(
                model, test_loader, criterion, device,
            )
            scheduler.step(val_loss)

            train_f1 = f1_score(train_labels, train_preds, average="weighted")
            print(f"  Epoch {epoch+1}/{phase1_end} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train F1: {train_f1:.4f}")

            # Save checkpoint
            save_checkpoint(
                config, fold_idx, epoch, model, optimizer, scheduler,
                best_val_loss, patience_counter,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # ── Phase 2: Unfreeze backbone, train with different LR + warmup ──
        # Save Phase 1 best model as fallback
        phase1_best_state = best_model_state
        print(f"\n  Phase 2: Fine-tuning (backbone unfrozen, warmup={config.warmup_epochs} epochs)")
        model.unfreeze_backbone()
        param_groups = model.get_param_groups(
            backbone_lr=0.0,  # Start at 0, warmup will increase
            classifier_lr=config.phase2_classifier_lr,
        )
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )

        # Reset early stopping state for Phase 2
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        phase2_epochs = phase1_end + config.phase2_epochs

        for epoch in range(phase1_end, phase2_epochs):
            # ── Warmup: linearly increase backbone LR ──
            warmup_epoch = epoch - phase1_end
            if warmup_epoch < config.warmup_epochs:
                warmup_factor = (warmup_epoch + 1) / config.warmup_epochs
                backbone_lr = config.phase2_backbone_lr * warmup_factor
                # Update backbone LR in param groups
                for pg in optimizer.param_groups:
                    if pg.get("is_backbone", False):
                        pg["lr"] = backbone_lr
                if warmup_epoch == 0:
                    print(f"  Warmup: backbone LR {0:.2e} → {config.phase2_backbone_lr:.2e} "
                          f"over {config.warmup_epochs} epochs")

            train_loss, train_preds, train_labels = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
            )
            val_loss, val_preds, val_labels, val_probs = evaluate(
                model, test_loader, criterion, device,
            )
            scheduler.step(val_loss)

            # Print current backbone LR
            current_backbone_lr = optimizer.param_groups[0]["lr"]
            train_f1 = f1_score(train_labels, train_preds, average="weighted")
            print(f"  Epoch {epoch+1}/{phase2_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Backbone LR: {current_backbone_lr:.2e}")

            # Save checkpoint
            save_checkpoint(
                config, fold_idx, epoch, model, optimizer, scheduler,
                best_val_loss, patience_counter,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model: prefer Phase 2 best, fall back to Phase 1 best
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"  Using Phase 2 best model (val_loss={best_val_loss:.4f})")
        elif phase1_best_state:
            model.load_state_dict(phase1_best_state)
            print(f"  Phase 2 did not improve, using Phase 1 best model")
        model.to(device)

        _, fold_preds, fold_labels, fold_probs = evaluate(
            model, test_loader, criterion, device,
        )

        # Collect predictions
        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
        all_probs.extend(fold_probs)

        # Per-fold metrics
        fold_f1 = f1_score(fold_labels, fold_preds, average="weighted")
        print(f"\n  {fold_name} ({test_subject}) | Weighted F1: {fold_f1:.4f}")

        # Save progress
        completed_folds.append(fold_name)
        save_progress(config, "train", completed_folds)

    # ── Final Metrics ──
    if len(all_preds) == 0:
        print("No predictions collected. Check data paths.")
        return

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs, config.num_classes)
    print_metrics(metrics, config.num_classes)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Save results
    results = {
        "metrics": {
            "weighted_f1": metrics["weighted_f1"],
            "macro_f1": metrics["macro_f1"],
            "cohens_kappa": metrics["cohens_kappa"],
            "auroc_weighted": metrics["auroc_weighted"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_auc": metrics["per_class_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "config": {
            "backbone": config.backbone,
            "num_classes": config.num_classes,
            "sequence_length": config.sequence_length,
            "batch_size": config.batch_size,
            "phase1_epochs": config.phase1_epochs,
            "phase2_epochs": config.phase2_epochs,
            "lstm_hidden_dim": config.lstm_hidden_dim,
        },
    }

    results_path = os.path.join(config.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save numpy arrays
    np.save(os.path.join(config.output_dir, "predictions.npy"), all_preds)
    np.save(os.path.join(config.output_dir, "labels.npy"), all_labels)
    np.save(os.path.join(config.output_dir, "probabilities.npy"), all_probs)
    np.save(os.path.join(config.output_dir, "confusion_matrix.npy"),
            np.array(metrics["confusion_matrix"]))
