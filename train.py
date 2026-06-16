"""Training and evaluation for pain intensity recognition.

Pipeline:
    1. Load LOSO splits from scanned dataset
    2. For each fold: generate frame windows → train model → collect predictions
    3. After all folds: compute comprehensive metrics on aggregated predictions

Supports two-phase training:
    - Phase 1: backbone frozen, train classifier only
    - Phase 2: unfreeze backbone with lower LR + warmup
"""

import os
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import swanlab

from model import PainRecognitionModel
from utils.dataset import (
    FrameSequenceDataset,
    get_train_transforms,
    get_test_transforms,
    undersample_windows,
    compute_class_weights,
)
from utils.checkpoint import save_checkpoint, load_checkpoint, save_progress, load_progress
from utils.losses import build_loss, corn_logits_to_preds, corn_logits_to_probs
from utils.data_loader import scan_dataset, remap_to_binary, build_loso_folds, generate_windows
from utils.metrics import compute_metrics, print_metrics

# GPU optimization
torch.backends.cudnn.benchmark = True  # 加速固定输入尺寸的卷积


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, corn_mode=False):
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
        if corn_mode:
            preds.extend(corn_logits_to_preds(logits).cpu().numpy())
        else:
            preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), preds, labels


def evaluate(model, dataloader, criterion, device, corn_mode=False):
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
            if corn_mode:
                probs.extend(corn_logits_to_probs(logits).cpu().numpy())
                preds.extend(corn_logits_to_preds(logits).cpu().numpy())
            else:
                probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
                preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), np.array(preds), np.array(labels), np.array(probs)


# ─── Main Training Loop ─────────────────────────────────────────────────────

def train_and_evaluate(config, resume=False):
    """Full LOSO cross-validation training and evaluation."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  cudnn.benchmark: {torch.backends.cudnn.benchmark}")

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

    # ── Initialize SwanLab ──
    run = swanlab.init(
        project="pain-intensity-recognition",
        experiment_name=f"ResNet18-LSTM_{config.loss_type}_{'binary' if config.binary_mode else '5class'}",
        config={
            "backbone": config.backbone,
            "pretrained_source": config.pretrained_source,
            "num_classes": config.num_classes,
            "sequence_length": config.sequence_length,
            "batch_size": config.batch_size,
            "phase1_epochs": config.phase1_epochs,
            "phase2_epochs": config.phase2_epochs,
            "loss_type": config.loss_type,
            "lstm_hidden_dim": config.lstm_hidden_dim,
            "dropout": config.dropout,
            "use_attention": config.use_attention_pooling,
            "binary_mode": config.binary_mode,
            "undersample": config.undersample,
            "num_folds": config.num_folds or len(fold_names),
        },
    )

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
        # Resolve weights path for external pretrained models.
        # VGGFace2 loads weights internally via facenet-pytorch (InceptionResnetV1),
        # so no external weights_path is needed.
        weights_path = None
        if config.pretrained_source == "arcface":
            weights_path = os.path.join(config.pretrained_weights_path, "w600k_r50.onnx")

        # Build loss function (supports CE, Corn ordinal, Focal)
        criterion, corn_mode = build_loss(config, class_weights)
        print(f"  Loss: {config.loss_type}" + (" (corn ordinal)" if corn_mode else ""))

        # Create model (Corn mode outputs K-1 logits for K-class ordinal regression)
        model = PainRecognitionModel(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            pretrained_source=config.pretrained_source,
            weights_path=weights_path,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_num_layers=config.lstm_num_layers,
            dropout=config.dropout,
            corn_mode=corn_mode,
            use_attention_pooling=config.use_attention_pooling,
        ).to(device)

        scaler = GradScaler()

        best_val_loss = 0.0  # tracks val F1 (higher is better)
        best_model_state = None
        patience_counter = 0
        start_epoch = 0

        # Load checkpoint if resuming
        if resume:
            ckpt, start_epoch = load_checkpoint(config, fold_idx, device)
            if ckpt is not None:
                try:
                    model.load_state_dict(ckpt["model_state_dict"])
                    best_val_loss = ckpt.get("best_val_loss", 0.0)
                    patience_counter = ckpt.get("patience_counter", 0)
                    print(f"  Resumed from epoch {start_epoch}, best_val_f1={best_val_loss:.4f}")
                except (RuntimeError, KeyError) as e:
                    print(f"  Checkpoint incompatible (different architecture), starting fresh: {e}")
                    ckpt = None
                    start_epoch = 0

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
                model, train_loader, optimizer, criterion, device, scaler, corn_mode,
            )
            val_loss, val_preds, val_labels, val_probs = evaluate(
                model, test_loader, criterion, device, corn_mode,
            )
            scheduler.step(val_loss)

            val_f1 = f1_score(val_labels, val_preds, average="weighted")
            train_f1 = f1_score(train_labels, train_preds, average="weighted")
            print(f"  Epoch {epoch+1}/{phase1_end} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

            # Log to SwanLab
            swanlab.log({
                f"fold_{fold_idx}/phase1/train_loss": train_loss,
                f"fold_{fold_idx}/phase1/val_loss": val_loss,
                f"fold_{fold_idx}/phase1/train_f1": train_f1,
                f"fold_{fold_idx}/phase1/val_f1": val_f1,
            }, step=epoch + 1)

            # Save checkpoint
            save_checkpoint(
                config, fold_idx, epoch, model, optimizer, scheduler,
                best_val_loss, patience_counter,
            )

            if val_f1 > best_val_loss:  # track best val F1 (higher is better)
                best_val_loss = val_f1
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

        # Reduce batch size for Phase 2 to avoid OOM when backbone is unfrozen
        if config.pretrained_source == "arcface" and config.batch_size > 32:
            phase2_batch_size = max(16, config.batch_size // 3)
            print(f"  Reducing batch size for ArcFace fine-tuning: {config.batch_size} → {phase2_batch_size}")
            train_loader = DataLoader(
                train_dataset, batch_size=phase2_batch_size, shuffle=True,
                num_workers=config.num_workers, pin_memory=True,
                persistent_workers=True if config.num_workers > 0 else False,
            )
            test_loader = DataLoader(
                test_dataset, batch_size=phase2_batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True,
                persistent_workers=True if config.num_workers > 0 else False,
            )
            torch.cuda.empty_cache()
        param_groups = model.get_param_groups(
            backbone_lr=0.0,  # Start at 0, warmup will increase
            classifier_lr=config.phase2_classifier_lr,
        )
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )

        # Reset early stopping state for Phase 2
        best_val_loss = 0.0  # reset val F1 tracking for phase 2
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
                model, train_loader, optimizer, criterion, device, scaler, corn_mode,
            )
            val_loss, val_preds, val_labels, val_probs = evaluate(
                model, test_loader, criterion, device, corn_mode,
            )
            scheduler.step(val_loss)

            # Print current backbone LR
            current_backbone_lr = optimizer.param_groups[0]["lr"]
            train_f1 = f1_score(train_labels, train_preds, average="weighted")
            val_f1 = f1_score(val_labels, val_preds, average="weighted")
            print(f"  Epoch {epoch+1}/{phase2_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"Backbone LR: {current_backbone_lr:.2e}")

            # Log to SwanLab
            swanlab.log({
                f"fold_{fold_idx}/phase2/train_loss": train_loss,
                f"fold_{fold_idx}/phase2/val_loss": val_loss,
                f"fold_{fold_idx}/phase2/train_f1": train_f1,
                f"fold_{fold_idx}/phase2/val_f1": val_f1,
                f"fold_{fold_idx}/phase2/backbone_lr": current_backbone_lr,
            }, step=epoch + 1)

            # Save checkpoint
            save_checkpoint(
                config, fold_idx, epoch, model, optimizer, scheduler,
                best_val_loss, patience_counter,
            )

            if val_f1 > best_val_loss:  # track best val F1
                best_val_loss = val_f1
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
            print(f"  Using Phase 2 best model (val_f1={best_val_loss:.4f})")
        elif phase1_best_state:
            model.load_state_dict(phase1_best_state)
            print(f"  Phase 2 did not improve, using Phase 1 best model")
        model.to(device)

        _, fold_preds, fold_labels, fold_probs = evaluate(
            model, test_loader, criterion, device, corn_mode,
        )

        # Collect predictions
        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
        all_probs.extend(fold_probs)

        # Per-fold metrics
        fold_f1 = f1_score(fold_labels, fold_preds, average="weighted")
        print(f"\n  {fold_name} ({test_subject}) | Weighted F1: {fold_f1:.4f}")

        # Log fold result to SwanLab
        swanlab.log({
            f"fold_results/{fold_name}/weighted_f1": fold_f1,
            f"fold_results/{fold_name}/test_subject": test_subject,
        }, step=fold_idx + 1)

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

    # ── Log final metrics to SwanLab ──
    swanlab.log({
        "final/weighted_f1": metrics["weighted_f1"],
        "final/macro_f1": metrics["macro_f1"],
        "final/cohens_kappa": metrics["cohens_kappa"],
        "final/auroc_weighted": metrics["auroc_weighted"],
    })

    # Log confusion matrix as image
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=range(config.num_classes),
                    yticklabels=range(config.num_classes))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        swanlab.log({"final/confusion_matrix": swanlab.Image(fig)})
        plt.close(fig)
    except ImportError:
        print("  Skipping confusion matrix image: matplotlib/seaborn not available")

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

    # ── Finish SwanLab run ──
    swanlab.finish()
