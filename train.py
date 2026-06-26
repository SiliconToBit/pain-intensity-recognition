"""Training and evaluation for pain intensity recognition.

Pipeline:
    1. Load LOSO splits from scanned dataset
    2. For each fold: generate frame windows -> train model -> collect predictions
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
    SingleFrameDataset,
    get_train_transforms,
    get_test_transforms,
    undersample_windows,
    compute_class_weights,
)
from utils.checkpoint import save_checkpoint, load_checkpoint, save_progress, load_progress
from utils.losses import build_loss, corn_logits_to_preds, corn_logits_to_probs
from utils.schedulers import WarmupReduceLROnPlateau
from utils.data_loader import scan_dataset, remap_to_binary, build_loso_folds, generate_windows, generate_single_frames, split_train_val_by_subject
from utils.metrics import compute_metrics, print_metrics
from utils.repro import set_seed, seed_worker


# --- Training helpers ---

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, ordinal_mode=False,
                max_grad_norm=1.0, grad_accum_steps=1):
    """Train one epoch.

    Args:
        grad_accum_steps: accumulate gradients over N micro-batches before
            stepping optimizer. Effective batch_size = batch_size * grad_accum_steps.
    """
    model.train()
    total_loss = 0
    preds, labels = [], []
    optimizer.zero_grad()

    for step, (sequences, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        sequences = sequences.to(device)
        targets = targets.to(device)

        with autocast():
            logits = model(sequences)
            loss = criterion(logits, targets) / grad_accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps  # un-scale for logging
        if ordinal_mode:
            preds.extend(corn_logits_to_preds(logits).cpu().numpy())
        else:
            preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), preds, labels


def evaluate(model, dataloader, criterion, device, ordinal_mode=False):
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
            if ordinal_mode:
                probs.extend(corn_logits_to_probs(logits).cpu().numpy())
                preds.extend(corn_logits_to_preds(logits).cpu().numpy())
            else:
                probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
                preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), np.array(preds), np.array(labels), np.array(probs)


# --- Experiment setup ---

def _setup_experiment(config, resume=False):
    """Initialize experiment: seed, dataset scan, LOSO folds, SwanLab.

    Returns:
        Tuple of (device, loso_folds, fold_names, num_folds, phase1_batch_size,
                  exp_name, task_str, completed_folds).
    """
    set_seed(config.seed, deterministic=config.deterministic)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  seed: {config.seed}  |  "
          f"deterministic: {config.deterministic}")

    # Scan dataset directory
    print("Scanning dataset...")
    all_sweeps = scan_dataset(config)
    print(f"Found {len(all_sweeps)} sweeps across {len(set(s['subject_id'] for s in all_sweeps))} subjects")

    # Binary mode: remap labels 0->0, 1-4->1
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

    # -- Initialize SwanLab --
    task_str = "binary" if config.binary_mode else "5class"
    exp_name = f"{config.pretrained_source}_{task_str}_{config.loss_type}"
    variants = []
    if config.use_attention_pooling:
        variants.append("attention")
    if variants:
        exp_name += "_" + "_".join(variants)

    group_name = exp_name
    tags = [
        config.pretrained_source,
        task_str,
        config.loss_type,
    ]
    if config.use_attention_pooling:
        tags.append("attention")
    if config.undersample:
        tags.append("undersample")

    # Compute effective batch size based on actual VRAM & per-sample cost.
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        usable = vram_gb - 1.0  # 1 GB safety margin for CUDA context
        if config.pretrained_source in ("arcface", "affectnet"):
            base_per_sample = 0.17
        else:
            base_per_sample = 0.06
        if config.single_frame:
            per_sample_gb = base_per_sample / config.sequence_length
            target_vram = usable * 0.90
            phase1_batch_size = max(config.batch_size,
                                    max(4, int(target_vram / per_sample_gb)))
        else:
            per_sample_gb = base_per_sample
            target_vram = usable * 0.90
            phase1_batch_size = min(config.batch_size,
                                    max(4, int(target_vram / per_sample_gb)))
        print(f"  Batch size: {phase1_batch_size} "
              f"(VRAM {vram_gb:.1f} GB, per-sample {per_sample_gb:.3f} GB)")
    else:
        phase1_batch_size = config.batch_size

    swanlab.init(
        project="pain-intensity-recognition",
        experiment_name=exp_name,
        group=group_name,
        tags=tags,
        config={
            "pretrained_source": config.pretrained_source,
            "num_classes": config.num_classes,
            "sequence_length": config.sequence_length,
            "num_windows_per_sweep": config.num_windows_per_sweep,
            "batch_size": phase1_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": phase1_batch_size * config.gradient_accumulation_steps,
            "phase1_epochs": config.phase1_epochs,
            "phase2_epochs": config.phase2_epochs,
            "phase1_lr": config.phase1_lr,
            "phase2_backbone_lr": config.phase2_backbone_lr,
            "phase2_classifier_lr": config.phase2_classifier_lr,
            "warmup_epochs": config.warmup_epochs,
            "loss_type": config.loss_type,
            "focal_gamma": config.focal_gamma,
            "lstm_hidden_dim": config.lstm_hidden_dim,
            "lstm_num_layers": config.lstm_num_layers,
            "dropout": config.dropout,
            "use_attention": config.use_attention_pooling,
            "binary_mode": config.binary_mode,
            "undersample": config.undersample,
            "class_weight": config.class_weight,
            "patience": config.patience,
            "num_folds": config.num_folds or len(fold_names),
            "seed": config.seed,
            "classifier_hidden_dim": config.classifier_hidden_dim,
            "label_smoothing": config.label_smoothing,
            "deterministic": config.deterministic,
        },
    )

    return (device, loso_folds, fold_names, num_folds,
            phase1_batch_size, exp_name, task_str, completed_folds)


# --- Phase training ---

def train_phase1(model, train_loader, val_loader, criterion, device, scaler,
                 config, fold_idx, global_step, start_epoch=0, ordinal_mode=False):
    """Phase 1: Train classifier only (backbone frozen).

    Returns:
        Tuple of (best_model_state, best_val_f1, patience_counter, global_step).
    """
    print(f"\n  Phase 1: Training classifier (backbone frozen)")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.phase1_lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience,
    )

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    phase1_start = max(0, start_epoch)
    phase1_end = config.phase1_epochs

    for epoch in range(phase1_start, phase1_end):
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, ordinal_mode,
            grad_accum_steps=config.gradient_accumulation_steps,
        )
        val_loss, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, ordinal_mode,
        )
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        scheduler.step(val_f1)

        train_f1 = f1_score(train_labels, train_preds, average="weighted")
        print(f"  Epoch {epoch+1}/{phase1_end} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # Log to SwanLab
        global_step += 1
        log_dict = {
            "train/loss": train_loss,
            "train/f1": train_f1,
            "val/loss": val_loss,
            "val/f1": val_f1,
            "train/phase": 1,
            f"fold/{fold_idx}/phase1/train_loss": train_loss,
            f"fold/{fold_idx}/phase1/val_loss": val_loss,
            f"fold/{fold_idx}/phase1/train_f1": train_f1,
            f"fold/{fold_idx}/phase1/val_f1": val_f1,
        }
        swanlab.log(log_dict, step=global_step)

        save_checkpoint(
            config, fold_idx, epoch, model, optimizer, scheduler,
            best_val_f1, patience_counter,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_model_state, best_val_f1, patience_counter, global_step


def train_phase2(model, train_dataset, val_dataset, test_dataset,
                 criterion, device, scaler, config, fold_idx, global_step,
                 phase1_end, phase1_batch_size, phase1_best_state, phase1_best_f1,
                 ordinal_mode=False):
    """Phase 2: Unfreeze backbone, train with different LR + warmup.

    Creates new dataloaders with potentially reduced batch size to avoid OOM
    when backbone gradients are enabled.

    Returns:
        Tuple of (best_model_state, best_val_f1, train_loader, val_loader,
                  test_loader, global_step).
    """
    print(f"\n  Phase 2: Fine-tuning (backbone unfrozen, warmup={config.warmup_epochs} epochs)")

    # Load Phase 1 best model as starting point
    if phase1_best_state:
        model.load_state_dict(phase1_best_state)
        print(f"  Loaded Phase 1 best model (val_f1={phase1_best_f1:.4f}) as starting point")

    model.unfreeze_backbone()

    # Verify backbone is actually unfrozen
    backbone_params = list(model.feature_extractor.parameters())
    trainable = sum(p.requires_grad for p in backbone_params)
    total = len(backbone_params)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Backbone: {trainable}/{total} params trainable")
    print(f"  Model: {total_trainable/1e6:.1f}M / {total_params/1e6:.1f}M params trainable "
          f"({100*total_trainable/total_params:.1f}%)")
    print(f"  Backbone LR: 0 -> {config.phase2_backbone_lr:.2e} over {config.warmup_epochs} epochs")

    # Reduce batch size for Phase 2 to avoid OOM when backbone is unfrozen
    phase2_batch_size = phase1_batch_size
    if config.pretrained_source in ("arcface", "affectnet"):
        phase2_batch_size = max(4, phase1_batch_size // 2)
        if phase2_batch_size < phase1_batch_size:
            print(f"  Reducing batch size for {config.pretrained_source} fine-tuning: "
                  f"{phase1_batch_size} -> {phase2_batch_size}")
        train_loader = DataLoader(
            train_dataset, batch_size=phase2_batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(config.seed),
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=phase2_batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=phase2_batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
        )
        torch.cuda.empty_cache()
    else:
        # Reuse Phase 1 loaders
        train_loader = DataLoader(
            train_dataset, batch_size=phase1_batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(config.seed),
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=phase1_batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=phase1_batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
        )

    param_groups = model.get_param_groups(
        backbone_lr=config.phase2_backbone_lr,
        classifier_lr=config.phase2_classifier_lr,
    )
    optimizer = torch.optim.Adam(param_groups)
    scheduler = WarmupReduceLROnPlateau(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        warmup_group_indices=[0],  # group 0 = backbone
        mode="max",
        factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience,
    )

    # Continue tracking from Phase 1 best
    best_val_f1 = phase1_best_f1
    best_model_state = phase1_best_state
    patience_counter = 0
    phase2_epochs = phase1_end + config.phase2_epochs

    for epoch in range(phase1_end, phase2_epochs):
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, ordinal_mode,
            grad_accum_steps=config.gradient_accumulation_steps,
        )
        val_loss, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, ordinal_mode,
        )
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        scheduler.step(val_f1, epoch=epoch - phase1_end)

        current_backbone_lr = optimizer.param_groups[0]["lr"]
        train_f1 = f1_score(train_labels, train_preds, average="weighted")
        print(f"  Epoch {epoch+1}/{phase2_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
              f"Backbone LR: {current_backbone_lr:.2e}")

        global_step += 1
        log_dict = {
            "train/loss": train_loss,
            "train/f1": train_f1,
            "val/loss": val_loss,
            "val/f1": val_f1,
            "val/backbone_lr": current_backbone_lr,
            "train/phase": 2,
            f"fold/{fold_idx}/phase2/train_loss": train_loss,
            f"fold/{fold_idx}/phase2/val_loss": val_loss,
            f"fold/{fold_idx}/phase2/train_f1": train_f1,
            f"fold/{fold_idx}/phase2/val_f1": val_f1,
            f"fold/{fold_idx}/phase2/backbone_lr": current_backbone_lr,
        }
        swanlab.log(log_dict, step=global_step)

        save_checkpoint(
            config, fold_idx, epoch, model, optimizer, scheduler,
            best_val_f1, patience_counter,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_model_state, best_val_f1, train_loader, val_loader, test_loader, global_step


# --- Fold execution ---

def run_fold(config, fold_idx, fold_name, fold_data, device, phase1_batch_size,
             global_step, resume=False):
    """Execute one LOSO fold: prepare data, train both phases, evaluate.

    Returns:
        Tuple of (fold_preds, fold_labels, fold_probs, global_step) or
        (None, None, None, global_step) if fold is skipped.
    """
    print(f"\n{'='*60}")
    print(f"Fold: {fold_name} ({fold_idx + 1})")
    print(f"{'='*60}")

    train_sweeps = fold_data["train_sweeps"]
    test_sweeps = fold_data["test_sweeps"]
    test_subject = fold_data["test_subject"]
    print(f"Test subject: {test_subject}")
    print(f"Train sweeps: {len(train_sweeps)}, Test sweeps: {len(test_sweeps)}")

    # Split training sweeps into train/validation by subject
    train_sweeps_actual, val_sweeps = split_train_val_by_subject(
        train_sweeps, val_ratio=0.15, seed=config.seed,
    )
    print(f"  Train subjects: {len(set(s['subject_id'] for s in train_sweeps_actual))}, "
          f"Val subjects: {len(set(s['subject_id'] for s in val_sweeps))}")

    # Generate samples: single-frame or sequence windows
    if config.single_frame:
        train_items = generate_single_frames(train_sweeps_actual)
        val_items = generate_single_frames(val_sweeps)
        test_items = generate_single_frames(test_sweeps)
        print(f"Train frames: {len(train_items)}, Val frames: {len(val_items)}, "
              f"Test frames: {len(test_items)}")
    else:
        train_items = generate_windows(train_sweeps_actual, window_size=config.sequence_length, slide_step=config.slide_step)
        val_items = generate_windows(val_sweeps, window_size=config.sequence_length, slide_step=config.slide_step)
        test_items = generate_windows(test_sweeps, window_size=config.sequence_length, slide_step=config.slide_step)
        print(f"Train windows: {len(train_items)}, Val windows: {len(val_items)}, "
              f"Test windows: {len(test_items)}")

    # Print class distribution
    train_dist = Counter(w["label"] for w in train_items)
    val_dist = Counter(w["label"] for w in val_items)
    test_dist = Counter(w["label"] for w in test_items)
    print(f"  Train class distribution: {dict(sorted(train_dist.items()))}")
    print(f"  Val   class distribution: {dict(sorted(val_dist.items()))}")
    print(f"  Test  class distribution: {dict(sorted(test_dist.items()))}")

    if len(train_items) == 0 or len(test_items) == 0:
        print(f"Skipping {fold_name}: no samples generated")
        return None, None, None, global_step

    # Compute class weights from ORIGINAL distribution (before any undersampling)
    if config.class_weight != "none":
        class_weights = compute_class_weights(
            train_items, num_classes=config.num_classes, mode=config.class_weight,
        ).to(device)
        print(f"  Class weights ({config.class_weight}): {class_weights.tolist()}")
    else:
        class_weights = None

    # Undersample training data
    if config.undersample:
        train_items = undersample_windows(train_items, num_classes=config.num_classes)
        print(f"Undersampled train samples: {len(train_items)}")

    # Create datasets and loaders
    if config.single_frame:
        train_dataset = SingleFrameDataset(train_items, transform=get_train_transforms())
        val_dataset = SingleFrameDataset(val_items, transform=get_test_transforms())
        test_dataset = SingleFrameDataset(test_items, transform=get_test_transforms())
    else:
        train_dataset = FrameSequenceDataset(train_items, transform=get_train_transforms())
        val_dataset = FrameSequenceDataset(val_items, transform=get_test_transforms())
        test_dataset = FrameSequenceDataset(test_items, transform=get_test_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=phase1_batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        generator=torch.Generator().manual_seed(config.seed),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=phase1_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=phase1_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )

    # Create model
    weights_path = None
    if config.pretrained_source == "arcface":
        weights_path = os.path.join(config.pretrained_weights_path, config.arcface_weights_file)
    elif config.pretrained_source == "affectnet":
        weights_path = os.path.join(config.pretrained_weights_path, config.affectnet_weights_file)

    criterion, ordinal_mode = build_loss(config, class_weights)
    print(f"  Loss: {config.loss_type}" + (" (ordinal K-1 output)" if ordinal_mode else ""))

    model = PainRecognitionModel(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        pretrained_source=config.pretrained_source,
        weights_path=weights_path,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_num_layers=config.lstm_num_layers,
        dropout=config.dropout,
        corn_mode=ordinal_mode,
        use_attention_pooling=config.use_attention_pooling,
        single_frame=config.single_frame,
        classifier_hidden_dim=config.classifier_hidden_dim,
    ).to(device)

    scaler = GradScaler()

    # Load checkpoint if resuming
    start_epoch = 0
    if resume:
        ckpt, start_epoch = load_checkpoint(config, fold_idx, device)
        if ckpt is not None:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"  Resumed from epoch {start_epoch}")
            except (RuntimeError, KeyError) as e:
                print(f"  Checkpoint incompatible (different architecture), starting fresh: {e}")
                start_epoch = 0

    # -- Phase 1: Train classifier only (backbone frozen) --
    phase1_best_state, phase1_best_f1, _, global_step = train_phase1(
        model, train_loader, val_loader, criterion, device, scaler,
        config, fold_idx, global_step, start_epoch=start_epoch, ordinal_mode=ordinal_mode,
    )

    # -- Phase 2: Unfreeze backbone, train with different LR + warmup --
    phase1_end = config.phase1_epochs
    best_model_state, best_val_f1, train_loader, val_loader, test_loader, global_step = train_phase2(
        model, train_dataset, val_dataset, test_dataset,
        criterion, device, scaler, config, fold_idx, global_step,
        phase1_end, phase1_batch_size, phase1_best_state, phase1_best_f1,
        ordinal_mode=ordinal_mode,
    )

    # Load best model across both phases
    if best_model_state:
        model.load_state_dict(best_model_state)
        if best_val_f1 > phase1_best_f1:
            print(f"  Using Phase 2 best model (val_f1={best_val_f1:.4f})")
        else:
            print(f"  Phase 2 did not improve over Phase 1 "
                  f"(val_f1={best_val_f1:.4f} <= {phase1_best_f1:.4f}), using Phase 1 best model")
    model.to(device)

    # Evaluate on test set
    _, fold_preds, fold_labels, fold_probs = evaluate(
        model, test_loader, criterion, device, ordinal_mode,
    )

    # Per-fold metrics
    fold_f1 = f1_score(fold_labels, fold_preds, average="weighted")
    print(f"\n  {fold_name} ({test_subject}) | Weighted F1: {fold_f1:.4f}")

    subject_num = int(test_subject.replace("Sub", ""))
    swanlab.log({
        "fold/weighted_f1": fold_f1,
        "fold/test_subject_id": subject_num,
    }, step=global_step)

    # Per-fold echarts confusion matrix
    fold_class_names = (
        ["无痛 (No Pain)", "疼痛 (Pain)"] if config.binary_mode
        else ["无痛 (0)", "轻微疼痛 (1)", "中度疼痛 (2)", "较强疼痛 (3)", "剧烈疼痛 (4)"]
    )
    try:
        swanlab.log({
            f"fold/confusion_matrix_{test_subject}": swanlab.echarts.confusion_matrix(
                fold_labels, fold_preds, fold_class_names
            )
        }, step=global_step)
    except Exception:
        pass  # pyecharts may not be installed

    return fold_preds, fold_labels, fold_probs, global_step


# --- Final metrics ---

def _log_final_metrics(config, all_preds, all_labels, all_probs, exp_name, task_str,
                       completed_folds, num_folds):
    """Compute and log final metrics after all folds."""
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs, config.num_classes)
    print_metrics(metrics, config.num_classes)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Log final metrics to SwanLab
    final_log = {
        "final/weighted_f1": metrics["weighted_f1"],
        "final/macro_f1": metrics["macro_f1"],
        "final/cohens_kappa": metrics["cohens_kappa"],
        "final/auroc_weighted": metrics["auroc_weighted"],
    }
    for i, r in enumerate(metrics["per_class_recall"]):
        final_log[f"final/recall_class_{i}"] = r
    for i, a in enumerate(metrics["per_class_auc"]):
        final_log[f"final/auc_class_{i}"] = a
    swanlab.log(final_log)

    # SwanLab ECharts: Confusion Matrix
    if config.binary_mode:
        class_names = ["无痛 (No Pain)", "疼痛 (Pain)"]
    else:
        class_names = [
            "无痛 (0)", "轻微疼痛 (1)", "中度疼痛 (2)", "较强疼痛 (3)", "剧烈疼痛 (4)"
        ]
    try:
        swanlab.log({
            "final/confusion_matrix_echarts": swanlab.echarts.confusion_matrix(
                all_labels.tolist(), all_preds.tolist(), class_names
            )
        })
    except Exception as e:
        print(f"  Skipping echarts confusion matrix: {e}")

    # SwanLab ECharts: ROC & PR Curves (binary mode)
    if config.binary_mode:
        try:
            y_prob_pos = all_probs[:, 1]
            swanlab.log({
                "final/roc_curve": swanlab.echarts.roc_curve(
                    all_labels.tolist(), y_prob_pos.tolist(), title="ROC Curve"
                ),
                "final/pr_curve": swanlab.echarts.pr_curve(
                    all_labels.tolist(), y_prob_pos.tolist(), title="PR Curve"
                ),
            })
        except Exception as e:
            print(f"  Skipping echarts ROC/PR curves: {e}")

    # Log confusion matrix as image (matplotlib fallback)
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

    # SwanLab Text Summary
    summary_lines = [
        f"## {exp_name} -- Results Summary",
        "",
        f"- **Backbone:** {config.pretrained_source} ({config.backbone})",
        f"- **Task:** {task_str} ({config.num_classes} classes)",
        f"- **Loss:** {config.loss_type}",
        f"- **Folds:** {len(completed_folds)} / {num_folds}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Weighted F1 | {metrics['weighted_f1']:.4f} |",
        f"| Macro F1 | {metrics['macro_f1']:.4f} |",
        f"| Cohen's Kappa | {metrics['cohens_kappa']:.4f} |",
        f"| AUROC (weighted) | {metrics['auroc_weighted']:.4f} |",
        "",
        f"**Per-class Recall:** {', '.join(f'{r:.3f}' for r in metrics['per_class_recall'])}",
        f"**Per-class AUC:** {', '.join(f'{a:.3f}' for a in metrics['per_class_auc'])}",
    ]
    swanlab.log({"summary": swanlab.Text("\n".join(summary_lines))})


# --- Main entry point ---

def train_and_evaluate(config, resume=False):
    """Full LOSO cross-validation training and evaluation."""
    # Setup
    (device, loso_folds, fold_names, num_folds,
     phase1_batch_size, exp_name, task_str, completed_folds) = _setup_experiment(config, resume)

    # Collect all predictions across folds
    all_preds = []
    all_labels = []
    all_probs = []
    global_step = 0

    for fold_idx, fold_name in enumerate(fold_names):
        # Skip completed folds
        if resume and fold_name in completed_folds:
            print(f"\n  Skipping {fold_name} (already completed)")
            continue

        fold_data = loso_folds[fold_name]
        fold_preds, fold_labels, fold_probs, global_step = run_fold(
            config, fold_idx, fold_name, fold_data, device, phase1_batch_size,
            global_step, resume=resume,
        )

        if fold_preds is None:
            continue

        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
        all_probs.extend(fold_probs)

        # Save progress
        completed_folds.append(fold_name)
        save_progress(config, "train", completed_folds)

    # Final metrics
    if len(all_preds) == 0:
        print("No predictions collected. Check data paths.")
        return

    _log_final_metrics(config, all_preds, all_labels, all_probs, exp_name, task_str,
                       completed_folds, num_folds)

    swanlab.finish()
