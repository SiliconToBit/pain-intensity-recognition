"""Checkpoint utilities for resumable training."""

import os
import json
import glob
import torch


def get_checkpoint_dir(config):
    """Get checkpoint directory."""
    ckpt_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def save_checkpoint(config, fold_idx, epoch, model, optimizer, scheduler,
                    best_val_loss, patience_counter):
    """Save training checkpoint."""
    ckpt_dir = get_checkpoint_dir(config)
    filepath = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_epoch{epoch:03d}.pth")

    state = {
        "fold_idx": fold_idx,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "config": config.to_dict(),
    }
    torch.save(state, filepath)

    # Also save as "latest" for easy resume
    latest_path = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_latest.pth")
    torch.save(state, latest_path)

    return filepath


def load_checkpoint(config, fold_idx, device="cpu"):
    """Load training checkpoint for a specific fold.

    Returns:
        tuple: (checkpoint_state, start_epoch) or (None, 0) if no checkpoint
    """
    ckpt_dir = get_checkpoint_dir(config)
    latest_path = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_latest.pth")

    if not os.path.exists(latest_path):
        return None, 0

    print(f"  Loading checkpoint: {os.path.basename(latest_path)}")
    ckpt = torch.load(latest_path, map_location=device)

    # Validate config compatibility
    _validate_checkpoint_config(ckpt, config)

    start_epoch = ckpt["epoch"] + 1
    return ckpt, start_epoch


def _validate_checkpoint_config(ckpt, config):
    """Warn if checkpoint was saved with a different config."""
    saved = ckpt.get("config")
    if saved is None:
        print("  ⚠️  Checkpoint has no config snapshot — skipping validation")
        return

    # Keys that affect model architecture (mismatches break state_dict loading)
    arch_keys = [
        "num_classes", "backbone", "pretrained_source", "lstm_hidden_dim",
        "lstm_num_layers", "single_frame", "use_attention_pooling",
        "classifier_hidden_dim",
    ]
    current = config.to_dict()
    mismatches = []
    for key in arch_keys:
        if key in saved and key in current and saved[key] != current[key]:
            mismatches.append(f"    {key}: checkpoint={saved[key]}, current={current[key]}")

    if mismatches:
        print("  ⚠️  Config mismatch (checkpoint may be incompatible):")
        for line in mismatches:
            print(line)


def save_progress(config, stage, completed_folds):
    """Save overall training progress."""
    progress_dir = get_checkpoint_dir(config)
    progress_file = os.path.join(progress_dir, f"{stage}_progress.json")

    progress = {
        "stage": stage,
        "completed_folds": completed_folds,
    }
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"  Progress saved: {len(completed_folds)} folds completed")


def load_progress(config, stage):
    """Load overall training progress.

    Returns:
        tuple: (completed_folds, all_results) or ([], {}) if no progress file
    """
    progress_dir = get_checkpoint_dir(config)
    progress_file = os.path.join(progress_dir, f"{stage}_progress.json")

    if not os.path.exists(progress_file):
        return [], {}

    with open(progress_file, "r") as f:
        progress = json.load(f)

    completed_folds = progress.get("completed_folds", [])
    print(f"  Loaded progress: {len(completed_folds)} folds completed")
    return completed_folds, {}


def cleanup_old_checkpoints(config, keep_last_n=2):
    """Remove old checkpoint files to save disk space."""
    ckpt_dir = get_checkpoint_dir(config)

    fold_files = {}
    for f in glob.glob(os.path.join(ckpt_dir, "fold*_epoch*.pth")):
        basename = os.path.basename(f)
        fold_idx = int(basename.split("_")[0].replace("fold", ""))
        if fold_idx not in fold_files:
            fold_files[fold_idx] = []
        fold_files[fold_idx].append(f)

    for fold_idx, files in fold_files.items():
        files.sort()
        if len(files) > keep_last_n:
            for old_file in files[:-keep_last_n]:
                os.remove(old_file)
                print(f"  Removed old checkpoint: {os.path.basename(old_file)}")
