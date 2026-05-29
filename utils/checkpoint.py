"""Checkpoint utilities for resumable training.

Supports saving and loading training state for both:
1. Feature extraction (per-fold VGGFace fine-tuning)
2. Ensemble training (per-fold EDLM training)
"""

import os
import json
import glob
import torch
import numpy as np


def get_checkpoint_dir(config, stage="feature_extraction"):
    """Get checkpoint directory for a specific training stage."""
    ckpt_dir = os.path.join(config.output_dir, "checkpoints", stage)
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def save_checkpoint(state, filepath):
    """Save training checkpoint.
    
    Args:
        state: dict containing all state to save
        filepath: path to save the checkpoint
    """
    torch.save(state, filepath)
    print(f"  💾 Checkpoint saved: {os.path.basename(filepath)}")


def load_checkpoint(filepath, device="cpu"):
    """Load training checkpoint.
    
    Args:
        filepath: path to the checkpoint file
        device: device to map tensors to
        
    Returns:
        dict containing saved state, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    print(f"  📂 Loading checkpoint: {os.path.basename(filepath)}")
    return torch.load(filepath, map_location=device)


def save_feature_extraction_checkpoint(config, fold_idx, epoch, model, optimizer, scheduler, scaler, best_loss, patience_counter):
    """Save feature extraction checkpoint."""
    ckpt_dir = get_checkpoint_dir(config, "feature_extraction")
    filepath = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_epoch{epoch:03d}.pth")
    
    state = {
        "fold_idx": fold_idx,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_loss": best_loss,
        "patience_counter": patience_counter,
    }
    save_checkpoint(state, filepath)
    
    # Also save as "latest" for easy resume
    latest_path = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_latest.pth")
    save_checkpoint(state, latest_path)
    
    return filepath


def load_feature_extraction_checkpoint(config, fold_idx, device="cpu"):
    """Load feature extraction checkpoint for a specific fold.
    
    Returns:
        tuple: (checkpoint_state, start_epoch) or (None, 0) if no checkpoint
    """
    ckpt_dir = get_checkpoint_dir(config, "feature_extraction")
    
    # Try latest checkpoint first
    latest_path = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_latest.pth")
    ckpt = load_checkpoint(latest_path, device)
    
    if ckpt is not None:
        start_epoch = ckpt["epoch"] + 1  # Resume from next epoch
        print(f"  Resuming fold {fold_idx} from epoch {start_epoch}")
        return ckpt, start_epoch
    
    return None, 0


def save_ensemble_checkpoint(config, fold_idx, epoch, model, optimizer, scheduler, best_val_loss, patience_counter, fold_results=None):
    """Save ensemble training checkpoint."""
    ckpt_dir = get_checkpoint_dir(config, "ensemble")
    filepath = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_epoch{epoch:03d}.pth")
    
    state = {
        "fold_idx": fold_idx,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "fold_results": fold_results or {},
    }
    save_checkpoint(state, filepath)
    
    # Also save as "latest" for easy resume
    latest_path = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_latest.pth")
    save_checkpoint(state, latest_path)
    
    return filepath


def load_ensemble_checkpoint(config, fold_idx, device="cpu"):
    """Load ensemble training checkpoint for a specific fold.
    
    Returns:
        tuple: (checkpoint_state, start_epoch) or (None, 0) if no checkpoint
    """
    ckpt_dir = get_checkpoint_dir(config, "ensemble")
    
    # Try latest checkpoint first
    latest_path = os.path.join(ckpt_dir, f"fold{fold_idx:02d}_latest.pth")
    ckpt = load_checkpoint(latest_path, device)
    
    if ckpt is not None:
        start_epoch = ckpt["epoch"] + 1  # Resume from next epoch
        print(f"  Resuming fold {fold_idx} from epoch {start_epoch}")
        return ckpt, start_epoch
    
    return None, 0


def save_progress(config, stage, completed_folds, all_results=None):
    """Save overall training progress.
    
    Args:
        config: Config object
        stage: "feature_extraction" or "ensemble"
        completed_folds: list of completed fold names
        all_results: dict of results per fold
    """
    progress_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(progress_dir, exist_ok=True)
    
    progress_file = os.path.join(progress_dir, f"{stage}_progress.json")
    progress = {
        "stage": stage,
        "completed_folds": completed_folds,
        "all_results": all_results or {},
    }
    
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"  📊 Progress saved: {len(completed_folds)} folds completed")


def load_progress(config, stage):
    """Load overall training progress.
    
    Returns:
        tuple: (completed_folds, all_results) or ([], {}) if no progress file
    """
    progress_dir = os.path.join(config.output_dir, "checkpoints")
    progress_file = os.path.join(progress_dir, f"{stage}_progress.json")
    
    if not os.path.exists(progress_file):
        return [], {}
    
    with open(progress_file, "r") as f:
        progress = json.load(f)
    
    completed_folds = progress.get("completed_folds", [])
    all_results = progress.get("all_results", {})
    print(f"  📊 Loaded progress: {len(completed_folds)} folds completed")
    return completed_folds, all_results


def get_latest_fold_checkpoint(config, stage):
    """Find the latest fold checkpoint across all folds.
    
    Returns:
        tuple: (fold_idx, checkpoint_path) or (None, None) if no checkpoint
    """
    ckpt_dir = get_checkpoint_dir(config, stage)
    
    # Search for latest checkpoints
    pattern = os.path.join(ckpt_dir, "fold*_latest.pth")
    ckpt_files = sorted(glob.glob(pattern))
    
    if not ckpt_files:
        return None, None
    
    # Return the last one (highest fold index)
    latest_file = ckpt_files[-1]
    
    # Extract fold index from filename
    basename = os.path.basename(latest_file)
    fold_idx = int(basename.replace("fold", "").replace("_latest.pth", ""))
    
    return fold_idx, latest_file


def cleanup_old_checkpoints(config, stage, keep_last_n=2):
    """Remove old checkpoint files to save disk space.
    
    Args:
        config: Config object
        stage: "feature_extraction" or "ensemble"
        keep_last_n: number of recent checkpoints to keep per fold
    """
    ckpt_dir = get_checkpoint_dir(config, stage)
    
    # Group by fold
    fold_files = {}
    for f in glob.glob(os.path.join(ckpt_dir, "fold*_epoch*.pth")):
        basename = os.path.basename(f)
        fold_idx = int(basename.split("_")[0].replace("fold", ""))
        if fold_idx not in fold_files:
            fold_files[fold_idx] = []
        fold_files[fold_idx].append(f)
    
    # Keep only last N per fold
    for fold_idx, files in fold_files.items():
        files.sort()
        if len(files) > keep_last_n:
            for old_file in files[:-keep_last_n]:
                os.remove(old_file)
                print(f"  🗑️  Removed old checkpoint: {os.path.basename(old_file)}")