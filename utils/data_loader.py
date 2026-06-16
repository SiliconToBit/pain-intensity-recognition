"""Data loading utilities for pain intensity recognition.

Handles:
    - Scanning the preprocessed dataset directory
    - Building Leave-One-Subject-Out (LOSO) cross-validation folds
    - Generating overlapping frame windows from video sweeps
    - Binary label remapping
"""

import os
import re

import numpy as np


FRAME_PATTERN = re.compile(r"RGB-(\d+)-(\d+)-(\d+)-(\d+)\.\w+")


def parse_frame_timestamp(filename):
    """Parse timestamp from frame filename for sorting.

    Expected format: RGB-HH-MM-SS-MS.ext
    Returns milliseconds since midnight.
    """
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
