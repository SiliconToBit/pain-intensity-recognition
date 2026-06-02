"""Dataset for loading frame sequences from image files."""

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_train_transforms():
    """Training data augmentation.

    Pipeline:
        1. RandomResizedCrop: 随机裁剪后缩放，增加尺度变化
        2. RandomHorizontalFlip: 水平翻转
        3. ColorJitter: 亮度/对比度/饱和度随机变化，模拟光照变化
        4. RandomAffine: 旋转±10° + 平移±5%
        5. ToTensor + ImageNet normalization
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms():
    """Test data transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FrameSequenceDataset(Dataset):
    """Dataset that loads image sequences on-the-fly.

    Each sample is a window of T consecutive frames with a pain label.
    Images are loaded from disk and transformed on access.
    """

    def __init__(self, windows, transform=None):
        """
        Args:
            windows: list of dicts, each with keys:
                - "frame_paths": list of T image file paths
                - "label": int pain intensity label
            transform: torchvision transform to apply to each frame
        """
        self.windows = windows
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        frame_paths = window["frame_paths"]
        label = window["label"]

        # Load and transform each frame
        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack to (T, C, H, W)
        sequence = torch.stack(frames, dim=0)

        return sequence, torch.tensor(label, dtype=torch.long)


def undersample_windows(windows, num_classes=5):
    """Undersample windows to balance classes.

    Args:
        windows: list of window dicts with "label" key
        num_classes: number of classes

    Returns:
        balanced: list of undersampled window dicts
    """
    label_to_windows = {i: [] for i in range(num_classes)}
    for w in windows:
        label_to_windows[w["label"]].append(w)

    min_count = min(len(ws) for ws in label_to_windows.values())

    balanced = []
    np.random.seed(42)
    for lbl in range(num_classes):
        ws = label_to_windows[lbl]
        selected = np.random.choice(len(ws), size=min_count, replace=False)
        balanced.extend([ws[i] for i in selected])

    np.random.shuffle(balanced)
    return balanced


def compute_class_weights(windows, num_classes=5, mode="inverse"):
    """Compute class weights for imbalanced dataset.

    Args:
        windows: list of window dicts with "label" key
        num_classes: number of classes
        mode: "inverse" → N / (N_c * K)
              "sqrt_inverse" → sqrt(N / (N_c * K))
              "none" → all ones

    Returns:
        torch.FloatTensor of shape (num_classes,)
    """
    if mode == "none":
        return torch.ones(num_classes)

    counts = Counter(w["label"] for w in windows)
    total = sum(counts.values())

    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        if mode == "inverse":
            w = total / (c * num_classes)
        elif mode == "sqrt_inverse":
            w = (total / (c * num_classes)) ** 0.5
        else:
            w = 1.0
        weights.append(w)

    return torch.FloatTensor(weights)
