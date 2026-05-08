import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from model import FeatureExtractor


FRAME_PATTERN = re.compile(r"RGB-(\d+)-(\d+)-(\d+)-(\d+)\.\w+")


def parse_frame_timestamp(filename):
    m = FRAME_PATTERN.match(os.path.basename(filename))
    if m:
        h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600000 + mi * 60000 + s * 1000 + ms
    return 0


def remap_frame_paths(obj, config):
    """
    Recursively remap hardcoded /home/gm paths to the correct config path.
    Handles nested dicts and lists containing frame paths.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "frame_paths" and isinstance(value, list):
                obj[key] = [
                    path.replace("/home/gm/dataset/mintpain", config.preprocessed_dir.replace("/rgb_preprocessed", ""))
                    if isinstance(path, str) else path
                    for path in value
                ]
            elif isinstance(value, (dict, list)):
                remap_frame_paths(value, config)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                remap_frame_paths(item, config)


def normalize_subject_name(name):
    m = re.match(r"Sub(\d+)", name)
    if m:
        num = int(m.group(1))
        return f"Subject_{num:02d}"
    return name.replace(" ", "_")


def load_loso_splits(config):
    with open(config.loso_splits_path, "rb") as f:
        loso_splits = pickle.load(f)
    # Remap hardcoded /home/gm paths to current config paths
    remap_frame_paths(loso_splits, config)
    return loso_splits


def collect_sweep_frames_from_samples(samples):
    sweep_frames = {}
    for sample in samples:
        sweep_id = sample["sweep_id"]
        subject_id = sample["subject_id"]
        key = (subject_id, sweep_id)
        if key not in sweep_frames:
            sweep_frames[key] = {"frames": [], "label": sample["label"]}
        for fp in sample["frame_paths"]:
            if fp not in sweep_frames[key]["frames"]:
                sweep_frames[key]["frames"].append(fp)
    for key in sweep_frames:
        sweep_frames[key]["frames"].sort(key=lambda x: parse_frame_timestamp(x))
    return sweep_frames


def generate_5frame_windows(sweep_frames, window_size=5, slide_step=5):
    windows = []
    for (subject_id, sweep_id), info in sweep_frames.items():
        frames = info["frames"]
        label = info["label"]
        n_frames = len(frames)
        if n_frames < window_size:
            continue
        n_windows = (n_frames - window_size) // slide_step + 1
        for i in range(n_windows):
            start = i * slide_step
            end = start + window_size
            window_frames = frames[start:end]
            sample_id = f"{subject_id}_{sweep_id}_Win5_{i:03d}"
            windows.append({
                "sample_id": sample_id,
                "subject_id": subject_id,
                "sweep_id": sweep_id,
                "frame_paths": window_frames,
                "label": label,
            })
    return windows


class FrameDataset(Dataset):
    def __init__(self, frame_paths, labels=None, transform=None):
        self.frame_paths = frame_paths
        self.labels = labels if labels is not None else [0] * len(frame_paths)
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_unique_frames_from_samples(samples):
    frame_paths = []
    frame_labels = []
    seen = set()
    for sample in samples:
        for fp in sample["frame_paths"]:
            if fp not in seen:
                seen.add(fp)
                frame_paths.append(fp)
                frame_labels.append(sample["label"])
    return frame_paths, frame_labels


def undersample_to_balance(frame_paths, frame_labels, num_classes=5):
    label_to_indices = {i: [] for i in range(num_classes)}
    for idx, lbl in enumerate(frame_labels):
        label_to_indices[lbl].append(idx)
    min_count = min(len(indices) for indices in label_to_indices.values())
    balanced_indices = []
    for lbl in range(num_classes):
        indices = label_to_indices[lbl]
        np.random.seed(42)
        selected = np.random.choice(indices, size=min_count, replace=False)
        balanced_indices.extend(selected.tolist())
    np.random.shuffle(balanced_indices)
    balanced_paths = [frame_paths[i] for i in balanced_indices]
    balanced_labels = [frame_labels[i] for i in balanced_indices]
    return balanced_paths, balanced_labels


def finetune_feature_extractor(config, train_frame_paths, train_labels, fold_idx):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    transform = get_transform(train=True)

    if config.undersample:
        train_frame_paths, train_labels = undersample_to_balance(
            train_frame_paths, train_labels, num_classes=config.num_classes
        )
        print(f"  Undersampled to {len(train_frame_paths)} frames (balanced across {config.num_classes} classes)")

    dataset = FrameDataset(train_frame_paths, train_labels, transform=transform)
    # 降低 num_workers 以避免多进程问题，使用 pin_memory 加速数据传输
    loader = DataLoader(dataset, batch_size=config.feature_extractor_batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = FeatureExtractor(
        num_classes=config.num_classes,
        bottleneck_dim=config.bottleneck_dim,
        vggface_weights_path=config.vggface_weights_path,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.feature_extractor_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_loss = float("inf")
    patience = 5
    patience_counter = 0

    print(f"  Starting fine-tuning for fold {fold_idx}...")
    for epoch in range(config.feature_extractor_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        with tqdm(loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{config.feature_extractor_epochs}", leave=False) as pbar:
            for imgs, lbls in pbar:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += lbls.size(0)
                correct += predicted.eq(lbls).sum().item()
                batch_count += 1
                # 更新进度条显示
                avg_loss = total_loss / batch_count
                acc = 100. * correct / total if total > 0 else 0
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})
        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total if total > 0 else 0
        print(f"✓ Fold {fold_idx} | Epoch {epoch+1}/{config.feature_extractor_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_path = os.path.join(config.weights_dir, f"feature_extractor_fold{fold_idx:02d}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Best model saved (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break

    print(f"Saved best feature extractor for fold {fold_idx}")
    model.load_state_dict(torch.load(os.path.join(config.weights_dir, f"feature_extractor_fold{fold_idx:02d}.pth"), map_location=device))
    return model


def extract_4d_features(config, model, frame_paths, batch_size=48):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    transform = get_transform(train=False)
    model.eval()

    dataset = FrameDataset(frame_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting 4D features"):
            imgs = imgs.to(device)
            feats = model(imgs, return_features=True)
            all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def extract_features(config):
    loso_splits = load_loso_splits(config)
    fold_names = sorted(loso_splits.keys())

    for fold_idx, fold_name in enumerate(fold_names):
        print(f"\n{'='*50}")
        print(f"Processing {fold_name} ({fold_idx + 1}/{len(fold_names)})")
        print(f"{'='*50}")

        fold_data = loso_splits[fold_name]
        train_samples = fold_data["train_samples"]
        test_samples = fold_data["test_samples"]
        test_subject = fold_data["test_subject"]

        print(f"Test subject: {test_subject}")
        print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

        train_sweep_frames = collect_sweep_frames_from_samples(train_samples)
        test_sweep_frames = collect_sweep_frames_from_samples(test_samples)

        train_windows = generate_5frame_windows(train_sweep_frames, window_size=config.sequence_length, slide_step=5)
        test_windows = generate_5frame_windows(test_sweep_frames, window_size=config.sequence_length, slide_step=5)

        print(f"Generated {len(train_windows)} train windows, {len(test_windows)} test windows (5-frame)")

        train_frame_paths, train_labels = get_unique_frames_from_samples(train_windows)
        test_frame_paths, test_labels = get_unique_frames_from_samples(test_windows)

        all_frame_paths = list(set(train_frame_paths) | set(test_frame_paths))
        all_frame_paths.sort()

        model = finetune_feature_extractor(config, train_frame_paths, train_labels, fold_idx)

        print("Extracting 4D features for all frames...")
        all_features_4d = extract_4d_features(config, model, all_frame_paths)

        frame_to_feat = {fp: feat for fp, feat in zip(all_frame_paths, all_features_4d)}

        train_features_4d = np.array([frame_to_feat[fp] for fp in train_frame_paths])
        test_features_4d = np.array([frame_to_feat[fp] for fp in test_frame_paths])

        pca = PCA(n_components=config.pca_dim)
        train_features_3d = pca.fit_transform(train_features_4d)
        test_features_3d = pca.transform(test_features_4d)

        train_feat_map = {fp: feat for fp, feat in zip(train_frame_paths, train_features_3d)}
        test_feat_map = {fp: feat for fp, feat in zip(test_frame_paths, test_features_3d)}

        fold_output_dir = os.path.join(config.features_3d_dir, fold_name)
        os.makedirs(fold_output_dir, exist_ok=True)

        def save_windows_to_npy(windows, feat_map, output_prefix):
            for window in tqdm(windows, desc=f"Saving {output_prefix} sequences"):
                frame_paths = window["frame_paths"]
                label = window["label"]
                sample_id = window["sample_id"]
                feats = []
                for fp in frame_paths:
                    if fp in feat_map:
                        feats.append(feat_map[fp])
                    else:
                        feats.append(np.zeros(config.pca_dim, dtype=np.float32))
                if len(feats) == config.sequence_length:
                    feats_arr = np.array(feats, dtype=np.float32)
                    np.save(
                        os.path.join(fold_output_dir, f"{output_prefix}_{sample_id}.npy"),
                        {"features": feats_arr, "label": label, "sample_id": sample_id},
                        allow_pickle=True,
                    )

        save_windows_to_npy(train_windows, train_feat_map, "train")
        save_windows_to_npy(test_windows, test_feat_map, "test")

        pca_save_path = os.path.join(config.features_3d_dir, f"pca_{fold_name}.pkl")
        with open(pca_save_path, "wb") as f:
            pickle.dump(pca, f)

        print(f"Fold {fold_name} features saved to {fold_output_dir}")

    print("\nFeature extraction and PCA complete for all folds.")
