import os
import pickle
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from config import Config
from feature_extraction import (
    load_loso_splits, collect_sweep_frames_from_samples,
    generate_5frame_windows, get_unique_frames_from_samples,
    finetune_feature_extractor, extract_4d_features,
)
from model import FeatureExtractor
from train import train_and_evaluate


def save_windows_to_npy(windows, feat_map, output_prefix, output_dir, pca_dim):
    for window in tqdm(windows, desc=f"Saving {output_prefix} sequences"):
        frame_paths = window["frame_paths"]
        label = window["label"]
        sample_id = window["sample_id"]
        feats = []
        for fp in frame_paths:
            if fp in feat_map:
                feats.append(feat_map[fp])
            else:
                feats.append(np.zeros(pca_dim, dtype=np.float32))
        if len(feats) == len(frame_paths):
            feats_arr = np.array(feats, dtype=np.float32)
            np.save(
                os.path.join(output_dir, f"{output_prefix}_{sample_id}.npy"),
                {"features": feats_arr, "label": label, "sample_id": sample_id},
                allow_pickle=True,
            )


def extract_features_skip_existing(config):
    loso_splits = load_loso_splits(config)
    all_fold_names = sorted(loso_splits.keys())
    if config.num_folds is not None and config.num_folds > 0:
        fold_names = all_fold_names[:min(config.num_folds, len(all_fold_names))]
    else:
        fold_names = all_fold_names

    print(f"Using {len(fold_names)} folds (configured num_folds={config.num_folds}).")

    for fold_idx, fold_name in enumerate(fold_names):
        fold_output_dir = os.path.join(config.features_3d_dir, fold_name)
        weight_path = os.path.join(config.weights_dir, f"feature_extractor_fold{fold_idx:02d}.pth")

        existing_files = 0
        if os.path.exists(fold_output_dir):
            existing_files = len([f for f in os.listdir(fold_output_dir) if f.endswith(".npy")])

        if existing_files > 0 and os.path.exists(weight_path):
            print(f"\n{'='*50}")
            print(f"Skipping {fold_name} ({fold_idx + 1}/{len(fold_names)}) - {existing_files} features already exist")
            print(f"{'='*50}")
            continue

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

        train_windows = generate_5frame_windows(train_sweep_frames, window_size=config.sequence_length, slide_step=1)
        test_windows = generate_5frame_windows(test_sweep_frames, window_size=config.sequence_length, slide_step=1)

        print(f"Generated {len(train_windows)} train windows, {len(test_windows)} test windows (5-frame)")

        train_frame_paths, train_labels = get_unique_frames_from_samples(train_windows)
        test_frame_paths, test_labels = get_unique_frames_from_samples(test_windows)

        all_frame_paths = list(set(train_frame_paths) | set(test_frame_paths))
        all_frame_paths.sort()

        weight_size = os.path.getsize(weight_path) if os.path.exists(weight_path) else 0
        # VGG16 weights are typically ~500MB+; skip fine-tuning if weights exist
        is_vgg_weight = weight_size > 200e6

        if os.path.exists(weight_path) and existing_files == 0 and is_vgg_weight:
            print(f"Loading existing weights from {weight_path} ({weight_size/1e6:.0f}MB, skipping fine-tuning)...")
            device = torch.device(config.device if torch.cuda.is_available() else "cpu")
            model = FeatureExtractor(
                num_classes=config.num_classes,
                bottleneck_dim=config.bottleneck_dim,
                backbone=config.feature_backbone,
            ).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
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

        os.makedirs(fold_output_dir, exist_ok=True)

        save_windows_to_npy(train_windows, train_feat_map, "train", fold_output_dir, config.pca_dim)
        save_windows_to_npy(test_windows, test_feat_map, "test", fold_output_dir, config.pca_dim)

        pca_save_path = os.path.join(config.features_3d_dir, f"pca_{fold_name}.pkl")
        with open(pca_save_path, "wb") as f:
            pickle.dump(pca, f)

        print(f"Fold {fold_name} features saved to {fold_output_dir}")

    print("\nFeature extraction complete for all folds.")


if __name__ == "__main__":
    config = Config()
    print("Starting continued feature extraction (skipping completed folds)...")
    extract_features_skip_existing(config)

    print("\nStarting ensemble training and evaluation...")
    train_and_evaluate(config)

    print("Pipeline completed successfully.")
