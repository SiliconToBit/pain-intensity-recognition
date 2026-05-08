import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FoldSequenceDataset(Dataset):
    def __init__(self, features_dir, prefix="train"):
        self.features_dir = features_dir
        self.prefix = prefix
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        if not os.path.exists(self.features_dir):
            return samples
        for fname in sorted(os.listdir(self.features_dir)):
            if fname.startswith(self.prefix) and fname.endswith(".npy"):
                fpath = os.path.join(self.features_dir, fname)
                data = np.load(fpath, allow_pickle=True).item()
                samples.append({
                    "features": data["features"].astype(np.float32),
                    "label": int(data["label"]),
                    "sample_id": data["sample_id"],
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "features": torch.FloatTensor(sample["features"]),
            "label": torch.LongTensor([sample["label"]])[0],
            "sample_id": sample["sample_id"],
        }
