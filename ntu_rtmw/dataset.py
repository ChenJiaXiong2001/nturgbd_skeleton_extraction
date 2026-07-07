import csv

import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    def __init__(self, manifest, split, frames=64):
        with open(manifest, newline="", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.frames = frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        data = np.load(row["path"], allow_pickle=False)
        keypoints = data["keypoints"].astype(np.float32)
        scores = data["scores"].astype(np.float32)
        x = self.sample(keypoints, scores)
        return torch.from_numpy(x), torch.tensor(int(row["label"]), dtype=torch.long)

    def sample(self, keypoints, scores):
        total = keypoints.shape[0]
        if total == 0:
            return np.zeros((self.frames, 2 * 133 * 3), dtype=np.float32)
        index = np.linspace(0, total - 1, self.frames).round().astype(np.int64)
        k = keypoints[index]
        s = scores[index][..., None]
        valid = np.isfinite(k)
        if valid.any():
            mean = np.nanmean(k, axis=(0, 1, 2), keepdims=True)
            scale = np.nanstd(k, axis=(0, 1, 2), keepdims=True) + 1e-6
            k = (k - mean) / scale
        k = np.nan_to_num(k, nan=0.0)
        x = np.concatenate([k, s], axis=-1)
        return x.reshape(self.frames, -1).astype(np.float32)
