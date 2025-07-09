# utils/dataset_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Generic Log-Mel Feature Dataset
# -----------------------------
class LogMelDataset(Dataset):
    def __init__(self, feature_dir, label=None):
        self.feature_paths = [
            os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".npy")
        ]
        self.label = label  # Optional fixed label (e.g., 0 for noise, 1 for speech)

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feat_path = self.feature_paths[idx]
        features = np.load(feat_path)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, time]

        if self.label is not None:
            label = torch.tensor(self.label, dtype=torch.float32)
        else:
            label = 0  # default dummy

        return features, label

# -----------------------------
# ASR Dataset with Transcript
# -----------------------------
class ASRDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_paths = [
            os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feat_path = self.feature_paths[idx]
        txt_path = feat_path.replace(".npy", ".txt")

        features = np.load(feat_path)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                transcript = f.read().strip()
        else:
            transcript = ""

        return features, transcript

# -----------------------------
# Utility to create DataLoader
# -----------------------------
def get_loader(dataset_type, feature_dir, batch_size=16, shuffle=True, label=None):
    if dataset_type == "vad" or dataset_type == "trigger":
        dataset = LogMelDataset(feature_dir, label=label)
    elif dataset_type == "asr":
        dataset = ASRDataset(feature_dir)
    else:
        raise ValueError("Unsupported dataset type")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)