import os
import numpy as np
import torch
from torch.utils.data import Dataset

# =========================
# Temporal Dataset
# =========================
class temporal_dataset(Dataset):
    def __init__(self, subject_dir, fold_id=0, mode='train'):
        assert mode in ['train', 'val']

        fold_dir = os.path.join(subject_dir, f'fold_{fold_id}')
        self.seg_path = os.path.join(fold_dir, f'{mode}_segments.npy')
        self.label_path = os.path.join(fold_dir, f'{mode}_labels.npy')

        self.segments = np.load(self.seg_path)
        self.labels = np.load(self.label_path)

        assert len(self.labels) == len(self.segments)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # load segment (T, C) or (C, T)
        segment = torch.tensor(self.segments[idx], dtype=torch.float32)

        # ensure (T, C) format
        if segment.shape[0] == 14:
            segment = segment.transpose(0, 1)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'segment': segment,
            'label': label
        }


# =========================
# Frequency Dataset
# =========================
class frequency_dataset(Dataset):
    def __init__(self, subject_dir, fold_id=0, mode='train'):
        assert mode in ['train', 'val']

        fold_dir = os.path.join(subject_dir, f'fold_{fold_id}')
        self.de_path = os.path.join(fold_dir, f'{mode}_de_grids.npy')
        self.label_path = os.path.join(fold_dir, f'{mode}_labels.npy')

        self.de_grids = np.load(self.de_path)
        self.labels = np.load(self.label_path)

        assert len(self.labels) == len(self.de_grids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # load frequency grid (C, H, W)
        de_grid = torch.tensor(self.de_grids[idx], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'de_grid': de_grid,
            'label': label
        }


# =========================
# Multi-modal Dataset
# =========================
class multi_dataset(Dataset):
    def __init__(self, subject_dir, fold_id=0, mode='train'):
        assert mode in ['train', 'val']

        fold_dir = os.path.join(subject_dir, f'fold_{fold_id}')
        self.de_path = os.path.join(fold_dir, f'{mode}_de_grids.npy')
        self.seg_path = os.path.join(fold_dir, f'{mode}_segments.npy')
        self.label_path = os.path.join(fold_dir, f'{mode}_labels.npy')

        self.de_grids = np.load(self.de_path)
        self.segments = np.load(self.seg_path)
        self.labels = np.load(self.label_path)

        assert len(self.de_grids) == len(self.labels) == len(self.segments)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # load inputs
        de_grid = torch.tensor(self.de_grids[idx], dtype=torch.float32)
        segment = torch.tensor(self.segments[idx], dtype=torch.float32)

        # ensure (T, C) format
        if segment.shape[0] == 14:
            segment = segment.transpose(0, 1)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'de_grid': de_grid,
            'segment': segment,
            'label': label
        }