import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torcheeg.datasets.constants import DREAMER_CHANNEL_LOCATION_DICT


# =========================
# Channel positions
# =========================
channel_names = list(DREAMER_CHANNEL_LOCATION_DICT.keys())
positions = np.array([DREAMER_CHANNEL_LOCATION_DICT[ch] for ch in channel_names])
num_channels = len(positions)


# =========================
# Distance-based adjacency
# =========================
dist_matrix = np.linalg.norm(
    positions[:, None, :] - positions[None, :, :],
    axis=-1
)


# binary adjacency (threshold-based)
distance_threshold = 3.5
binary_adj = (dist_matrix <= distance_threshold).astype(int)
np.fill_diagonal(binary_adj, 0)


# gaussian weighted adjacency
gaussian_adj = np.exp(-dist_matrix)
np.fill_diagonal(gaussian_adj, 0)


# =========================
# Learned adjacency (from file)
# =========================
learned_adj = np.load("path/to/learned_adj.npy")


# =========================
# Visualization
# =========================
plt.figure(figsize=(24, 8))


# (1) binary adjacency
plt.subplot(1, 3, 1)
sns.heatmap(
    binary_adj,
    annot=True,
    square=True,
    cmap="Blues",
    xticklabels=channel_names,
    yticklabels=channel_names,
    fmt="d"
)
plt.title("Binary Adjacency")


# (2) gaussian adjacency
plt.subplot(1, 3, 2)
sns.heatmap(
    gaussian_adj,
    annot=True,
    square=True,
    cmap="YlGnBu",
    xticklabels=channel_names,
    yticklabels=channel_names,
    fmt=".2f"
)
plt.title("Gaussian Adjacency")


# (3) learned adjacency
plt.subplot(1, 3, 3)
sns.heatmap(
    learned_adj,
    annot=True,
    square=True,
    cmap="coolwarm",
    xticklabels=channel_names,
    yticklabels=channel_names,
    fmt=".2f"
)
plt.title("Learned Adjacency")