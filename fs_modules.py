import torch
import numpy as np
from torcheeg.datasets.constants import DREAMER_CHANNEL_LOCATION_DICT

# =========================
# Gaussian adjacency
# =========================
def gaussian_adj():
    channel_list = list(DREAMER_CHANNEL_LOCATION_DICT.keys())

    # channel coordinates
    positions = np.array([DREAMER_CHANNEL_LOCATION_DICT[ch] for ch in channel_list])

    # pairwise distance matrix
    dist_matrix = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1
    )

    # gaussian-weighted adjacency
    adj_matrix = np.exp(-dist_matrix)
    np.fill_diagonal(adj_matrix, 0)

    return torch.tensor(adj_matrix, dtype=torch.float32)


# =========================
# Symmetric adjacency normalization
# =========================
def normalize_adjacency(A: torch.Tensor, eps=1e-6):
    deg = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + eps))
    return D_inv_sqrt @ A @ D_inv_sqrt