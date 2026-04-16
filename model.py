import torch
import torch.nn as nn
from s4_modules import S4D
from fs_modules import normalize_adjacency

# =========================
# S4 Encoder (Time Series)
# Reference: https://github.com/state-spaces/s4
# =========================
class S4encoder(nn.Module):
    def __init__(self, d_input=14, d_output=256, d_model=256, n_layers=4, dropout=0.3, prenorm=False):
        super().__init__()
        self.prenorm = prenorm

        # input projection (channel → hidden)
        self.encoder = nn.Linear(d_input, d_model)

        # stacked S4 layers
        self.s4_layers = nn.ModuleList([
            S4D(d_model, dropout=dropout, transposed=True, lr=0.001)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout1d(dropout) for _ in range(n_layers)])

        # output projection
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        # x: (B, T, C)
        x = self.encoder(x)

        # S4 expects (B, H, T)
        x = x.transpose(-1, -2)

        # residual S4 blocks
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x

            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = dropout(z)

            x = z + x

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        # back to (B, T, H)
        x = x.transpose(-1, -2)

        # global temporal pooling
        x = x.mean(dim=1)

        x = self.decoder(x)
        return x


# =========================
# S4 Classifier
# =========================
class S4classifier(nn.Module):
    def __init__(self, s4_config):
        super().__init__()
        self.s4_model = S4encoder(**s4_config)

        # feature normalization
        self.norm = nn.LayerNorm(256)

        # binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, eeg_tem):
        x = self.s4_model(eeg_tem)
        x = self.norm(x)
        return self.classifier(x)


# =========================
# Frequency-Spatial Encoder
# =========================
class FSencoder(nn.Module):
    def __init__(self, in_channels=5, dropout=0.3, initial_adj=None, adj_train=None):
        super().__init__()

        # adjacency matrix (trainable or fixed)
        if adj_train:
            self.A_raw = nn.Parameter(initial_adj.clone().detach())
        else:
            self.register_buffer("A_raw", initial_adj.clone().detach())

        # CNN backbone
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )

        # attention branch
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(2)
        )

        # CNN → node features
        self.cnn_to_node = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 14 * 16),
            nn.ReLU(inplace=True)
        )

        # node → feature embedding
        self.node_to_feat = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # feature extraction + attention fusion
        trunk = self.trunk(x)
        attn = self.attn(x)
        x = trunk * attn

        # reshape to node features
        x = self.cnn_to_node(x)
        x = x.view(-1, 14, 16)

        # graph filtering (A @ X)
        A = normalize_adjacency(self.A_raw)
        x = torch.matmul(A, x)

        # node aggregation
        x = self.node_to_feat(x)
        x = x.mean(dim=1)

        return x


# =========================
# FS Classifier
# =========================
class FSclassifier(nn.Module):
    def __init__(self, fs_config):
        super().__init__()
        self.fs_model = FSencoder(**fs_config)

        # feature normalization
        self.norm = nn.LayerNorm(256)

        # binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, eeg_freq):
        x = self.fs_model(eeg_freq)
        x = self.norm(x)
        return self.classifier(x)


# =========================
# Multi-domain Classifier
# =========================
class Multiclassifier(nn.Module):
    def __init__(self, s4_config, fs_config):
        super().__init__()

        self.s4_model = S4encoder(**s4_config)
        self.fs_model = FSencoder(**fs_config)

        self.norm_time = nn.LayerNorm(256)
        self.norm_freq = nn.LayerNorm(256)
        self.norm_combined = nn.LayerNorm(512)

        # fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, eeg_tem, eeg_freq):
        # temporal branch
        x_time = self.norm_time(self.s4_model(eeg_tem))

        # frequency branch
        x_freq = self.norm_freq(self.fs_model(eeg_freq))

        # feature fusion
        x = torch.cat([x_time, x_freq], dim=1)
        x = self.norm_combined(x)

        return self.classifier(x)