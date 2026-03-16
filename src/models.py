# src/models.py
import torch
import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.net(x)        # (B, hidden, T)
        h = h[:, :, -1]        # last timestep
        out = self.head(h).squeeze(-1)
        return out