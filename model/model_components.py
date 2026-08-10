"""Shared PyTorch runtime and neural-network components for active clean models."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    Tensor = torch.Tensor
    TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    TORCH_IMPORT_ERROR = exc

    class _MissingCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _MissingTorch:
        Tensor = object
        cuda = _MissingCuda()

        @staticmethod
        def no_grad():
            def decorator(function):
                return function

            return decorator

    class _MissingModule:
        pass

    class _MissingNN:
        Module = _MissingModule

    class _MissingDataset:
        pass

    torch = _MissingTorch()
    nn = _MissingNN()
    F = None
    DataLoader = None
    Dataset = _MissingDataset
    Tensor = object


class ImageEncoder(nn.Module):
    """Encode a fixed-size raster modality without changing legacy weights."""

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaskedSequenceEncoder(nn.Module):
    """Encode and mask a variable-length environmental sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        present = mask > 0
        x = self.input(values)
        encoded = self.encoder(x, src_key_padding_mask=~present)
        weights = present.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.out(pooled)


class PhenologyEncoder(nn.Module):
    """Encode a fixed-length environmental feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)
