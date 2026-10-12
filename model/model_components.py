"""Shared neural-network building blocks for the manuscript model pipeline.

The discriminator and stage-two classifier import these encoders directly so
their raster normalization, temporal masking, and vector encoding behavior do
not diverge. PyTorch is optional at import time to allow lightweight CLI and
documentation inspection; constructing a model still requires PyTorch.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "DataLoader",
    "Dataset",
    "F",
    "ImageEncoder",
    "MaskedSequenceEncoder",
    "PhenologyEncoder",
    "TORCH_IMPORT_ERROR",
    "Tensor",
    "make_spatial_normalization",
    "nn",
    "torch",
]

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
        """Minimal CUDA namespace used when PyTorch is unavailable."""

        @staticmethod
        def is_available() -> bool:
            """Report that CUDA cannot be used without PyTorch."""
            return False

    class _MissingTorch:
        """Import-time placeholder for the small torch surface used by CLIs."""

        Tensor = object
        cuda = _MissingCuda()

        @staticmethod
        def no_grad() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            """Return an identity decorator for import-only code paths."""

            def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
                return function

            return decorator

    class _MissingModule:
        """Placeholder base class that keeps module definitions importable."""

    class _MissingNN:
        """Placeholder namespace exposing only ``Module``."""

        Module = _MissingModule

    class _MissingDataset:
        """Placeholder dataset base class for import-only code paths."""

    torch = _MissingTorch()
    nn = _MissingNN()
    F = None
    DataLoader = None
    Dataset = _MissingDataset
    Tensor = object


def make_spatial_normalization(
    channels: int,
    normalization: str,
    *,
    max_groups: int = 8,
) -> Any:
    """Construct the requested 2-D normalization layer.

    Group normalization uses the largest divisor of ``channels`` no greater
    than ``max_groups``. This is the normalization used by the final NAIP, CHM,
    and NAIP–CHM interaction encoders. Batch normalization remains available
    for compatibility and controlled comparisons.
    """
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError("PyTorch is required to construct model layers") from TORCH_IMPORT_ERROR
    if channels <= 0:
        raise ValueError("channels must be positive")
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    if normalization == "group":
        groups = min(max_groups, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown image normalization: {normalization!r}")


class ImageEncoder(nn.Module):
    """Encode a fixed-size raster modality.

    The normalization is explicit because historical comparison models used
    batch normalization, whereas the final manuscript raster encoders use
    group normalization. The generic default remains ``batch`` for backward
    compatibility; frozen final configurations always pass ``group``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        dropout: float,
        normalization: str = "batch",
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            make_spatial_normalization(32, normalization),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            make_spatial_normalization(64, normalization),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            make_spatial_normalization(hidden_dim, normalization),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaskedSequenceEncoder(nn.Module):
    """Encode and mask a variable-length environmental sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
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

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
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
