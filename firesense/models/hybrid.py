"""Hybrid CNN+CAFIM+Transformer architecture.

A dual-branch CNN stem extracts local fuel and weather features; CAFIM blocks
fuse them at two scales; the fused features are then routed through a
transformer back-end for long-range reasoning.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .firesense import CAFIM
from .transformer import SegFormerDecoder, TransformerStage, _init_weights


class _BranchCNN(nn.Module):
    def __init__(self, in_channels: int, dims: Sequence[int], wide_kernel: bool) -> None:
        super().__init__()
        k = 5 if wide_kernel else 3
        p = k // 2
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], k, padding=p, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[1], dims[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(x)
        return f1, self.stage2(f1)


class FireHybridCAFIM(nn.Module):
    """CNN stems (per modality) -> CAFIM x2 -> Transformer x2 -> SegFormer decoder."""

    def __init__(self, fuel_channels: int = 4, weather_channels: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        dims = [48, 96, 192, 384]
        self.fuel_branch = _BranchCNN(fuel_channels, dims[:2], wide_kernel=False)
        self.weather_branch = _BranchCNN(weather_channels, dims[:2], wide_kernel=True)

        self.cafim1 = CAFIM(dims[0])
        self.cafim2 = CAFIM(dims[1])

        self.trans3 = TransformerStage(dims[1] * 2, dims[2], num_blocks=2, num_heads=4, sr_ratio=2, patch_size=2, dropout=dropout)
        self.trans4 = TransformerStage(dims[2], dims[3], num_blocks=2, num_heads=8, sr_ratio=1, patch_size=2, dropout=dropout)

        decoder_dims = [dims[0] * 2, dims[1] * 2, dims[2], dims[3]]
        self.decoder = SegFormerDecoder(decoder_dims, decoder_dim=128)
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Conv2d(128, 1, 1)
        _init_weights(self)

    def forward(self, fuel: torch.Tensor, weather: torch.Tensor, mc_sampling: bool = False) -> torch.Tensor:
        f1, f2 = self.fuel_branch(fuel)
        w1, w2 = self.weather_branch(weather)
        skip1 = self.cafim1(f1, w1)
        skip2 = self.cafim2(f2, w2)
        f3 = self.trans3(skip2)
        f4 = self.trans4(f3)
        x = self.decoder([skip1, skip2, f3, f4], (fuel.shape[2], fuel.shape[3]))
        x = F.dropout(x, p=self.dropout.p, training=True) if mc_sampling else self.dropout(x)
        return self.head(x)
