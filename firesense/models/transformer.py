"""SegFormer-style transformer architectures and lightweight variants.

Defines the shared building blocks (PatchEmbedding, EfficientSelfAttention,
MixFFN, TransformerBlock, TransformerStage, SegFormerDecoder) and four
architectures used in the comparison:

* ``FireTransformer``           -- the full SegFormer-B0-scale model.
* ``FireTransformerSmall``      -- halved channel widths and block counts.
* ``FireTransformerRegularized``-- the small variant with stronger dropout
                                   and Cutout input augmentation.
* ``FireHybridNet``             -- two CNN stages followed by two transformer
                                   stages.

Self-attention is global by construction, so there is no need for the CAFIM
module that compensates for limited receptive fields in the CNN model. We do
keep a dual-branch input projection so fuel and weather start from independent
embeddings before the shared encoder mixes them.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class EfficientSelfAttention(nn.Module):
    """Self-attention with spatial reduction on Key/Value (SegFormer)."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            s = x.permute(0, 2, 1).reshape(b, c, h, w)
            s = self.sr(s).reshape(b, c, -1).permute(0, 2, 1)
            s = self.sr_norm(s)
            kv = self.kv(s).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    """FFN with an embedded 3x3 depthwise conv that injects positional info."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = F.gelu(self.dwconv(x))
        x = x.reshape(b, c, n).transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion, dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class TransformerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        sr_ratio: int,
        patch_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, sr_ratio, dropout=dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, h, w)
        x = self.norm(x)
        b, _, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w)


class SegFormerDecoder(nn.Module):
    """All-MLP decoder that aligns multi-scale features and fuses them."""

    def __init__(self, embed_dims: Sequence[int], decoder_dim: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(d, decoder_dim, 1),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True),
                )
                for d in embed_dims
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(embed_dims), decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: Sequence[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        aligned = []
        for feat, lat in zip(features, self.lateral):
            f = lat(feat)
            f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=True)
            aligned.append(f)
        return self.fuse(torch.cat(aligned, dim=1))


def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class _DualBranchProj(nn.Module):
    def __init__(self, fuel_channels: int, weather_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuel = nn.Sequential(
            nn.Conv2d(fuel_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.weather = nn.Sequential(
            nn.Conv2d(weather_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, fuel: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.fuel(fuel), self.weather(weather)], dim=1)


class _SegFormerBackbone(nn.Module):
    """Generic SegFormer-style backbone parameterised by stage configs."""

    def __init__(
        self,
        fuel_channels: int,
        weather_channels: int,
        proj_channels: int,
        embed_dims: Sequence[int],
        num_blocks: Sequence[int],
        num_heads: Sequence[int],
        sr_ratios: Sequence[int],
        patch_sizes: Sequence[int],
        decoder_dim: int,
        mc_dropout_p: float,
        encoder_dropout: float,
    ) -> None:
        super().__init__()
        self.proj = _DualBranchProj(fuel_channels, weather_channels, proj_channels)
        in_c = proj_channels * 2
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(
                TransformerStage(
                    in_channels=in_c if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    num_blocks=num_blocks[i],
                    num_heads=num_heads[i],
                    sr_ratio=sr_ratios[i],
                    patch_size=patch_sizes[i],
                    dropout=encoder_dropout,
                )
            )
        self.decoder = SegFormerDecoder(embed_dims, decoder_dim=decoder_dim)
        self.dropout = nn.Dropout(p=mc_dropout_p)
        self.head = nn.Conv2d(decoder_dim, 1, 1)
        _init_weights(self)

    def encode(self, fuel: torch.Tensor, weather: torch.Tensor):
        x = self.proj(fuel, weather)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats

    def forward(self, fuel: torch.Tensor, weather: torch.Tensor, mc_sampling: bool = False) -> torch.Tensor:
        feats = self.encode(fuel, weather)
        x = self.decoder(feats, (fuel.shape[2], fuel.shape[3]))
        x = F.dropout(x, p=self.dropout.p, training=True) if mc_sampling else self.dropout(x)
        return self.head(x)


class FireTransformer(_SegFormerBackbone):
    """SegFormer-B0-scale transformer with dual-branch input projection."""

    def __init__(self, fuel_channels: int = 4, weather_channels: int = 8, dropout: float = 0.2) -> None:
        super().__init__(
            fuel_channels=fuel_channels,
            weather_channels=weather_channels,
            proj_channels=32,
            embed_dims=[64, 128, 256, 512],
            num_blocks=[2, 2, 2, 2],
            num_heads=[1, 2, 4, 8],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[4, 2, 2, 2],
            decoder_dim=256,
            mc_dropout_p=0.3,
            encoder_dropout=dropout,
        )


class FireTransformerSmall(_SegFormerBackbone):
    """Halved channels and block counts; ~2.5M parameters."""

    def __init__(self, fuel_channels: int = 4, weather_channels: int = 8, dropout: float = 0.2) -> None:
        super().__init__(
            fuel_channels=fuel_channels,
            weather_channels=weather_channels,
            proj_channels=16,
            embed_dims=[32, 64, 128, 256],
            num_blocks=[1, 1, 2, 1],
            num_heads=[1, 2, 4, 8],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[4, 2, 2, 2],
            decoder_dim=128,
            mc_dropout_p=0.3,
            encoder_dropout=dropout,
        )


class FireTransformerRegularized(_SegFormerBackbone):
    """Small backbone with stronger dropout and Cutout input augmentation."""

    def __init__(
        self,
        fuel_channels: int = 4,
        weather_channels: int = 8,
        dropout: float = 0.4,
        cutout_holes_fuel: int = 2,
        cutout_size_fuel: int = 12,
        cutout_holes_weather: int = 1,
        cutout_size_weather: int = 8,
    ) -> None:
        super().__init__(
            fuel_channels=fuel_channels,
            weather_channels=weather_channels,
            proj_channels=16,
            embed_dims=[32, 64, 128, 256],
            num_blocks=[1, 1, 2, 1],
            num_heads=[1, 2, 4, 8],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[4, 2, 2, 2],
            decoder_dim=128,
            mc_dropout_p=0.4,
            encoder_dropout=dropout,
        )
        self.cutout_fuel = (cutout_holes_fuel, cutout_size_fuel)
        self.cutout_weather = (cutout_holes_weather, cutout_size_weather)

    @staticmethod
    def _apply_cutout(x: torch.Tensor, n_holes: int, length: int) -> torch.Tensor:
        b, _, h, w = x.shape
        mask = torch.ones_like(x)
        for _ in range(n_holes):
            cy = torch.randint(0, h, (b,))
            cx = torch.randint(0, w, (b,))
            for i in range(b):
                y1, y2 = (cy[i] - length // 2).clamp(0, h), (cy[i] + length // 2).clamp(0, h)
                x1, x2 = (cx[i] - length // 2).clamp(0, w), (cx[i] + length // 2).clamp(0, w)
                mask[i, :, y1:y2, x1:x2] = 0
        return x * mask

    def forward(self, fuel: torch.Tensor, weather: torch.Tensor, mc_sampling: bool = False) -> torch.Tensor:
        if self.training:
            fuel = self._apply_cutout(fuel, *self.cutout_fuel)
            weather = self._apply_cutout(weather, *self.cutout_weather)
        return super().forward(fuel, weather, mc_sampling=mc_sampling)


class _CNNStem(nn.Module):
    def __init__(self, in_channels: int, dims: Sequence[int]) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=False),
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
        f2 = self.stage2(f1)
        return f1, f2


class FireHybridNet(nn.Module):
    """Two CNN stages followed by two transformer stages.

    The CNN front-end exploits convolutional inductive bias on the high-resolution
    feature maps; the transformer back-end captures long-range dependencies on the
    coarsened maps where the token sequence is short and self-attention is cheap.
    """

    def __init__(self, fuel_channels: int = 4, weather_channels: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        dims = [48, 96, 192, 384]
        self.proj = _DualBranchProj(fuel_channels, weather_channels, 16)
        self.cnn = _CNNStem(32, dims[:2])
        self.trans3 = TransformerStage(dims[1], dims[2], num_blocks=2, num_heads=4, sr_ratio=2, patch_size=2, dropout=dropout)
        self.trans4 = TransformerStage(dims[2], dims[3], num_blocks=2, num_heads=8, sr_ratio=1, patch_size=2, dropout=dropout)
        self.decoder = SegFormerDecoder(dims, decoder_dim=128)
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Conv2d(128, 1, 1)
        _init_weights(self)

    def forward(self, fuel: torch.Tensor, weather: torch.Tensor, mc_sampling: bool = False) -> torch.Tensor:
        x = self.proj(fuel, weather)
        f1, f2 = self.cnn(x)
        f3 = self.trans3(f2)
        f4 = self.trans4(f3)
        x = self.decoder([f1, f2, f3, f4], (fuel.shape[2], fuel.shape[3]))
        x = F.dropout(x, p=self.dropout.p, training=True) if mc_sampling else self.dropout(x)
        return self.head(x)
