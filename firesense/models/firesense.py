"""FireSenseNet: dual-branch CNN with Cross-Attentive Feature Interaction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        # ``downsample`` is always present (an empty Sequential when no
        # projection is needed); this matches the layout used in the
        # released checkpoint so state-dicts load without remapping.
        self.downsample: nn.Module = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CAFIM(nn.Module):
    """Cross-Attentive Feature Interaction Module.

    Projects fuel and weather features to a shared space, predicts a spatial
    attention map ``alpha in [0, 1]^{H x W}``, and emits the complementary
    gating ``[alpha . F_fuel ; (1 - alpha) . F_weather]``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_f = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(channels, channels, 1, bias=False)
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, fuel: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([self.conv_f(fuel), self.conv_w(weather)], dim=1)
        alpha = self.attn(joint)
        return torch.cat([fuel * alpha, weather * (1.0 - alpha)], dim=1)


class _UpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad in case of off-by-one mismatch between upsampled and skip features.
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class FireSenseNet(nn.Module):
    """Dual-branch encoder + CAFIM at every scale + lightweight U-Net decoder.

    The fuel branch uses 3x3 residual blocks tuned to local terrain texture;
    the weather branch uses 5x5 convolutions, reflecting the larger spatial
    coherence of atmospheric fields. A CAFIM block fuses the two branches at
    every encoder scale before the decoder reconstructs the per-pixel logits.
    Monte Carlo Dropout before the prediction head supports predictive
    uncertainty estimation.
    """

    def __init__(
        self,
        fuel_channels: int = 4,
        weather_channels: int = 8,
        base_c: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c = base_c

        # Fuel branch
        self.f_inc = self._stem(fuel_channels, c, kernel=3)
        self.f_enc1 = _ResBlock(c, c)
        self.f_enc2 = _ResBlock(c, c * 2, stride=2)
        self.f_enc3 = _ResBlock(c * 2, c * 4, stride=2)

        # Weather branch (larger receptive field)
        self.w_inc = self._stem(weather_channels, c, kernel=5)
        self.w_enc1 = self._weather_block(c, c, downsample=False)
        self.w_enc2 = self._weather_block(c, c * 2, downsample=True)
        self.w_enc3 = self._weather_block(c * 2, c * 4, downsample=True)

        # CAFIM at every encoder scale
        self.cafim1 = CAFIM(c)
        self.cafim2 = CAFIM(c * 2)
        self.cafim3 = CAFIM(c * 4)

        # Bottleneck on the deepest skip
        self.bottle = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c * 8, c * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 8, c * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec3 = _UpBlock(c * 8 + c * 8, c * 4)
        self.dec2 = _UpBlock(c * 4 + c * 4, c * 2)
        self.dec1 = _UpBlock(c * 2 + c * 2, c)

        self.mc_dropout = nn.Dropout(p=dropout)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    @staticmethod
    def _stem(in_c: int, out_c: int, kernel: int) -> nn.Sequential:
        pad = kernel // 2
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _weather_block(in_c: int, out_c: int, downsample: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        if downsample:
            layers.append(nn.MaxPool2d(2))
        kernel = 3 if downsample else 5
        pad = kernel // 2
        layers.extend(
            [
                nn.Conv2d(in_c, out_c, kernel, padding=pad, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
        )
        return nn.Sequential(*layers)

    def forward(
        self,
        fuel: torch.Tensor,
        weather: torch.Tensor,
        mc_sampling: bool = False,
    ) -> torch.Tensor:
        f1 = self.f_enc1(self.f_inc(fuel))
        w1 = self.w_enc1(self.w_inc(weather))
        f2 = self.f_enc2(f1)
        w2 = self.w_enc2(w1)
        f3 = self.f_enc3(f2)
        w3 = self.w_enc3(w2)

        s1 = self.cafim1(f1, w1)
        s2 = self.cafim2(f2, w2)
        s3 = self.cafim3(f3, w3)

        x = self.bottle(s3)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        if mc_sampling:
            x = F.dropout(x, p=self.mc_dropout.p, training=True)
        else:
            x = self.mc_dropout(x)
        return self.head(x)
