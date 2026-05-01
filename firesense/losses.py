"""Composite loss for next-day wildfire spread segmentation.

The loss is a weighted sum of weighted-BCE, Dice, and Focal terms. All three
components honour an ignore mask: pixels with ``target < 0`` (the dataset's
unobserved regions) do not contribute to the loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """``L = w_bce * BCE + w_dice * Dice + w_focal * Focal``."""

    def __init__(
        self,
        pos_weight: float = 3.0,
        gamma: float = 2.0,
        w_bce: float = 0.4,
        w_dice: float = 0.3,
        w_focal: float = 0.3,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]), reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = (targets >= 0).float()
        safe = targets.clamp(min=0.0)

        bce = (self.bce(logits, safe) * valid).sum() / (valid.sum() + 1e-7)

        probs = torch.sigmoid(logits)
        intersection = (probs * safe * valid).sum()
        union = (probs * valid).sum() + (safe * valid).sum()
        dice = 1.0 - (2.0 * intersection + 1e-7) / (union + 1e-7)

        bce_pp = F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
        pt = torch.exp(-bce_pp)
        focal = (((1 - pt) ** self.gamma) * bce_pp * valid).sum() / (valid.sum() + 1e-7)

        return self.w_bce * bce + self.w_dice * dice + self.w_focal * focal
