"""Dataset, dataloaders, and normalization statistics for the Next Day Wildfire
Spread benchmark.

The 12 input channels are split into a fuel/terrain group (channels 0-3) and a
weather group (channels 4-11).  The target FireMask uses {-1, 0, 1}, where -1
denotes unobserved pixels that must be ignored by the loss and every metric.

Two preprocessing modes are exposed:

* ``use_augmentation=True`` -- the pipeline used for FireSenseNet and the
  transformer-based models: Gaussian-mixture smoothing on the sparse
  PrevFireMask and wind-speed channels, random horizontal/vertical flips, and
  soft-label transformation 0 -> U(0.01, 0.03), 1 -> U(0.80, 0.99).
* ``use_augmentation=False`` -- the baseline pipeline: z-score normalization
  only, hard binary labels, no flips.

Normalization statistics are pre-computed on the training split.  Re-derive
them by running this module as a script: ``python -m firesense.data``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Per-channel global statistics computed on the training split.
GLOBAL_MEANS = np.array(
    [
        896.571320, 5350.680988, 30.460330, -0.002723,
        146.646508, 3.627848, 281.852110, 297.716643,
        0.006526, 0.323428, -0.772871, 53.469103,
    ],
    dtype=np.float32,
).reshape(12, 1, 1)

GLOBAL_STDS = np.array(
    [
        842.610351, 2185.218566, 214.200051, 0.138312,
        3435.083853, 1.309215, 18.497169, 19.458073,
        0.003736, 1.533664, 2.440719, 25.097984,
    ],
    dtype=np.float32,
).reshape(12, 1, 1)

CHANNEL_NAMES = [
    "elevation", "NDVI", "population", "PrevFireMask",
    "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "erc",
]
FUEL_CHANNELS = [0, 1, 2, 3]
WEATHER_CHANNELS = [4, 5, 6, 7, 8, 9, 10, 11]


def _gaussian_kernel_2d(sigma: float) -> torch.Tensor:
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return (k[:, None] * k[None, :]).unsqueeze(0).unsqueeze(0)


def _gaussian_blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a single-channel ``(H, W)`` tensor with a 2D Gaussian kernel."""
    kernel = _gaussian_kernel_2d(sigma)
    pad = kernel.shape[-1] // 2
    return F.conv2d(t.unsqueeze(0).unsqueeze(0), kernel, padding=pad).squeeze(0).squeeze(0)


class WildfireDataset(Dataset):
    """HDF5-backed dataset for the Next Day Wildfire Spread benchmark."""

    def __init__(
        self,
        h5_path: str | Path,
        is_train: bool = False,
        use_augmentation: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["inputs"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_path, "r") as f:
            inp = f["inputs"][idx]
            tgt = f["target"][idx]

        inp = (inp - GLOBAL_MEANS) / GLOBAL_STDS
        inp = np.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0)

        fuel = torch.from_numpy(inp[FUEL_CHANNELS]).float()
        weather = torch.from_numpy(inp[WEATHER_CHANNELS]).float()
        tgt = torch.from_numpy(tgt).float()

        if not self.use_augmentation:
            return fuel, weather, tgt

        # Gaussian-mixture smoothing on the two sparse channels.
        for tensor, ch in ((fuel, 3), (weather, 1)):
            data = tensor[ch]
            tensor[ch] = 0.5 * (_gaussian_blur(data, 0.4) + _gaussian_blur(data, 0.8))

        # Random flips during training.
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                fuel, weather, tgt = (torch.flip(t, dims=[1]) for t in (fuel, weather, tgt))
            if torch.rand(1).item() > 0.5:
                fuel, weather, tgt = (torch.flip(t, dims=[2]) for t in (fuel, weather, tgt))

        # Soft-label transformation; -1 is preserved so the loss can mask it.
        ignore = tgt == -1
        zero = tgt == 0
        one = tgt == 1
        soft = torch.zeros_like(tgt)
        soft[zero] = torch.empty(zero.sum().item()).uniform_(0.01, 0.03)
        soft[one] = torch.empty(one.sum().item()).uniform_(0.80, 0.99)
        soft[ignore] = -1.0
        return fuel, weather, soft


def get_dataloaders(
    data_dir: str | Path = "pytorch_data",
    batch_size: int = 64,
    num_workers: int = 4,
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    common = dict(num_workers=num_workers, pin_memory=True)
    train = WildfireDataset(data_dir / "train.h5", is_train=True, use_augmentation=use_augmentation)
    val = WildfireDataset(data_dir / "eval.h5", is_train=False, use_augmentation=use_augmentation)
    test = WildfireDataset(data_dir / "test.h5", is_train=False, use_augmentation=use_augmentation)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=batch_size, shuffle=False, **common),
    )


def compute_normalization_stats(train_h5: str | Path = "pytorch_data/train.h5") -> None:
    """Recompute per-channel mean/std on the training split (excluding NaNs)
    and print them in a copy-pastable format."""
    with h5py.File(train_h5, "r") as f:
        data = f["inputs"][:].astype(np.float64)
    means, stds = [], []
    for i, name in enumerate(CHANNEL_NAMES):
        valid = data[:, i][~np.isnan(data[:, i])]
        m, s = float(valid.mean()), float(valid.std())
        if s < 1e-6:
            s = 1.0
        means.append(m)
        stds.append(s)
        print(f"{name:>14}: mean={m:.6f}, std={s:.6f}")
    print("\nGLOBAL_MEANS = [")
    for m, name in zip(means, CHANNEL_NAMES):
        print(f"    {m:.6f},  # {name}")
    print("]\nGLOBAL_STDS = [")
    for s, name in zip(stds, CHANNEL_NAMES):
        print(f"    {s:.6f},  # {name}")
    print("]")


if __name__ == "__main__":
    compute_normalization_stats()
