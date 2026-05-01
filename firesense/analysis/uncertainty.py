"""MC-Dropout uncertainty maps for fire-active samples.

Runs ``n`` stochastic forward passes through ``FireSenseNet`` (with dropout
forced on at inference time) and visualises the per-pixel mean prediction and
standard deviation alongside the ground truth.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import get_dataloaders
from ..models import FireSenseNet


def run(
    checkpoint: str | Path = "checkpoints/firesense_best.pth",
    output_dir: str | Path = "visualizations",
    n_samples: int = 3,
    n_mc_passes: int = 30,
    fire_threshold: int = 25,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FireSenseNet(base_c=32)
    if Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"Loaded checkpoint {checkpoint}")
    else:
        print(f"Warning: {checkpoint} not found; visualising with random weights.")
    model.to(device).eval()

    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=0)
    found = 0
    for fuel, weather, target in test_loader:
        if target.sum() < fire_threshold:
            continue
        fuel, weather = fuel.to(device), weather.to(device)

        preds = []
        with torch.no_grad():
            for _ in range(n_mc_passes):
                probs = torch.sigmoid(model(fuel, weather, mc_sampling=True))
                preds.append(probs.cpu().numpy()[0, 0])
        preds = np.stack(preds)
        mean_p = preds.mean(axis=0)
        var_p = preds.var(axis=0)

        prev_fire = fuel.cpu().numpy()[0, 3]
        true_fire = target.cpu().numpy()[0, 0]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        axes[0].imshow(prev_fire, cmap="Reds"); axes[0].set_title("Previous Day Fire Extent"); axes[0].axis("off")
        axes[1].imshow(true_fire, cmap="Reds", vmin=0, vmax=1); axes[1].set_title("Next Day Fire (Ground Truth)"); axes[1].axis("off")
        im2 = axes[2].imshow(mean_p, cmap="jet", vmin=0, vmax=1); axes[2].set_title(f"Mean prediction (N={n_mc_passes})")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04); axes[2].axis("off")
        im3 = axes[3].imshow(var_p, cmap="turbo"); axes[3].set_title("Predictive variance (uncertainty)")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04); axes[3].axis("off")

        save_path = output_dir / f"mc_dropout_{found}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
        found += 1
        if found >= n_samples:
            break


if __name__ == "__main__":
    run()
