"""Channel-masking ablation that ranks the 12 input channels by F1 drop.

Setting a channel to zero is equivalent to setting it to its training-set mean
under z-score normalization, which removes its spatial signal entirely while
preserving the input shape. The drop in test-set F1 is reported as that
channel's importance.

Usage::

    python -m firesense.analysis.feature_importance
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ..data import CHANNEL_NAMES, FUEL_CHANNELS, WEATHER_CHANNELS, get_dataloaders
from ..models import build_model
from ..trainer import _confusion


def _f1_at(model, loader, device, fuel_idx: int | None = None, weather_idx: int | None = None, threshold: float = 0.59) -> float:
    model.eval()
    TP = FP = FN = 0.0
    with torch.no_grad():
        for fuel, weather, target in tqdm(loader, leave=False, desc="eval"):
            fuel, weather, target = fuel.to(device), weather.to(device), target.to(device)
            if fuel_idx is not None:
                fuel[:, fuel_idx, :, :] = 0.0
            if weather_idx is not None:
                weather[:, weather_idx, :, :] = 0.0
            logits = model(fuel, weather)
            tp, fp, fn = _confusion(logits, target, threshold=threshold)
            TP += tp; FP += fp; FN += fn
    prec = TP / (TP + FP + 1e-7)
    rec = TP / (TP + FN + 1e-7)
    return 2 * prec * rec / (prec + rec + 1e-7)


def run(
    model_name: str = "firesense",
    checkpoint: str | Path = "checkpoints/firesense_best.pth",
    output: str | Path = "visualizations/feature_importance.png",
    threshold: float = 0.59,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))

    _, _, test_loader = get_dataloaders(batch_size=256, num_workers=8)
    base = _f1_at(model, test_loader, device, threshold=threshold)
    print(f"baseline F1: {base:.4f}")

    fuel_names = [CHANNEL_NAMES[i] for i in FUEL_CHANNELS]
    weather_names = [CHANNEL_NAMES[i] for i in WEATHER_CHANNELS]

    importances: Dict[str, float] = {}
    for i, name in enumerate(fuel_names):
        f1 = _f1_at(model, test_loader, device, fuel_idx=i, threshold=threshold)
        importances[name] = max(0.0, base - f1)
        print(f"  fuel/{name:>14}: dF1 = {base - f1:+.4f}")
    for i, name in enumerate(weather_names):
        f1 = _f1_at(model, test_loader, device, weather_idx=i, threshold=threshold)
        importances[name] = max(0.0, base - f1)
        print(f"  weather/{name:>11}: dF1 = {base - f1:+.4f}")

    order = sorted(importances, key=importances.get)
    values = [importances[n] for n in order]
    plt.figure(figsize=(12, 8))
    bars = plt.barh(order, values, color="darkorange")
    plt.xlabel("F1 drop when the channel is zeroed")
    plt.title("Feature importance via channel masking (test set)")
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            plt.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")
    return importances


if __name__ == "__main__":
    run()
