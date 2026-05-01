"""Evaluate trained checkpoints on the test set.

By default every model in ``MODEL_REGISTRY`` whose checkpoint exists at
``checkpoints/<name>_best.pth`` is evaluated. To restrict the run, edit
``MODEL_NAMES`` below.

Run from the project root::

    python -m firesense.evaluate
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .data import get_dataloaders
from .models import MODEL_REGISTRY, build_model
from .trainer import threshold_sweep

# Edit this to evaluate a subset; ``None`` means "every key in MODEL_REGISTRY".
MODEL_NAMES: Iterable[str] | None = None


def evaluate(
    names: Iterable[str] | None = None,
    ckpt_dir: str | Path = "checkpoints",
    batch_size: int = 64,
    num_workers: int = 8,
) -> None:
    names = list(names) if names is not None else list(MODEL_REGISTRY)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    _, _, test_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    print(f"{'Model':<22} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC-PR':>8}")
    print("-" * 70)
    for name in names:
        path = ckpt_dir / f"{name}_best.pth"
        if not path.exists():
            print(f"{name:<22} (missing checkpoint at {path})")
            continue
        model = build_model(name).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        result = threshold_sweep(model, test_loader, device)
        print(
            f"{name:<22} {result['threshold']:>10.2f} {result['precision']:>10.4f} "
            f"{result['recall']:>10.4f} {result['f1']:>8.4f} {result['auc_pr']:>8.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    evaluate(MODEL_NAMES)
