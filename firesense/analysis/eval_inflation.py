"""Re-evaluate every checkpoint under the inflated "Both Days + -1->0" protocol
and quantify how much the F1 score grows.

The clean protocol (used throughout the manuscript) excludes -1 pixels and
predicts only the next-day fire mask. The inflated protocol used by some
prior work (i) treats -1 pixels as background and (ii) sets the target to
``max(PrevFireMask, NextDayFireMask)`` so the model can boost its score by
copying the input mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data import GLOBAL_MEANS, GLOBAL_STDS
from ..models import MODEL_REGISTRY


class _DualEvalDataset(Dataset):
    """Yields fuel, weather, both-days target, clean target."""

    def __init__(self, h5_path: str | Path) -> None:
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["inputs"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            inp = f["inputs"][idx]
            tgt = f["target"][idx]

        raw_prev_fire = inp[3].copy()
        inp = (inp - GLOBAL_MEANS) / GLOBAL_STDS
        inp = np.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0)
        fuel = torch.from_numpy(inp[:4]).float()
        weather = torch.from_numpy(inp[4:]).float()

        prev_binary = (raw_prev_fire > 0).astype(np.float32)
        next_day = tgt.copy()
        next_day[next_day == -1] = 0.0
        prev_binary[raw_prev_fire == -1] = 0.0
        both_days = torch.from_numpy(np.maximum(prev_binary, next_day)).float().squeeze(0)
        clean = torch.from_numpy(tgt).float().squeeze(0)
        return fuel, weather, both_days, clean


def _f1(model, loader, device, mode: str) -> float:
    model.eval()
    TP = FP = FN = 0.0
    with torch.no_grad():
        for fuel, weather, both_days, clean in loader:
            fuel, weather = fuel.to(device), weather.to(device)
            probs = torch.sigmoid(model(fuel, weather)).cpu().squeeze(1)
            preds = (probs > 0.5).float()
            if mode == "tricked":
                tp = (preds * both_days).sum().item()
                fp = (preds * (1 - both_days)).sum().item()
                fn = ((1 - preds) * both_days).sum().item()
            else:
                valid = clean >= 0
                if valid.sum() == 0:
                    continue
                p = preds[valid]
                t = (clean[valid] > 0.5).float()
                tp = (p * t).sum().item()
                fp = (p * (1 - t)).sum().item()
                fn = ((1 - p) * t).sum().item()
            TP += tp; FP += fp; FN += fn
    prec = TP / (TP + FP + 1e-7)
    rec = TP / (TP + FN + 1e-7)
    return 2 * prec * rec / (prec + rec + 1e-7)


def run(
    checkpoints: Iterable[Tuple[str, str, str]] | None = None,
    test_h5: str | Path = "pytorch_data/test.h5",
    ckpt_dir: str | Path = "checkpoints",
    batch_size: int = 64,
    num_workers: int = 4,
) -> None:
    """Compare clean vs inflated F1 for every model in ``checkpoints``.

    ``checkpoints`` is an iterable of ``(display_name, registry_key,
    checkpoint_filename)`` tuples; if omitted, every entry in
    :data:`firesense.models.MODEL_REGISTRY` is evaluated against
    ``<ckpt_dir>/<key>_best.pth``.
    """
    if checkpoints is None:
        checkpoints = [(k, k, f"{k}_best.pth") for k in MODEL_REGISTRY]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _DualEvalDataset(test_h5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ckpt_dir = Path(ckpt_dir)

    print(f"{'Model':<28} | {'Clean F1':>10} | {'Tricked F1':>11} | {'Inflation':>10}")
    print("-" * 70)
    for display, key, fname in checkpoints:
        path = ckpt_dir / fname
        if not path.exists():
            print(f"{display:<28} | (missing checkpoint at {path})")
            continue
        try:
            model = MODEL_REGISTRY[key]().to(device)
            state = torch.load(path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
        except Exception as exc:  # noqa: BLE001
            print(f"{display:<28} | failed to load: {exc}")
            continue
        clean = _f1(model, loader, device, mode="clean")
        tricked = _f1(model, loader, device, mode="tricked")
        infl = (tricked - clean) / max(clean, 1e-7) * 100
        print(f"{display:<28} | {clean:>10.4f} | {tricked:>11.4f} | {infl:>+9.1f}%")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
