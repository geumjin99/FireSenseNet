"""Self-contained smoke test for the refactored package.

Exercises every public component end-to-end without touching the real
``checkpoints/`` directory:

    1. Forward + backward pass for every registered model on random tensors.
    2. ``WildfireDataset`` returns shapes and dtypes we expect, in both
       augmentation modes.
    3. ``CompositeLoss`` produces a finite, positive value on a real batch.
    4. ``Trainer.fit`` runs end-to-end (one epoch, tiny subset, temp ckpt
       dir) for the lightweight ``small_trans`` model.
    5. ``threshold_sweep`` returns the expected dict on a tiny test loader.

Run from the project root::

    python -m firesense.smoke_test
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import WildfireDataset, get_dataloaders
from .losses import CompositeLoss
from .models import MODEL_REGISTRY, build_model, get_config
from .trainer import Trainer, threshold_sweep


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "pytorch_data"


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def step1_forward_backward() -> None:
    banner("[1/5] forward + backward pass for every model on random tensors")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fuel = torch.randn(2, 4, 64, 64, device=device)
    weather = torch.randn(2, 8, 64, 64, device=device)
    target = torch.randint(0, 2, (2, 1, 64, 64), device=device).float()
    loss_fn = CompositeLoss().to(device)

    for name in MODEL_REGISTRY:
        model = build_model(name).to(device)
        model.train()
        out = model(fuel, weather)
        loss = loss_fn(out, target)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        assert out.shape == (2, 1, 64, 64), f"{name}: bad output shape {tuple(out.shape)}"
        assert torch.isfinite(loss), f"{name}: non-finite loss {loss.item()}"
        print(f"  {name:<20} params={n_params/1e6:5.2f}M  loss={loss.item():.4f}  out={tuple(out.shape)}  OK")
        del model


def step2_dataset() -> None:
    banner("[2/5] WildfireDataset returns the expected shapes (both aug modes)")
    h5 = DATA_DIR / "eval.h5"
    if not h5.exists():
        print(f"  SKIP: {h5} not found")
        return
    for use_aug in (True, False):
        ds = WildfireDataset(h5, is_train=False, use_augmentation=use_aug)
        fuel, weather, target = ds[0]
        assert fuel.shape == (4, 64, 64), f"fuel shape {tuple(fuel.shape)}"
        assert weather.shape == (8, 64, 64), f"weather shape {tuple(weather.shape)}"
        assert target.shape == (1, 64, 64), f"target shape {tuple(target.shape)}"
        # Soft labels in (0, 1) when augmentation is on; raw {-1, 0, 1} otherwise
        valid = target[target >= 0]
        if use_aug:
            assert (valid >= 0).all() and (valid <= 1).all(), "softened labels out of [0,1]"
        else:
            unique = sorted(set(valid.unique().tolist()))
            assert set(unique) <= {0.0, 1.0}, f"unexpected raw label values: {unique}"
        print(f"  use_augmentation={use_aug}  len={len(ds)}  OK")


def step3_loss_on_real_batch() -> None:
    banner("[3/5] CompositeLoss on a real batch")
    h5 = DATA_DIR / "eval.h5"
    if not h5.exists():
        print(f"  SKIP: {h5} not found")
        return
    ds = WildfireDataset(h5, is_train=False, use_augmentation=True)
    loader = DataLoader(Subset(ds, range(8)), batch_size=4, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("small_trans").to(device)
    loss_fn = CompositeLoss().to(device)
    fuel, weather, target = next(iter(loader))
    fuel, weather, target = fuel.to(device), weather.to(device), target.to(device)
    logits = model(fuel, weather)
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss) and loss.item() > 0, f"unexpected loss {loss.item()}"
    print(f"  composite loss = {loss.item():.4f}  OK")


def _swap_in_subset_loaders(trainer: Trainer, n: int = 32) -> None:
    """Replace the trainer's full-dataset loaders with tiny subsets."""
    bs = max(2, min(n // 4, trainer.config.batch_size))
    trainer.train_loader = DataLoader(
        Subset(trainer.train_loader.dataset, range(min(n, len(trainer.train_loader.dataset)))),
        batch_size=bs, shuffle=True, num_workers=0, pin_memory=True,
    )
    trainer.val_loader = DataLoader(
        Subset(trainer.val_loader.dataset, range(min(n // 2, len(trainer.val_loader.dataset)))),
        batch_size=bs, shuffle=False, num_workers=0, pin_memory=True,
    )
    trainer.test_loader = DataLoader(
        Subset(trainer.test_loader.dataset, range(min(n // 2, len(trainer.test_loader.dataset)))),
        batch_size=bs, shuffle=False, num_workers=0, pin_memory=True,
    )


def step4_trainer_one_epoch(tmp_ckpt_dir: Path) -> None:
    banner(f"[4/5] Trainer.fit for 1 epoch on a 32-sample subset (ckpts in {tmp_ckpt_dir})")
    if not (DATA_DIR / "train.h5").exists():
        print(f"  SKIP: {DATA_DIR / 'train.h5'} not found")
        return
    name = "small_trans"
    config = get_config(name)
    config.epochs = 1  # one epoch
    config.patience = 1
    model = build_model(name)
    trainer = Trainer(
        model=model,
        config=config,
        name="smoke_" + name,
        data_dir=DATA_DIR,
        ckpt_dir=tmp_ckpt_dir,
        num_workers=0,
    )
    _swap_in_subset_loaders(trainer, n=32)
    result = trainer.fit()
    assert math.isfinite(result["f1"]), f"non-finite F1: {result}"
    print(f"  test F1 on subset = {result['f1']:.4f}  threshold={result['threshold']:.2f}  OK")


def step5_threshold_sweep(tmp_ckpt_dir: Path) -> None:
    banner("[5/5] threshold_sweep on a tiny test loader")
    if not (DATA_DIR / "test.h5").exists():
        print(f"  SKIP: {DATA_DIR / 'test.h5'} not found")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("small_trans").to(device)
    ds = WildfireDataset(DATA_DIR / "test.h5", is_train=False, use_augmentation=True)
    loader = DataLoader(Subset(ds, range(16)), batch_size=4, shuffle=False, num_workers=0)
    result = threshold_sweep(model, loader, device)
    for key in ("threshold", "f1", "precision", "recall", "auc_pr"):
        assert key in result, f"missing key {key}"
        assert math.isfinite(result[key]), f"non-finite {key}={result[key]}"
    print(f"  threshold={result['threshold']:.2f}  f1={result['f1']:.4f}  auc_pr={result['auc_pr']:.4f}  OK")


def main() -> int:
    real_ckpt = PROJECT_ROOT / "checkpoints"
    tmp_ckpt = Path(tempfile.mkdtemp(prefix="firesense_smoke_"))
    assert tmp_ckpt != real_ckpt, "smoke test would clobber real checkpoints"
    print(f"smoke ckpt dir = {tmp_ckpt}")
    print(f"real ckpt dir  = {real_ckpt}  (will not be touched)")

    try:
        step1_forward_backward()
        step2_dataset()
        step3_loss_on_real_batch()
        step4_trainer_one_epoch(tmp_ckpt)
        step5_threshold_sweep(tmp_ckpt)
        banner("ALL SMOKE TESTS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp_ckpt, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
