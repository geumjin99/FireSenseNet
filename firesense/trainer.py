"""Single-model trainer with early stopping, optional AMP, and a test-time
threshold sweep.

A ``Trainer`` instance encapsulates one full training run: it owns the model,
loss, optimizer, scheduler, dataloaders, and checkpoint paths. The same class
is used for every architecture; per-architecture differences (optimizer,
learning rate, batch size, AMP, augmentation) are passed in as a config dict.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import get_dataloaders
from .losses import CompositeLoss


@dataclass
class TrainConfig:
    """Per-architecture hyper-parameters."""

    lr: float = 3e-4
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "adam" or "adamw"
    batch_size: int = 64
    epochs: int = 100
    patience: int = 20
    amp: bool = False
    grad_clip: Optional[float] = None
    use_augmentation: bool = True
    pos_weight: float = 3.0
    extra: Dict[str, float] = field(default_factory=dict)


class _EarlyStopping:
    def __init__(self, patience: int, verbose: bool = True, delta: float = 0.0) -> None:
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_loss = math.inf
        self.early_stop = False

    def step(self, val_loss: float, model: nn.Module, save_path: Path) -> None:
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self._save(val_loss, model, save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, val_loss: float, model: nn.Module, save_path: Path) -> None:
        if self.verbose:
            print(f"Val loss improved {self.best_loss:.6f} -> {val_loss:.6f}; saving.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        self.best_loss = val_loss


def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _confusion(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Return (TP, FP, FN) with -1 pixels masked out."""
    valid = targets >= 0
    if valid.sum() == 0:
        return 0.0, 0.0, 0.0
    probs = torch.sigmoid(logits[valid])
    hard = (targets[valid] > 0.5).float()  # softened labels -> hard
    pred = (probs > threshold).float()
    tp = (pred * hard).sum().item()
    fp = (pred * (1 - hard)).sum().item()
    fn = ((1 - pred) * hard).sum().item()
    return tp, fp, fn


def threshold_sweep(model: nn.Module, loader, device) -> Dict[str, float]:
    """Sweep classification thresholds in [0.05, 0.95] and return the F1-best
    operating point along with AUC-PR (threshold-free)."""
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for fuel, weather, target in tqdm(loader, desc="Sweep", leave=False):
            fuel, weather = fuel.to(device), weather.to(device)
            logits = model(fuel, weather)
            valid = target >= 0
            probs_all.append(torch.sigmoid(logits).cpu()[valid])
            labels_all.append((target[valid] > 0.5).float())
    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()

    best = dict(threshold=0.5, f1=0.0, precision=0.0, recall=0.0)
    for th in np.arange(0.05, 0.95, 0.01):
        pred = (probs > th).astype(np.float32)
        tp = (pred * labels).sum()
        fp = (pred * (1 - labels)).sum()
        fn = ((1 - pred) * labels).sum()
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2 * prec * rec / (prec + rec + 1e-7)
        if f1 > best["f1"]:
            best.update(threshold=float(th), f1=float(f1), precision=float(prec), recall=float(rec))
    best["auc_pr"] = float(average_precision_score(labels, probs))
    return best


class Trainer:
    """Trains a single model end-to-end and records its best test-set metrics."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        name: str,
        data_dir: str | Path = "pytorch_data",
        ckpt_dir: str | Path = "checkpoints",
        num_workers: int = 8,
    ) -> None:
        self.name = name
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = CompositeLoss(pos_weight=config.pos_weight).to(self.device)
        self.optimizer = _build_optimizer(self.model, config)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)

        self.scaler = torch.amp.GradScaler("cuda") if config.amp else None
        self.early_stopping = _EarlyStopping(patience=config.patience, verbose=True)

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=config.batch_size,
            num_workers=num_workers,
            use_augmentation=config.use_augmentation,
        )

        self.ckpt_dir = Path(ckpt_dir)
        self.best_path = self.ckpt_dir / f"{name}_best.pth"
        self.resume_path = self.ckpt_dir / f"{name}_resume.pth"
        self.start_epoch = 0
        if self.resume_path.exists():
            self._load_resume()

        params = sum(p.numel() for p in self.model.parameters())
        print(f"[{name}] device={self.device} params={params/1e6:.2f}M config={config}")

    def _load_resume(self) -> None:
        chk = torch.load(self.resume_path, map_location=self.device)
        self.model.load_state_dict(chk["model_state_dict"])
        self.optimizer.load_state_dict(chk["optimizer_state_dict"])
        self.start_epoch = chk["epoch"] + 1
        print(f"[{self.name}] Resumed from epoch {self.start_epoch}")

    def _save_resume(self, epoch: int, val_loss: float) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": val_loss,
            },
            self.resume_path,
        )

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        bar = tqdm(self.train_loader, desc=f"[{self.name}] Epoch {epoch+1:02d} train", dynamic_ncols=True)
        for fuel, weather, target in bar:
            fuel = fuel.to(self.device)
            weather = weather.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(fuel, weather)
                    loss = self.criterion(logits, target)
                self.scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(fuel, weather)
                loss = self.criterion(logits, target)
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            running += loss.item() * fuel.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        return running / len(self.train_loader.dataset)

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        running = 0.0
        TP = FP = FN = 0.0
        with torch.no_grad():
            for fuel, weather, target in self.val_loader:
                fuel = fuel.to(self.device)
                weather = weather.to(self.device)
                target = target.to(self.device)
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(fuel, weather)
                        loss = self.criterion(logits, target)
                else:
                    logits = self.model(fuel, weather)
                    loss = self.criterion(logits, target)
                running += loss.item() * fuel.size(0)
                tp, fp, fn = _confusion(logits.float(), target)
                TP += tp; FP += fp; FN += fn
        loss = running / len(self.val_loader.dataset)
        prec = TP / (TP + FP + 1e-7)
        rec = TP / (TP + FN + 1e-7)
        f1 = 2 * prec * rec / (prec + rec + 1e-7)
        return {"loss": loss, "precision": prec, "recall": rec, "f1": f1}

    def fit(self) -> Dict[str, float]:
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            metrics = self._validate()
            print(
                f"[{self.name}] Epoch {epoch+1:02d}/{self.config.epochs} "
                f"train={train_loss:.4f} val={metrics['loss']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
            )
            self._save_resume(epoch, metrics["loss"])
            self.early_stopping.step(metrics["loss"], self.model, self.best_path)
            if self.early_stopping.early_stop:
                print(f"[{self.name}] Early stopping at epoch {epoch+1}.")
                break
        return self.test()

    def test(self) -> Dict[str, float]:
        if self.best_path.exists():
            self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        result = threshold_sweep(self.model, self.test_loader, self.device)
        params = sum(p.numel() for p in self.model.parameters())
        print(
            f"\n=== {self.name} test results ===\n"
            f"  params    : {params:,}\n"
            f"  threshold : {result['threshold']:.2f}\n"
            f"  precision : {result['precision']:.4f}\n"
            f"  recall    : {result['recall']:.4f}\n"
            f"  f1        : {result['f1']:.4f}\n"
            f"  auc_pr    : {result['auc_pr']:.4f}"
        )
        return result
