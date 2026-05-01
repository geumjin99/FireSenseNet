"""Model registry.

Add a new model in three steps:

1. Implement the ``nn.Module`` in a file under this package.
2. Add a builder under :data:`MODEL_REGISTRY` keyed by a short name.
3. Add a matching :class:`firesense.trainer.TrainConfig` under
   :data:`TRAIN_CONFIGS`.

The training script (``train.py``) selects a model by setting ``MODEL_NAME``
to one of the registry keys.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from ..trainer import TrainConfig
from .baseline import BaselineCNN
from .firesense import CAFIM, FireSenseNet
from .hybrid import FireHybridCAFIM
from .transformer import (
    FireHybridNet,
    FireTransformer,
    FireTransformerRegularized,
    FireTransformerSmall,
)

__all__ = [
    "MODEL_REGISTRY",
    "TRAIN_CONFIGS",
    "build_model",
    "get_config",
    "available_models",
    "BaselineCNN",
    "CAFIM",
    "FireSenseNet",
    "FireHybridCAFIM",
    "FireHybridNet",
    "FireTransformer",
    "FireTransformerRegularized",
    "FireTransformerSmall",
]


MODEL_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    "firesense":         lambda: FireSenseNet(base_c=32),
    "baseline_cnn":      lambda: BaselineCNN(),
    "transformer":       lambda: FireTransformer(),
    "small_trans":       lambda: FireTransformerSmall(),
    "reg_trans":         lambda: FireTransformerRegularized(),
    "hybrid_cnn_trans":  lambda: FireHybridNet(),
    "hybrid_cafim":      lambda: FireHybridCAFIM(),
}


# Per-architecture training configuration. Common settings (epochs=100,
# patience=20, pos_weight=3.0, cosine schedule with eta_min=1e-6) are inherited
# from TrainConfig defaults; only the differences are listed here.
TRAIN_CONFIGS: Dict[str, TrainConfig] = {
    "firesense": TrainConfig(
        lr=3e-4, optimizer="adam", batch_size=128,
    ),
    "baseline_cnn": TrainConfig(
        lr=3e-4, optimizer="adam", batch_size=128,
        use_augmentation=False,
    ),
    "transformer": TrainConfig(
        lr=1e-4, weight_decay=0.01, optimizer="adamw",
        batch_size=64, amp=True, grad_clip=1.0,
    ),
    "small_trans": TrainConfig(
        lr=1e-4, weight_decay=0.01, optimizer="adamw",
        batch_size=64, amp=True, grad_clip=1.0,
    ),
    "reg_trans": TrainConfig(
        lr=1e-4, weight_decay=0.01, optimizer="adamw",
        batch_size=64, amp=True, grad_clip=1.0,
    ),
    "hybrid_cnn_trans": TrainConfig(
        lr=1e-4, weight_decay=0.01, optimizer="adamw",
        batch_size=64, amp=True, grad_clip=1.0,
    ),
    "hybrid_cafim": TrainConfig(
        lr=3e-4, weight_decay=0.01, optimizer="adamw",
        batch_size=64, amp=True, grad_clip=1.0,
    ),
}


def available_models() -> list[str]:
    return list(MODEL_REGISTRY)


def build_model(name: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return MODEL_REGISTRY[name]()


def get_config(name: str) -> TrainConfig:
    if name not in TRAIN_CONFIGS:
        raise KeyError(f"No training config for {name!r}.")
    return TRAIN_CONFIGS[name]
