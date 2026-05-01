"""Train a single model.

To train a different architecture, edit ``MODEL_NAME`` below. The available
keys are listed by ``firesense.models.available_models()`` and are
also visible in ``firesense/models/__init__.py``::

    firesense           -- FireSenseNet (CAFIM)            -- our best model
    baseline_cnn        -- Single-stream CNN baseline      -- no augmentation
    transformer         -- SegFormer-B0-scale transformer
    small_trans         -- Lightweight transformer (~2.5M params)
    reg_trans           -- Regularized transformer (high dropout + Cutout)
    hybrid_cnn_trans    -- CNN front-end + transformer back-end
    hybrid_cafim        -- CNN stems + CAFIM + transformer back-end

Run from the project root::

    python -m firesense.train
"""
from __future__ import annotations

from .models import build_model, get_config
from .trainer import Trainer

MODEL_NAME = "firesense"  # <-- edit this to train a different architecture


def main() -> None:
    model = build_model(MODEL_NAME)
    config = get_config(MODEL_NAME)
    trainer = Trainer(model=model, config=config, name=MODEL_NAME)
    trainer.fit()


if __name__ == "__main__":
    main()
