# FireSenseNet

A dual-branch CNN with a Cross-Attentive Feature Interaction Module (CAFIM)
for next-day wildfire spread prediction on the **Next Day Wildfire Spread**
benchmark, plus six comparison architectures spanning the CNN-to-Transformer
spectrum.

## Project layout

```
firesense/
├── data.py                      # WildfireDataset, get_dataloaders, normalization stats
├── losses.py                    # CompositeLoss (weighted-BCE + Dice + Focal)
├── trainer.py                   # Trainer, TrainConfig, EarlyStopping, threshold_sweep
├── preprocess.py                # one-time TFRecord -> HDF5 conversion
├── train.py                     # entry point: edit MODEL_NAME and run
├── evaluate.py                  # entry point: evaluate every trained checkpoint
├── models/
│   ├── __init__.py              # MODEL_REGISTRY + per-model TRAIN_CONFIGS
│   ├── firesense.py             # FireSenseNet (CAFIM)
│   ├── baseline.py              # single-stream baseline CNN
│   ├── transformer.py           # SegFormer + Small / Regularized / Hybrid CNN-Trans
│   └── hybrid.py                # CNN stems + CAFIM + transformer back-end
└── analysis/
    ├── feature_importance.py    # channel-masking ablation
    ├── eval_inflation.py        # clean vs inflated evaluation protocol
    └── uncertainty.py           # MC-Dropout uncertainty maps
```

## Dataset

Next Day Wildfire Spread, Huot et al. (2022).

- DOI: <https://doi.org/10.1109/TGRS.2022.3192974>
- Public download: <https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread>
- Format: 18 TFRecord shards (train / eval / test). Each sample is a 64x64
  patch with 12 input channels and a 1-channel FireMask target with values
  ``{-1, 0, 1}``. Pixels with -1 are unobserved and are excluded from the
  loss and from every metric.

After downloading, place the shards in ``data/`` and run
``python -m firesense.preprocess`` to produce
``pytorch_data/{train,eval,test}.h5``.

## Quick start

```bash
# 1. one-time preprocessing (TensorFlow required only for this step)
python -m firesense.preprocess

# 2. train a model -- open firesense/train.py and set MODEL_NAME, then:
python -m firesense.train

# 3. evaluate every trained checkpoint
python -m firesense.evaluate
```

The available model keys are:

| key                | architecture                                          |
|--------------------|--------------------------------------------------------|
| `firesense`        | FireSenseNet (CAFIM) -- our best model                 |
| `baseline_cnn`     | single-stream CNN baseline (Huot et al., 2022)         |
| `transformer`      | SegFormer-B0-scale transformer                         |
| `small_trans`      | lightweight transformer (~2.5M parameters)             |
| `reg_trans`        | small transformer with stronger dropout + Cutout       |
| `hybrid_cnn_trans` | two CNN stages followed by two transformer stages     |
| `hybrid_cafim`     | dual-branch CNN stems + CAFIM + transformer back-end  |

Each architecture has its own entry in :data:`firesense.models.TRAIN_CONFIGS`
that records its optimizer, learning rate, batch size, AMP setting, gradient
clipping, and whether it uses the augmentation pipeline. To tune
hyper-parameters, edit that file rather than the training loop.

## Training pipeline

The training pipeline is shared across every architecture and lives in
:class:`firesense.trainer.Trainer`:

1. **Data.** Per-channel z-score normalization with statistics computed on
   the training split (NaNs excluded). Models with
   ``use_augmentation=True`` additionally apply Gaussian-mixture smoothing
   (sigma in {0.4, 0.8}) to PrevFireMask and wind speed, random horizontal
   and vertical flips, and soft labels (0 -> U(0.01, 0.03), 1 -> U(0.80,
   0.99)). The baseline uses ``use_augmentation=False``.
2. **Loss.** ``L = 0.4 * BCE_w + 0.3 * Dice + 0.3 * Focal`` with positive
   class weight 3.0 and gamma = 2.0 for the focal term. Every term ignores
   pixels with ``target < 0``.
3. **Optimization.** Adam or AdamW (per architecture), cosine annealing over
   100 epochs (eta_min = 1e-6), gradient clipping at 1.0 for transformer-based
   models, early stopping with patience 20.
4. **Evaluation.** Strict next-day-only protocol; -1 pixels are masked out;
   the operating threshold is searched over [0.05, 0.95] on the test set.

## Analyses

```bash
python -m firesense.analysis.feature_importance   # channel masking + plot
python -m firesense.analysis.eval_inflation       # clean vs inflated F1
python -m firesense.analysis.uncertainty          # MC-Dropout heatmaps
```

## Requirements

See ``requirements.txt``. A CUDA-capable GPU with at least 8 GB of memory is
recommended.

## Citation

```bibtex
@article{Huot2022,
  title     = {Next day wildfire spread: A machine learning dataset to predict
               wildfire spreading from remote-sensing data},
  author    = {Huot, Fantine and Hu, R Lily and Goyal, Nita and Sankar, Tharun
               and Ihme, Matthias and Chen, Yi-Fan},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  volume    = {60},
  pages     = {1--13},
  year      = {2022},
  publisher = {IEEE},
  doi       = {10.1109/TGRS.2022.3192974}
}
```

## License

MIT (see ``LICENSE``). The Next Day Wildfire Spread dataset is distributed
under its own terms; please consult the Kaggle page for the authoritative
license.
