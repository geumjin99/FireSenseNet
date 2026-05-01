"""Convert TFRecord shards into HDF5 files.

This is a one-time step. TensorFlow is required for parsing; PyTorch reads
the resulting HDF5 files directly via :class:`firesense.data.WildfireDataset`.

Run once after placing the TFRecord shards in ``data/``::

    python -m firesense.preprocess
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

INPUT_FEATURES = [
    "elevation", "NDVI", "population", "PrevFireMask",
    "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "erc",
]
TARGET_FEATURE = "FireMask"


def preprocess(data_dir: str | Path = "data", output_dir: str | Path = "pytorch_data") -> None:
    import tensorflow as tf  # imported lazily; only needed for preprocessing

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_spec = {
        name: tf.io.FixedLenFeature(shape=(64, 64), dtype=tf.float32)
        for name in INPUT_FEATURES + [TARGET_FEATURE]
    }

    def parse(example):
        parsed = tf.io.parse_single_example(example, feature_spec)
        inputs = tf.stack([parsed[name] for name in INPUT_FEATURES], axis=0)
        target = tf.expand_dims(parsed[TARGET_FEATURE], axis=0)
        return inputs, target

    for split in ("train", "eval", "test"):
        files = sorted(str(p) for p in data_dir.glob(f"next_day_wildfire_spread_{split}_*.tfrecord"))
        if not files:
            print(f"[skip] no TFRecord shards found for split={split}")
            continue
        ds = tf.data.TFRecordDataset(files).map(parse)
        inputs, targets = [], []
        for inp, tgt in ds.as_numpy_iterator():
            inputs.append(inp)
            targets.append(tgt)
        inputs_np = np.stack(inputs).astype(np.float32)
        targets_np = np.stack(targets).astype(np.float32)
        out_path = output_dir / f"{split}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("inputs", data=inputs_np, compression="gzip")
            f.create_dataset("target", data=targets_np, compression="gzip")
        print(f"[{split}] wrote {len(inputs_np)} samples -> {out_path}")


if __name__ == "__main__":
    preprocess()
