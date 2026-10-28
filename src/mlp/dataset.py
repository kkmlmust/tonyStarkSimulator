"""Load recorded gesture CSVs into arrays and split them without leakage.

Why grouped splitting matters
-----------------------------
Consecutive  of a held gesture are almost identical. If we split the
dataset row-by-row at random, near-duplicate frames would end up in *both*
training and validation, the model would memorise them, and validation
accuracy would be reassuringly high while real-life accuracy is poor.
frames
To avoid that, every recording burst carries a ``clip_id`` (see
``data/data_collector.py``). Here we split by whole clips: all frames of a
clip go to training, or all go to validation -- never a mix.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Feature columns only (the metadata columns follow them in the CSV).
NUM_FEATURES = 63
META_START = NUM_FEATURES  # first metadata column index


@dataclass
class Dataset:
    X: np.ndarray  # (N, 63) float32
    y: np.ndarray  # (N,) int64 class indices
    groups: np.ndarray  # (N,) str clip key = "<session>/<clip_id>"
    labels: list[str]  # index -> human-readable label


@dataclass
class Split:
    train: Dataset
    val: Dataset
    test: Dataset | None = None


def load_dataset(sessions_dir: Path | str) -> Dataset:
    """Read every ``*.csv`` under ``sessions_dir`` into a single Dataset."""
    sessions_dir = Path(sessions_dir)
    files = sorted(sessions_dir.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found under {sessions_dir}")

    features: list[list[float]] = []
    labels_raw: list[str] = []
    groups: list[str] = []

    label_to_index: dict[str, int] = {}

    for path in files:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            for row in reader:
                if len(row) < META_START + 4:
                    continue
                features.append([float(v) for v in row[:NUM_FEATURES]])
                label = row[META_START]
                clip_id = row[META_START + 1]
                session = row[META_START + 2]
                labels_raw.append(label)
                groups.append(f"{session}/{clip_id}")
                label_to_index.setdefault(label, len(label_to_index))

    if not features:
        raise ValueError(f"no samples found under {sessions_dir}")

    labels = [None] * len(label_to_index)
    for label, index in label_to_index.items():
        labels[index] = label

    return Dataset(
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray([label_to_index[l] for l in labels_raw], dtype=np.int64),
        groups=np.asarray(groups),
        labels=labels,
    )


def _subset(data: Dataset, mask: np.ndarray) -> Dataset:
    return Dataset(
        X=data.X[mask],
        y=data.y[mask],
        groups=data.groups[mask],
        labels=data.labels,
    )


def grouped_split(
    data: Dataset,
    val_frac: float = 0.2,
    seed: int = 42,
    test_session: str | None = None,
) -> Split:
    """Split ``data`` by whole clips.

    Args:
        data: the full dataset.
        val_frac: fraction of *clips* (not rows) reserved for validation.
        seed: RNG seed so the split is reproducible.
        test_session: if given, every clip from this session becomes the
            test set -- the most honest evaluation (a whole recording day
            the model never saw).
    """
    groups = data.groups

    if test_session is not None:
        test_mask = np.array([g.startswith(f"{test_session}/") for g in groups])
        if not test_mask.any():
            raise ValueError(f"no clips found for test session {test_session!r}")
    else:
        test_mask = np.zeros(len(groups), dtype=bool)

    remaining = ~test_mask
    unique_groups = np.unique(groups[remaining])

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_val = max(1, int(round(len(unique_groups) * val_frac)))
    val_groups = set(unique_groups[:n_val])

    val_mask = np.array([g in val_groups for g in groups]) & remaining
    train_mask = remaining & ~val_mask

    test = _subset(data, test_mask) if test_mask.any() else None
    return Split(
        train=_subset(data, train_mask),
        val=_subset(data, val_mask),
        test=test,
    )
