"""Train the MLP gesture classifier and save a self-contained checkpoint.

Run from the repository root:

    uv run python src/mlp/train.py

The checkpoint (``models/gesture_mlp.pt``) bundles everything inference
needs: the model weights, the class label order, and the feature
mean/std used for standardisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mlp.dataset import load_dataset, grouped_split  # noqa: E402
from mlp.model import MLP  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SESSIONS_DIR = BASE_DIR / "src" / "data" / "sessions"
CHECKPOINT_PATH = BASE_DIR / "models" / "gesture_mlp.pt"

EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
VAL_FRAC = 0.2
TEST_SESSION = None  # e.g. "session_20260101_120000" to hold a day out


def fit_standardizer(train_X: np.ndarray):
    """Compute mean/std on the training split and return an apply function."""
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    def apply(x: np.ndarray) -> np.ndarray:
        return (x - mean) / std

    return apply, mean, std


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def evaluate(model, X: torch.Tensor, y) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = float((preds == np.asarray(y)).mean())
    return acc, preds


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    data = load_dataset(SESSIONS_DIR)
    split = grouped_split(data, val_frac=VAL_FRAC, test_session=TEST_SESSION)

    print(f"classes   : {data.labels}")
    print(f"samples   : train={len(split.train.y)} val={len(split.val.y)}"
          + (f" test={len(split.test.y)}" if split.test else ""))
    print(f"clips     : train={len(np.unique(split.train.groups))} "
          f"val={len(np.unique(split.val.groups))}")

    apply, mean, std = fit_standardizer(split.train.X)
    train_X = torch.from_numpy(apply(split.train.X).astype(np.float32))
    val_X = torch.from_numpy(apply(split.val.X).astype(np.float32))
    train_y = torch.from_numpy(split.train.y)
    val_y = torch.from_numpy(split.val.y)

    n_classes = len(data.labels)
    model = MLP(input_dim=63, output_dim=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(train_X))
        total_loss = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            # BatchNorm needs >1 sample per batch.
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            logits = model(train_X[idx])
            loss = criterion(logits, train_y[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(idx)

        val_acc, _ = evaluate(model, val_X, val_y)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss={total_loss / len(train_X):.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_acc, preds = evaluate(model, val_X, val_y)
    print(f"\nbest val accuracy: {val_acc:.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{l:>9}" for l in data.labels))
    cm = confusion_matrix(split.val.y, preds, n_classes)
    for label, row in zip(data.labels, cm):
        print(f"{label:>9} " + "  ".join(f"{v:>9d}" for v in row))

    if split.test is not None:
        test_X = apply(split.test.X).astype(np.float32)
        test_acc, _ = evaluate(model, torch.from_numpy(test_X), torch.from_numpy(split.test.y))
        print(f"\nheld-out session accuracy: {test_acc:.3f}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": data.labels,
            "input_dim": 63,
            "mean": mean.reshape(-1).astype(np.float32),
            "std": std.reshape(-1).astype(np.float32),
        },
        CHECKPOINT_PATH,
    )
    print(f"\nsaved checkpoint -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
