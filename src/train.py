"""Training harness for the period-classification MLP (PRE_REGISTRATION §3, frozen).

Frozen specification implemented here:
  - Input: raw (Re c, Im c) — no Fourier features or other feature engineering.
  - Architecture: input projection Linear(2, 256) + 4 hidden Linear(256, 256) layers,
    GELU after every hidden linear, linear 10-way classification head.
    Exact parameter count: 266,506 (the pre-registered figure "≈3.0e5" is a
    one-significant-figure approximation of this count; the architecture clause
    "4 hidden layers × width 256" is the operative spec and is pinned in tests).
  - Optimizer: AdamW, lr 1e-3 with cosine decay to 0 over the full step budget,
    weight decay 1e-4, batch 1024, ≤300 epochs with early stopping on val loss.
  - Seeds {0, 1, 2, 3, 4}, all reported.
  - Checkpoint every 200 optimizer steps (training-dynamics replay for H3).
  - CPU/MPS-safe (float32 everywhere; MPS has no float64).

The probe-power control net (PRE_REGISTRATION §4) is trained with --shuffle-labels,
which permutes the *training* labels with the run seed (val/test untouched).

NOTE: the full 5-seed sweep on the 2M-point dataset is gated and is NOT run by importing
this module; training happens only via an explicit CLI invocation.

Usage (at activation, not before):
  python -m src.train --data-dir data --out-dir checkpoints            # seeds 0..4
  python -m src.train --data-dir data --out-dir checkpoints --seeds 0 --shuffle-labels
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Frozen configuration (PRE_REGISTRATION §3). Tests assert these values; do not
# edit without a dated amendment to PRE_REGISTRATION §8.
# ---------------------------------------------------------------------------
FROZEN = {
    "input_features": ["re", "im"],          # raw coordinates only
    "width": 256,
    "n_hidden": 4,                            # hidden Linear(256,256) layers
    "n_classes": 10,
    "activation": "gelu",
    "optimizer": "adamw",
    "lr": 1e-3,
    "lr_schedule": "cosine",
    "weight_decay": 1e-4,
    "batch_size": 1024,
    "max_epochs": 300,
    "early_stop_metric": "val_loss",
    "checkpoint_every_steps": 200,
    "seeds": [0, 1, 2, 3, 4],
}

# Implementation default (the pre-registration fixes the early-stop *metric*, not the
# patience; this value is logged in every run config).
EARLY_STOP_PATIENCE = 20

# Class layout (PRE_REGISTRATION §2): periods 1-8 -> classes 0-7,
# "deeper interior" (bounded, period > 8 or unresolved) -> 8, "escapes" -> 9.
CLASS_NAMES = [f"period_{p}" for p in range(1, 9)] + ["deeper_interior", "escapes"]
N_CLASSES = len(CLASS_NAMES)


def label_to_class(row: dict) -> int:
    """Map a generator row (src.build_dataset) to the 10-class label."""
    if row["escaped"]:
        return 9
    p = row.get("period")
    if p is not None and 1 <= p <= 8:
        return p - 1
    return 8  # bounded, period > 8 or unresolved (includes boundary-unresolved points)


def load_split(data_dir: str | Path, split: str):
    """Load one split written by src.build_dataset.

    Returns (X float32 (N,2), y int64 (N,), rows list[dict]).
    """
    data_dir = Path(data_dir)
    X = np.load(data_dir / f"{split}.npy").astype(np.float32)
    rows = json.loads((data_dir / f"{split}_labels.json").read_text())
    y = np.array([label_to_class(r) for r in rows], dtype=np.int64)
    if len(X) != len(y):
        raise ValueError(f"{split}: points/labels length mismatch ({len(X)} vs {len(y)})")
    return X, y, rows


class MLP(nn.Module):
    """Frozen architecture (PRE_REGISTRATION §3).

    Layers: Linear(2,256) input projection -> GELU -> 4 × [Linear(256,256) -> GELU]
    -> Linear(256,10) head. 266,506 parameters.

    `forward_with_activations` exposes the five post-GELU hidden activations
    (h1 = after the input projection, ..., h5 = penultimate) for the probing and
    ablation protocols of PRE_REGISTRATION §4.
    """

    def __init__(self, in_dim: int = 2, width: int = FROZEN["width"],
                 n_hidden: int = FROZEN["n_hidden"], n_classes: int = FROZEN["n_classes"]):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(n_hidden))
        self.act = nn.GELU()
        self.head = nn.Linear(width, n_classes)

    def forward_with_activations(self, x: torch.Tensor):
        acts = []
        h = self.act(self.input_proj(x))
        acts.append(h)
        for layer in self.hidden:
            h = self.act(layer(h))
            acts.append(h)
        return self.head(h), acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_activations(x)
        return logits

    @property
    def n_activation_sites(self) -> int:
        return 1 + len(self.hidden)


def init_model(seed: int) -> MLP:
    """Seeded model factory. The UNTRAINED probe-power control (PRE_REGISTRATION §4a)
    must reproduce a run's initialization exactly: training uses this same factory."""
    torch.manual_seed(seed)
    return MLP()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_optimizer(model: nn.Module, lr: float = FROZEN["lr"],
                    weight_decay: float = FROZEN["weight_decay"]) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    """Cosine decay of lr over the full (max-epoch) step budget."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             batch_size: int = 8192) -> tuple[float, float]:
    """Mean cross-entropy loss and accuracy over (X, y) already on the model device."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total_loss += float(loss_fn(logits, y[i:i + batch_size]))
        correct += int((logits.argmax(dim=1) == y[i:i + batch_size]).sum())
    return total_loss / len(X), correct / len(X)


def _save_checkpoint(path: Path, model: nn.Module, step: int, epoch: int) -> None:
    torch.save({"model": model.state_dict(), "step": step, "epoch": epoch}, path)


def load_model(checkpoint: str | Path, device: torch.device | str = "cpu") -> MLP:
    state = torch.load(checkpoint, map_location=device)
    model = MLP()
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device)
    model.eval()
    return model


def train_one_seed(seed: int, data_dir: str | Path, out_dir: str | Path,
                   device: str | None = None,
                   max_epochs: int = FROZEN["max_epochs"],
                   batch_size: int = FROZEN["batch_size"],
                   lr: float = FROZEN["lr"],
                   weight_decay: float = FROZEN["weight_decay"],
                   checkpoint_every: int = FROZEN["checkpoint_every_steps"],
                   patience: int = EARLY_STOP_PATIENCE,
                   shuffle_labels: bool = False,
                   verbose: bool = True) -> dict:
    """Train one seed. Returns a summary dict; writes to out_dir:
      config.json   — full effective config + seed, written BEFORE training starts
      step_*.pt     — checkpoints every `checkpoint_every` optimizer steps (incl. init)
      best.pt       — best-val-loss model
      final.pt      — model at stop time
      history.json  — per-epoch train/val loss curves (run artifact, not committed)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)

    config = dict(FROZEN)
    config.update({
        "seed": seed,
        "shuffle_labels": shuffle_labels,
        "effective": {
            "max_epochs": max_epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "checkpoint_every_steps": checkpoint_every,
            "early_stop_patience": patience,
        },
        "device": str(dev),
        "data_dir": str(data_dir),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "started_unix": time.time(),
    })
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    X_tr, y_tr, _ = load_split(data_dir, "train")
    X_va, y_va, _ = load_split(data_dir, "val")
    if shuffle_labels:
        # Probe-power control (PRE_REGISTRATION §4b): permute TRAIN labels only.
        rng = np.random.default_rng(seed)
        y_tr = y_tr[rng.permutation(len(y_tr))]

    model = init_model(seed).to(dev)
    _save_checkpoint(out_dir / "step_0000000.pt", model, 0, 0)

    X_tr_t = torch.from_numpy(X_tr).to(dev)
    y_tr_t = torch.from_numpy(y_tr).to(dev)
    X_va_t = torch.from_numpy(X_va).to(dev)
    y_va_t = torch.from_numpy(y_va).to(dev)

    steps_per_epoch = math.ceil(len(X_tr) / batch_size)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=steps_per_epoch * max_epochs)
    loss_fn = nn.CrossEntropyLoss()
    shuffle_gen = torch.Generator().manual_seed(seed)  # CPU generator: MPS-safe

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, since_best, global_step, stopped_at = float("inf"), 0, 0, max_epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=shuffle_gen).to(dev)
        epoch_loss = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.detach()) * len(idx)
            if global_step % checkpoint_every == 0:
                _save_checkpoint(out_dir / f"step_{global_step:07d}.pt",
                                 model, global_step, epoch)
        val_loss, val_acc = evaluate(model, X_va_t, y_va_t)
        history["train_loss"].append(epoch_loss / len(X_tr))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"[seed {seed}] epoch {epoch:3d}  train {epoch_loss / len(X_tr):.4f}"
                  f"  val {val_loss:.4f}  val_acc {val_acc:.4f}")
        if val_loss < best_val - 1e-6:
            best_val, since_best = val_loss, 0
            _save_checkpoint(out_dir / "best.pt", model, global_step, epoch)
        else:
            since_best += 1
            if since_best >= patience:
                stopped_at = epoch
                if verbose:
                    print(f"[seed {seed}] early stop at epoch {epoch} (val loss)")
                break

    _save_checkpoint(out_dir / "final.pt", model, global_step, stopped_at)
    (out_dir / "history.json").write_text(json.dumps(history))
    return {"seed": seed, "stopped_epoch": stopped_at, "global_steps": global_step,
            "best_val_loss": best_val, "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--seeds", type=int, nargs="+", default=FROZEN["seeds"])
    ap.add_argument("--device", default=None, help="cpu | mps (default: auto)")
    ap.add_argument("--max-epochs", type=int, default=FROZEN["max_epochs"])
    ap.add_argument("--batch-size", type=int, default=FROZEN["batch_size"])
    ap.add_argument("--checkpoint-every", type=int,
                    default=FROZEN["checkpoint_every_steps"])
    ap.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="probe-power control: train on seed-permuted labels")
    args = ap.parse_args()

    for seed in args.seeds:
        run_name = f"seed{seed}" + ("_shuffled" if args.shuffle_labels else "")
        summary = train_one_seed(
            seed, args.data_dir, Path(args.out_dir) / run_name,
            device=args.device, max_epochs=args.max_epochs,
            batch_size=args.batch_size, checkpoint_every=args.checkpoint_every,
            patience=args.patience, shuffle_labels=args.shuffle_labels)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
