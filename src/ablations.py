"""Causal ablations and activation patching (PRE_REGISTRATION §4, frozen protocol; H2).

Ablation protocol:
  1. Fit the top-k (pre-registered k = 3) PROBE DIRECTIONS for a target (default |λ|)
     at the penultimate layer: iteratively fit a ridge probe, take its normalized weight
     vector, project it out of the activations, and refit (nullspace-projection style),
     yielding k orthonormal directions ranked by probe importance.
  2. Zero-ablate (remove the activation component along each direction) and mean-ablate
     (replace each component with its train-split mean) at the penultimate layer, and
     measure the drop in period-classification accuracy on the test split.
  3. Compare against RANK-MATCHED random ablations: k random orthonormal directions
     (seeded), same ablation modes, repeated over `n_random` draws.
H2 (locked): top-3 probe-direction ablation drops accuracy >= 20 points while the
rank-matched random ablation drops < 5.

Activation patching (frozen): swap PENULTIMATE activations between pairs of points in
the SAME hyperbolic component (same nearest component center and same period label) at
DIFFERENT |λ|, and measure the period-prediction change. Component membership is
operationalized as (nearest center of period <= 8, identical period label); the pair
list and |λ| gap threshold are logged with every run.

Usage (at activation, not before):
  python -m src.ablations --checkpoint checkpoints/seed0/best.pt --data-dir data \
      --out results/ablations_seed0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train import MLP, ROOT, load_model, load_split, pick_device
from .probes import (RIDGE_ALPHAS, component_centers, distance_to_nearest_center,
                     get_activations, r2_score, ridge_fit, target_arrays)

TOP_K = 3            # pre-registered: top-3 probe directions
N_RANDOM_DRAWS = 10  # rank-matched random-direction draws (seeded)
MIN_DLAMBDA = 0.3    # |λ| gap for "different internal coordinate" patch pairs (logged)


# ---------------------------------------------------------------------------
# Probe directions and ablation edits
# ---------------------------------------------------------------------------

def _fit_direction(A: np.ndarray, y: np.ndarray, A_val: np.ndarray, y_val: np.ndarray,
                   alphas=RIDGE_ALPHAS) -> np.ndarray:
    """Best ridge probe (alpha on val) for y from activations A; unit weight vector."""
    best_w, best_r2 = None, -np.inf
    for alpha in alphas:
        w, b = ridge_fit(A, y, alpha)
        r2 = r2_score(y_val, A_val @ w + b)
        if r2 > best_r2:
            best_w, best_r2 = w, r2
    return best_w / np.linalg.norm(best_w)


def topk_probe_directions(A_tr: np.ndarray, y_tr: np.ndarray,
                          A_va: np.ndarray, y_va: np.ndarray,
                          k: int = TOP_K) -> np.ndarray:
    """k orthonormal probe directions, ranked: fit probe -> take unit weight vector ->
    project out of the activations -> refit. Returns (k, d)."""
    A_tr, A_va = A_tr.astype(np.float64).copy(), A_va.astype(np.float64).copy()
    dirs: list[np.ndarray] = []
    for _ in range(k):
        d = _fit_direction(A_tr, y_tr, A_va, y_va)
        for prev in dirs:  # numerical re-orthogonalization
            d = d - (d @ prev) * prev
        d = d / np.linalg.norm(d)
        dirs.append(d)
        A_tr -= np.outer(A_tr @ d, d)
        A_va -= np.outer(A_va @ d, d)
    return np.stack(dirs)


def random_orthonormal_directions(k: int, dim: int, seed: int) -> np.ndarray:
    """Rank-matched control: k random orthonormal directions in R^dim (seeded)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, k))
    Q, _ = np.linalg.qr(M)
    return Q[:, :k].T


def ablate_directions(A: np.ndarray, directions: np.ndarray, mode: str = "zero",
                      mean_coeffs: np.ndarray | None = None) -> np.ndarray:
    """Edit activations along a set of directions.
      zero: remove the component along each direction.
      mean: replace the component with `mean_coeffs` (per-direction reference means,
            computed on the train split)."""
    A = np.asarray(A, dtype=np.float64)
    coeffs = A @ directions.T                       # (N, k)
    if mode == "zero":
        replacement = np.zeros_like(coeffs)
    elif mode == "mean":
        if mean_coeffs is None:
            raise ValueError("mean ablation needs reference mean_coeffs")
        replacement = np.broadcast_to(mean_coeffs, coeffs.shape)
    else:
        raise ValueError(f"unknown ablation mode: {mode}")
    return A + (replacement - coeffs) @ directions


@torch.no_grad()
def head_accuracy(model: MLP, penult: np.ndarray, y: np.ndarray,
                  device: torch.device | str = "cpu", batch_size: int = 8192) -> float:
    """Period accuracy when the (possibly edited) penultimate activations are pushed
    through the frozen classification head."""
    model.eval()
    correct = 0
    A = torch.from_numpy(np.asarray(penult, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    for i in range(0, len(A), batch_size):
        logits = model.head(A[i:i + batch_size].to(device)).cpu()
        correct += int((logits.argmax(dim=1) == yt[i:i + batch_size]).sum())
    return correct / len(A)


def ablation_experiment(model: MLP, data_dir: str | Path,
                        device: torch.device | str = "cpu", target: str = "abs_lambda",
                        k: int = TOP_K, n_random: int = N_RANDOM_DRAWS, seed: int = 0,
                        max_points: int | None = None) -> dict:
    """Full H2 protocol at the penultimate layer. Returns accuracies for: clean,
    zero/mean ablation of top-k probe directions, and zero/mean ablation of
    rank-matched random directions (n_random seeded draws)."""
    rng = np.random.default_rng(seed)
    splits = {}
    for name in ("train", "val", "test"):
        X, y, rows = load_split(data_dir, name)
        if max_points is not None and len(X) > max_points:
            keep = rng.choice(len(X), max_points, replace=False)
            X, y, rows = X[keep], y[keep], [rows[j] for j in keep]
        pts = X[:, 0].astype(np.float64) + 1j * X[:, 1].astype(np.float64)
        acts = get_activations(model, X, device)
        splits[name] = {"X": X, "y": y, "rows": rows, "pts": pts,
                        "penult": acts[-1]}

    # probe directions for the target, fit on train/val (test held out for accuracy)
    tgt_tr = target_arrays(splits["train"]["rows"], splits["train"]["pts"])[target]
    tgt_va = target_arrays(splits["val"]["rows"], splits["val"]["pts"])[target]
    m_tr, y_tr_t = tgt_tr
    m_va, y_va_t = tgt_va
    dirs = topk_probe_directions(splits["train"]["penult"][m_tr], y_tr_t[m_tr],
                                 splits["val"]["penult"][m_va], y_va_t[m_va], k=k)

    A_test, y_test = splits["test"]["penult"].astype(np.float64), splits["test"]["y"]
    mean_ref = (splits["train"]["penult"].astype(np.float64) @ dirs.T).mean(axis=0)

    out = {"config": {"target": target, "k": k, "n_random": n_random, "seed": seed,
                      "layer": "penultimate", "max_points": max_points},
           "acc_clean": head_accuracy(model, A_test, y_test, device),
           "probe_dirs": {
               "zero": head_accuracy(model, ablate_directions(A_test, dirs, "zero"),
                                     y_test, device),
               "mean": head_accuracy(model, ablate_directions(A_test, dirs, "mean",
                                                              mean_ref), y_test, device),
           },
           "random_dirs": {"zero": [], "mean": []}}
    dim = A_test.shape[1]
    A_train = splits["train"]["penult"].astype(np.float64)
    for draw in range(n_random):
        R = random_orthonormal_directions(k, dim, seed=seed * 1000 + draw)
        r_mean_ref = (A_train @ R.T).mean(axis=0)
        out["random_dirs"]["zero"].append(
            head_accuracy(model, ablate_directions(A_test, R, "zero"), y_test, device))
        out["random_dirs"]["mean"].append(
            head_accuracy(model, ablate_directions(A_test, R, "mean", r_mean_ref),
                          y_test, device))
    return out


# ---------------------------------------------------------------------------
# Activation patching (same component, different |λ|)
# ---------------------------------------------------------------------------

def find_patch_pairs(rows: list[dict], pts: np.ndarray,
                     min_dlambda: float = MIN_DLAMBDA, max_pairs: int = 500,
                     centers: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Index pairs (i, j) of interior points in the same component (same nearest
    component center AND same period) with | |λ(i)| - |λ(j)| | >= min_dlambda."""
    if centers is None:
        centers = component_centers()
    interior = [i for i, r in enumerate(rows)
                if not r["escaped"] and r["abs_lambda"] is not None
                and r["period"] is not None]
    if not interior:
        return []
    ipts = pts[interior]
    nearest = np.empty(len(ipts), dtype=np.int64)
    for s in range(0, len(ipts), 65536):
        nearest[s:s + 65536] = np.abs(ipts[s:s + 65536, None]
                                      - centers[None, :]).argmin(axis=1)
    groups: dict[tuple[int, int], list[int]] = {}
    for local, i in enumerate(interior):
        groups.setdefault((int(nearest[local]), rows[i]["period"]), []).append(i)
    pairs: list[tuple[int, int]] = []
    for members in groups.values():
        members = sorted(members, key=lambda i: rows[i]["abs_lambda"])
        lo, hi = 0, len(members) - 1
        while lo < hi and len(pairs) < max_pairs:
            a, b = members[lo], members[hi]
            if rows[b]["abs_lambda"] - rows[a]["abs_lambda"] >= min_dlambda:
                pairs.append((a, b))
                lo, hi = lo + 1, hi - 1
            else:
                break
        if len(pairs) >= max_pairs:
            break
    return pairs


@torch.no_grad()
def activation_patching(model: MLP, X: np.ndarray, pairs: list[tuple[int, int]],
                        device: torch.device | str = "cpu") -> dict:
    """Swap penultimate activations within each pair and measure the period-prediction
    change (PRE_REGISTRATION §4). Reports, over donor->receiver patches in both
    directions: fraction of changed predicted classes and mean max-|Δlogit|."""
    if not pairs:
        return {"n_pairs": 0, "note": "no qualifying same-component pairs"}
    model.eval()
    idx = sorted({i for p in pairs for i in p})
    pos = {i: n for n, i in enumerate(idx)}
    Xt = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
    logits, acts = model.forward_with_activations(Xt)
    logits = logits.cpu().numpy()
    penult = acts[-1]
    preds = logits.argmax(axis=1)

    changed, deltas = 0, []
    for a, b in pairs:
        for src, dst in ((a, b), (b, a)):
            patched = model.head(penult[pos[src]:pos[src] + 1]).cpu().numpy()[0]
            base = logits[pos[dst]]
            changed += int(patched.argmax() != preds[pos[dst]])
            deltas.append(float(np.max(np.abs(patched - base))))
    n_patches = 2 * len(pairs)
    return {"n_pairs": len(pairs), "n_patches": n_patches,
            "frac_pred_changed": changed / n_patches,
            "mean_max_abs_logit_delta": float(np.mean(deltas))}


def patching_experiment(model: MLP, data_dir: str | Path,
                        device: torch.device | str = "cpu",
                        min_dlambda: float = MIN_DLAMBDA, max_pairs: int = 500,
                        max_points: int | None = None, seed: int = 0) -> dict:
    X, _, rows = load_split(data_dir, "test")
    if max_points is not None and len(X) > max_points:
        keep = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
        X, rows = X[keep], [rows[j] for j in keep]
    pts = X[:, 0].astype(np.float64) + 1j * X[:, 1].astype(np.float64)
    pairs = find_patch_pairs(rows, pts, min_dlambda=min_dlambda, max_pairs=max_pairs)
    result = activation_patching(model, X, pairs, device)
    result["config"] = {"min_dlambda": min_dlambda, "max_pairs": max_pairs,
                        "split": "test", "max_points": max_points,
                        "component_id": "nearest center (period <= 8) + period label"}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--target", default="abs_lambda",
                    choices=("abs_lambda", "green", "dist_center"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    results = {
        "source": {"checkpoint": args.checkpoint},
        "ablation": ablation_experiment(model, args.data_dir, device,
                                        target=args.target, seed=args.seed),
        "patching": patching_experiment(model, args.data_dir, device, seed=args.seed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
