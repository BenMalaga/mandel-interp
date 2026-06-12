"""Probing harness (PRE_REGISTRATION §4, frozen protocol).

For every hidden layer of a trained period-classification MLP we fit
  - a LINEAR probe (ridge regression; L2 strength selected on the val split), and
  - a 2-HIDDEN-LAYER MLP probe (256 -> 64 -> 64 -> 1, GELU; L2 = AdamW weight decay,
    selected on the val split),
predicting three targets:
  - abs_lambda : |λ(c)|, the multiplier-map internal coordinate (interior points only),
  - green      : the Green's function G(c) (escape-side points only),
  - dist_center: CONTROL target — Euclidean distance from c to the nearest
                 hyperbolic-component center of period ≤ 8 (defined for all points;
                 centers computed deterministically by Newton's method on the
                 critical-orbit polynomial P_p(c), the c with superattracting p-cycles).
Metric: held-out R² on the test split (selection only ever touches val).

Probe-power controls (PRE_REGISTRATION §4, load-bearing):
  (a) UNTRAINED control — identical probes on a same-init untrained net
      (`--untrained --seed N` reproduces the init of training seed N exactly, because
      training uses the same seeded factory `src.train.init_model`).
  (b) SHUFFLED-LABEL control — identical probes on a net trained with
      `python -m src.train --shuffle-labels` (point `--checkpoint` at that run).
A representation claim (H1) requires the trained net to beat BOTH controls by the
pre-registered margin.

Usage (at activation, not before):
  python -m src.probes --checkpoint checkpoints/seed0/best.pt --data-dir data \
      --out results/probes_seed0.json
  python -m src.probes --untrained --seed 0 --data-dir data \
      --out results/probes_seed0_untrained.json
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .train import MLP, ROOT, init_model, load_model, load_split, pick_device

# L2 grids (selection on val; grids logged with every result).
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MLP_PROBE_WEIGHT_DECAYS = (1e-6, 1e-4, 1e-2)
MLP_PROBE_HIDDEN = 64
TARGET_NAMES = ("abs_lambda", "green", "dist_center")
CONTROL_MAX_PERIOD = 8  # centers of period <= 8, matching the labeled period classes


# ---------------------------------------------------------------------------
# Component centers (control target) — deterministic, no external data.
# ---------------------------------------------------------------------------

def critical_orbit_poly(c: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized evaluation of P_p(c) and dP_p/dc, where P_1 = c, P_{n+1} = P_n^2 + c.
    Roots of P_p are the centers of hyperbolic components of period dividing p."""
    z = c.copy()
    dz = np.ones_like(c)
    for _ in range(period - 1):
        dz = 2.0 * z * dz + 1.0
        z = z * z + c
    return z, dz


@functools.lru_cache(maxsize=4)
def component_centers(max_period: int = CONTROL_MAX_PERIOD,
                      grid: tuple[int, int] = (241, 201),
                      newton_iters: int = 80, tol: float = 1e-10) -> np.ndarray:
    """All hyperbolic-component centers with period 1..max_period (236 for max_period=8),
    found by vectorized Newton iteration on P_p(c) = 0 from a grid of starting points
    over the sampling window. Deterministic; cached."""
    re = np.linspace(-2.5, 1.0, grid[0])
    im = np.linspace(-1.5, 1.5, grid[1])
    starts = (re[:, None] + 1j * im[None, :]).ravel()
    found: list[np.ndarray] = []
    for p in range(1, max_period + 1):
        c = starts.copy()
        for _ in range(newton_iters):
            f, df = critical_orbit_poly(c, p)
            with np.errstate(all="ignore"):
                step = f / df
            step = np.where(np.isfinite(step), step, 0.0)
            c = c - step
        f, _ = critical_orbit_poly(c, p)
        ok = np.isfinite(c) & (np.abs(f) < tol)
        found.append(c[ok])
    roots = np.concatenate(found)
    # dedupe (roots of P_p include all periods dividing p, so there is heavy overlap)
    order = np.lexsort((roots.imag, roots.real))
    roots = roots[order]
    keep = [roots[0]]
    for r in roots[1:]:
        if abs(r - keep[-1]) > 1e-8 and all(abs(r - k) > 1e-8 for k in keep[-64:]):
            keep.append(r)
    return np.array(keep)


def distance_to_nearest_center(points: np.ndarray,
                               centers: np.ndarray | None = None) -> np.ndarray:
    """Control probe target: Euclidean distance in the c-plane from each point to the
    nearest component center. `points` is complex (N,)."""
    if centers is None:
        centers = component_centers()
    # chunked to keep memory modest on the full dataset
    out = np.empty(len(points), dtype=np.float64)
    for i in range(0, len(points), 65536):
        chunk = points[i:i + 65536]
        out[i:i + 65536] = np.abs(chunk[:, None] - centers[None, :]).min(axis=1)
    return out


# ---------------------------------------------------------------------------
# Activations and probe targets
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_activations(model: MLP, X: np.ndarray, device: torch.device | str = "cpu",
                    batch_size: int = 8192) -> list[np.ndarray]:
    """Post-GELU activations at every hidden layer. Returns a list of (N, width)
    float32 arrays, one per site (h1 = input projection, ..., h5 = penultimate)."""
    model.eval()
    chunks: list[list[np.ndarray]] = [[] for _ in range(model.n_activation_sites)]
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    for i in range(0, len(Xt), batch_size):
        _, acts = model.forward_with_activations(Xt[i:i + batch_size].to(device))
        for site, a in enumerate(acts):
            chunks[site].append(a.cpu().numpy())
    return [np.concatenate(c) for c in chunks]


def target_arrays(rows: list[dict], points: np.ndarray,
                  centers: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each probe target, (mask, y): which rows the target is defined on, and its
    value there. |λ| only exists on the interior, G only on the escape side
    (PRE_REGISTRATION §4 keeps the two probes clean); the control target is global."""
    lam = np.array([r["abs_lambda"] if r["abs_lambda"] is not None else np.nan
                    for r in rows])
    grn = np.array([r["green"] if (r["escaped"] and r["green"] is not None) else np.nan
                    for r in rows])
    dist = distance_to_nearest_center(points, centers)
    return {
        "abs_lambda": (~np.isnan(lam), np.nan_to_num(lam)),
        "green": (~np.isnan(grn), np.nan_to_num(grn)),
        "dist_center": (np.ones(len(rows), dtype=bool), dist),
    }


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination on held-out data."""
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Linear (ridge) probe
# ---------------------------------------------------------------------------

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge with intercept on centered data. Returns (w, b)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    d = X.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)
    return w, float(y_mean - x_mean @ w)


def fit_linear_probe(A_tr, y_tr, A_va, y_va, A_te, y_te,
                     alphas=RIDGE_ALPHAS) -> dict:
    """Ridge probe; alpha selected on val R²; reported metric is held-out test R²."""
    best = None
    for alpha in alphas:
        w, b = ridge_fit(A_tr, y_tr, alpha)
        val_r2 = r2_score(y_va, A_va @ w + b)
        if best is None or val_r2 > best["val_r2"]:
            best = {"alpha": alpha, "val_r2": val_r2, "w": w, "b": b}
    test_r2 = r2_score(y_te, A_te @ best["w"] + best["b"])
    return {"probe": "linear", "alpha": best["alpha"],
            "val_r2": best["val_r2"], "test_r2": test_r2,
            "w": best["w"], "b": best["b"]}


# ---------------------------------------------------------------------------
# 2-hidden-layer MLP probe
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """2-hidden-layer regression probe (PRE_REGISTRATION §4): d -> 64 -> 64 -> 1, GELU."""

    def __init__(self, in_dim: int, hidden: int = MLP_PROBE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp_probe(A_tr, y_tr, weight_decay: float, epochs: int, lr: float,
                     batch_size: int, seed: int, device) -> MLPProbe:
    torch.manual_seed(seed)
    probe = MLPProbe(A_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(np.asarray(A_tr, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.asarray(y_tr, dtype=np.float32)).to(device)
    gen = torch.Generator().manual_seed(seed)
    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=gen).to(device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def _mlp_probe_predict(probe: MLPProbe, A, device, batch_size: int = 8192) -> np.ndarray:
    Xt = torch.from_numpy(np.asarray(A, dtype=np.float32))
    out = []
    for i in range(0, len(Xt), batch_size):
        out.append(probe(Xt[i:i + batch_size].to(device)).cpu().numpy())
    return np.concatenate(out)


def fit_mlp_probe(A_tr, y_tr, A_va, y_va, A_te, y_te,
                  weight_decays=MLP_PROBE_WEIGHT_DECAYS, epochs: int = 200,
                  lr: float = 1e-3, batch_size: int = 4096, seed: int = 0,
                  device: torch.device | str = "cpu") -> dict:
    """2-hidden-layer probe; weight decay (L2) selected on val R²; reports test R²."""
    best = None
    for wd in weight_decays:
        probe = _train_mlp_probe(A_tr, y_tr, wd, epochs, lr, batch_size, seed, device)
        val_r2 = r2_score(y_va, _mlp_probe_predict(probe, A_va, device))
        if best is None or val_r2 > best["val_r2"]:
            best = {"weight_decay": wd, "val_r2": val_r2, "probe": probe}
    test_r2 = r2_score(y_te, _mlp_probe_predict(best["probe"], A_te, device))
    return {"probe": "mlp2", "weight_decay": best["weight_decay"],
            "val_r2": best["val_r2"], "test_r2": test_r2}


# ---------------------------------------------------------------------------
# Full protocol
# ---------------------------------------------------------------------------

def run_probes(model: MLP, data_dir: str | Path, device: torch.device | str = "cpu",
               probe_types: tuple[str, ...] = ("linear", "mlp2"),
               mlp_epochs: int = 200, seed: int = 0,
               max_points: int | None = None) -> dict:
    """Fit all probes on all hidden layers and all targets. Returns the nested results
    dict {layer: {target: {probe_type: {.., 'test_r2': float}}}} (weights stripped).

    `max_points` (logged when set) subsamples each split deterministically — used by
    the plumbing smoke test only; the pre-registered analysis runs on the full splits.
    """
    splits = {}
    for name in ("train", "val", "test"):
        X, y, rows = load_split(data_dir, name)
        if max_points is not None and len(X) > max_points:
            keep = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
            X, rows = X[keep], [rows[k] for k in keep]
        pts = X[:, 0].astype(np.float64) + 1j * X[:, 1].astype(np.float64)
        splits[name] = {"X": X, "rows": rows,
                        "acts": get_activations(model, X, device),
                        "targets": target_arrays(rows, pts)}

    n_layers = model.n_activation_sites
    results: dict = {"config": {"ridge_alphas": list(RIDGE_ALPHAS),
                                "mlp_weight_decays": list(MLP_PROBE_WEIGHT_DECAYS),
                                "mlp_epochs": mlp_epochs, "probe_types": list(probe_types),
                                "max_points": max_points, "n_layers": n_layers,
                                "penultimate_layer": f"layer_{n_layers}"},
                     "layers": {}}
    for li in range(n_layers):
        layer_name = f"layer_{li + 1}"
        results["layers"][layer_name] = {}
        for target in TARGET_NAMES:
            entry = {}
            masks = {s: splits[s]["targets"][target][0] for s in splits}
            if min(int(m.sum()) for m in masks.values()) < 10:
                results["layers"][layer_name][target] = {"skipped": "too few points"}
                continue
            arrs = {s: (splits[s]["acts"][li][masks[s]],
                        splits[s]["targets"][target][1][masks[s]]) for s in splits}
            if "linear" in probe_types:
                fit = fit_linear_probe(arrs["train"][0], arrs["train"][1],
                                       arrs["val"][0], arrs["val"][1],
                                       arrs["test"][0], arrs["test"][1])
                entry["linear"] = {k: v for k, v in fit.items() if k not in ("w", "b")}
            if "mlp2" in probe_types:
                entry["mlp2"] = fit_mlp_probe(arrs["train"][0], arrs["train"][1],
                                              arrs["val"][0], arrs["val"][1],
                                              arrs["test"][0], arrs["test"][1],
                                              epochs=mlp_epochs, seed=seed, device=device)
            results["layers"][layer_name][target] = entry
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="trained (or shuffled-label-trained) model .pt")
    ap.add_argument("--untrained", action="store_true",
                    help="probe-power control (a): same-init untrained net")
    ap.add_argument("--seed", type=int, default=0,
                    help="init seed for --untrained; also seeds the MLP probes")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mlp-epochs", type=int, default=200)
    args = ap.parse_args()
    if bool(args.checkpoint) == bool(args.untrained):
        ap.error("exactly one of --checkpoint / --untrained is required")

    device = pick_device(args.device)
    if args.untrained:
        model = init_model(args.seed).to(device)
        source = {"control": "untrained", "seed": args.seed}
    else:
        model = load_model(args.checkpoint, device)
        source = {"checkpoint": args.checkpoint}

    results = run_probes(model, args.data_dir, device=device,
                         mlp_epochs=args.mlp_epochs, seed=args.seed)
    results["source"] = source
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
