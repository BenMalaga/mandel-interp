"""Build the Mandel-Interp labeled dataset (PRE_REGISTRATION §2).

Deterministic (NumPy PCG64, seed 20260611). Samples c uniformly over the window plus a
boundary-enriched stratum (rejection toward |lambda| ~ 1 / the escape boundary), labels each
point via src.dynamics, and writes train/val/test splits with a committed manifest of counts
and content hashes. Raw arrays go under data/ (gitignored); only the manifest + a small
released sample live in results/.

Usage:
  python -m src.build_dataset --n 2000000 --boundary-frac 0.35
  python -m src.build_dataset --smoke           # tiny, for validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .dynamics import label_point

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 20260611
WINDOW = {"re": (-2.5, 1.0), "im": (-1.5, 1.5)}   # frozen sampling region (§2)
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}


def uniform_points(rng: np.random.Generator, n: int) -> np.ndarray:
    re = rng.uniform(*WINDOW["re"], size=n)
    im = rng.uniform(*WINDOW["im"], size=n)
    return re + 1j * im


def boundary_enriched(rng: np.random.Generator, n: int, max_iter: int = 2000) -> np.ndarray:
    """Rejection-sample points whose smooth escape count is moderate (near the boundary):
    these are the hard, decision-relevant points where memorization vs structure diverges."""
    out = np.empty(n, dtype=np.complex128)
    filled = 0
    while filled < n:
        cand = uniform_points(rng, max(n, 4096))
        for c in cand:
            L = label_point(complex(c), max_iter=max_iter)
            near = (L.escaped and 8 < (L.smooth_iters or 0) < 200) or \
                   (not L.escaped and L.abs_lambda is not None and L.abs_lambda > 0.6)
            if near:
                out[filled] = c
                filled += 1
                if filled >= n:
                    break
    return out


def label_array(points: np.ndarray, max_iter: int) -> list[dict]:
    return [label_point(complex(c), max_iter=max_iter).as_row() for c in points]


def _hash_rows(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(repr(sorted(r.items())).encode())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--boundary-frac", type=float, default=0.35)
    ap.add_argument("--max-iter", type=int, default=50_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.max_iter = 600, 2000

    rng = np.random.default_rng(SEED)
    n_bound = int(args.n * args.boundary_frac)
    n_unif = args.n - n_bound
    pts = np.concatenate([uniform_points(rng, n_unif),
                          boundary_enriched(rng, n_bound, max_iter=min(args.max_iter, 5000))])
    rng.shuffle(pts)

    rows = label_array(pts, args.max_iter)
    # split by index (seeded shuffle already applied)
    n = len(rows)
    i_tr, i_va = int(n * SPLIT["train"]), int(n * (SPLIT["train"] + SPLIT["val"]))
    splits = {"train": rows[:i_tr], "val": rows[i_tr:i_va], "test": rows[i_va:]}

    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    manifest = {"seed": SEED, "window": WINDOW, "n_total": n,
                "boundary_frac": args.boundary_frac, "max_iter": args.max_iter, "splits": {}}
    for name, rs in splits.items():
        np.save(DATA / f"{name}.npy", np.array([(r["re"], r["im"]) for r in rs]))
        (DATA / f"{name}_labels.json").write_text(json.dumps(rs))
        n_esc = sum(r["escaped"] for r in rs)
        manifest["splits"][name] = {
            "n": len(rs), "n_escaped": n_esc, "n_interior": len(rs) - n_esc,
            "sha16": _hash_rows(rs),
        }
    (RESULTS / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
