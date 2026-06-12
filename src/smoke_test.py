"""End-to-end PLUMBING smoke test — explicitly NOT a research run.

Purpose: verify that the pipeline pieces fit together (tiny dataset slice -> brief
training -> probes incl. both probe-power controls -> ablations + patching) without
errors. It is run with ~2,000 points, reduced max_iter, ONE seed, and a handful of
epochs, all far below the pre-registered scope (PRE_REGISTRATION §2-§4), so its
outputs are scientifically meaningless and are treated accordingly:

  HARD RULE: no accuracy, loss, R², or any other outcome number from this script is
  ever printed, recorded, committed, or reported. The only result it is allowed to
  produce is "ran without error" (or a traceback). All artifacts (data slice,
  checkpoints, probe/ablation outputs) live in a temporary directory that is deleted
  on exit, success or failure.

The real 5-seed run on the 2M-point dataset is gated and is NOT started here.

Usage:  python -m src.smoke_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

SMOKE_N = 2000          # ~2k points (vs pre-registered 2,000,000)
SMOKE_MAX_ITER = 2000   # reduced (vs pre-registered 50,000)
SMOKE_SEED = 0          # ONE seed (vs pre-registered {0..4})
SMOKE_EPOCHS = 3


def _say(msg: str) -> None:
    print(f"[smoke] {msg}")


def _build_tiny_slice(data_dir: Path) -> None:
    """Tiny dataset slice in the exact on-disk format of src.build_dataset (which is
    validated separately); written to the temp dir so the real data/ and the committed
    results/dataset_manifest.json are never touched."""
    from .build_dataset import SPLIT, boundary_enriched, label_array, uniform_points

    rng = np.random.default_rng(SMOKE_SEED)
    n_bound = int(SMOKE_N * 0.35)
    pts = np.concatenate([
        uniform_points(rng, SMOKE_N - n_bound),
        boundary_enriched(rng, n_bound, max_iter=1000),
    ])
    rng.shuffle(pts)
    rows = label_array(pts, SMOKE_MAX_ITER)
    n = len(rows)
    i_tr = int(n * SPLIT["train"])
    i_va = int(n * (SPLIT["train"] + SPLIT["val"]))
    splits = {"train": rows[:i_tr], "val": rows[i_tr:i_va], "test": rows[i_va:]}
    data_dir.mkdir(parents=True)
    for name, rs in splits.items():
        np.save(data_dir / f"{name}.npy", np.array([(r["re"], r["im"]) for r in rs]))
        (data_dir / f"{name}_labels.json").write_text(json.dumps(rs))


def main() -> None:
    from .ablations import ablation_experiment, patching_experiment
    from .probes import run_probes
    from .train import init_model, load_model, train_one_seed

    _say("PLUMBING TEST ONLY — outcome numbers are neither recorded nor reported.")
    with tempfile.TemporaryDirectory(prefix="mandel_smoke_") as td:
        tmp = Path(td)
        data_dir = tmp / "data"

        _say("stage 1/5: building tiny dataset slice (reduced max_iter) ...")
        _build_tiny_slice(data_dir)

        _say("stage 2/5: brief training, one seed (+ shuffled-label control net) ...")
        train_one_seed(SMOKE_SEED, data_dir, tmp / "ckpt", device="cpu",
                       max_epochs=SMOKE_EPOCHS, batch_size=256, checkpoint_every=5,
                       patience=SMOKE_EPOCHS + 1, verbose=False)
        train_one_seed(SMOKE_SEED, data_dir, tmp / "ckpt_shuffled", device="cpu",
                       max_epochs=1, batch_size=256, checkpoint_every=5,
                       patience=2, shuffle_labels=True, verbose=False)

        _say("stage 3/5: probes on the trained net (linear + 2-layer, all layers) ...")
        model = load_model(tmp / "ckpt" / "final.pt", "cpu")
        r = run_probes(model, data_dir, device="cpu", mlp_epochs=5, seed=SMOKE_SEED)
        assert r["layers"], "probe results empty"

        _say("stage 4/5: probe-power controls (untrained init + shuffled-label net) ...")
        r = run_probes(init_model(SMOKE_SEED), data_dir, device="cpu",
                       probe_types=("linear",), seed=SMOKE_SEED)
        assert r["layers"], "untrained-control probe results empty"
        shuffled = load_model(tmp / "ckpt_shuffled" / "final.pt", "cpu")
        r = run_probes(shuffled, data_dir, device="cpu",
                       probe_types=("linear",), seed=SMOKE_SEED)
        assert r["layers"], "shuffled-control probe results empty"

        _say("stage 5/5: ablations (top-3 vs rank-matched random) + patching ...")
        a = ablation_experiment(model, data_dir, device="cpu", n_random=3,
                                seed=SMOKE_SEED)
        assert "acc_clean" in a and a["random_dirs"]["zero"], "ablation results empty"
        p = patching_experiment(model, data_dir, device="cpu", min_dlambda=0.2)
        assert "n_pairs" in p, "patching results missing"

        # Results dicts go out of scope unread; the temp dir is deleted on exit.
    _say("OK — full pipeline ran without error (plumbing only; no outcome numbers "
         "recorded; all smoke artifacts deleted).")


if __name__ == "__main__":
    main()
