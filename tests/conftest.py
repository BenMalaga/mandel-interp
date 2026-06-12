"""Shared fixtures: tiny SYNTHETIC datasets for plumbing/contract tests.

These rows are random placeholders in the generator's on-disk format — they carry no
dynamics information and exist only so shape/config/contract tests never touch the
real (gated) dataset build or assert anything about research outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def make_synthetic_rows(rng: np.random.Generator, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        re = float(rng.uniform(-2.5, 1.0))
        im = float(rng.uniform(-1.5, 1.5))
        if i % 2 == 0:  # "escaped" placeholder
            rows.append({"re": re, "im": im, "escaped": True,
                         "smooth_iters": float(rng.uniform(1.0, 100.0)),
                         "green": float(rng.uniform(0.0, 1.0)),
                         "period": None, "abs_lambda": None})
        else:           # "interior" placeholder
            rows.append({"re": re, "im": im, "escaped": False,
                         "smooth_iters": None, "green": None,
                         "period": int(rng.integers(1, 9)),
                         "abs_lambda": float(rng.uniform(0.0, 0.99))})
    return rows


def write_split(data_dir: Path, name: str, rows: list[dict]) -> None:
    np.save(data_dir / f"{name}.npy", np.array([(r["re"], r["im"]) for r in rows]))
    (data_dir / f"{name}_labels.json").write_text(json.dumps(rows))


@pytest.fixture(scope="session")
def synth_data_dir(tmp_path_factory) -> Path:
    """train/val/test splits of synthetic rows in the build_dataset on-disk format."""
    d = tmp_path_factory.mktemp("synth_data")
    rng = np.random.default_rng(123)
    for name, n in (("train", 64), ("val", 24), ("test", 24)):
        write_split(d, name, make_synthetic_rows(rng, n))
    return d
