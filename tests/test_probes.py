"""Contract tests for the probing harness: math contracts of the estimators, shapes,
masks, result structure, and the deterministic component-center finder.

NO research-outcome assertions: synthetic data only; nothing here asserts what a
trained network represents.
"""

import numpy as np
import pytest

from src.probes import (CONTROL_MAX_PERIOD, MLP_PROBE_WEIGHT_DECAYS, MLPProbe,
                        RIDGE_ALPHAS, component_centers, critical_orbit_poly,
                        distance_to_nearest_center, fit_linear_probe, get_activations,
                        r2_score, ridge_fit, run_probes, target_arrays)
from src.train import init_model


# ---------------------------------------------------------------------------
# Component centers (control target)
# ---------------------------------------------------------------------------

def test_component_centers_complete_and_exact():
    """For periods 1..8 there are exactly 1+1+3+6+15+27+63+120 = 236 component centers
    (Moebius count of roots of the critical-orbit polynomials); the Newton finder must
    recover all of them, including the analytically-known low-period ones."""
    C = component_centers(CONTROL_MAX_PERIOD)
    assert len(C) == 236
    for known in (0 + 0j,                                   # main cardioid
                  -1 + 0j,                                  # period-2 disk
                  -1.7548776662466927 + 0j,                 # period-3 "airplane"
                  -0.12256116687665 + 0.74486176661974j):   # period-3 "rabbit"
        assert np.min(np.abs(C - known)) < 1e-8
    # all are genuine roots of some critical-orbit polynomial P_p, p <= 8
    resid = np.full(len(C), np.inf)
    for p in range(1, CONTROL_MAX_PERIOD + 1):
        f, _ = critical_orbit_poly(C, p)
        resid = np.minimum(resid, np.abs(f))
    assert np.all(resid < 1e-8)


def test_distance_to_nearest_center_contract():
    pts = np.array([0 + 0j, -1 + 0j, 0.01 + 0j])
    d = distance_to_nearest_center(pts)
    assert d.shape == (3,)
    assert d[0] == pytest.approx(0.0, abs=1e-8)
    assert d[1] == pytest.approx(0.0, abs=1e-8)
    assert d[2] == pytest.approx(0.01, abs=1e-6)


# ---------------------------------------------------------------------------
# Estimator math contracts (synthetic)
# ---------------------------------------------------------------------------

def test_r2_score_definition():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert r2_score(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert r2_score(np.ones(4), np.ones(4) * 2) == 0.0  # zero-variance guard


def test_ridge_recovers_synthetic_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16))
    w_true = rng.standard_normal(16)
    y = X @ w_true + 0.7
    w, b = ridge_fit(X, y, alpha=1e-8)
    assert np.allclose(w, w_true, atol=1e-6)
    assert b == pytest.approx(0.7, abs=1e-6)


def test_fit_linear_probe_selects_on_val_reports_test():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 16))
    w_true = rng.standard_normal(16)
    y = X @ w_true
    out = fit_linear_probe(X[:200], y[:200], X[200:250], y[200:250],
                           X[250:], y[250:])
    assert out["probe"] == "linear"
    assert out["alpha"] in RIDGE_ALPHAS  # L2 selected from the declared grid
    assert isinstance(out["val_r2"], float) and isinstance(out["test_r2"], float)
    assert out["w"].shape == (16,)


def test_mlp_probe_is_two_hidden_layers():
    import torch
    probe = MLPProbe(in_dim=256)
    linears = [m for m in probe.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3  # 2 hidden + 1 output = 2-hidden-layer probe
    assert linears[0].in_features == 256
    assert linears[-1].out_features == 1
    out = probe(torch.randn(9, 256))
    assert out.shape == (9,)
    assert len(MLP_PROBE_WEIGHT_DECAYS) >= 2  # L2 grid exists for val selection


# ---------------------------------------------------------------------------
# Activations, targets, protocol structure
# ---------------------------------------------------------------------------

def test_get_activations_shapes():
    model = init_model(0)
    X = np.random.default_rng(2).uniform(-2, 1, size=(33, 2)).astype(np.float32)
    acts = get_activations(model, X, device="cpu", batch_size=8)
    assert len(acts) == 5
    assert all(a.shape == (33, 256) for a in acts)


def test_target_arrays_masks_keep_domains_clean():
    """|λ| only on the interior, G only on the escape side (PRE_REGISTRATION §4)."""
    rows = [
        {"re": 0.0, "im": 0.0, "escaped": False, "smooth_iters": None, "green": None,
         "period": 1, "abs_lambda": 0.2},
        {"re": 1.0, "im": 1.0, "escaped": True, "smooth_iters": 3.0, "green": 0.5,
         "period": None, "abs_lambda": None},
        {"re": 0.25, "im": 0.0, "escaped": False, "smooth_iters": None, "green": None,
         "period": None, "abs_lambda": None},  # bounded, unresolved
    ]
    pts = np.array([0 + 0j, 1 + 1j, 0.25 + 0j])
    t = target_arrays(rows, pts)
    assert set(t) == {"abs_lambda", "green", "dist_center"}
    np.testing.assert_array_equal(t["abs_lambda"][0], [True, False, False])
    np.testing.assert_array_equal(t["green"][0], [False, True, False])
    np.testing.assert_array_equal(t["dist_center"][0], [True, True, True])
    assert t["abs_lambda"][1][0] == pytest.approx(0.2)
    assert t["green"][1][1] == pytest.approx(0.5)


def test_run_probes_result_structure(synth_data_dir):
    """Structure only, values are never asserted (they are outcomes)."""
    model = init_model(0)  # untrained: structure is identical to the real protocol
    res = run_probes(model, synth_data_dir, device="cpu",
                     probe_types=("linear", "mlp2"), mlp_epochs=1, seed=0)
    assert res["config"]["penultimate_layer"] == "layer_5"
    assert sorted(res["layers"]) == [f"layer_{i}" for i in range(1, 6)]
    for layer in res["layers"].values():
        assert set(layer) == {"abs_lambda", "green", "dist_center"}
        for entry in layer.values():
            assert "skipped" in entry or (
                isinstance(entry["linear"]["test_r2"], float)
                and isinstance(entry["mlp2"]["test_r2"], float))
