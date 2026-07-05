"""Contract tests for ablations + activation patching: orthonormality, the algebra of
the activation edits, pair-finding rules, and patching mechanics.

NO research-outcome assertions, synthetic activations and untrained nets only.
"""

import numpy as np
import pytest
import torch

from src.ablations import (MIN_DLAMBDA, N_RANDOM_DRAWS, TOP_K, ablate_directions,
                           activation_patching, find_patch_pairs, head_accuracy,
                           random_orthonormal_directions, topk_probe_directions)
from src.train import init_model


def test_frozen_protocol_constants():
    assert TOP_K == 3          # pre-registered: top-3 probe directions
    assert N_RANDOM_DRAWS >= 2  # rank-matched random control is multi-draw


def test_topk_probe_directions_orthonormal():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((400, 32))
    y = A @ rng.standard_normal(32)
    A_va = rng.standard_normal((100, 32))
    y_va = A_va @ np.zeros(32)
    D = topk_probe_directions(A, y, A_va, y_va, k=3)
    assert D.shape == (3, 32)
    np.testing.assert_allclose(D @ D.T, np.eye(3), atol=1e-8)


def test_random_directions_seeded_and_orthonormal():
    D1 = random_orthonormal_directions(3, 64, seed=7)
    D2 = random_orthonormal_directions(3, 64, seed=7)
    D3 = random_orthonormal_directions(3, 64, seed=8)
    assert D1.shape == (3, 64)
    np.testing.assert_allclose(D1 @ D1.T, np.eye(3), atol=1e-10)
    np.testing.assert_array_equal(D1, D2)         # reproducible
    assert not np.allclose(D1, D3)                # seed-sensitive


def test_zero_ablation_removes_component():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((50, 16))
    D = random_orthonormal_directions(3, 16, seed=0)
    A_abl = ablate_directions(A, D, mode="zero")
    np.testing.assert_allclose(A_abl @ D.T, 0.0, atol=1e-10)
    # off-subspace components untouched
    perp = np.eye(16) - D.T @ D
    np.testing.assert_allclose(A_abl @ perp, A @ perp, atol=1e-10)


def test_mean_ablation_sets_reference_coefficients():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((50, 16))
    D = random_orthonormal_directions(2, 16, seed=1)
    mean_ref = np.array([0.3, -1.2])
    A_abl = ablate_directions(A, D, mode="mean", mean_coeffs=mean_ref)
    coeffs = A_abl @ D.T
    np.testing.assert_allclose(coeffs, np.broadcast_to(mean_ref, coeffs.shape),
                               atol=1e-10)
    with pytest.raises(ValueError):
        ablate_directions(A, D, mode="mean")  # reference means are mandatory


def test_head_accuracy_contract():
    model = init_model(0)
    A = np.random.default_rng(3).standard_normal((40, 256)).astype(np.float32)
    with torch.no_grad():
        y = model.head(torch.from_numpy(A)).argmax(dim=1).numpy()
    acc = head_accuracy(model, A, y, device="cpu", batch_size=16)
    assert acc == pytest.approx(1.0)  # mechanical identity, not an outcome
    acc2 = head_accuracy(model, A, (y + 1) % 10, device="cpu")
    assert 0.0 <= acc2 <= 1.0


def _interior_row(re, im, period, lam):
    return {"re": re, "im": im, "escaped": False, "smooth_iters": None,
            "green": None, "period": period, "abs_lambda": lam}


def test_find_patch_pairs_same_component_different_lambda():
    # synthetic points clustered at two known component centers (0: period 1; -1: period 2)
    rows = [_interior_row(0.01, 0.00, 1, 0.05),
            _interior_row(0.00, 0.02, 1, 0.90),
            _interior_row(-0.01, 0.01, 1, 0.50),
            _interior_row(-1.00, 0.02, 2, 0.10),
            _interior_row(-1.02, 0.00, 2, 0.80),
            {"re": 1.0, "im": 1.0, "escaped": True, "smooth_iters": 2.0,
             "green": 0.4, "period": None, "abs_lambda": None}]
    pts = np.array([complex(r["re"], r["im"]) for r in rows])
    pairs = find_patch_pairs(rows, pts, min_dlambda=MIN_DLAMBDA)
    assert pairs, "qualifying same-component pairs must be found"
    for a, b in pairs:
        assert rows[a]["period"] == rows[b]["period"]          # same component
        assert not rows[a]["escaped"] and not rows[b]["escaped"]
        assert abs(rows[a]["abs_lambda"] - rows[b]["abs_lambda"]) >= MIN_DLAMBDA
    # escaped point never participates
    assert all(5 not in p for p in pairs)


def test_find_patch_pairs_empty_when_no_interior():
    rows = [{"re": 1.0, "im": 1.0, "escaped": True, "smooth_iters": 2.0,
             "green": 0.4, "period": None, "abs_lambda": None}]
    assert find_patch_pairs(rows, np.array([1 + 1j])) == []


def test_activation_patching_mechanics():
    """Full penultimate swap means the patched prediction equals the donor's
    prediction, verified against a direct forward pass (mechanics, not outcomes)."""
    model = init_model(0)
    X = np.random.default_rng(4).uniform(-2, 1, size=(4, 2)).astype(np.float32)
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).argmax(dim=1).numpy()
    res = activation_patching(model, X, [(0, 1)], device="cpu")
    assert res["n_pairs"] == 1 and res["n_patches"] == 2
    assert res["frac_pred_changed"] == pytest.approx(float(preds[0] != preds[1]))
    assert res["mean_max_abs_logit_delta"] >= 0.0
    # empty pair list degrades gracefully
    assert activation_patching(model, X, [], device="cpu")["n_pairs"] == 0
