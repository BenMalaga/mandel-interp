"""Contract tests for the training harness: shapes, frozen config, checkpoint cadence.

NO outcome assertions (no accuracy/loss/R² thresholds), those are reserved for the
pre-registered analysis. Training here happens only on tiny SYNTHETIC data (conftest)
to exercise plumbing.
"""

import json

import pytest
import torch

from src.train import (EARLY_STOP_PATIENCE, FROZEN, MLP, N_CLASSES, build_optimizer,
                       build_scheduler, count_params, init_model, label_to_class,
                       load_model, load_split, train_one_seed)


def test_frozen_config_matches_prereg():
    """PRE_REGISTRATION §3, verbatim."""
    assert FROZEN["input_features"] == ["re", "im"]      # raw coordinates only
    assert FROZEN["width"] == 256
    assert FROZEN["n_hidden"] == 4
    assert FROZEN["n_classes"] == 10
    assert FROZEN["activation"] == "gelu"
    assert FROZEN["optimizer"] == "adamw"
    assert FROZEN["lr"] == 1e-3
    assert FROZEN["lr_schedule"] == "cosine"
    assert FROZEN["weight_decay"] == 1e-4
    assert FROZEN["batch_size"] == 1024
    assert FROZEN["max_epochs"] == 300
    assert FROZEN["early_stop_metric"] == "val_loss"
    assert FROZEN["checkpoint_every_steps"] == 200
    assert FROZEN["seeds"] == [0, 1, 2, 3, 4]
    assert EARLY_STOP_PATIENCE == 20


def test_param_count_matches_prereg():
    n = count_params(MLP())
    # Exact count of the frozen architecture (input projection Linear(2,256) +
    # 4 hidden Linear(256,256) + Linear(256,10) head), pinned so refactors can't
    # silently change capacity:
    assert n == 266_506
    # The pre-registered "≈3.0e5" is a one-significant-figure approximation of this
    # count; assert we are in that ballpark (within 12%, the exact value above is
    # 11.2% below the rounded figure).
    assert abs(n - 3.0e5) / 3.0e5 < 0.12


def test_architecture_modules():
    m = MLP()
    assert m.input_proj.in_features == 2 and m.input_proj.out_features == 256
    assert len(m.hidden) == 4
    assert all(h.in_features == 256 and h.out_features == 256 for h in m.hidden)
    assert m.head.out_features == N_CLASSES == 10
    assert isinstance(m.act, torch.nn.GELU)


def test_forward_shapes_and_activation_sites():
    m = MLP()
    x = torch.randn(17, 2)
    logits, acts = m.forward_with_activations(x)
    assert logits.shape == (17, 10)
    assert len(acts) == m.n_activation_sites == 5  # h1..h5 (h5 = penultimate)
    assert all(a.shape == (17, 256) for a in acts)
    assert torch.equal(m(x), logits)


def test_init_model_seeded_and_reproducible():
    a, b, c = init_model(0), init_model(0), init_model(1)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)  # untrained probe-power control reproduces the init
    assert any(not torch.equal(pa, pc)
               for pa, pc in zip(a.parameters(), c.parameters()))


def test_optimizer_and_scheduler_match_prereg():
    m = MLP()
    opt = build_optimizer(m)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    sched = build_scheduler(opt, total_steps=1234)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 1234


def test_label_to_class_contract():
    """PRE_REGISTRATION §2 class layout."""
    esc = {"escaped": True, "period": None, "abs_lambda": None}
    assert label_to_class(esc) == 9
    for p in range(1, 9):
        assert label_to_class({"escaped": False, "period": p, "abs_lambda": 0.5}) == p - 1
    assert label_to_class({"escaped": False, "period": None, "abs_lambda": None}) == 8
    assert label_to_class({"escaped": False, "period": 12, "abs_lambda": 0.3}) == 8


def test_load_split_shapes(synth_data_dir):
    X, y, rows = load_split(synth_data_dir, "train")
    assert X.shape == (64, 2) and X.dtype.name == "float32"
    assert y.shape == (64,) and y.dtype.name == "int64"
    assert len(rows) == 64
    assert set(y.tolist()) <= set(range(10))


def test_checkpoint_cadence_and_config(tmp_path, synth_data_dir):
    """64 train points, batch 16 -> 4 steps/epoch; 2 epochs -> 8 steps; cadence 2
    must yield checkpoints at steps 0 (init), 2, 4, 6, 8, plus best/final + config."""
    out = tmp_path / "run"
    summary = train_one_seed(0, synth_data_dir, out, device="cpu", max_epochs=2,
                             batch_size=16, checkpoint_every=2, patience=99,
                             verbose=False)
    assert summary["global_steps"] == 8
    steps = sorted(p.name for p in out.glob("step_*.pt"))
    assert steps == [f"step_{s:07d}.pt" for s in (0, 2, 4, 6, 8)]
    assert (out / "best.pt").exists() and (out / "final.pt").exists()
    assert (out / "history.json").exists()

    config = json.loads((out / "config.json").read_text())
    # frozen values are logged verbatim alongside the effective (overridden) ones
    assert config["lr"] == 1e-3 and config["weight_decay"] == 1e-4
    assert config["batch_size"] == 1024 and config["max_epochs"] == 300
    assert config["checkpoint_every_steps"] == 200 and config["seeds"] == [0, 1, 2, 3, 4]
    assert config["seed"] == 0 and config["shuffle_labels"] is False
    assert config["effective"]["batch_size"] == 16
    assert config["effective"]["checkpoint_every_steps"] == 2

    # checkpoints are loadable and the right architecture
    m = load_model(out / "final.pt", "cpu")
    assert count_params(m) == 266_506


def test_shuffled_label_control_run(tmp_path, synth_data_dir):
    """The probe-power control net (PRE_REGISTRATION §4b) trains and logs its flag."""
    out = tmp_path / "run_shuffled"
    train_one_seed(0, synth_data_dir, out, device="cpu", max_epochs=1, batch_size=32,
                   checkpoint_every=100, patience=99, shuffle_labels=True,
                   verbose=False)
    config = json.loads((out / "config.json").read_text())
    assert config["shuffle_labels"] is True
    assert (out / "final.pt").exists()
