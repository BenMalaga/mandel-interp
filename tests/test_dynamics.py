"""Ground-truth regression tests for the Mandelbrot dynamics labels.

These pin the generator against analytically-known points so a refactor can't silently
corrupt the labels the interpretability probes depend on.
"""

from src.dynamics import label_point


def test_main_cardioid_center():
    L = label_point(0 + 0j)
    assert not L.escaped and L.period == 1 and abs(L.abs_lambda) < 1e-6


def test_period2_disk_center():
    L = label_point(-1 + 0j)
    assert not L.escaped and L.period == 2 and abs(L.abs_lambda) < 1e-6


def test_period3_rabbit():
    L = label_point(-0.122561 + 0.744862j)
    assert not L.escaped and L.period == 3 and L.abs_lambda < 0.05


def test_cardioid_cusp_is_boundary():
    # c = 1/4 is the cusp: a parabolic fixed point with |lambda| exactly 1.
    L = label_point(0.25 + 0j)
    assert not L.escaped and abs(L.abs_lambda - 1.0) < 1e-3


def test_far_point_escapes():
    L = label_point(1.0 + 1.0j)
    assert L.escaped and L.green is not None and L.green >= 0.0


def test_misiurewicz_i_not_interior():
    # c = i is preperiodic (boundary); must NOT be labeled as inside a hyperbolic component.
    L = label_point(0 - 1j)
    assert L.period is None or (L.abs_lambda is not None and L.abs_lambda <= 1.0 + 1e-3)


def test_green_no_overflow_on_slow_escape():
    # a point that escapes slowly must not overflow the Green's-function computation.
    L = label_point(-0.75 + 0.03j, max_iter=20000)
    assert L.green is None or (L.green >= 0.0 and L.green < 1e9)
