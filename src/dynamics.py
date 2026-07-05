"""Exact-ish computation of Mandelbrot dynamics labels for c = a + bi under z -> z^2 + c.

Everything here is deterministic and reproducible. For a point c we compute:
  - escape: does the orbit of 0 escape (|z| > 2)?  + smooth (continuous) iteration count.
  - green:  Green's function G(c) = lim 2^-n log|z_n|, the escape-side equipotential.
  - period: the period of the attracting cycle when c is inside a hyperbolic component,
            found by iterating to the attractor then detecting the cycle length.
  - multiplier: lambda = prod over the cycle of f'(z_i) = prod 2*z_i; |lambda| is the
            canonical internal coordinate (|lambda| < 1 strictly inside, = 1 on the boundary).

These are the ground-truth targets the interpretability probes (PRE_REGISTRATION H1/H2/H3)
will look for inside a trained network. No network here, labels only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ESCAPE_R = 2.0
ESCAPE_R2 = ESCAPE_R * ESCAPE_R


@dataclass
class Label:
    re: float
    im: float
    escaped: bool
    smooth_iters: float | None   # continuous escape count (escape side only)
    green: float | None          # Green's function G(c) (escape side only)
    period: int | None           # attracting-cycle period (interior only)
    abs_lambda: float | None     # |multiplier| (interior only)

    def as_row(self) -> dict:
        return {"re": self.re, "im": self.im, "escaped": self.escaped,
                "smooth_iters": self.smooth_iters, "green": self.green,
                "period": self.period, "abs_lambda": self.abs_lambda}


def escape_info(c: complex, max_iter: int) -> tuple[bool, float | None, float | None]:
    """Return (escaped, smooth_iter_count, green_value). Non-escaping -> (False, None, None)."""
    z = 0.0 + 0.0j
    for n in range(max_iter):
        z = z * z + c
        az2 = z.real * z.real + z.imag * z.imag
        if az2 > ESCAPE_R2:
            mod = np.sqrt(az2)
            # smooth iteration count (continuous dwell): n + 1 - log2(log|z|/log R)
            nu = n + 1 - np.log(np.log(mod) / np.log(ESCAPE_R)) / np.log(2.0)
            # Green's function G(c) ~ 2^-n log|z_n|. Multiply by 2^-(n+1) (underflows to
            # 0.0 for large n) rather than dividing by 2^(n+1) (overflows for n > ~1023).
            green = float(np.log(mod) * 2.0 ** (-(n + 1)))
            return True, float(nu), green
    return False, None, None


def _orbit_tail(c: complex, settle: int, probe: int) -> np.ndarray:
    """Iterate `settle` steps to fall onto the attractor, then record `probe` more."""
    z = 0.0 + 0.0j
    for _ in range(settle):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag > 1e6:  # diverged
            return np.empty(0, dtype=np.complex128)
    tail = np.empty(probe, dtype=np.complex128)
    for i in range(probe):
        z = z * z + c
        tail[i] = z
    return tail


def detect_period(c: complex, settle: int = 2000, probe: int = 64,
                  tol: float = 1e-6, max_period: int = 64) -> int | None:
    """Period of the attracting cycle of z->z^2+c, or None if it appears to escape."""
    tail = _orbit_tail(c, settle, probe)
    if tail.size == 0:
        return None
    z0 = tail[0]
    for p in range(1, min(max_period, probe)):
        if abs(tail[p] - z0) < tol:
            # confirm it's a genuine p-cycle over the rest of the tail
            if np.all(np.abs(tail[p:] - tail[:-p]) < tol * 10):
                return p
    return None


def newton_refine_cycle(c: complex, period: int, z_guess: complex,
                        iters: int = 40) -> np.ndarray | None:
    """Refine a periodic point z0 solving f^p(z) = z via Newton, then return the cycle."""
    z = z_guess
    for _ in range(iters):
        w = z
        dw = 1.0 + 0.0j  # d/dz of f^p
        for _ in range(period):
            dw = 2.0 * w * dw
            w = w * w + c
        denom = dw - 1.0
        if abs(denom) < 1e-30:
            return None
        step = (w - z) / denom
        z = z - step
        if abs(step) < 1e-14:
            break
    cycle = np.empty(period, dtype=np.complex128)
    w = z
    for i in range(period):
        cycle[i] = w
        w = w * w + c
    if abs(cycle[0] - w) > 1e-9:  # didn't close up
        return None
    return cycle


def multiplier_abs(c: complex, period: int) -> float | None:
    """|lambda| = |prod 2*z_i| over the attracting p-cycle. None if refinement fails."""
    tail = _orbit_tail(c, 2000, period + 1)
    if tail.size == 0:
        return None
    cycle = newton_refine_cycle(c, period, tail[0])
    if cycle is None:
        return None
    lam = np.prod(2.0 * cycle)
    return float(abs(lam))


def label_point(c: complex, max_iter: int = 50_000) -> Label:
    escaped, nu, green = escape_info(c, max_iter)
    if escaped:
        return Label(c.real, c.imag, True, nu, green, None, None)
    period = detect_period(c)
    abs_lam = multiplier_abs(c, period) if period is not None else None
    # An attracting cycle has |lambda| < 1. If detection latched onto a repelling cycle
    # (|lambda| >= 1, e.g. a boundary / Misiurewicz point like c = i), it is NOT inside a
    # hyperbolic component: drop the interior labels so the "interior" class stays clean.
    if abs_lam is not None and abs_lam >= 1.0 - 1e-6:
        # keep |lambda| ~ 1 boundary points labeled as boundary (period/lambda = None)
        if abs_lam > 1.0 + 1e-3:
            period, abs_lam = None, None
    return Label(c.real, c.imag, False, None, None, period, abs_lam)
