"""Render the project showpiece figure: the ground-truth label structure of the
Mandelbrot set that a tiny network is trained to classify.

Two panels over c in [-2.5, 1.0] x [-1.5, 1.5]:

  (left)  interior hyperbolic components colored by PERIOD  -- the class labels
          the network predicts. Period is computed exactly via cycle detection
          (src.dynamics.detect_period) on a coarse grid of confirmed-interior points.

  (right) the same region colored by the multiplier modulus |lambda| INSIDE each
          component (the canonical internal coordinate, 0 at the center -> 1 at the
          boundary), and a smooth escape-time shading OUTSIDE. |lambda| is exactly
          computed via Newton refinement (src.dynamics.multiplier_abs). It is the
          quantity the interpretability probes search for inside the trained net.

Everything here is exactly-computable ground-truth structure (the label space). No
network is involved. A fast vectorized escape-time pass supplies the outside shading
and a fast interior mask; the exact dynamics core supplies period and |lambda| labels.

Run:  python -m src.figures.showpiece
Output: docs/figures/mandelbrot_label_structure.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.dynamics import _orbit_tail, detect_period, newton_refine_cycle

# --- view + render resolution (coarse-but-clean; renders in a couple of seconds) ---
RE_MIN, RE_MAX = -2.5, 1.0
IM_MIN, IM_MAX = -1.5, 1.5
NX, NY = 900, 600
MAX_ITER = 200

# coarse grid on which exact period / |lambda| labels are computed for interior pixels
LABEL_NX, LABEL_NY = 280, 180

# settle/probe for the figure's exact labeling. Lower than the dataset builder's
# defaults (settle=2000) because the figure only labels low-period components in a
# coarse view; a few hundred settle steps reliably land on these attractors. The
# labels are still computed by genuine cycle detection + Newton multiplier refinement.
FIG_SETTLE = 400
FIG_PROBE = 48

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "mandelbrot_label_structure.png"


def vectorized_escape(re_min, re_max, im_min, im_max, nx, ny, max_iter):
    """Vectorized z->z^2+c escape time with continuous (smooth) iteration count.

    Returns (smooth, inside_mask): smooth is the continuous dwell on the escape side
    (NaN where the point did not escape); inside_mask is True where the orbit stayed
    bounded for max_iter steps (a fast superset of the true interior).
    """
    re = np.linspace(re_min, re_max, nx)
    im = np.linspace(im_min, im_max, ny)
    C = re[np.newaxis, :] + 1j * im[:, np.newaxis]
    Z = np.zeros_like(C)
    escaped_at = np.full(C.shape, np.nan)
    mod_at = np.zeros(C.shape)
    alive = np.ones(C.shape, dtype=bool)
    for n in range(max_iter):
        Z[alive] = Z[alive] * Z[alive] + C[alive]
        mag2 = Z.real * Z.real + Z.imag * Z.imag
        just_escaped = alive & (mag2 > 4.0)
        escaped_at[just_escaped] = n
        mod_at[just_escaped] = np.sqrt(mag2[just_escaped])
        alive &= ~just_escaped
    inside_mask = alive
    # smooth (continuous) iteration count for a clean band-free outside shading
    with np.errstate(divide="ignore", invalid="ignore"):
        smooth = escaped_at + 1.0 - np.log(np.log(mod_at) / np.log(2.0)) / np.log(2.0)
    smooth[inside_mask] = np.nan
    return smooth, inside_mask


def _fast_multiplier_abs(c, period):
    """|lambda| over the attracting p-cycle, using the figure's lighter settle budget.

    Same computation as src.dynamics.multiplier_abs (Newton-refine the cycle, then
    |prod 2*z_i|) but with FIG_SETTLE instead of the dataset builder's settle=2000.
    """
    tail = _orbit_tail(c, FIG_SETTLE, period + 1)
    if tail.size == 0:
        return None
    cycle = newton_refine_cycle(c, period, tail[0])
    if cycle is None:
        return None
    return float(abs(np.prod(2.0 * cycle)))


def exact_interior_labels(re_min, re_max, im_min, im_max, nx, ny, inside_mask=None):
    """Exact period and |lambda| on a coarse interior grid (NaN outside / on escape side).

    If `inside_mask` (a boolean array on the same coarse grid) is given, only its True
    pixels are labeled -- skipping the escape side entirely makes the pass fast.
    """
    re = np.linspace(re_min, re_max, nx)
    im = np.linspace(im_min, im_max, ny)
    period = np.full((ny, nx), np.nan)
    abs_lam = np.full((ny, nx), np.nan)
    for j, b in enumerate(im):
        for i, a in enumerate(re):
            if inside_mask is not None and not inside_mask[j, i]:
                continue
            c = complex(a, b)
            p = detect_period(c, settle=FIG_SETTLE, probe=FIG_PROBE)
            if p is None:
                continue
            lam = _fast_multiplier_abs(c, p)
            if lam is None or lam >= 1.0 + 1e-3:
                continue  # repelling / boundary latch -> not a clean interior point
            period[j, i] = p
            abs_lam[j, i] = lam
    return period, abs_lam


def main():
    print("Rendering escape-time field (vectorized)...")
    smooth, inside = vectorized_escape(RE_MIN, RE_MAX, IM_MIN, IM_MAX, NX, NY, MAX_ITER)

    print("Computing exact period / |lambda| labels on coarse interior grid...")
    # fast coarse bounded mask: only label pixels whose orbit stays bounded (the interior),
    # so the exact-dynamics pass skips the escape side entirely.
    _, inside_coarse = vectorized_escape(
        RE_MIN, RE_MAX, IM_MIN, IM_MAX, LABEL_NX, LABEL_NY, MAX_ITER
    )
    period_c, lam_c = exact_interior_labels(
        RE_MIN, RE_MAX, IM_MIN, IM_MAX, LABEL_NX, LABEL_NY, inside_mask=inside_coarse
    )

    # upsample the coarse label grids to the render resolution (nearest -> crisp components)
    yi = (np.linspace(0, LABEL_NY - 1, NY)).round().astype(int)
    xi = (np.linspace(0, LABEL_NX - 1, NX)).round().astype(int)
    period_full = period_c[np.ix_(yi, xi)]
    lam_full = lam_c[np.ix_(yi, xi)]
    # restrict exact labels to the high-res bounded mask so component edges stay sharp
    period_full = np.where(inside, period_full, np.nan)
    lam_full = np.where(inside, lam_full, np.nan)

    extent = [RE_MIN, RE_MAX, IM_MIN, IM_MAX]

    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "savefig.facecolor": "#0d1117",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "font.size": 11,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # ---------------- LEFT: period labels (the classes) ----------------
    # a clean, colorblind-friendly qualitative ramp for periods 1..8 (+ deeper interior)
    period_colors = [
        "#3b6fb6",  # 1
        "#4fa3a0",  # 2
        "#6fbf73",  # 3
        "#c7d44e",  # 4
        "#f2b134",  # 5
        "#ee7733",  # 6
        "#d94f70",  # 7
        "#a05fb4",  # 8
        "#6c7a89",  # >8 / deeper interior
    ]
    cmap_p = ListedColormap(period_colors)
    cmap_p.set_bad("#05070a")  # escape side -> near-black backdrop
    bounds = np.arange(0.5, 10.5, 1.0)
    norm_p = BoundaryNorm(bounds, cmap_p.N)

    period_plot = np.where(np.isnan(period_full), np.nan, np.clip(period_full, 1, 9))
    axL.imshow(period_plot, extent=extent, origin="lower", cmap=cmap_p,
               norm=norm_p, interpolation="nearest", aspect="equal")
    axL.set_title("Interior components colored by PERIOD\n(the class labels)",
                  fontsize=12.5, pad=10)
    axL.set_xlabel("Re(c)")
    axL.set_ylabel("Im(c)")

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_p, cmap=cmap_p), ax=axL,
                      ticks=range(1, 10), fraction=0.046, pad=0.03)
    cb.ax.set_yticklabels([str(p) for p in range(1, 9)] + [">8"])
    cb.set_label("attracting-cycle period", fontsize=10)
    cb.outline.set_edgecolor("#30363d")

    # ---------------- RIGHT: |lambda| inside + escape-time outside ----------------
    # outside: smooth escape time on a muted blue ramp (low alpha so it reads as backdrop)
    out = np.where(inside, np.nan, smooth)
    axR.imshow(np.log1p(out), extent=extent, origin="lower", cmap="bone",
               interpolation="bilinear", aspect="equal", alpha=0.85)
    # inside: |lambda| on a perceptually-uniform magma ramp (0 center -> 1 boundary)
    lam_masked = np.ma.masked_invalid(lam_full)
    im_lam = axR.imshow(lam_masked, extent=extent, origin="lower", cmap="magma",
                        vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    axR.set_title(r"Inside: multiplier modulus $|\lambda|$   |   Outside: escape time",
                  fontsize=12.5, pad=10)
    axR.set_xlabel("Re(c)")
    axR.set_ylabel("Im(c)")

    cb2 = fig.colorbar(im_lam, ax=axR, fraction=0.046, pad=0.03)
    cb2.set_label(r"$|\lambda|$  (0 = component center  $\rightarrow$  1 = boundary)",
                  fontsize=10)
    cb2.outline.set_edgecolor("#30363d")

    for ax in (axL, axR):
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle(
        "The ground-truth structure a tiny network is trained to classify "
        "— we then probe whether it rediscovers $|\\lambda|$ internally",
        fontsize=13.5, y=1.005,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
