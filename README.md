# mandel-interp

**What algorithm does a tiny neural network learn when you train it on a fractal?**

A small MLP is trained to classify points of the Mandelbrot set by the *period* of the
hyperbolic component they fall in. Then we open the network up and ask a sharp question:
did it rediscover the real mathematics of the problem — the multiplier map |λ| that working
mathematicians use as each component's natural internal coordinate — or did it just memorize
a lookup table of which side of the boundary each point is on?

---

## The question

The Mandelbrot set looks impossibly intricate, but its structure is *exactly computable*:
for any point `c` you can determine, with no ambiguity, whether it escapes, what period its
interior cycle has, and the precise value of a hidden coordinate called the **multiplier
modulus |λ|** that runs from 0 at a component's center to 1 at its boundary.

That exactness is the whole point. Most interpretability research has to guess at what a
network "should" represent. Here the ground truth is a theorem, not a hunch — so when we look
inside a trained net and ask "is |λ| in here?", we can give a clean, causal answer. The result
is interesting either way:

- **If the network builds |λ|**, a tiny net rediscovered a piece of complex-dynamics theory
  from nothing but coordinates and labels — a concrete "world model" on a continuous task.
- **If it doesn't**, we have a crisp, publishable account of what small networks learn
  *instead* (a boundary-memorizer) and why.

![Two-panel render of the Mandelbrot set: left, interior components colored by period; right, multiplier modulus inside and escape-time shading outside](docs/figures/mandelbrot_label_structure.png)

*The ground-truth structure a tiny network is trained to classify — we then probe whether it
rediscovers |λ| internally. **Left:** interior hyperbolic components colored by the period of
their attracting cycle (these are the class labels). **Right:** the multiplier modulus |λ|
inside each component (dark center → bright boundary) with smooth escape-time shading outside.
Both panels are computed exactly from the dynamics, not from any model.*

---

## How it works

- **Exact labels, no dataset to download.** A small, validated dynamics core
  (`src/dynamics.py`) computes period, |λ| (via Newton-refined cycles), and the escape-side
  Green's function G(c) for any point. Labels are generated deterministically from a fixed seed.
- **A deliberately plain network.** The MLP sees only the raw coordinates `(Re c, Im c)` — no
  Fourier features, no hand-engineered inputs — so any structure it represents it had to
  *build* itself. The architecture is frozen in the pre-registration.
- **Linear and nonlinear probes.** We fit probes for |λ|, G(c), and a control coordinate on
  every hidden layer, and measure held-out R².
- **Probe-power controls.** The same probes are run on an *untrained* net and on a net trained
  on *shuffled* labels. A representation claim only counts if the trained net beats both — this
  rules out "a probe can fit anything from rich-enough features."
- **Causal tests, not just correlations.** We ablate the top probe directions and patch
  activations between points at different |λ|, and check whether the network's predictions
  actually depend on the structure we claim to have found.

---

## What makes it rigorous

- **Pre-registered before training.** Every hypothesis, threshold, and probe protocol was
  committed to [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) *before any network was trained*,
  with a git timestamp. That is what makes a "the net encodes |λ|" claim credible rather than
  a story fit after the fact.
- **A clean null is a result.** The "boundary-memorizer" outcome is fully specified in advance
  and reported with the same weight as a positive finding.
- **Controls are load-bearing, not decorative.** Untrained-net and shuffled-label baselines are
  required passes, not afterthoughts.
- **Reproducible from scratch.** Labels are generated from a fixed seed; the dynamics core is
  validated against analytically-known points; results run on a laptop CPU.
- **Reported across five seeds**, with bootstrap confidence intervals, and no silently-dropped
  classes (the hard boundary region is always shown).

---

## Status

Pre-registration locked (2026-06-11); the dynamics core and the training/probing/ablation
harness are built and tested. **No network has been trained yet** — the figure above is exact
ground-truth structure, not a model output. Training and the interpretability analysis run next.

---

## Reproduce / links

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# regenerate the figure above (exact dynamics; renders in seconds)
python -m src.figures.showpiece
```

- **Pre-registration & method:** [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md)
- **Dynamics core (exact labels):** [`src/dynamics.py`](src/dynamics.py)
- **Figure generator:** [`src/figures/showpiece.py`](src/figures/showpiece.py)

License: code MIT (planned); released dataset CC BY 4.0.
