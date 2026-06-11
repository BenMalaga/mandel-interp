# mandel-interp

**What algorithm does a tiny neural network learn when trained on the Mandelbrot set?**

We train small MLPs to predict the period of the hyperbolic component containing a point
`c` in the Mandelbrot set, then look *inside* the trained network to ask whether it
rediscovers the known mathematical structure of complex dynamics — the multiplier map
|λ(c)| (each component's natural internal coordinate) and the Green's function / equipotential
G(c) on the escape side — or whether it merely memorizes a piecewise boundary lookup. The
question is about the network, not the fractal: the ground truth is exactly computable, which
is what makes clean, causal interpretability possible.

This is a mechanistic-interpretability study (linear probes, causal ablations, grokking-style
training-dynamics measures) on a continuous *holomorphic-dynamics* task — a setting the
interpretability literature has only studied on discrete algebraic and board-game tasks.

- **Status:** pre-registration locked (2026-06-11); generator validated. No network trained yet.
- **Hypotheses, method, and prior art:** see [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md).
- **Compute:** laptop-scale (tiny networks, CPU/MPS). Data is generated, not downloaded.
- **Deliverables:** the training + probing harness, an interpretability-ready labeled dataset
  `(c, period, |λ|, G(c), component id)`, trained checkpoints, and a writeup.

License: code MIT (planned); released dataset CC BY 4.0.
