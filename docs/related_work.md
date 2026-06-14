# Related work

Living prior-art map for mandel-interp. The project asks a question nobody has asked yet:
when a small MLP is trained to classify the period of the hyperbolic component containing a
point c, does it internally represent the multiplier map |lambda(c)| (and the Green's
function G(c) on the escape side), or only a memorized boundary lookup? The claim is about
the network, with the ground truth exactly computable. This file tracks the closest existing
work in three buckets and states, for each, why this project is not derivative.

Last refreshed: 2026-06-14. Prior-art also rechecked at pre-registration lock (2026-06-11);
see PRE_REGISTRATION.md preamble.

## 1. Learning on Mandelbrot / Julia (the nearest topical neighbor)

- **Tjahjono, Feng, Putri, Susanto, "Learning with Mandelbrot and Julia"
  (arXiv:2509.00903; Nonlinear Dynamics, 2025).** Trains CART / KNN / MLP / LSTM / BiLSTM /
  RF / CNN to classify fractal-set membership and reports predictive accuracy and reduced
  computational cost versus direct numerical iteration. It is the closest topical work.
  *Differentiation:* it reports outcome metrics only (accuracy, cost). It runs zero probing,
  zero causal analysis, and makes no claim about what any network internally represents. They
  note comparative model behavior "suggests the presence of novel regularity properties" but
  treat that as a conjecture, not an investigation. This project supplies exactly the
  interpretability layer they omit: linear / nonlinear probes for |lambda| and G(c), the
  load-bearing untrained and shuffled-label probe-power controls, causal ablations, activation
  patching, and a grokking-style training-dynamics measure.

- **Hobby / rendering nets (e.g. MaxRobinsonTheGreat/mandelbrotnn and similar).** Fit a
  network to *render* the set (predict pixel color / escape time at high resolution).
  *Differentiation:* the goal there is approximating the image of M; this design deliberately
  inverts that. We do not care how well the net renders M. We ask what internal coordinate the
  net builds, and the whole point of using a trivially-computable ground truth is that it makes
  that question causally answerable rather than a rendering benchmark.

## 2. Interpretability of "did the net learn the real algorithm / world model"

The method lineage this project inherits, all on DISCRETE algebraic or board tasks, never on
holomorphic / continuous dynamics:

- **Nanda, Chan, Lieberum, Smith, Steinhardt, "Progress measures for grokking via mechanistic
  interpretability" (ICLR 2023; arXiv:2301.05217).** Source of the H3 progress-measure idea
  (a structural probe metric that rises before held-out accuracy does).
- **Power et al., "Grokking" (2022)** and **Zhong et al., "The Clock and the Pizza" (2023).**
  Multiple internal algorithms for the same modular-arithmetic task; motivates "memorize vs
  build structure" framing.
- **Li, Hopkins, Bau, Viegas, Pfister, Wattenberg, "Emergent world representations:
  Othello-GPT" (ICLR 2023).** Probing + intervention for an internal world model on a discrete
  board game; the canonical "linear probe + causal intervention" template this project ports.
- **Hewitt & Liang, "Designing and interpreting probes with control tasks" (EMNLP 2019).**
  The reason the untrained-net and shuffled-label controls are mandatory and load-bearing: a
  probe fitting |lambda| proves a representation claim only if it beats those controls by the
  pre-registered margin. Without them H1 is unfalsifiable.

*Differentiation:* every one of these is a discrete / algebraic / symbolic task. None applies
the probe-plus-causal-intervention method to a network trained on a continuous holomorphic
dynamical system, and none asks about a complex-dynamics internal coordinate (the multiplier
map) or an equipotential (the Green's function). That continuous-dynamics target is the gap.

## 3. New since the 2026-06-11 lock (this refresh)

- **Holomorphic Neural ODEs with Kolmogorov-Arnold Networks for Interpretable Discovery of
  Complex Dynamics (arXiv:2605.22235, May 2026).** A KAN-ODE under Cauchy-Riemann
  regularization recovers governing symbolic equations of holomorphic systems (velocity-field
  R^2 > 0.95 on six systems with ~280 params) and reconstructs Julia-set fractal boundaries to
  ~98% agreement. It is the closest NEW neighbor: holomorphic structure + interpretability +
  fractals in one paper.
  *Differentiation:* it is a different question with a different method. (a) It does symbolic
  *system identification* of a dynamical law from trajectory data; this project does post-hoc
  *probing and causal ablation* of a plain MLP trained on static classification labels.
  (b) Its interpretability is architectural (the KAN is built to be readable, and holomorphy is
  enforced by the loss); ours is the opposite stance, a deliberately plain raw-coordinate MLP
  with no structure imposed, where any |lambda| representation must be *built by the net* and
  then *discovered* by external probes. (c) Its target is the governing ODE / Julia boundary;
  ours is a specific complex-dynamics internal coordinate (the multiplier modulus) and the
  escape-side equipotential inside a classifier. The two are complementary, not overlapping;
  cite it as concurrent related work, not a scoop.

- **General field signal:** mechanistic interpretability was named an MIT Technology Review
  "10 Breakthrough Technologies of 2026," confirming the method bucket (Section 2) is highly
  active. No 2026 work was found that probes a Mandelbrot/complex-dynamics-trained classifier
  for the multiplier map. The specific angle remains unclaimed as of this refresh.

## Verdict (unchanged by the refresh)

The building blocks all exist (fractal classifiers; probe-plus-intervention interpretability;
interpretable holomorphic-dynamics modeling), which is what keeps this from being a crank
reinvention. But the specific combination, probing a plain net trained on Mandelbrot period
labels for the multiplier map / Green's function with pre-registered causal controls, plus the
released (c, period, |lambda|, G(c), component id) dataset, is still unclaimed. Re-verify again
before any public release.

## How to refresh this file

Re-run the two prior-art searches (Mandelbrot/Julia learning; probing/world-model
interpretability on dynamics) on arXiv + Google Scholar, plus a check of
arxiv.org/list/cs.LG and math.DS for "multiplier map" / "equipotential" + "probe". Add any new
hit here with a one-line differentiation. Never relax a pre-registered threshold to match a
competitor; if genuinely scooped, document it honestly (as the portfolio's shelved
rank-fingerprint project did).
