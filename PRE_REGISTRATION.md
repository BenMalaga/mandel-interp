# Pre-Registration — Mandel-Interp

**Locked:** 2026-06-11 (git commit timestamp is authoritative). No network has been trained
and no probe/ablation has been run at lock time. The dataset is generated deterministically
from a fixed seed and contains no model outcomes; locking the analysis plan before any
*training* is what makes the representation claims (H1–H3) credible rather than post-hoc.

Prior-art rechecked 2026-06-11: no published work applies probing/causal interpretability to a
network trained on Mandelbrot/complex-dynamics labels (closest: arXiv:2509.00903, fractal
classification, accuracy-only; the grokking/world-model interpretability lineage is all on
discrete algebraic/board tasks). Amendments after this commit are labeled, dated, and appended
to §8, never edited in place.

## 1. Research question
Trained to classify the period of the hyperbolic component containing a point c, does a small
MLP internally represent the multiplier map |λ(c)| (and the Green's function G(c) on the escape
side), or only a memorized boundary lookup? The claim is about the network, not the set: the
ground truth is exactly computable (`src/dynamics.py`, validated against analytically-known
points), which is what enables clean causal interpretability.

## 2. Data (computed, deterministic — `src/build_dataset.py`)
- Map z → z² + c; escape radius 2; `max_iter = 50000` for escape labels.
- Period via cycle detection; multiplier λ = ∏ 2·zᵢ over the attracting cycle (|λ| is the
  internal coordinate, < 1 strictly inside, = 1 on the boundary); a detected repelling cycle
  (|λ| > 1, e.g. c = i) is NOT labeled interior. Green's function G(c) = limₙ 2⁻ⁿ log|zₙ|.
- Sampling region (frozen): c ∈ [-2.5, 1.0] × [-1.5, 1.5]; **target N = 2,000,000**, of which a
  **boundary-enriched stratum (fraction 0.35)** is rejection-sampled toward |λ|≈1 / moderate
  escape time. NumPy PCG64, **seed 20260611**.
- Classes (10): periods 1–8, "deeper interior" (bounded, period > 8 or unresolved), "escapes".
- Split: 80/10/10 train/val/test by the seeded shuffle. Exact per-class/per-stratum counts and
  per-split SHA are recorded by the (deterministic) generator in
  `results/dataset_manifest.json`; because generation is seeded and outcome-free, these are
  inputs, not peeked outcomes.
- Label cross-check: component centers validated against Zenodo record 10.5281/zenodo.15527027.

## 3. Models (frozen)
- **Architecture:** MLP, input (Re c, Im c) **raw** (no Fourier/feature engineering — the point
  is whether the net *builds* |λ| internally from coordinates), 4 hidden layers × width 256,
  GELU activations, linear 10-way classification head. ≈3.0×10⁵ params.
- **Training:** AdamW, lr 1e-3 with cosine decay, weight decay 1e-4, batch 1024, ≤300 epochs
  with early stop on val loss; PyTorch CPU/MPS. **Seeds {0,1,2,3,4}** (all reported).
- **Sanity gate:** ≥95% accuracy on the OFF-boundary test subset before any interpretability
  analysis is run; models failing the gate are reported but excluded from H1–H3.

## 4. Probes, controls, ablations (frozen before training)
- **Probes:** linear and 2-hidden-layer probes predicting |λ(c)| (interior), G(c) (escape), and
  a **control target** = distance-to-nearest-component-center, fit on every hidden layer with an
  L2 penalty selected on val. Metric: held-out R².
- **Probe-power controls (load-bearing):** identical probes on (a) an UNTRAINED net (same init)
  and (b) a net trained on SHUFFLED labels. A representation claim requires the trained net to
  beat BOTH by the H1 margin (this is what separates "the net encodes |λ|" from "a probe can fit
  |λ| from any sufficiently rich features"; Hewitt–Liang style).
- **Causal ablation:** zero- and mean-ablate the top-3 probe directions; compare accuracy drop
  to rank-matched random-direction ablations.
- **Activation patching:** swap penultimate activations between same-component c-points at
  different |λ|; measure period-prediction change.
- **Training-dynamics replay:** checkpoint every 200 steps for the H3 progress measure.

## 5. Hypotheses & decision rules (locked thresholds)
- **H1 (representation):** penultimate-layer linear-probe R² for |λ| ≥ 0.80, AND ≥ 0.15 R² above
  the control target, AND above both probe-power controls by ≥ 0.15 R². Else → "boundary-
  memorizer" (a clean, publishable null about what tiny nets do NOT learn).
- **H2 (causal):** top-3 probe-direction ablation drops period accuracy ≥ 20 pts while the
  rank-matched random-direction ablation drops < 5 pts.
- **H3 (training dynamics):** |λ| probe R² crosses 0.80 at an earlier training step than
  held-out boundary-region accuracy crosses 90% (a grokking-style progress measure).
- **Escape task:** identical rules with G(c) as the probe target.
- Pooled across the 5 seeds: a hypothesis is "supported" only if it holds (point estimate past
  threshold) in ≥ 4 of 5 seeds; bootstrap 95% CIs over the test set reported throughout.

## 6. What we report regardless of outcome
The headline (H1 supported or the boundary-memorizer null), all probe-power controls, per-class
and per-stratum accuracy (the boundary class is never silently dropped), the causal-ablation
table, and the training-dynamics curves.

## 7. Prior art engaged
arXiv:2509.00903 (fractal classification, accuracy-only — we add the interpretability it omits);
Nanda et al. "Progress measures for grokking" (ICLR 2023); Power et al. 2022; Zhong et al.
"Clock and Pizza" 2023; Li et al. Othello-GPT world models (ICLR 2023); Hewitt & Liang 2019
(probe-power baselines). Component-center reference: Vigneron & Mihalache (Zenodo 15527027).

## 8. Deviations
None at lock. Amendments appear below this line, dated and labeled, never edited in place.
