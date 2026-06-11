# Data — generated, not downloaded

This project's data is **computed deterministically**, not fetched. Run the generator (built
in `src/`, idempotent, NumPy PCG64 seed 20260611) to produce the labeled set
`(c, period, |λ|, G(c), component id)` over c ∈ [-2.5, 1.0] × [-1.5, 1.5] plus a
boundary-enriched stratum. Raw arrays land under `data/` (gitignored); the small released
dataset goes in `results/` with a DOI at publication.

One external cross-check (read-only): hyperbolic-component centers to period 32, public Zenodo
record **10.5281/zenodo.15527027** (Vigneron & Mihalache, CC-BY-4.0) — used only to validate
generated period/center labels, never as a training input.
