# Proposal: Challenges and Improvements for Richer Evaluations Plan
Date: 2026-01-21

## Overall assessment
The plan is well structured and aligns experiments to the RQs, with clear diagnostics and ablations. The core methodology is plausible, but success depends on careful control of proposal distributions, detector reliability, and avoiding estimator bias.

## Challenges / risks
- Q construction is under-specified; if Q does not reliably increase the target mode, extrapolation can fail or drift into incoherence.
- Mode-collapse and cluster metrics are sensitive to thresholds; without pre-registered criteria, gains may be unstable.
- Importance sampling weights can collapse, especially if Q is too strong or if top-k/top-p truncation is used without correcting the proposal distribution.
- Coherence filtering can bias prevalence estimates; using a judge gate inside estimation changes the target event.
- Z detectors (heuristic + judge) can dominate error; judge/model coupling can introduce systematic bias.

## Proposals / adjustments
- Add a Q-screening step: measure delta in target mode rate from P to candidate Qs on a small sample, then pick the minimal-shift Q that measurably increases the mode.
- For IS, prefer multiple-alpha mixture proposals and report ESS plus PSIS k diagnostics; fall back to clipped weights with explicit sensitivity reporting.
- For RQ2, use time-to-event analysis with censoring and report hazard ratio or median-with-CI, not just samples-to-first-Z.
- For RQ3, either disable top-k/top-p for IS runs or incorporate truncation/renormalization into q to avoid biased weights.
- Separate discovery filtering from estimation; if coherence gating is used, report both unconditional p(Z|P) and p(Z|P, coherent).
- Pre-register hero-prompt selection criteria to avoid cherry-picking; keep a held-out prompt set for confirmation.
- Add a toy-model sanity check where exact p(Z|P) is computable to validate the IS pipeline end-to-end.

## Related work to cite (for positioning/baselines)
- Contrastive Decoding (ACL 2023)
- DExperts (ACL 2021)
- Stay on Topic with Classifier-Free Guidance (ICML 2024)
- Pareto Smoothed Importance Sampling (JMLR 2024)
- Rare event simulation and importance sampling (Handbook of Statistics 2015)
