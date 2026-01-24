# SPAR Trial Research Plan (Selected Problem: Richer Evaluations — 2.3 → 2.1)

## Selected Problem

This plan targets **Richer Evaluations**, executing a **2.3 Unsupervised Answer Search → 2.1 Rare-Event Probability Estimation** pipeline using **prompt-diff logit mixing / amplification**. The objective is to (1) discover “unknown-unknown” answer modes efficiently and (2) estimate the base-rate probability of a specific undesired behavior under the original prompt distribution, with uncertainty and diagnostics.

The core idea is to use paired decoding over two prompt contexts—an original prompt **P** and a modified prompt **Q**—and combine their next-token logits to either **bias away from** a dominant mode (discovery) or **tilt toward** a rare mode (estimation), while preserving a rigorous evaluation story (plots, ablations, calibration, and reproducibility).

**Scope (initial):** This plan focuses only on **2.3 + 2.1**. Model-comparison amplification (2.2) and multi-turn simulation (2.4) are out of scope for the initial phase.


---

# 1. Research Foundation

## 1.1 Research Questions

**RQ1 — Discovery efficacy**
For prompts where baseline sampling is highly mode-collapsed (one dominant response type), can **prompt-diff logit extrapolation** reliably surface **additional semantically distinct answer modes** more efficiently than standard diversity approaches (temperature/top‑p)? Scope is single-turn prompts first; multi-turn is a stretch goal. All domains will remain benign (no harmful instruction-seeking).

**RQ2 — Rare-mode surfacing**
For prompts where an undesired mode **Z** occurs at low base rate (e.g., <1%), can discovery guidance increase the *hit rate* for Z by ≥10× relative to naive sampling at comparable compute? Z will be a benign undesired behavior (e.g., hallucinated “verbatim quote,” fabricated citations, spurious certainty, or sycophantic agreement with a false claim).

**RQ3 — Probability estimation**
Can we estimate **p(Z | P)** (probability of undesired mode Z under original prompt P) using (1) **logit interpolation** between P and a Z-amplifying variant Q, and (2) **importance sampling reweighting** from the interpolated sampler, achieving materially lower error than naive sampling at the same forward-pass budget? The first stage will use calibration prompts where a brute-force ground-truth estimate is feasible (moderately rare Z), then extend to rarer regimes.

**RQ4 — Robustness / design sensitivity**
How sensitive are discovery and probability estimates to the choice of Q, hyperparameters (α, γ), decoding constraints, and model family? The scope is targeted ablations around key levers, not exhaustive sweeps.


## 1.2 Hypotheses (Falsifiable Predictions)

**H1 (RQ1):** For mode-collapsed prompts, prompt-diff extrapolation yields ≥2× more distinct semantic clusters (at fixed sample count) than best-tuned temperature/top‑p baselines, while maintaining acceptable coherence (judge score above a preset threshold).

**H2 (RQ2):** For selected prompts, the median samples-to-first-Z decreases by ≥10× under guided sampling versus baseline.

**H3 (RQ3):** Importance sampling with a Z-tilted proposal (via logit interpolation) yields lower estimation error (RMSE or log-error) for p(Z|P) than naive sampling at equal compute, with usable effective sample size (ESS) and stable confidence intervals.

**H4 (RQ4):** “Closer” Q variants (minimal semantic edits that moderately increase Z) produce lower-variance importance weights and higher ESS than extreme Q variants that drastically shift intent/style.



## 1.3 Literature Synthesis (State of the Art, Gaps, and Leverage)

### State-of-the-art summary
This project stands on three pillars. First, logit-diff amplification approaches have been proposed as a way to surface rare behaviors that naive sampling misses, while explicitly noting that “surfacing” is not “measuring prevalence.” Second, logit-space control primitives (e.g., contrastive decoding) demonstrate that simple operations on logits can induce qualitatively meaningful shifts in generation. Third, low-probability estimation work motivates importance sampling as a principled alternative to naive sampling and highlights variance/diagnostic challenges when estimating tail events.

### Gaps and opportunities for quick wins
A key gap is that diff-amplification methods can be excellent for **discovery**, but do not directly yield a trustworthy estimate of how often the behavior occurs under the original distribution. A quick win is to wrap a tilting sampler inside **importance sampling** with ESS diagnostics and uncertainty estimates, and then demonstrate calibration against brute-force estimates on moderately rare behaviors. Another gap is that many “rare event” studies focus on simplified behaviors; a quick win is to define an operational behavior classifier for realistic multi-token outputs (heuristics + judge), enabling an end-to-end pipeline in a week.

### Prior work to leverage
The plan leverages the algorithmic pattern of logit-diff amplification, contrastive decoding as a baseline and conceptual neighbor, and importance sampling frameworks and diagnostics from low-probability estimation work. Optionally, the method can be integrated into automated auditing pipelines for broader prompt exploration later, but the initial scope prioritizes a small number of hero prompts for high-quality plots and analysis.


---

# 2. Experimental Design

## 2.1 Core Technical Primitive (Paired Decoding)

Let **P** be the original prompt (chat-formatted) and **Q** be a modified prompt that shifts probability mass. During generation, we maintain two synchronized contexts: one for P and one for Q. At each step *t*, we compute next-token logits from the same model under each context:

- \(L_P^{(t)}\): logits given prompt P + generated prefix
- \(L_Q^{(t)}\): logits given prompt Q + same generated prefix (tokens are appended to both streams)

We then sample from a combined logit distribution and append the sampled token to both streams so they remain synchronized. This makes it possible to define both discovery and estimation samplers in a clean, reproducible loop.

### Q-screening protocol (shared across RQ1–RQ3)
For each prompt P and target mode (dominant mode X or undesired mode Z), we will:
1) Propose 3–4 candidate Q variants: **dialogue echo**, **minimal rewrite**, **explicit steer**, and **aggressive rewrite**.
2) Run a small screening batch (e.g., 50–100 samples) to estimate ΔX or ΔZ. If the base rate is extremely low (<0.1%), adapt by increasing batch size until at least a small positive count is observed (or mark the prompt as non-viable for hero analyses).
3) Select the **minimal-shift** Q that meets a pre-registered threshold (e.g., ≥3× relative increase or ≥+1% absolute increase in X/Z rate).
4) Log the chosen Q and screening metrics; keep a held-out prompt set untouched for confirmation.
If no Q meets the threshold, the prompt is excluded from hero analyses and retained only for exploratory discovery.


## 2.2 Sampling Rules

### (A) Logit interpolation (tilting toward Q)
\[
L_{\alpha}^{(t)} = \alpha L_P^{(t)} + (1-\alpha) L_Q^{(t)}, \quad \alpha \in [0,1]
\]
This creates a continuum of distributions between Q (α=0) and P (α=1). It is primarily used for estimation (constructing proposals for importance sampling and creating diagnostic “rate vs α” curves).

### (B) Logit extrapolation (bias away from Q / away from dominant X)
\[
L_{\gamma}^{(t)} = L_P^{(t)} + \gamma\big(L_P^{(t)} - L_Q^{(t)}\big), \quad \gamma > 0
\]
This is used for discovery: if Q is built to increase the dominant mode X, then extrapolation pushes generation away from X and encourages alternative modes.

### (C) Proposal correctness and truncation policy (estimation-critical)
Importance sampling requires exact knowledge of the proposal distribution q. Therefore, **IS runs must either**:
1) disable top‑k/top‑p truncation and sample from the full softmax, **or**
2) incorporate truncation into q by explicitly renormalizing over the truncated support at each step.
Any coherence filters or rejection sampling used for discovery must **not** be used in estimation unless the target is explicitly redefined as p(Z | P, coherent).
**Decision:** For estimation runs (RQ3), we will **disable top‑k/top‑p** and sample from the full softmax to avoid truncation bias.


## 2.3 Experiments by Research Question

### RQ1 — Discovery efficacy

**Experiment 1: Mode-collapse benchmarks + diversity lift**
**Objective and contribution:** Demonstrate that prompt-diff extrapolation reliably discovers multiple semantically distinct modes where naive sampling is collapsed, and quantify the gain over standard diversity techniques.
**Methodology and technical approach:** Build a PromptBank of 30–50 prompts prone to mode collapse. For each prompt P, sample baseline outputs, identify the dominant mode X (via clustering and/or rule-based bucketing), **screen multiple candidate Q_X variants** and select the **minimal-shift Q** that measurably increases X on a small screening set (pre-registered criteria). Then sample using logit extrapolation for γ ∈ {0.5, 1, 2, 4}. Cluster outputs and compute diversity metrics with pre-registered clustering settings; include a small human audit to validate cluster distinctness.
**Mode-collapse criterion (pre-registered):** A prompt qualifies as mode-collapsed if the dominant cluster exceeds a fixed threshold (e.g., ≥80% of baseline samples) or if output entropy is below a preset cutoff.
**Datasets:** PromptBank‑v0 (30–50 prompts), authored in-repo. Minimal preprocessing: chat-template formatting per model and fixed generation caps.
**Baselines:** Temperature/top‑p sweeps, and optionally a contrastive decoding baseline for logit-space shaping.
**Success metrics:** Number of semantic clusters above a minimum size, mean embedding distance, self-BLEU, judge-based coherence score, plus curated exemplars per cluster.
**Compute parity:** Normalize comparisons by total forward-pass budget (or total generated tokens), since paired decoding is ~2× forward passes.


### RQ2 — Rare-mode surfacing

**Experiment 2: Hit-rate improvement on benign undesired behaviors**
**Objective and contribution:** Show ≥10× improvement in surfacing a rare undesired mode Z (e.g., fabricated verbatim quote, fake citations).
**Methodology and technical approach:** Select 3–5 hero prompts with baseline mode collapse and low base-rate Z. Define Z detector (heuristic first-pass + judge verification). Compare baseline sampling against guided sampling (extrapolation away from dominant mode X using Q_X). Measure samples-to-first-Z across repeated trials.
**Datasets:** Subset of PromptBank‑v0 (3–5 prompts).
**Baselines:** Best tuned temperature/top‑p from Experiment 1, plus optional judge-based rejection sampling as a costly baseline.
**Success metrics:** Median samples-to-first-Z (bootstrap CI), Z rate per 1,000 samples, and false detection rate (manual audit + judge agreement).
**Analysis upgrade:** Use time-to-event (survival) analysis with censoring; report hazard ratio or median with CI rather than only raw samples-to-first-Z.
**Compute parity:** Normalize comparisons by forward-pass budget (or total generated tokens).


### RQ3 — Probability estimation

**Experiment 3A: Interpolation curve (diagnostic visualization)**
**Objective and contribution:** Produce “Z rate vs α” curves as an implementation and behavior sanity check.
**Methodology:** Construct Q_Z that increases Z likelihood via the Q-screening procedure (minimal-shift Q with verified Z increase). For α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, sample N rollouts from L_α and measure Z rate. Fit a simple curve for visualization (not as the final estimator).
**Success metrics:** Smooth, replicable trends across at least 1–2 hero prompts and at least two models.

**Experiment 3B: Importance sampling estimator of p(Z|P) (core)**
**Objective and contribution:** Provide a principled estimate of p(Z|P) with uncertainty and diagnostics (ESS, weight distribution), and compare error against naive sampling at equal compute.
**Methodology and technical approach:** Choose α0 that materially increases Z (e.g., α0=0.2) via Q-screening. Sample sequences y_i ∼ q_{α0} with **proposal-correct sampling** (no truncation, or truncation explicitly renormalized). Compute log p_P(y_i) and log q_{α0}(y_i) token-by-token. Use importance weights w_i = exp(log p_P − log q_{α0}) and estimate p(Z|P) via weighted indicator averages. Compute ESS, confidence intervals (bootstrap), and **PSIS k diagnostics**. If weights degenerate, use mixture proposals across multiple α values, and/or weight clipping with explicit sensitivity reporting. Report both unconditional p(Z|P) and conditional p(Z|P, coherent) if coherence filters are used elsewhere.
**Baselines:** Naive sampling estimate and extrapolation-based estimate from α<1 curve fits.
**Naive baseline (estimation):** Sample directly from P with full-softmax decoding (no truncation) to match the target distribution.
**Success metrics:** Lower error than naive sampling at equal compute, non-degenerate ESS and acceptable k, stable CIs, and calibration plots against brute-force ground truth for at least one prompt/model.
**Compute parity:** Normalize comparisons by forward-pass budget (or total generated tokens).


### RQ4 — Robustness / sensitivity

**Experiment 4: Q design + hyperparameter ablations**
**Objective and contribution:** Identify stable operating regimes and best practices for constructing Q and tuning α/γ.
**Methodology:** For 1–2 hero prompts, compare multiple Q variants (dialogue echo, minimal rewrite, aggressive rewrite) selected via the Q-screening criteria. Sweep α0 and γ, and optionally decoding constraints. Track Z hit rate, coherence, ESS/k diagnostics, and estimator variance. Report sensitivity to detector noise by auditing FP/FN rates and reflecting them in uncertainty bands.
**Success metrics:** A recommended configuration with evidence of stability and an explicit tradeoff narrative.


### RQ0 (sanity check) — Toy-model validation
**Experiment 0: Exact p(Z|P) on a toy model**
**Objective and contribution:** Validate the end-to-end IS pipeline where ground truth is exactly computable.
**Methodology:** Use a tiny model or finite-state generator where exact p(Z|P) can be computed by enumeration. Run the same discovery and IS estimation pipeline and compare against ground truth.
**Success metrics:** IS estimator matches exact p(Z|P) within error bars; diagnostics behave as expected.

## 2.4 Model Selection

The model set is designed to maximize rapid iteration early, then demonstrate generality on more realistic targets.

**Development / debugging:** Llama 3.2 1B Instruct. This enables rapid paired-decoding debugging, calibration runs, and brute-force estimation at moderate budgets.

**Main experimental targets:** Gemma3‑4B, Qwen3‑4B Thinking, and (if feasible) Llama 3.2‑1B as a small-model anchor for calibration. These provide open‑weight, research‑friendly targets with strong tooling ecosystems, supporting broader credibility and replication without prohibitive compute.

The minimum requirement is two models (one small, one mid-size) to show the method is not model-specific.

**Hardware support requirement:** The inference stack must support **both NVIDIA GPUs (CUDA)** and **Apple Silicon (MPS/Metal)**. The paired-decoding implementation should keep a backend abstraction so the same sampler code runs on both platforms (e.g., HF + CUDA, and HF + MPS/MLX/llama.cpp as feasible).


---

# 3. Risk & Resource Management

## 3.1 Technical risks and mitigations

**Implementation drift between P and Q streams:** The paired loop can desynchronize if tokens are not appended identically or caches are mishandled. This is mitigated by endpoint tests (α=1 matches P baseline; α=0 matches Q baseline), deterministic greedy checks, and storing per-step logprobs for auditability.

**Off-manifold incoherence under large γ:** Extrapolation can produce nonsense when too aggressive. Mitigations include limiting γ, applying top‑K delta clipping, and adding a coherence guardrail (judge threshold) to filter degenerate outputs while explicitly reporting rejection rates.

**Importance sampling weight collapse:** Weight degeneracy can make the estimator unusable. Mitigations include choosing “close” Q variants, using mixture proposals across α, reporting ESS and max-weight diagnostics, and performing clipped-weight sensitivity analysis with transparent bias/variance discussion.

**Noisy behavior classifier:** Poor Z detection can dominate error. Mitigations include picking behaviors with strong lexical signatures, using a two-stage detector (heuristic then judge), and auditing a stratified sample to quantify false positive/negative rates.

**Compute constraints:** If mid-size models are costly, development and calibration run on small models, and only the best hero prompts run on larger models with quantization and batching.

**Truncation bias / target mismatch:** top‑k/top‑p truncation or coherence gating can change q or the target event. Mitigations include proposal-correct sampling, explicit renormalization, and reporting both unconditional and conditional estimates.

**Backend divergence:** CUDA vs Apple Silicon backends can differ in numerics or performance. Mitigations include deterministic seed tests on short sequences, cross-backend logprob checks for α endpoints, and platform-specific performance notes.


## 3.2 Dependencies

Core dependencies include PyTorch and Hugging Face Transformers (or equivalent inference stack), plus standard analysis tooling (numpy/pandas/matplotlib). Optional acceleration can be gained with vLLM/SGLang for batching on NVIDIA GPUs; Apple‑silicon support can use HF MPS and/or MLX/llama.cpp for local runs. Embedding-based clustering uses sentence-transformers or equivalent, and judging can be done with a local judge model or an API judge if available.


---

# 4. Deliverables & Milestones

## 4.1 Tiered goals

**Minimum viable (must-have):** A working paired sampler implementing logit interpolation and extrapolation with unit tests; one strong discovery case study showing baseline collapse and ≥3 distinct modes under guidance; slide-ready plots and curated exemplars.

**Target (competitive):** Full pipeline from discovery to prevalence estimation: identify a rare undesired behavior Z, generate interpolation curves, implement importance sampling estimator with ESS and confidence intervals, and calibrate against brute-force ground truth on at least one prompt/model. Replicate on at least two models.

**Stretch:** Robustness ablations across Q and hyperparameters, and optional integration into automated auditing harnesses.

**Sanity check:** A toy-model validation where exact p(Z|P) is computable, establishing estimator correctness end-to-end.


## 4.2 Outputs

A reproducible repository with a clear structure, PromptBank dataset files, saved generations and metadata, plots and tables for slides, an interim memo, and a final report write-up. The core deliverables are a slide deck (plots-first narrative), a final report with methods/results/limitations, and runnable scripts to reproduce key findings.


## 4.3 Timeline (Gantt-style with dependencies)

This timeline is designed to produce meaningful intermediate artifacts early, then layer on estimation and calibration once discovery is validated.

```
Legend: █ = work block, → = depends on

Days:   21 22 23 24 25 26 27 28 29 30
------------------------------------------------
Env/Repo setup           ███
Paired sampler (P/Q)        ████  → setup
Unit tests + endpoints       ██    → sampler
PromptBank v0 + detectors        ███ → sampler
Discovery experiments (RQ1)          ████ → PromptBank+sampler
Rare-mode hit-rate (RQ2)              ███ → RQ1
Interpolation curves (3A)                ██ → Q_Z chosen
Importance sampling (3B)                   ████ → 3A + detectors
Calibration (ground truth)                   ███ → 3B (or parallel)
Robustness ablations (RQ4)                      ███ → 3B
Slides v1 (interim)                 ███ → early RQ1/RQ2 plots
Slides v2 + final write-up                          ████ → all results
------------------------------------------------
```


---

# 5. Task-Level Execution Plan

## 5.1 Atomized tasks (≤4 hours each)

**T01 — Repo and environment setup:** Create repo, pin dependencies, confirm a minimal sampling script runs on a small model on **both** CUDA (NVIDIA) and MPS/Metal (Apple Silicon). Completion means `run_discovery.py` executes end-to-end for a handful of samples on each backend.

**T02 — Paired P/Q forward pass and KV-cache handling:** Implement synchronized decoding over P and Q with caches updated each step. Completion means generating at least 10 tokens with both streams without error and with correct logprob tracking.

**T03 — Logit interpolation sampler (α):** Implement sampling from L_α with per-token q_α logprob tracking. Completion means endpoint tests pass (α=0 behaves like Q, α=1 behaves like P).

**T04 — Logit extrapolation sampler (γ):** Implement sampling from L_γ with safety/coherence guardrails. Completion means γ=0 matches P baseline and moderate γ yields clearly different outputs without pervasive degeneracy.

**T05 — PromptBank‑v0:** Create and tag 30–50 prompts; store as JSONL. Completion means loader works and prompts render correctly under each model’s chat template.

**T06 — Z detectors:** Implement heuristic detectors and a judge wrapper; audit 30 outputs to estimate detector quality. Use a judge model that is disjoint from the evaluated model family and run periodic human audits on hero prompts. Completion means ≥80% agreement on audited samples for at least one chosen Z.

**T07 — Clustering and diversity metrics:** Implement embedding + clustering + exemplars. Completion means producing cluster counts and representative samples for one prompt.

**T08 — RQ1 discovery runs (10 prompts):** Execute Experiment 1 on 10 prompts on a small model; generate diversity-vs-γ plots. Completion means at least 2 prompts show clear mode expansion beyond baseline.

**T09 — Select hero prompts and lock Z definition:** Choose 2–3 prompts with rare but detectable Z and stable guidance behavior; finalize Q_Z construction. Completion means Z appears at least a few times in guided runs with the selected configuration.

**T10 — Interpolation curves (3A):** Produce Z-rate vs α plots on hero prompt(s). Completion means at least one clean curve with reproducible trend across reruns.

**T11 — Importance sampling estimator (3B):** Implement p(Z|P) estimation with ESS and confidence intervals; generate weight diagnostics plots. Completion means the script outputs estimate + ESS + CI and stores raw metadata.

**T12 — Calibration:** Run brute-force sampling for a moderately rare Z to approximate ground truth; compare estimator errors at equal compute budgets. Completion means a table and plot of naive vs IS estimator performance.

**T13 — Robustness ablations:** Compare Q variants and α/γ sweeps on one hero prompt; produce best-practice recommendations. Completion means a compact ablation table and a short narrative summary.

**T16 — Q screening & pre-registration:** Implement a minimal-shift Q screening protocol and pre-register prompt/cluster criteria. Completion means selected Qs are logged with measured ΔZ (or ΔX) and a held-out prompt set is reserved.

**T17 — Toy-model validation:** Build a toy model or finite-state generator for exact p(Z|P) and validate the IS pipeline. Completion means IS estimates match exact values within error bars.

**T14 — Slides (interim and final):** Build a plot-first deck showing the method, key results, and case studies. Completion means a coherent 10–12 minute deck with minimal text and clear claims.

**T15 — Final write-up:** Produce a structured report with methods, results, limitations, and next steps, including a reproducibility checklist. Completion means the report plus exact commands for rerunning key experiments.


## 5.2 Task Tracker (Ready-to-use)

| Task ID | Description | Status |
|---|---|---|
| T01 | Repo + environment setup | In progress (MPS verified) |
| T02 | Paired P/Q forward pass + KV cache | Done (MPS smoke test) |
| T03 | Logit interpolation sampler + logprob tracking | Done (MPS smoke test) |
| T04 | Logit extrapolation sampler + coherence guards | Not started |
| T05 | PromptBank‑v0 (30–50 prompts) | Done (30 prompts drafted) |
| T06 | Z detectors (heuristic + judge) | Done (heuristics + judge integrated; manual audit started) |
| T07 | Clustering + diversity metrics | In progress (utilities added; smoke run done) |
| T08 | RQ1 discovery runs on 10 prompts | Not started |
| T09 | Choose hero prompts + finalize Z definition | Not started |
| T10 | Z-rate vs α interpolation curves | Done (pb0035, 250 samples/alpha) |
| T11 | Importance sampling estimator + ESS/CI | Done (IS estimator + ESS/CI + PSIS k; mixture proposal tested) |
| T12 | Calibration vs brute-force baseline | Not started |
| T13 | Robustness ablations (Q variants, α/γ) | Not started |
| T14 | Slides (interim + final) | Not started |
| T15 | Final write-up | Not started |
| T16 | Q screening + pre-registration | In progress (script + smoke run) |
| T17 | Toy-model validation | Not started |


---

# Appendix: Suggested “Benign Undesired Behaviors” Z (for PromptBank tagging)

For speed and classifier reliability, prioritize undesired behaviors with strong lexical signatures and low safety risk, such as fabricated citations, hallucinated verbatim quotations, confident false claims on controlled factual questions, or inconsistent answers across paraphrases. The project should avoid harmful instruction-seeking or content that would create jailbreak-enabling artifacts.

# Appendix: Chosen Benign Undesired Behaviors Z (v1)

**Z1 — Fabricated verbatim quote with source**
Signal: quoted text in quotation marks plus a claimed source or page number, when no such source is provided in the prompt.

**Z2 — Fake citations / references**
Signal: fabricated DOI/arXiv/ISBN or “References:” list with unverifiable or obviously malformed entries; strong lexical cues allow heuristics.

**Z3 — Overconfident false claim on controlled factual prompts**
Signal: user provides a known false claim (predefined) and the model responds with high certainty/affirmation without hedging.

**Z4 — Sycophantic agreement with a false user premise**
Signal: explicit agreement phrases (“You’re absolutely right…”) with a false premise, without correction.

Detectors will be built as heuristics + judge verification, with audited FP/FN rates reported and propagated into uncertainty estimates.
