# Experiment Plan Checklist

## 0) Setup and Data Preparation
- [x] Initialize the repository and `uv` environment.
- [x] Install core dependencies (Transformers, Torch, Accelerate, BitsAndBytes, Datasets).
- [x] Verify Qwen3-4B-Instruct-2507 can run with a small smoke test.
- [x] Build source-locked datasets from official TruthfulQA.
- [x] Generate and validate:
  - [x] `eval_binary`
  - [x] `calib_contrastive`
  - [x] `drift_benign`
- [x] Run automatic quality checks (integrity, label consistency, prompt linting, deduplication).

## 1) Baseline Evaluation (Deterministic Binary Choice)
- [x] Run full baseline evaluation on `eval_binary` with Qwen3-4B (4-bit).
- [x] Sweep candidate token prefix style (`space`, `newline`, `none`) and select one fixed protocol.
- [x] Record baseline accuracy and 95% bootstrap confidence interval.
- [x] Save baseline artifacts and a short findings note.

## 2) Direction Extraction
- [x] Use `calib_contrastive` to extract layer-wise directions:
  - [x] `v_l = mean(h_l | P_h) - mean(h_l | P_g)`
- [x] Save direction vectors and metadata.
- [x] Compute per-layer direction norms.
- [x] Compute inter-layer cosine similarity summary.
- [x] Create one visualization for layer-wise direction strength.

## 3) Activation Probe (Causal Signal Check)
- [x] Run projection-removal probe:
  - [x] `h' = h - beta * (h dot v_hat) * v_hat`
- [x] Test middle and late layers first (single-layer runs).
- [x] Run a small beta grid and extend it to the useful range discovered empirically.
- [x] Compare base vs probe accuracy and logit deltas.
- [x] Select top 1-3 effective layers for permanent patch.

## 4) Minimal Weight Patch (Permanent Edit)
- [x] Switch to non-quantized weights for patching (BF16 recommended).
- [x] Patch attention output projection only (minimal scope).
- [x] Apply rank-one orthogonalization:
  - [x] `W' = W - alpha * v_hat * (v_hat^T W)`
- [x] Run a small alpha grid and extend to stronger values where informative.
- [x] Re-evaluate on `eval_binary` and compare against baseline.

## 5) Capability Drift and Safety Checks
- [x] Evaluate on `drift_benign` prompts.
- [x] Measure a preliminary capability-drift proxy (string similarity / output divergence surrogate).
- [x] Track abstention/refusal tendency changes.
- [x] Summarize trade-off: truthfulness gain vs drift.

## 6) Milestone Reporting Package
- [x] Prepare one results table:
  - [x] Baseline
  - [x] Activation probe (best setting)
  - [x] Weight patch (best setting)
- [x] Prepare one figure for layer/beta analysis.
- [x] Prepare one figure or table for drift proxy.
- [x] Write a concise milestone narrative:
  - [x] What was implemented
  - [x] What improved
  - [x] What remains for final comparison (e.g., DoLa as planned baseline)

## 6.1) Remaining Proposal Comparison
- [ ] Run DoLa as a planned comparison baseline if a compatible implementation/model path is available.

## 7) Stretch Goals (Only If Time Permits)
- [x] Add module ablation (`attn` vs `mlp` vs `both`).
- [x] Add direction variant ablation (task-aligned vs original direction file).
- [x] Expand calibration size sensitivity analysis.
- [x] Add a second seed run for robustness.
