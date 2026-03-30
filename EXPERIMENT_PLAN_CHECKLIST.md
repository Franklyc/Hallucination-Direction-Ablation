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
- [ ] Run full baseline evaluation on `eval_binary` with Qwen3-4B (4-bit).
- [ ] Sweep candidate token prefix style (`space`, `newline`, `none`) and select one fixed protocol.
- [ ] Record baseline accuracy and 95% bootstrap confidence interval.
- [ ] Save baseline artifacts and a short findings note.

## 2) Direction Extraction
- [ ] Use `calib_contrastive` to extract layer-wise directions:
  - [ ] `v_l = mean(h_l | P_h) - mean(h_l | P_g)`
- [ ] Save direction vectors and metadata.
- [ ] Compute per-layer direction norms.
- [ ] Compute inter-layer cosine similarity summary.
- [ ] Create one visualization for layer-wise direction strength.

## 3) Activation Probe (Causal Signal Check)
- [ ] Run projection-removal probe:
  - [ ] `h' = h - beta * (h dot v_hat) * v_hat`
- [ ] Test middle and late layers first (single-layer runs).
- [ ] Run a small beta grid (e.g., `0.25, 0.5, 0.75, 1.0`).
- [ ] Compare base vs probe accuracy and logit deltas.
- [ ] Select top 1-3 effective layers for permanent patch.

## 4) Minimal Weight Patch (Permanent Edit)
- [ ] Switch to non-quantized weights for patching (BF16 recommended).
- [ ] Patch attention output projection only (minimal scope).
- [ ] Apply rank-one orthogonalization:
  - [ ] `W' = W - alpha * v_hat * (v_hat^T W)`
- [ ] Run a small alpha grid (e.g., `0.1, 0.25, 0.5`).
- [ ] Re-evaluate on `eval_binary` and compare against baseline.

## 5) Capability Drift and Safety Checks
- [ ] Evaluate on `drift_benign` prompts.
- [ ] Measure a preliminary capability-drift proxy (e.g., next-token KL or output divergence).
- [ ] Track abstention/refusal tendency changes.
- [ ] Summarize trade-off: truthfulness gain vs drift.

## 6) Milestone Reporting Package
- [ ] Prepare one results table:
  - [ ] Baseline
  - [ ] Activation probe (best setting)
  - [ ] Weight patch (best setting)
- [ ] Prepare one figure for layer/beta analysis.
- [ ] Prepare one figure or table for drift proxy.
- [ ] Write a concise milestone narrative:
  - [ ] What was implemented
  - [ ] What improved
  - [ ] What remains for final comparison (e.g., DoLa as planned baseline)

## 7) Stretch Goals (Only If Time Permits)
- [ ] Add module ablation (`attn` vs `mlp` vs `both`).
- [ ] Add direction variant ablation (layer-wise vs shared/global direction).
- [ ] Expand calibration size sensitivity analysis.
- [ ] Add a second seed run for robustness.
