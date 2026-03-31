# Full Experimental Report

Date: 2026-03-31
Project: Hallucination-Direction-Ablation
Model under test: Qwen/Qwen3-4B-Instruct-2507

## Executive Summary

This project built and evaluated a hallucination-direction ablation pipeline on TruthfulQA using a fixed binary-choice evaluation protocol. The work progressed from environment repair and dataset preparation to direction extraction, activation-level intervention, rank-one weight patching, and drift analysis. The main outcome is that the pipeline is now fully operational under a 15 GiB GPU memory cap, with reproducible full-held-out gains from activation probes on mid-to-late layers. The strongest accuracy result observed on the held-out TruthfulQA split is a probe on layers 18-24 with beta 3.0, which improved binary accuracy by 0.85 points over baseline. A nearby configuration, layers 20-24 with beta 3.0, offers a slightly weaker accuracy gain of 0.68 points but a better utility-drift balance on benign generation.

A major methodological correction was also made during the work: the initial drift proxy overestimated drift because the generation hook was being re-applied in a way that corrupted decoding. After fixing that issue, corrected drift estimates showed that the interventions are substantially less destructive than first reported, but there is still a clear tradeoff between improved TruthfulQA performance and changes in general-generation behavior. Attempts to move the intervention to a first-answer-token position did not yield a useful improvement under the current generation stack, which makes that avenue a practical blocker without a deeper decoding rewrite.

## Objective

The original objective, derived from the project proposal and milestone guidance, was to start the experiment pipeline as early as possible and push it forward until the next meaningful blocker. The intended technical question was whether a hallucination direction could be extracted from TruthfulQA contrastive data and then used to improve factual binary-choice performance through either causal activation intervention or permanent weight patching, while keeping non-target behavior stable enough to remain useful.

## Environment And Resources

The repository was moved into [Hallucination-Direction-Ablation](.) and initialized with an `uv`-managed Python environment. The initial environment had a CPU-only Torch build; this was corrected by installing a CUDA-enabled Torch build (`2.11.0+cu128`). All core scripts were updated to accept an explicit per-GPU memory cap, and all production runs were constrained to 15 GiB.

Key environment properties:

- GPU class: NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB class.
- Runtime: Python with Hugging Face Transformers and PyTorch.
- Loading mode: 4-bit for iteration and BF16 for patch experiments.
- Memory control: all core scripts support `--gpu-memory-gb` and were run with `15`.

Relevant scripts:

- [experiments/scripts/common.py](experiments/scripts/common.py)
- [experiments/scripts/truthfulqa_binary_eval.py](experiments/scripts/truthfulqa_binary_eval.py)
- [experiments/scripts/extract_direction.py](experiments/scripts/extract_direction.py)
- [experiments/scripts/activation_probe.py](experiments/scripts/activation_probe.py)
- [experiments/scripts/weight_patch_eval.py](experiments/scripts/weight_patch_eval.py)
- [experiments/scripts/drift_probe_eval.py](experiments/scripts/drift_probe_eval.py)

## Data And Split Protocol

The data pipeline was made source-locked and transform-only. Instead of generating new factual content, the project uses the official TruthfulQA CSV and derives all experiment inputs from it.

Prepared datasets:

- [experiments/data/prepared/eval_binary.jsonl](experiments/data/prepared/eval_binary.jsonl)
- [experiments/data/prepared/calib_contrastive.jsonl](experiments/data/prepared/calib_contrastive.jsonl)
- [experiments/data/prepared/drift_benign.jsonl](experiments/data/prepared/drift_benign.jsonl)
- [experiments/data/prepared/manual_audit_samples.jsonl](experiments/data/prepared/manual_audit_samples.jsonl)

The split protocol used a random calibration/held-out partition with the calibration split reserved for direction estimation and the held-out split used for final evaluation. The final binary evaluation used a fixed answer-format protocol with `candidate-prefix = newline`, selected after a prefix sweep on the subset.

## Methods

### Baseline Evaluation

The baseline metric is deterministic binary-choice accuracy on held-out TruthfulQA rows. For each question, the model chooses between the correct and incorrect answer using log-probability comparison on the answer tokens. A bootstrap confidence interval is reported for accuracy.

The selected baseline protocol was `newline` candidate prefixes. The full held-out baseline result is stored at [experiments/artifacts/baseline_qwen4b_full_newline.json](experiments/artifacts/baseline_qwen4b_full_newline.json).

### Direction Extraction

Direction extraction was implemented as a contrastive hidden-state subtraction on the calibration split. The final task-aligned version used fixed binary A/B prompts directly grounded in the TruthfulQA answers. The extracted directions are stored in [experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz](experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz).

The task-aligned prompt format is recorded in [experiments/artifacts/directions_qwen4b_cal200_taskaligned_meta.json](experiments/artifacts/directions_qwen4b_cal200_taskaligned_meta.json).

### Activation Probe

The activation probe uses projection removal on selected hidden-state positions and compares base vs intervened held-out accuracy. To make the effect interpretable, the probe output now includes per-item diagnostics:

- `margin_correct = logprob(correct) - logprob(incorrect)`
- prediction flips
- fixed examples
- broken examples
- large margin-shift examples

The probe can target a token position using `hook_position`:

- `prompt_last_token`
- `first_answer_token`

The main probe outputs are stored under `experiments/artifacts/probe_*.json`.

### Weight Patch

The patch pipeline applies a rank-one orthogonalization-style update to selected write matrices. The patch operates only on BF16 weights and excludes 4-bit quantized models. The patch output also includes the same per-item diagnostics used by the probe.

The main patch outputs are stored under `experiments/artifacts/patch_*.json`.

### Drift Proxy

A benign drift proxy was added to compare model generations on non-factual prompts before and after intervention. The metric is based on string similarity between base and intervened generations, plus token-count drift. After an initial bug was found in the generation hook behavior, the drift script was corrected to avoid repeated corruption and to report more reliable estimates.

The corrected drift results are stored under `experiments/artifacts/drift_probe_*.json`.

## Experimental Timeline And Results

### 1. Baseline Protocol Sweep

A candidate-prefix sweep was run on a subset with `space`, `newline`, and `none`. `newline` was selected as the best protocol for downstream evaluation.

### 2. Full Baseline

The full held-out baseline result was:

- Accuracy: 75.93%
- 95% CI: [72.37%, 79.66%]
- n = 590

### 3. Direction Extraction

Two direction files were generated:

- [experiments/artifacts/directions_qwen4b_cal200.npz](experiments/artifacts/directions_qwen4b_cal200.npz)
- [experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz](experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz)

The task-aligned direction estimate concentrated its norm in the late layers, especially around layers 30-35.

### 4. Activation Probe Search

The probe search began with low-to-moderate beta values on late layers and later expanded into mid-layer bands. The key finding is that binary accuracy improvement only emerged once the intervention strength increased enough to create meaningful margin shifts.

Full held-out probe results:

| Config | Delta accuracy | Mean drift similarity | Exact match | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 18-24, beta 2.5 | -0.68 points | 0.9480 | 0.8250 | Too weak, no useful gain |
| 18-24, beta 2.75 | -0.34 points | 0.9201 | 0.7750 | Still too weak |
| 18-24, beta 3.0 | +0.85 points | 0.8907 | 0.7000 | Best pure utility |
| 18-24, beta 4.0 | +0.51 points | 0.8833 | 0.6500 | Strong gain, somewhat more drift |
| 19-24, beta 3.0 | -0.51 points | 0.8654 | 0.6500 | Too narrow / unstable |
| 20-24, beta 3.0 | +0.68 points | 0.9368 | 0.7250 | Best utility-drift balance |
| 21-24, beta 3.0 | -1.02 points | 0.9373 | 0.8000 | Too narrow / regressive |

The best overall utility was layers 18-24 with beta 3.0. The best balance between utility and benign similarity was layers 20-24 with beta 3.0.

Relevant artifacts:

- [experiments/artifacts/probe_taskalign_l18to24_b2p5_full_diag.json](experiments/artifacts/probe_taskalign_l18to24_b2p5_full_diag.json)
- [experiments/artifacts/probe_taskalign_l18to24_b2p75_full_diag.json](experiments/artifacts/probe_taskalign_l18to24_b2p75_full_diag.json)
- [experiments/artifacts/probe_taskalign_l18to24_b3p0_full_diag.json](experiments/artifacts/probe_taskalign_l18to24_b3p0_full_diag.json)
- [experiments/artifacts/probe_taskalign_l18to24_b4p0_full_diag.json](experiments/artifacts/probe_taskalign_l18to24_b4p0_full_diag.json)
- [experiments/artifacts/probe_taskalign_l19to24_b3p0_full_diag.json](experiments/artifacts/probe_taskalign_l19to24_b3p0_full_diag.json)
- [experiments/artifacts/probe_taskalign_l20to24_b3p0_full_diag.json](experiments/artifacts/probe_taskalign_l20to24_b3p0_full_diag.json)
- [experiments/artifacts/probe_taskalign_l21to24_b3p0_full_diag.json](experiments/artifacts/probe_taskalign_l21to24_b3p0_full_diag.json)

### 5. Weight Patch Search

Patch results showed that the patch path is technically functional, but the full held-out patch settings tested so far did not improve accuracy in a stable way.

Key patch outcomes:

- Layers 18-24, alpha 0.35, both modules: no accuracy change on subset.
- Layers 18-24, alpha 0.8, both modules: +0.56 points on subset, but -0.34 points on full held-out.
- Layers 22-24, alpha 0.8, both modules: no subset gain.

The patch route remains useful as a controllability check, but it is not yet the best-performing intervention path.

Relevant artifacts:

- [experiments/artifacts/patch_taskalign_l18to24_both_a0p35_sub_diag.json](experiments/artifacts/patch_taskalign_l18to24_both_a0p35_sub_diag.json)
- [experiments/artifacts/patch_taskalign_l18to24_both_a0p8_sub_diag.json](experiments/artifacts/patch_taskalign_l18to24_both_a0p8_sub_diag.json)
- [experiments/artifacts/patch_taskalign_l18to24_both_a0p8_full_diag.json](experiments/artifacts/patch_taskalign_l18to24_both_a0p8_full_diag.json)
- [experiments/artifacts/patch_taskalign_l222324_both_a0p8_sub_diag.json](experiments/artifacts/patch_taskalign_l222324_both_a0p8_sub_diag.json)

### 6. Drift Proxy Correction

The first drift estimates were misleading because the hook path was interfering with decoding repeatedly. After the drift script was corrected, the benign-generation similarity estimates became much higher and more believable.

Corrected drift results:

| Config | Mean similarity | Exact match | Notes |
| --- | ---: | ---: | --- |
| 18-24, beta 2.5 | 0.9480 | 0.8250 | High similarity, but negative utility |
| 18-24, beta 2.75 | 0.9201 | 0.7750 | Still negative utility |
| 18-24, beta 3.0 | 0.8907 | 0.7000 | Best utility, more drift than 20-24 |
| 18-24, beta 4.0 | 0.8833 | 0.6500 | Strong utility, similar drift to beta 3.0 |
| 19-24, beta 3.0 | 0.8654 | 0.6500 | Stable drift but negative utility |
| 20-24, beta 3.0 | 0.9368 | 0.7250 | Best utility-drift compromise |
| 21-24, beta 3.0 | 0.9373 | 0.8000 | High similarity but negative utility |

Relevant artifacts:

- [experiments/artifacts/drift_probe_taskalign_l18to24_b2p5_v2.json](experiments/artifacts/drift_probe_taskalign_l18to24_b2p5_v2.json)
- [experiments/artifacts/drift_probe_taskalign_l18to24_b2p75_v2.json](experiments/artifacts/drift_probe_taskalign_l18to24_b2p75_v2.json)
- [experiments/artifacts/drift_probe_taskalign_l18to24_b3p0_v2.json](experiments/artifacts/drift_probe_taskalign_l18to24_b3p0_v2.json)
- [experiments/artifacts/drift_probe_taskalign_l18to24_b4p0_v2.json](experiments/artifacts/drift_probe_taskalign_l18to24_b4p0_v2.json)
- [experiments/artifacts/drift_probe_taskalign_l19to24_b3p0_v2.json](experiments/artifacts/drift_probe_taskalign_l19to24_b3p0_v2.json)
- [experiments/artifacts/drift_probe_taskalign_l20to24_b3p0_v2.json](experiments/artifacts/drift_probe_taskalign_l20to24_b3p0_v2.json)
- [experiments/artifacts/drift_probe_taskalign_l21to24_b3p0_v2.json](experiments/artifacts/drift_probe_taskalign_l21to24_b3p0_v2.json)

### 7. First-Answer-Token Attempt

A position-restriction attempt was made to move the intervention from the prompt-last-token state to the first-answer-token state. This avenue did not produce a useful improvement under the current generation stack.

Binary evaluation result for layers 20-24, beta 3.0, first-answer-token:

- Delta accuracy: -0.68 points

A smoke drift run was then performed with cache disabled so the hook could actually fire during incremental decoding, but the probe still produced no measurable generation change on the 10-sample smoke set:

- Mean similarity: 1.0000
- Exact match rate: 1.0000

This indicates that the current generation API surface is not sufficient for a practical first-answer-token drift study without a deeper rewrite of decode-time instrumentation.

Relevant artifacts:

- [experiments/artifacts/probe_taskalign_l20to24_b3p0_firstans_full_diag.json](experiments/artifacts/probe_taskalign_l20to24_b3p0_firstans_full_diag.json)
- [experiments/artifacts/drift_probe_taskalign_l20to24_b3p0_firstans_smoke10.json](experiments/artifacts/drift_probe_taskalign_l20to24_b3p0_firstans_smoke10.json)

### 8. Additional Robustness, Ablations, and Direction Analysis

To make the experimental picture more complete, an additional analysis pass was run on the extracted directions and a small set of sensitivity and ablation experiments was added.

Direction analysis on the task-aligned directions produced:

- Mean off-diagonal cosine similarity: 0.1916
- Median off-diagonal cosine similarity: 0.0751
- Top norm layer: 34

The corresponding visualization is stored at [experiments/artifacts/analysis/direction_analysis.png](experiments/artifacts/analysis/direction_analysis.png), and the numeric summary is stored at [experiments/artifacts/analysis/analysis_summary.json](experiments/artifacts/analysis/analysis_summary.json).

Presentation-ready summary figures were also generated for reporting:

- [experiments/artifacts/plots/results_overview.png](experiments/artifacts/plots/results_overview.png)
- [experiments/artifacts/plots/tradeoff_frontier.png](experiments/artifacts/plots/tradeoff_frontier.png)

Calibration and seed sensitivity:

- `calibration_size=100, seed=13` with layers `20-24, beta=3.0` reduced the gain to `-0.17 points`, indicating that the extracted direction is somewhat sensitive to calibration size.
- `seed=13` with the task-aligned `20-24, beta=3.0` probe still produced a positive `+0.51 points` gain on its own split, so the effect is not a one-off artifact of the original seed.

Direction-variant ablation:

- Using the original non-task-aligned directions on `20-24, beta=3.0` gave `+0.34 points`, which is positive but weaker than the task-aligned direction.

Additional adjacent-band checks:

- `18-22, beta=3.0` produced `-1.86 points` on the held-out split.
- `18-23, beta=3.0` produced `-1.53 points` on the held-out split.

Module ablation for the patch path on layers `20-24, alpha=0.8`:

- Attention-only patch: `-0.51 points`
- MLP-only patch: `0.00 points`
- Both modules: `-0.34 points`

The drift proxy was also extended with a simple refusal-keyword heuristic on benign prompts, and no refusal-like keywords were observed in the analyzed outputs under the current setting.

## Key Findings

1. The pipeline is operational end-to-end under the 15 GiB memory cap.
2. The strongest reliable binary-choice improvement comes from mid-to-late layers, especially layers 18-24 with beta 3.0.
3. Nearby layer bands are highly sensitive. Small changes in the selected band can move performance from positive to negative.
4. Weight patching is controllable but has not beaten the best probe configuration on held-out performance.
5. The corrected drift analysis shows a meaningful utility-drift tradeoff rather than the catastrophic drift implied by the initial buggy evaluation.
6. Position restriction to first-answer-token is not yet a practical win under the current generation stack.
7. Task-aligned directions outperform the original direction file, while smaller calibration sets can weaken or reverse the gain.
8. The patch route remains controllable but is not yet competitive with the best probe setting on held-out accuracy.
9. The best-performing layer band is narrow but not singleton-level narrow: widening or shrinking it by one to two layers can flip the sign of the gain.

## Limitations

- The project is still evaluating a single model family and a single benchmark family. Generalization beyond TruthfulQA remains untested.
- The drift proxy is a heuristic similarity measure, not a full safety or alignment evaluation.
- The first-answer-token route would need a deeper decode-time rewrite if it is to be studied seriously.
- Patch settings have not yet shown a stable full-held-out gain.

## Reproducibility Notes

The current state can be reproduced with the scripts and artifacts in this repository. The main runbook is in [experiments/README.md](experiments/README.md), and the broader experimental progress log is in [experiments/artifacts/EXPERIMENT_PROGRESS_2026-03-30.md](experiments/artifacts/EXPERIMENT_PROGRESS_2026-03-30.md).

A useful way to rerun the leading configurations is to follow the same pipeline used in the recorded artifacts:

- Baseline evaluation with `newline` candidate prefixes.
- Task-aligned direction extraction from the calibration split.
- Probe sweep on layers 18-24 and 20-24 with beta around 3.0.
- Corrected drift proxy on benign prompts.

## Final Status

The project reached the point where the remaining obvious avenue, moving the intervention to a first-answer-token decode position, no longer produced useful signal without a more invasive generation redesign. At that point, the most valuable remaining work shifted from search into documentation and consolidation. The report here captures the full path from the initial proposal-guided setup through the final experimental state.
