# HDA Two-Phase Runbook (Qwen3-4B)

This folder now follows the Qwen3-4B two-phase route:

1. Phase A (fast iteration): 4-bit model for baseline, direction extraction, and activation probe.
2. Phase B (permanent edit): BF16 model for minimal rank-one weight patch.

All scripts default to model `Qwen/Qwen3-4B-Instruct-2507`.

## 1) Environment sync

```bash
uv sync
```

## 1.5) Prepare datasets (source-locked)

```bash
uv run python experiments/scripts/prepare_truthfulqa.py \
  --calibration-size 200 \
  --drift-size 40
```

Optional asset download helper:

```bash
uv run python experiments/scripts/download_assets.py
```

## 2) Phase A: baseline (4-bit)

```bash
uv run python experiments/scripts/truthfulqa_binary_eval.py \
  --load-in-4bit \
  --candidate-prefix space \
  --max-samples 200 \
  --output-json experiments/artifacts/qwen_baseline_binary_eval.json
```

## 3) Phase A: direction extraction (4-bit)

```bash
uv run python experiments/scripts/extract_direction.py \
  --load-in-4bit \
  --calibration-size 120 \
  --max-samples 240 \
  --output experiments/artifacts/qwen_directions.npz \
  --metadata-json experiments/artifacts/qwen_directions_meta.json
```

Upgraded options:

- Balanced instruction contrastive direction from prepared calibration prompts:

```bash
uv run python experiments/scripts/extract_direction.py \
  --load-in-4bit \
  --method instruction \
  --contrastive-jsonl experiments/data/prepared/calib_contrastive.jsonl \
  --output experiments/artifacts/qwen_instruction_directions.npz \
  --metadata-json experiments/artifacts/qwen_instruction_directions_meta.json
```

- Answer-state direction from teacher-forced correct vs incorrect continuations:

```bash
uv run python experiments/scripts/extract_direction.py \
  --load-in-4bit \
  --method answer_state \
  --calibration-size 200 \
  --max-correct-variants 2 \
  --max-incorrect-variants 2 \
  --output experiments/artifacts/qwen_answerstate_directions.npz \
  --metadata-json experiments/artifacts/qwen_answerstate_directions_meta.json
```

## 4) Phase A: activation probe (4-bit)

```bash
uv run python experiments/scripts/activation_probe.py \
  --load-in-4bit \
  --directions experiments/artifacts/qwen_directions.npz \
  --layers 20,24,28 \
  --beta 0.50 \
  --candidate-prefix space \
  --max-samples 160 \
  --output-json experiments/artifacts/qwen_activation_probe_eval.json
```

The probe output now includes per-category accuracy summaries and paired sign-test diagnostics.

## 4.5) Multi-seed upgraded probe runner

```bash
uv run python experiments/scripts/run_multiseed_probe.py \
  --load-in-4bit \
  --direction-method instruction \
  --seeds 7,13 \
  --calibration-size 200 \
  --layers 20,21,22,23,24 \
  --beta 3.0 \
  --candidate-prefix newline \
  --output-dir experiments/artifacts/instruction_v3_l20to24_b3p0
```

## 4.6) Calibration-selected verifier reranking

This is now the strongest route for the current binary TruthfulQA setup. Method details are documented in `VERIFIER_RERANKING_PRINCIPLES.md`.

Single-seed full held-out run:

```bash
uv run python experiments/scripts/truthfulqa_verifier_eval.py \
  --load-in-4bit \
  --gpu-memory-gb 15 \
  --seed 7 \
  --calibration-size 200 \
  --candidate-prefix newline \
  --verdict-prefixes newline \
  --output-json experiments/artifacts/verifier_full_seed7.json
```

Multi-seed verifier runner:

```bash
uv run python experiments/scripts/run_multiseed_verifier.py \
  --load-in-4bit \
  --gpu-memory-gb 15 \
  --seeds 7,13,29 \
  --calibration-size 200 \
  --candidate-prefix newline \
  --verdict-prefixes newline \
  --output-dir experiments/artifacts/verifier_multiseed
```

## 5) Phase B: minimal weight patch (BF16)

```bash
uv run python experiments/scripts/weight_patch_eval.py \
  --dtype bfloat16 \
  --directions experiments/artifacts/qwen_directions.npz \
  --layers 24,28 \
  --alpha 0.25 \
  --modules attn \
  --candidate-prefix space \
  --max-samples 160 \
  --output-json experiments/artifacts/qwen_weight_patch_eval.json
```

## Notes

- TruthfulQA CSV is auto-downloaded if missing.
- Scoring is deterministic log-prob scoring over A/B candidate letters.
- `--candidate-prefix` controls whether the scored token is `A`, ` A`, or `\nA` style.
- Weight patch script intentionally blocks 4-bit mode to avoid editing quantized weights.
