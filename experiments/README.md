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
