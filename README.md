# Hallucination Direction Ablation (HDA)

This directory contains experiments for improving factual behavior in LLMs, centered on TruthfulQA and model-internal interventions.

The project started with direction extraction and ablation (activation edits and weight patches), then expanded into HERETIC-style transfer experiments, mechanistic open-generation analysis, and verifier reranking.

Current default model family is `Qwen/Qwen3-4B-Instruct-2507`.

## What This Project Covers

- Source-locked TruthfulQA data preparation and stratified splits
- Deterministic binary-choice evaluation (A/B scoring by log-probability)
- Direction extraction from calibration data
- Activation-level projection removal and weight-level orthogonal patching
- Benign drift proxy analysis
- HERETIC-style paired-prompt route and open-generation transfer checks
- Mechanistic reviewed intervention route
- Verifier reranking (currently the strongest route)

## Repository Layout

- `experiments/scripts/`: all runnable experiment code
- `experiments/data/`: prepared datasets and route-specific data
- `experiments/artifacts/`: outputs, summaries, and plots

## Snapshot of Findings

- Activation ablation can improve held-out binary accuracy, but effects are sensitive to layer band and strength.
- Minimal rank-one patching is functional but weaker than the best runtime interventions.
- Verifier reranking is currently the best-performing approach in this repository, with consistent multi-seed gains.

## Quick Start

```bash
uv sync
uv run python experiments/scripts/prepare_truthfulqa.py --calibration-size 200 --drift-size 40
uv run python experiments/scripts/truthfulqa_binary_eval.py --load-in-4bit --candidate-prefix newline --output-json experiments/artifacts/baseline_qwen4b_full_newline.json
```

For route-specific runs, use script-level help:

```bash
uv run python experiments/scripts/<script_name>.py --help
```

## Notes

- TruthfulQA CSV is auto-downloaded if missing.
- Most fast iteration runs use 4-bit loading; weight patch runs use non-quantized weights.
- Main logs, summaries, and figures are written to `experiments/artifacts/`.
