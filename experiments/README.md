# HDA Experiment Bootstrap

This folder contains a minimal, reproducible experiment scaffold that follows
the proposal and milestone guidance:

1. Run TruthfulQA binary-choice baseline.
2. Extract hallucination directions from contrastive prompts.
3. Run activation-level projection-removal probe.
4. Run minimal rank-one weight patch evaluation.

## 1) Install dependencies

```bash
pip install -r experiments/requirements.txt
```

## 2) Baseline evaluation

```bash
python experiments/scripts/truthfulqa_binary_eval.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --max-samples 200
```

## 3) Direction extraction

```bash
python experiments/scripts/extract_direction.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --calibration-size 120 \
  --max-samples 240 \
  --output experiments/artifacts/directions_mistral.npz
```

## 4) Activation probe

```bash
python experiments/scripts/activation_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --directions experiments/artifacts/directions_mistral.npz \
  --layers 20,21,22 \
  --beta 0.35 \
  --max-samples 160
```

## 5) Weight patch evaluation

```bash
python experiments/scripts/weight_patch_eval.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --directions experiments/artifacts/directions_mistral.npz \
  --layers 20,21,22 \
  --alpha 0.35 \
  --modules attn \
  --max-samples 160
```

## Notes

- The scripts auto-download `TruthfulQA.csv` from the official repository if it
  is missing.
- Scoring is deterministic and based on log probabilities.
- Results and artifacts are written to `experiments/artifacts/`.
