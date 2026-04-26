# Hallucination Direction Ablation

This repository contains my course project codebase for studying truthfulness-oriented interventions on large language models, centered on TruthfulQA. The main workflows in this repo are:

- deterministic binary TruthfulQA evaluation
- direction extraction from calibration data
- rank-one weight patch evaluation
- multiseed robustness runs
- verifier reranking
- lightweight regression and drift checks
- figure and summary generation

The default model family used in the main pipeline is `Qwen/Qwen3-4B-Instruct-2507`.

## Code Organization

The repository is organized around the `experiments/` directory.

- `experiments/scripts/`
  Runnable Python entry points for data preparation, evaluation, patching, verifier reranking, regression checks, export, and plotting.
- `experiments/data/`
  Input datasets and prepared JSONL files. This includes the TruthfulQA CSV and derived prepared splits.
- `experiments/artifacts/`
  Generated outputs such as JSON summaries, direction files, multiseed aggregates, regression results, exported patched models, and final plots.
- `experiments/plans/`
  Internal planning notes and execution checklists for the experiment pipeline.
- `experiments/requirements.txt`
  A plain requirements file matching the core Python dependencies.
- `pyproject.toml`
  Project metadata and the main dependency list for `uv`.
- `uv.lock`
  Locked dependency versions for reproducible `uv` environments.

Most users only need `experiments/scripts/`, `experiments/data/`, and `experiments/artifacts/`.

## Environment Setup

### Requirements

- Python 3.10 or newer
- A CUDA-capable GPU is strongly recommended
- Enough VRAM for the chosen model
  - fast exploratory runs can use 4-bit loading
  - weight patch runs for the 4B model should use non-quantized weights

### Install with `uv` (recommended)

```bash
uv sync
```

### Install with `pip`

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r experiments/requirements.txt
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## End-to-End Running Instructions

The core pipeline is:

1. prepare TruthfulQA into deterministic binary splits
2. run a baseline binary evaluation
3. extract a task-aligned direction
4. run a fixed weight patch evaluation
5. run multiseed patch analysis
6. run verifier reranking
7. run regression and drift checks
8. generate plots

All commands below are written relative to the repository root.

### 1. Prepare the dataset

```bash
uv run python experiments/scripts/prepare_truthfulqa.py \
  --calibration-size 200 \
  --drift-size 40
```

What this does:

- downloads `experiments/data/TruthfulQA.csv` automatically if it is missing
- writes prepared binary data into `experiments/data/prepared/`
- writes a preparation summary JSON into `experiments/artifacts/`

Important outputs:

- `experiments/data/prepared/calib_contrastive.jsonl`
- `experiments/data/prepared/eval_binary.jsonl`
- `experiments/data/prepared/drift_benign.jsonl`

### 2. Run a baseline binary evaluation

```bash
uv run python experiments/scripts/truthfulqa_binary_eval.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --load-in-4bit \
  --candidate-prefix newline \
  --output-json experiments/artifacts/baseline_qwen4b_full_newline.json
```

What this does:

- loads the base model
- scores the deterministic A/B TruthfulQA binary task
- writes a JSON summary under `experiments/artifacts/`

### 3. Extract a task-aligned direction

```bash
uv run python experiments/scripts/extract_taskaligned_direction.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --calibration-size 200 \
  --seed 7 \
  --load-in-4bit \
  --output experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz \
  --metadata-json experiments/artifacts/directions_qwen4b_cal200_taskaligned_meta.json
```

What this does:

- loads the calibration portion of the binary dataset
- extracts layer-wise prompt-state mean-difference directions
- saves the direction tensor to `.npz`
- saves extraction metadata to `.json`

Important outputs:

- `experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz`
- `experiments/artifacts/directions_qwen4b_cal200_taskaligned_meta.json`

### 4. Run a fixed weight patch evaluation

```bash
uv run python experiments/scripts/weight_patch_eval.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --directions experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz \
  --layers 18,19 \
  --modules mlp \
  --alpha 2.8 \
  --candidate-prefix newline \
  --seed 41 \
  --output-json experiments/artifacts/weight_patch_eval_qwen4b_l18to19_mlp_a2p8_seed41.json
```

What this does:

- loads a direction file
- applies a rank-one patch to the selected layers and modules
- evaluates the patched model on the held-out binary split
- writes per-run metrics and diagnostics to `experiments/artifacts/`

Note:

- do not combine `--load-in-4bit` with weight patching for the main 4B runs

### 5. Run multiseed patch analysis

```bash
uv run python experiments/scripts/run_multiseed_patch.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --directions experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz \
  --layers 18,19 \
  --modules mlp \
  --alpha 2.8 \
  --candidate-prefix newline \
  --seeds 7,13,23,29,31,41,53,67,89,123
```

What this does:

- reruns the same patch configuration across multiple split seeds
- writes one subdirectory per multiseed run into `experiments/artifacts/`
- saves aggregate JSON summaries for later plotting

### 6. Run verifier reranking

```bash
uv run python experiments/scripts/run_multiseed_verifier.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --candidate-prefix newline \
  --verdict-prefixes newline \
  --force-template-ids factual_correctness \
  --force-verdict-prefix newline \
  --seeds 7,13,29,41,53,67
```

What this does:

- scores each candidate answer with a yes/no factuality prompt
- compares truthful and unsupported candidates
- saves per-seed and aggregate verifier outputs to `experiments/artifacts/`

### 7. Run regression and drift checks

```bash
uv run python experiments/scripts/run_patch_regression_suite.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --directions experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz \
  --layers 18,19 \
  --modules mlp \
  --alpha 2.8
```

What this does:

- evaluates a patched model on lightweight non-target checks
- includes HellaSwag, an MMLU slice, GSM8K, and benign drift checks
- writes regression outputs and comparison tables under `experiments/artifacts/regression/`

### 8. Generate figures

```bash
uv run python experiments/scripts/make_final_project_figures.py
```

What this does:

- reads aggregate experiment outputs from `experiments/artifacts/`
- writes project figures into `experiments/artifacts/route_plots/`

## Expected Inputs and Outputs

### Inputs

The main inputs used by this repo are:

- `experiments/data/TruthfulQA.csv`
  The TruthfulQA source CSV. It is auto-downloaded if missing.
- Hugging Face model identifiers or local model paths
  Example: `Qwen/Qwen3-4B-Instruct-2507`
- optional manual annotation JSONL files for bridge or open-generation checks

### Output formats

Common output formats in this repo are:

- `.json`
  Summary metrics, bootstrap intervals, aggregate statistics, and run summaries
- `.jsonl`
  Prepared datasets, per-example evaluation rows, and annotation files
- `.npz`
  Saved direction tensors for patching
- `.png`
  Generated plots and final figures
- model folders
  Exported patched models under `experiments/artifacts/patched_models/`

## Common Script Entry Points

If you do not need the full pipeline, these are the main scripts to start with:

- `experiments/scripts/prepare_truthfulqa.py`
  Build deterministic binary TruthfulQA splits and benign drift prompts.
- `experiments/scripts/truthfulqa_binary_eval.py`
  Evaluate the unpatched model on the binary A/B protocol.
- `experiments/scripts/extract_taskaligned_direction.py`
  Extract layer-wise task-aligned directions from calibration data.
- `experiments/scripts/weight_patch_eval.py`
  Apply and evaluate a fixed rank-one patch.
- `experiments/scripts/run_multiseed_patch.py`
  Repeat one patch configuration across many split seeds.
- `experiments/scripts/truthfulqa_verifier_eval.py`
  Run verifier-based scoring on one split.
- `experiments/scripts/run_multiseed_verifier.py`
  Aggregate verifier results across multiple seeds.
- `experiments/scripts/run_patch_regression_suite.py`
  Evaluate non-target regression and drift behavior for one patch.
- `experiments/scripts/make_final_project_figures.py`
  Build the main comparison figures from saved artifacts.

For additional arguments on any script:

```bash
uv run python experiments/scripts/<script_name>.py --help
```
