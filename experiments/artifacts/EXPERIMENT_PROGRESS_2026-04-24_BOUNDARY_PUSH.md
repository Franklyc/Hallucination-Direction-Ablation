# Boundary Push Log

- Main claim target: permanent truthful editing is real but bounded
- Main model: Qwen/Qwen3-4B-Instruct-2507
- Main HDA seeds: 7,13,23,29,31,41,53,67,89,123
- Main verifier seeds: 7,13,29,41,53,67
- Canonical paired-analysis seed: 41
- Directions file: experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz
- Candidate prefix: newline
- Dtype: bfloat16
- GPU memory cap: 15 GiB
- Drift prompt set: experiments/data/prepared/drift_benign_100.jsonl

## Frozen starting point

- HDA tuned patch: +0.97 points over 10 seeds, 10/10 positive
- Verifier fixed factual: +5.23 points over 6 seeds, 6/6 positive
- Canonical tuned patch seed 41: 9 fixes, 2 breaks

## HDA frontier decision

- tuned sparse: +0.97 points over 10 seeds, 10/10 positive
- same-band alpha=1: +0.15 points over 10 seeds, 7/10 positive
- narrow-family 18-20: +0.92 points over 10 seeds, 10/10 positive
- wide late-MLP 18-24: +0.80 points over 10 seeds, 10/10 positive
- attention-only late band: -0.39 points over 10 seeds, 0/10 positive
- all-layer MLP: -0.00 points over 10 seeds, 5/10 positive
- keep claim as: the late-MLP narrow patch still dominates this fixed control family; keep the localization claim on the tuned sparse 18-19 MLP patch.
