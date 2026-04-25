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
