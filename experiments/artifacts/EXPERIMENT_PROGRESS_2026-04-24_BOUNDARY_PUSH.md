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
- keep claim as: the late-MLP narrow family survives this fixed control family; Task 3 regression should decide whether the final exact-band claim is 18-19 or 18-20.

## Regression frontier decision

- clean main patch: 18-19/a2.8 MLP, TruthfulQA 0.7797 -> 0.7915 (+0.0119), MMLU 0.7233 -> 0.7300, HellaSwag 0.8100 -> 0.8080, GSM8K 0.3100 -> 0.3150, drift similarity 0.7953 (exact 0.4600, token delta 0.2900)
- same-band alpha=1: 18-19/a1.0 MLP, TruthfulQA 0.7797 -> 0.7831 (+0.0034), MMLU 0.7233 -> 0.7267, HellaSwag 0.8100 -> 0.8100, GSM8K 0.3100 -> 0.3200, drift similarity 0.8743 (exact 0.6100, token delta 0.1700)
- narrow-family neighbor: 18-20/a2.4 MLP, TruthfulQA 0.7797 -> 0.7915 (+0.0119), MMLU 0.7233 -> 0.7300, HellaSwag 0.8100 -> 0.8120, GSM8K 0.3100 -> 0.3150, drift similarity 0.7964 (exact 0.4500, token delta 0.2800)
- broad late-MLP: 18-24/a1.6 MLP, TruthfulQA 0.7797 -> 0.7932 (+0.0136), MMLU 0.7233 -> 0.7333, HellaSwag 0.8100 -> 0.8080, GSM8K 0.3100 -> 0.3150, drift similarity 0.7915 (exact 0.4400, token delta 0.0800)
- late attention: 20-24/a0.8 attn, TruthfulQA 0.7797 -> 0.7746 (-0.0051), MMLU 0.7233 -> 0.7267, HellaSwag 0.8100 -> 0.8080, GSM8K 0.3100 -> 0.3200, drift similarity 0.9114 (exact 0.7100, token delta 0.1100)
- final paper patch family: Task 3 supports the sparse late-MLP family claim, and 18-19/a2.8 MLP is the chosen canonical operating point by parsimony/sparsity; the refreshed regression does not decisively beat the 18-20 neighbor.

## Verifier route decision

- selected verifier mean delta: +1.55 points over 6 seeds
- fixed verifier mean delta: +2.37 points over 6 seeds
- fixed minus selected gap: +0.82 points over shared seeds
- keep verifier story as: fixed factual verifier is the stronger route on the standardized six-seed comparison.

## Binary-to-open bridge decision

- base supported_answer rate: 0.225
- patched supported_answer rate: 0.200
- base contradicted_reference rate: 0.125
- patched contradicted_reference rate: 0.125
- unresolved change: 0.450 -> 0.525 (18 -> 21); bridge used 8 overlapping changed questions plus 32 controls because 3 binary-changed questions were absent from open_eval.jsonl.
- final interpretation: proxy-limited evidence; on the 40-row bridge, the patched model did not transfer the binary gain into stronger open-generation support.
