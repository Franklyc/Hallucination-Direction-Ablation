# Hallucination Direction Ablation

Current milestone implementation is organized around the Qwen3-4B two-phase plan.

- Proposal and report guidance: `proposal.tex`, `guidance_for_milestone_report.md`
- Experiment runbook: `experiments/README.md`
- Core scripts: `experiments/scripts/`

Quick start:

```bash
uv sync
uv run python experiments/scripts/truthfulqa_binary_eval.py --load-in-4bit --max-samples 200
```
