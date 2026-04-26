import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from common import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate paired regression results into summary artifacts.")
    parser.add_argument("--regression-root", required=True, help="artifacts/regression root directory")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap rounds for paired delta CI")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paired_bootstrap_delta(base_vals, patch_vals, rounds: int, seed: int):
    base = np.asarray(base_vals, dtype=np.float64)
    patch = np.asarray(patch_vals, dtype=np.float64)
    if base.shape != patch.shape:
        raise ValueError("Paired arrays must have the same shape.")
    deltas = patch - base
    point = float(deltas.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(rounds, dtype=np.float64)
    n = len(deltas)
    for idx in range(rounds):
        sample_idx = rng.integers(0, n, size=n)
        boots[idx] = deltas[sample_idx].mean()
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return point, lo, hi


def paired_sign_test(base_vals, patch_vals):
    better = 0
    worse = 0
    for base, patch in zip(base_vals, patch_vals):
        if patch > base:
            better += 1
        elif patch < base:
            worse += 1
    if better + worse == 0:
        return 1.0, better, worse
    pval = float(binomtest(better, better + worse, 0.5, alternative="two-sided").pvalue)
    return pval, better, worse


def find_single(pattern_root: Path, pattern: str):
    matches = list(pattern_root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern} under {pattern_root}, found {len(matches)}")
    return matches[0]


def read_lm_eval_samples(task_dir: Path):
    sample_files = sorted(task_dir.rglob("samples_*.jsonl"))
    rows = []
    for path in sample_files:
        rows.extend(load_jsonl(path))
    return rows


def metric_candidates(task_name: str):
    task_name = task_name.lower()
    if task_name.startswith("hellaswag"):
        return ["acc_norm", "acc"]
    if task_name.startswith("mmlu"):
        return ["acc", "acc_norm"]
    if task_name.startswith("gsm8k"):
        return [
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match",
            "acc",
        ]
    return ["acc"]


def choose_metric(rows, task_name: str):
    candidates = metric_candidates(task_name)
    if not rows:
        raise ValueError(f"No sample rows for task {task_name}")
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    fallback = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float)) and key not in {"doc_id"}:
            fallback.append(key)
    if not fallback:
        raise ValueError(f"No numeric metric fields found for task {task_name}")
    return fallback[0]


def align_lm_eval_rows(base_rows, patch_rows):
    base_map = {row["doc_hash"]: row for row in base_rows}
    patch_map = {row["doc_hash"]: row for row in patch_rows}
    shared = sorted(set(base_map) & set(patch_map))
    return [base_map[key] for key in shared], [patch_map[key] for key in shared]


def summarize_task(task_name: str, base_rows, patch_rows, metric_key: str, bootstrap: int, seed: int, notes: str = ""):
    base_vals = [float(row[metric_key]) for row in base_rows]
    patch_vals = [float(row[metric_key]) for row in patch_rows]
    point, lo, hi = paired_bootstrap_delta(base_vals, patch_vals, bootstrap, seed)
    pval, better, worse = paired_sign_test(base_vals, patch_vals)
    return {
        "task": task_name,
        "metric": metric_key,
        "n": len(base_vals),
        "base_score": float(np.mean(base_vals)),
        "patched_score": float(np.mean(patch_vals)),
        "delta": point,
        "paired_bootstrap_ci_low": lo,
        "paired_bootstrap_ci_high": hi,
        "sign_test_p": pval,
        "better_count": better,
        "worse_count": worse,
        "notes": notes,
    }


def compare_truthfulqa(root: Path, bootstrap: int, seed: int):
    base = load_json(root / "base" / "truthfulqa" / "base_truthfulqa.json")
    patch = load_json(root / "patched" / "truthfulqa" / "patched_truthfulqa.json")
    base_rows = base["rows"]
    patch_rows = patch["rows"]
    if len(base_rows) != len(patch_rows):
        raise ValueError("TruthfulQA row length mismatch")

    flips = []
    base_correct = []
    patch_correct = []
    category_stats = {}
    for base_row, patch_row in zip(base_rows, patch_rows):
        if base_row["question"] != patch_row["question"]:
            raise ValueError("TruthfulQA rows misaligned on question")
        b_ok = int(base_row["pred"] == base_row["correct"])
        p_ok = int(patch_row["pred"] == patch_row["correct"])
        base_correct.append(b_ok)
        patch_correct.append(p_ok)
        category = base_row["category"]
        bucket = category_stats.setdefault(category, {"base_correct": 0, "patch_correct": 0, "n": 0})
        bucket["base_correct"] += b_ok
        bucket["patch_correct"] += p_ok
        bucket["n"] += 1
        if base_row["pred"] != patch_row["pred"] or b_ok != p_ok:
            flips.append(
                {
                    "question": base_row["question"],
                    "category": category,
                    "base_pred": base_row["pred"],
                    "patched_pred": patch_row["pred"],
                    "correct": base_row["correct"],
                    "base_margin_correct": float(base_row["logprob_A"] - base_row["logprob_B"]) if base_row["correct"] == "A" else float(base_row["logprob_B"] - base_row["logprob_A"]),
                    "patched_margin_correct": float(patch_row["logprob_A"] - patch_row["logprob_B"]) if patch_row["correct"] == "A" else float(patch_row["logprob_B"] - patch_row["logprob_A"]),
                    "base_correct": b_ok,
                    "patched_correct": p_ok,
                }
            )

    summary = summarize_task("truthfulqa_binary", [{"acc": v} for v in base_correct], [{"acc": v} for v in patch_correct], "acc", bootstrap, seed)
    categories = []
    for category, stats in sorted(category_stats.items()):
        base_acc = stats["base_correct"] / stats["n"]
        patch_acc = stats["patch_correct"] / stats["n"]
        categories.append(
            {
                "category": category,
                "n": stats["n"],
                "base_accuracy": base_acc,
                "patched_accuracy": patch_acc,
                "delta": patch_acc - base_acc,
            }
        )
    flips = sorted(flips, key=lambda row: (row["patched_correct"] - row["base_correct"], row["question"]), reverse=True)
    compare = {
        "summary": summary,
        "category_delta": categories,
        "flip_count": len(flips),
    }
    save_json(root / "compare" / "truthfulqa_compare.json", compare)
    with (root / "compare" / "truthfulqa_flips.jsonl").open("w", encoding="utf-8") as f:
        for row in flips:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def compare_lm_eval(root: Path, task_dir_name: str, logical_name: str, bootstrap: int, seed: int, notes: str = ""):
    base_dir = root / "base" / task_dir_name
    patch_dir = root / "patched" / task_dir_name
    base_rows = read_lm_eval_samples(base_dir)
    patch_rows = read_lm_eval_samples(patch_dir)
    base_rows, patch_rows = align_lm_eval_rows(base_rows, patch_rows)
    metric = choose_metric(base_rows, logical_name)
    summary = summarize_task(logical_name, base_rows, patch_rows, metric, bootstrap, seed, notes=notes)
    save_json(root / "compare" / f"{task_dir_name}_compare.json", summary)
    return summary


def cosine_similarity_rows(base_embeddings, patch_embeddings):
    base = np.asarray(base_embeddings, dtype=np.float64)
    patch = np.asarray(patch_embeddings, dtype=np.float64)
    if base.shape != patch.shape:
        raise ValueError("Embedding arrays must align.")
    return np.sum(base * patch, axis=1)


def compare_drift(root: Path):
    base = load_json(root / "base" / "drift" / "base_drift.json")
    patch = load_json(root / "patched" / "drift" / "patched_drift.json")
    base_rows = base["responses"]
    patch_rows = patch["responses"]
    if len(base_rows) != len(patch_rows):
        raise ValueError("Drift row length mismatch")

    similarities = cosine_similarity_rows(base["embeddings"], patch["embeddings"])
    output_rows = []
    family_rollup = {}
    format_base = []
    format_patch = []
    refusal_base = []
    refusal_patch = []
    material_list = []

    for idx, (base_row, patch_row) in enumerate(zip(base_rows, patch_rows)):
        if base_row["prompt_id"] != patch_row["prompt_id"]:
            raise ValueError("Drift rows misaligned on prompt_id")
        base_tokens = max(1, int(base_row["token_count"]))
        patch_tokens = int(patch_row["token_count"])
        ratio = patch_tokens / base_tokens if base_tokens > 0 else 1.0
        refusal_flip = int(bool(base_row["is_refusal"]) != bool(patch_row["is_refusal"]))
        format_flip = int(bool(base_row["format_pass"]) != bool(patch_row["format_pass"]))
        exact_field_base = base_row.get("exact_field_match")
        exact_field_patch = patch_row.get("exact_field_match")
        exact_field_flip = int(exact_field_base != exact_field_patch) if exact_field_base is not None and exact_field_patch is not None else 0
        material = bool(
            similarities[idx] < 0.85
            or refusal_flip
            or format_flip
            or ratio > 1.5
            or ratio < 0.5
        )
        family = base_row["task_family"]
        family_bucket = family_rollup.setdefault(
            family,
            {
                "n": 0,
                "material_count": 0,
                "mean_similarity": [],
            },
        )
        family_bucket["n"] += 1
        family_bucket["material_count"] += int(material)
        family_bucket["mean_similarity"].append(float(similarities[idx]))
        format_base.append(int(base_row["format_pass"]))
        format_patch.append(int(patch_row["format_pass"]))
        refusal_base.append(int(base_row["is_refusal"]))
        refusal_patch.append(int(patch_row["is_refusal"]))
        material_list.append(int(material))
        output_rows.append(
            {
                "prompt_id": base_row["prompt_id"],
                "task_family": family,
                "format_type": base_row.get("format_type"),
                "base_len": int(base_row["token_count"]),
                "patched_len": patch_tokens,
                "length_ratio": float(ratio),
                "similarity": float(similarities[idx]),
                "refusal_flip": refusal_flip,
                "format_flip": format_flip,
                "base_format_pass": bool(base_row["format_pass"]),
                "patched_format_pass": bool(patch_row["format_pass"]),
                "base_exact_field_match": exact_field_base,
                "patched_exact_field_match": exact_field_patch,
                "exact_field_flip": exact_field_flip,
                "material_drift": material,
                "prompt_text": base_row["prompt_text"],
                "base_response": base_row["response_text"],
                "patched_response": patch_row["response_text"],
            }
        )

    output_rows.sort(key=lambda row: (int(row["material_drift"]), -row["similarity"], abs(row["length_ratio"] - 1.0)), reverse=True)
    with (root / "compare" / "drift_top_changes.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(output_rows, key=lambda row: (int(row["material_drift"]), 1.0 - row["similarity"], abs(row["length_ratio"] - 1.0)), reverse=True)[:50]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manual_rows = sorted(output_rows, key=lambda row: (int(row["material_drift"]), 1.0 - row["similarity"], abs(row["length_ratio"] - 1.0)), reverse=True)[:20]
    manual_lines = ["# Drift Manual Review", ""]
    for row in manual_rows:
        manual_lines.extend(
            [
                f"## {row['prompt_id']} ({row['task_family']})",
                f"- similarity: {row['similarity']:.4f}",
                f"- length_ratio: {row['length_ratio']:.3f}",
                f"- material_drift: {row['material_drift']}",
                f"- prompt: {row['prompt_text']}",
                f"- base: {row['base_response']}",
                f"- patched: {row['patched_response']}",
                "",
            ]
        )
    (root / "compare" / "drift_manual_review_20.md").write_text("\n".join(manual_lines), encoding="utf-8")

    family_summary = []
    for family, bucket in sorted(family_rollup.items()):
        family_summary.append(
            {
                "task_family": family,
                "n": bucket["n"],
                "material_drift_rate": bucket["material_count"] / bucket["n"],
                "mean_similarity": float(np.mean(bucket["mean_similarity"])),
            }
        )

    format_row = summarize_task(
        "benign_drift_format_pass_rate",
        [{"metric": v} for v in format_base],
        [{"metric": v} for v in format_patch],
        "metric",
        5000,
        0,
        notes="Higher is better; paired over 100 prompts.",
    )
    refusal_row = summarize_task(
        "benign_drift_refusal_rate",
        [{"metric": v} for v in refusal_base],
        [{"metric": v} for v in refusal_patch],
        "metric",
        5000,
        0,
        notes="Lower is better.",
    )
    drift_summary = {
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "material_drift_rate": float(np.mean(material_list)),
        "format_pass_rate_base": float(np.mean(format_base)),
        "format_pass_rate_patched": float(np.mean(format_patch)),
        "refusal_rate_base": float(np.mean(refusal_base)),
        "refusal_rate_patched": float(np.mean(refusal_patch)),
        "family_summary": family_summary,
        "top_changes_considered": 20,
    }
    save_json(root / "compare" / "drift_compare.json", drift_summary)
    similarity_row = {
        "task": "benign_drift_mean_similarity",
        "metric": "cosine_similarity",
        "n": len(similarities),
        "base_score": 1.0,
        "patched_score": drift_summary["mean_similarity"],
        "delta": drift_summary["mean_similarity"] - 1.0,
        "paired_bootstrap_ci_low": float("nan"),
        "paired_bootstrap_ci_high": float("nan"),
        "sign_test_p": float("nan"),
        "better_count": 0,
        "worse_count": 0,
        "notes": "Reference 1.0 means identical outputs; lower is more drift.",
    }
    material_row = {
        "task": "benign_drift_material_rate",
        "metric": "material_drift_rate",
        "n": len(material_list),
        "base_score": 0.0,
        "patched_score": drift_summary["material_drift_rate"],
        "delta": drift_summary["material_drift_rate"],
        "paired_bootstrap_ci_low": float("nan"),
        "paired_bootstrap_ci_high": float("nan"),
        "sign_test_p": float("nan"),
        "better_count": 0,
        "worse_count": 0,
        "notes": "Lower is better.",
    }
    return [similarity_row, format_row, refusal_row, material_row]


def write_summary_tables(root: Path, rows):
    csv_path = root / "compare" / "summary_table.csv"
    md_path = root / "compare" / "summary_table.md"
    fieldnames = [
        "task",
        "n",
        "base_score",
        "patched_score",
        "delta",
        "paired_bootstrap_ci_low",
        "paired_bootstrap_ci_high",
        "sign_test_p",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "| task | n | base_score | patched_score | delta | ci_low | ci_high | sign_test_p | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {n} | {base:.4f} | {patch:.4f} | {delta:.4f} | {lo} | {hi} | {p} | {notes} |".format(
                task=row["task"],
                n=row["n"],
                base=row["base_score"],
                patch=row["patched_score"],
                delta=row["delta"],
                lo="nan" if math.isnan(row["paired_bootstrap_ci_low"]) else f"{row['paired_bootstrap_ci_low']:.4f}",
                hi="nan" if math.isnan(row["paired_bootstrap_ci_high"]) else f"{row['paired_bootstrap_ci_high']:.4f}",
                p="nan" if math.isnan(row["sign_test_p"]) else f"{row['sign_test_p']:.4g}",
                notes=row["notes"],
            )
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.regression_root)
    (root / "compare").mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append(compare_truthfulqa(root, args.bootstrap, args.seed))
    rows.append(compare_lm_eval(root, "hellaswag", "hellaswag", args.bootstrap, args.seed, notes="Metric uses acc_norm when available."))
    rows.append(compare_lm_eval(root, "mmlu_slice", "mmlu_slice", args.bootstrap, args.seed, notes="Combined over 6 subjects x 50 items."))
    rows.append(compare_lm_eval(root, "gsm8k", "gsm8k", args.bootstrap, args.seed, notes="Generation metric from lm-eval samples."))
    rows.extend(compare_drift(root))

    save_json(root / "compare" / "per_task_delta.json", {"rows": rows})
    write_summary_tables(root, rows)
    print(f"Saved regression comparison package under: {root / 'compare'}")


if __name__ == "__main__":
    main()
