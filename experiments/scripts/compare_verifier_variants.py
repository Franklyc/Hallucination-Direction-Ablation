import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two verifier result sets across shared seeds.")
    parser.add_argument("--left-name", default="selected", help="Label for the first verifier variant")
    parser.add_argument("--left-paths", required=True, help="Comma-separated result JSON files or directories")
    parser.add_argument("--right-name", default="fixed", help="Label for the second verifier variant")
    parser.add_argument("--right-paths", required=True, help="Comma-separated result JSON files or directories")
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/verifier_variant_comparison.json",
        help="Where to save comparison output",
    )
    return parser.parse_args()


def expand_inputs(raw: str):
    paths = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if value:
            paths.append(Path(value))
    if not paths:
        raise ValueError("No input paths provided.")
    return paths


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_result_payload(payload: dict):
    return isinstance(payload, dict) and "selected_config" in payload and "seed" in payload


def collect_result_files(inputs):
    files = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
    result_files = []
    for file_path in files:
        try:
            payload = load_json(file_path)
        except Exception:
            continue
        if is_result_payload(payload):
            result_files.append((file_path, payload))
    if not result_files:
        raise ValueError("No verifier result JSON files found.")
    return result_files


def summarize_category_delta(payload: dict):
    base = payload.get("base", {}).get("category_accuracy", {})
    new = payload.get("verifier", {}).get("category_accuracy", {})
    out = {}
    for category in sorted(set(base) | set(new)):
        base_acc = base.get(category, {}).get("accuracy")
        new_acc = new.get(category, {}).get("accuracy")
        if base_acc is None or new_acc is None:
            continue
        out[category] = float(new_acc - base_acc)
    return out


def build_variant_summary(name: str, files_and_payloads):
    by_seed = {}
    category_values = {}
    rows = []
    for file_path, payload in files_and_payloads:
        seed = int(payload["seed"])
        category_delta = summarize_category_delta(payload)
        row = {
            "seed": seed,
            "path": str(file_path),
            "selected_config": payload.get("selected_config", {}).get("key"),
            "base_acc": float(payload.get("base", {}).get("acc", 0.0)),
            "verifier_acc": float(payload.get("verifier", {}).get("acc", 0.0)),
            "delta_acc": float(payload.get("delta_acc", 0.0)),
            "fixed_count": int(payload.get("diagnostics", {}).get("fixed_count", 0)),
            "broken_count": int(payload.get("diagnostics", {}).get("broken_count", 0)),
            "paired_sign_test_pvalue": float(
                payload.get("diagnostics", {}).get("paired_sign_test_pvalue", 1.0)
            ),
            "category_delta_accuracy": category_delta,
        }
        by_seed[seed] = row
        rows.append(row)
        for category, delta in category_delta.items():
            category_values.setdefault(category, []).append(delta)

    rows.sort(key=lambda row: row["seed"])
    deltas = [row["delta_acc"] for row in rows]
    summary = {
        "name": name,
        "n_seeds": len(rows),
        "mean_delta_acc": float(statistics.mean(deltas)),
        "median_delta_acc": float(statistics.median(deltas)),
        "std_delta_acc": float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0,
        "min_delta_acc": float(min(deltas)),
        "max_delta_acc": float(max(deltas)),
        "positive_seed_count": int(sum(1 for delta in deltas if delta > 0)),
        "selected_configs": sorted(
            {row["selected_config"] for row in rows if row["selected_config"]}
        ),
        "category_delta_accuracy_mean": {
            category: float(statistics.mean(values))
            for category, values in sorted(category_values.items())
        },
        "per_seed": rows,
    }
    return summary, by_seed


def main():
    args = parse_args()
    left_summary, left_by_seed = build_variant_summary(
        args.left_name,
        collect_result_files(expand_inputs(args.left_paths)),
    )
    right_summary, right_by_seed = build_variant_summary(
        args.right_name,
        collect_result_files(expand_inputs(args.right_paths)),
    )

    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        raise ValueError("The two variants do not share any seeds.")

    paired_rows = []
    paired_delta_gaps = []
    category_gap_values = {}
    for seed in shared_seeds:
        left_row = left_by_seed[seed]
        right_row = right_by_seed[seed]
        gap = float(right_row["delta_acc"] - left_row["delta_acc"])
        paired_delta_gaps.append(gap)
        category_gap = {}
        for category in sorted(
            set(left_row["category_delta_accuracy"]) | set(right_row["category_delta_accuracy"])
        ):
            left_value = left_row["category_delta_accuracy"].get(category)
            right_value = right_row["category_delta_accuracy"].get(category)
            if left_value is None or right_value is None:
                continue
            diff = float(right_value - left_value)
            category_gap[category] = diff
            category_gap_values.setdefault(category, []).append(diff)

        paired_rows.append(
            {
                "seed": seed,
                args.left_name: left_row,
                args.right_name: right_row,
                "delta_acc_gap": gap,
                "category_delta_gap": category_gap,
            }
        )

    output = {
        "variants": {
            args.left_name: left_summary,
            args.right_name: right_summary,
        },
        "paired_comparison": {
            "shared_seed_count": len(shared_seeds),
            "shared_seeds": shared_seeds,
            "mean_delta_acc_gap": float(statistics.mean(paired_delta_gaps)),
            "median_delta_acc_gap": float(statistics.median(paired_delta_gaps)),
            "fixed_better_count": int(sum(1 for gap in paired_delta_gaps if gap > 0)),
            "fixed_nonworse_count": int(sum(1 for gap in paired_delta_gaps if gap >= 0)),
            "category_delta_gap_mean": {
                category: float(statistics.mean(values))
                for category, values in sorted(category_gap_values.items())
            },
            "per_seed": paired_rows,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved comparison to: {output_path}")
    print(
        f"{args.right_name} - {args.left_name} mean delta gap: "
        f"{100.0 * output['paired_comparison']['mean_delta_acc_gap']:.2f} points "
        f"over {output['paired_comparison']['shared_seed_count']} shared seeds"
    )


if __name__ == "__main__":
    main()
