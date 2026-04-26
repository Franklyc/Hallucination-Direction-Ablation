import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate verifier result JSON files.")
    parser.add_argument("--name", default="verifier", help="Label for this result set")
    parser.add_argument("--paths", required=True, help="Comma-separated result JSON files or directories")
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/verifier_aggregate.json",
        help="Where to save the aggregate summary",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_result_payload(payload: dict):
    return isinstance(payload, dict) and "selected_config" in payload and "seed" in payload


def expand_inputs(raw: str):
    values = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if item:
            values.append(Path(item))
    if not values:
        raise ValueError("No input paths provided.")
    return values


def collect_result_files(inputs):
    files = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)

    results = []
    for file_path in files:
        try:
            payload = load_json(file_path)
        except Exception:
            continue
        if is_result_payload(payload):
            results.append((file_path, payload))
    if not results:
        raise ValueError("No verifier result JSON files found.")
    return results


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


def main():
    args = parse_args()
    files_and_payloads = collect_result_files(expand_inputs(args.paths))

    per_seed = {}
    category_values = {}
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
        per_seed[seed] = row
        for category, delta in category_delta.items():
            category_values.setdefault(category, []).append(delta)

    rows = [per_seed[seed] for seed in sorted(per_seed)]
    deltas = [row["delta_acc"] for row in rows]
    output = {
        "name": args.name,
        "summary": {
            "n_seeds": len(rows),
            "mean_delta_acc": float(statistics.mean(deltas)),
            "median_delta_acc": float(statistics.median(deltas)),
            "std_delta_acc": float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0,
            "min_delta_acc": float(min(deltas)),
            "max_delta_acc": float(max(deltas)),
            "positive_seed_count": int(sum(1 for delta in deltas if delta > 0)),
            "nonnegative_seed_count": int(sum(1 for delta in deltas if delta >= 0)),
            "selected_configs": sorted(
                {row["selected_config"] for row in rows if row["selected_config"]}
            ),
        },
        "category_delta_accuracy_mean": {
            category: float(statistics.mean(values))
            for category, values in sorted(category_values.items())
        },
        "per_seed": rows,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregate summary to: {output_path}")
    print(
        f"{args.name} mean delta acc: {100.0 * output['summary']['mean_delta_acc']:.2f} points "
        f"over {output['summary']['n_seeds']} seeds"
    )


if __name__ == "__main__":
    main()
