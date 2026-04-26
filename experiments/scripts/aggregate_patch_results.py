import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate HDA patch result JSON files.")
    parser.add_argument("--name", default="hda_patch", help="Label for this result set")
    parser.add_argument("--paths", required=True, help="Comma-separated result JSON files or directories")
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/analysis/hda_patch_aggregate.json",
        help="Where to save the aggregate summary",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_patch_payload(payload: dict):
    return (
        isinstance(payload, dict)
        and "seed" in payload
        and "base" in payload
        and "patched" in payload
        and "delta_acc" in payload
        and "alpha" in payload
        and "modules" in payload
    )


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
        if is_patch_payload(payload):
            results.append((file_path, payload))
    if not results:
        raise ValueError("No HDA patch result JSON files found.")
    return results


def summarize_category_delta(payload: dict):
    base = payload.get("base", {}).get("category_accuracy", {})
    new = payload.get("patched", {}).get("category_accuracy", {})
    out = {}
    for category in sorted(set(base) | set(new)):
        base_acc = base.get(category, {}).get("accuracy")
        new_acc = new.get(category, {}).get("accuracy")
        if base_acc is None or new_acc is None:
            continue
        out[category] = float(new_acc - base_acc)
    return out


def config_key(payload: dict):
    layers = payload.get("layers", [])
    layer_text = ",".join(str(v) for v in layers)
    alpha = float(payload.get("alpha", 0.0))
    modules = payload.get("modules", "unknown")
    return f"layers={layer_text}|modules={modules}|alpha={alpha:.4g}"


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
            "config_key": config_key(payload),
            "layers": [int(v) for v in payload.get("layers", [])],
            "modules": payload.get("modules"),
            "alpha": float(payload.get("alpha", 0.0)),
            "base_acc": float(payload.get("base", {}).get("acc", 0.0)),
            "patched_acc": float(payload.get("patched", {}).get("acc", 0.0)),
            "delta_acc": float(payload.get("delta_acc", 0.0)),
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
            "config_keys": sorted({row["config_key"] for row in rows}),
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
