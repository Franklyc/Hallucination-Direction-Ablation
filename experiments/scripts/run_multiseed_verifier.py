import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibration-selected verifier reranking across multiple seeds")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--gpu-memory-gb",
        type=int,
        default=15,
        help="Per-GPU memory cap in GiB for model loading",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit loading for verifier runs",
    )
    parser.add_argument(
        "--seeds",
        default="7,13,29",
        help="Comma-separated seeds to run",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200,
        help="Calibration split size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional row cap before split",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="newline",
        choices=["space", "newline", "none"],
        help="Prefix style for the baseline A/B scorer",
    )
    parser.add_argument(
        "--verdict-prefixes",
        default="newline",
        help="Comma-separated prefix styles for yes/no verifier scoring",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=400,
        help="Bootstrap rounds for per-seed eval",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/verifier_multiseed",
        help="Where to save per-seed and aggregate outputs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-seed outputs when present",
    )
    return parser.parse_args()


def parse_seed_list(raw: str):
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("No seeds specified.")
    return values


def run_command(command, workdir: Path):
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(workdir), check=True)


def resolve_cli_path(path_str: str, workdir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return workdir / path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_delta(result: dict):
    base = result.get("base", {}).get("category_accuracy", {})
    new = result.get("verifier", {}).get("category_accuracy", {})
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
    workdir = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent
    output_dir = resolve_cli_path(args.output_dir, workdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable
    seeds = parse_seed_list(args.seeds)

    per_seed = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result_path = seed_dir / "verifier_eval.json"

        if not (args.skip_existing and result_path.exists()):
            cmd = [
                python_exe,
                str(script_dir / "truthfulqa_verifier_eval.py"),
                "--model",
                args.model,
                "--dtype",
                args.dtype,
                "--gpu-memory-gb",
                str(args.gpu_memory_gb),
                "--seed",
                str(seed),
                "--calibration-size",
                str(args.calibration_size),
                "--candidate-prefix",
                args.candidate_prefix,
                "--verdict-prefixes",
                args.verdict_prefixes,
                "--bootstrap",
                str(args.bootstrap),
                "--output-json",
                str(result_path),
            ]
            if args.max_samples > 0:
                cmd.extend(["--max-samples", str(args.max_samples)])
            if args.load_in_4bit:
                cmd.append("--load-in-4bit")
            run_command(cmd, workdir=workdir)

        result = load_json(result_path)
        per_seed.append(
            {
                "seed": seed,
                "base_acc": float(result.get("base", {}).get("acc", 0.0)),
                "verifier_acc": float(result.get("verifier", {}).get("acc", 0.0)),
                "delta_acc": float(result.get("delta_acc", 0.0)),
                "selected_config": result.get("selected_config", {}).get("key"),
                "fixed_count": int(result.get("diagnostics", {}).get("fixed_count", 0)),
                "broken_count": int(result.get("diagnostics", {}).get("broken_count", 0)),
                "paired_sign_test_pvalue": float(
                    result.get("diagnostics", {}).get("paired_sign_test_pvalue", 1.0)
                ),
                "category_delta_accuracy": build_category_delta(result),
                "result_path": str(result_path),
            }
        )

    deltas = [row["delta_acc"] for row in per_seed]
    category_values = {}
    for row in per_seed:
        for category, delta in row["category_delta_accuracy"].items():
            category_values.setdefault(category, []).append(delta)

    aggregate = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
            "gpu_memory_gb": args.gpu_memory_gb,
            "calibration_size": args.calibration_size,
            "max_samples": args.max_samples,
            "candidate_prefix": args.candidate_prefix,
            "verdict_prefixes": args.verdict_prefixes,
            "bootstrap": args.bootstrap,
            "seeds": seeds,
        },
        "summary": {
            "n_seeds": len(per_seed),
            "mean_delta_acc": float(statistics.mean(deltas)),
            "median_delta_acc": float(statistics.median(deltas)),
            "std_delta_acc": float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0,
            "min_delta_acc": float(min(deltas)),
            "max_delta_acc": float(max(deltas)),
            "positive_seed_count": int(sum(1 for value in deltas if value > 0)),
            "nonnegative_seed_count": int(sum(1 for value in deltas if value >= 0)),
            "selected_configs": sorted(
                {row["selected_config"] for row in per_seed if row["selected_config"]}
            ),
        },
        "category_delta_accuracy_mean": {
            category: float(statistics.mean(values))
            for category, values in sorted(category_values.items())
        },
        "per_seed": per_seed,
    }

    aggregate_path = output_dir / "aggregate_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregate summary to: {aggregate_path}")
    print(
        "multi-seed summary: "
        f"mean_delta_acc={100.0 * aggregate['summary']['mean_delta_acc']:.2f} points "
        f"positive_seeds={aggregate['summary']['positive_seed_count']}/{aggregate['summary']['n_seeds']}"
    )


if __name__ == "__main__":
    main()
