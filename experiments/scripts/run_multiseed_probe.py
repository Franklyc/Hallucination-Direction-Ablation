import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run upgraded extraction+probe pipeline across multiple seeds")
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
        help="Use 4-bit loading for extraction and probe runs",
    )
    parser.add_argument(
        "--seeds",
        default="7,13,29",
        help="Comma-separated seeds to run",
    )
    parser.add_argument(
        "--direction-method",
        default="answer_state",
        choices=["instruction", "answer_state"],
        help="Direction extraction method",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200,
        help="Calibration split size",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices for the activation probe",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Projection removal strength for the activation probe",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="newline",
        choices=["space", "newline", "none"],
        help="Candidate prefix style for binary scoring",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap rounds for per-seed probe evaluation",
    )
    parser.add_argument(
        "--answer-pool",
        default="mean",
        choices=["mean", "first", "last"],
        help="Answer pooling mode for answer_state directions",
    )
    parser.add_argument(
        "--max-correct-variants",
        type=int,
        default=3,
        help="Max correct variants per question for answer_state directions",
    )
    parser.add_argument(
        "--max-incorrect-variants",
        type=int,
        default=3,
        help="Max incorrect variants per question for answer_state directions",
    )
    parser.add_argument(
        "--contrastive-jsonl",
        default="experiments/data/prepared/calib_contrastive.jsonl",
        help="Prepared contrastive calibration rows for instruction mode",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/multiseed_probe",
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


def maybe_load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_delta(probe_result: dict):
    base = probe_result.get("base", {}).get("category_accuracy", {})
    new = probe_result.get("intervened", {}).get("category_accuracy", {})
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)
    python_exe = sys.executable

    per_seed = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        directions_path = seed_dir / f"directions_{args.direction_method}.npz"
        directions_meta_path = seed_dir / f"directions_{args.direction_method}_meta.json"
        probe_path = seed_dir / f"probe_{args.direction_method}.json"
        seed_contrastive_path = Path(args.contrastive_jsonl)

        if args.direction_method == "instruction":
            prepared_dir = seed_dir / "prepared"
            prepared_report = seed_dir / "dataset_prepare_report.json"
            seed_contrastive_path = prepared_dir / "calib_contrastive.jsonl"
            if not (args.skip_existing and seed_contrastive_path.exists() and prepared_report.exists()):
                prepare_cmd = [
                    python_exe,
                    str(script_dir / "prepare_truthfulqa.py"),
                    "--seed",
                    str(seed),
                    "--calibration-size",
                    str(args.calibration_size),
                    "--output-dir",
                    str(prepared_dir),
                    "--report-json",
                    str(prepared_report),
                ]
                run_command(prepare_cmd, workdir=workdir)

        if not (args.skip_existing and directions_path.exists() and directions_meta_path.exists()):
            extract_cmd = [
                python_exe,
                str(script_dir / "extract_direction.py"),
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
                "--method",
                args.direction_method,
                "--output",
                str(directions_path),
                "--metadata-json",
                str(directions_meta_path),
            ]
            if args.load_in_4bit:
                extract_cmd.append("--load-in-4bit")
            if args.direction_method == "instruction":
                extract_cmd.extend(
                    [
                        "--contrastive-jsonl",
                        str(seed_contrastive_path),
                    ]
                )
            else:
                extract_cmd.extend(
                    [
                        "--answer-pool",
                        args.answer_pool,
                        "--max-correct-variants",
                        str(args.max_correct_variants),
                        "--max-incorrect-variants",
                        str(args.max_incorrect_variants),
                    ]
                )
            run_command(extract_cmd, workdir=workdir)

        if not (args.skip_existing and probe_path.exists()):
            probe_cmd = [
                python_exe,
                str(script_dir / "activation_probe.py"),
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
                "--directions",
                str(directions_path),
                "--layers",
                args.layers,
                "--beta",
                str(args.beta),
                "--candidate-prefix",
                args.candidate_prefix,
                "--bootstrap",
                str(args.bootstrap),
                "--output-json",
                str(probe_path),
            ]
            if args.load_in_4bit:
                probe_cmd.append("--load-in-4bit")
            run_command(probe_cmd, workdir=workdir)

        probe_result = maybe_load_json(probe_path)
        directions_meta = maybe_load_json(directions_meta_path)
        per_seed.append(
            {
                "seed": seed,
                "delta_acc": float(probe_result.get("delta_acc", 0.0)),
                "base_acc": float(probe_result.get("base", {}).get("acc", 0.0)),
                "intervened_acc": float(probe_result.get("intervened", {}).get("acc", 0.0)),
                "fixed_count": int(probe_result.get("diagnostics", {}).get("fixed_count", 0)),
                "broken_count": int(probe_result.get("diagnostics", {}).get("broken_count", 0)),
                "paired_sign_test_pvalue": float(
                    probe_result.get("diagnostics", {}).get("paired_sign_test_pvalue", 1.0)
                ),
                "category_delta_accuracy": build_category_delta(probe_result),
                "probe_result": str(probe_path),
                "directions_meta": str(directions_meta_path),
                "direction_method": directions_meta.get("method"),
                "direction_semantics": directions_meta.get("direction_semantics"),
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
            "direction_method": args.direction_method,
            "calibration_size": args.calibration_size,
            "layers": [int(x.strip()) for x in args.layers.split(",") if x.strip()],
            "beta": args.beta,
            "candidate_prefix": args.candidate_prefix,
            "answer_pool": args.answer_pool,
            "max_correct_variants": args.max_correct_variants,
            "max_incorrect_variants": args.max_incorrect_variants,
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
