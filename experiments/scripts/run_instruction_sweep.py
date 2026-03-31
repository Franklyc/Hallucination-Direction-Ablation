import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a multi-seed sweep over instruction-direction probe configs")
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
        default="7,13",
        help="Comma-separated seeds to run",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200,
        help="Calibration split size",
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
        "--configs",
        default="18,19,20,21,22,23,24@3.0;18,19,20,21,22,23,24@4.0;20,21,22,23,24@3.0;18,19,20,21,22,23,24@2.75",
        help="Semicolon-separated configs in the form layers@beta",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/instruction_sweep",
        help="Where to save outputs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-seed outputs when present",
    )
    return parser.parse_args()


def parse_seed_list(raw: str):
    seeds = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    if not seeds:
        raise ValueError("No seeds specified.")
    return seeds


def parse_configs(raw: str):
    configs = []
    for idx, chunk in enumerate(raw.split(";"), start=1):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "@" not in chunk:
            raise ValueError(f"Invalid config '{chunk}'. Expected layers@beta.")
        layer_text, beta_text = chunk.split("@", 1)
        layers = ",".join(x.strip() for x in layer_text.split(",") if x.strip())
        beta = float(beta_text.strip())
        slug = f"cfg{idx}_l{'-'.join(layers.split(','))}_b{str(beta).replace('.', 'p')}"
        configs.append(
            {
                "layers": layers,
                "beta": beta,
                "slug": slug,
            }
        )
    if not configs:
        raise ValueError("No configs specified.")
    return configs


def run_command(command, workdir: Path):
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(workdir), check=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    workdir = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable
    seeds = parse_seed_list(args.seeds)
    configs = parse_configs(args.configs)

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        prepared_dir = seed_dir / "prepared"
        prepared_report = seed_dir / "dataset_prepare_report.json"
        contrastive_path = prepared_dir / "calib_contrastive.jsonl"
        directions_path = seed_dir / "directions_instruction.npz"
        directions_meta_path = seed_dir / "directions_instruction_meta.json"

        if not (args.skip_existing and contrastive_path.exists() and prepared_report.exists()):
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
                "instruction",
                "--contrastive-jsonl",
                str(contrastive_path),
                "--output",
                str(directions_path),
                "--metadata-json",
                str(directions_meta_path),
            ]
            if args.load_in_4bit:
                extract_cmd.append("--load-in-4bit")
            run_command(extract_cmd, workdir=workdir)

        for config in configs:
            probe_path = seed_dir / f"{config['slug']}.json"
            if args.skip_existing and probe_path.exists():
                continue
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
                config["layers"],
                "--beta",
                str(config["beta"]),
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

    summary_rows = []
    for config in configs:
        per_seed = []
        for seed in seeds:
            seed_dir = output_dir / f"seed_{seed}"
            probe_path = seed_dir / f"{config['slug']}.json"
            result = load_json(probe_path)
            diag = result.get("diagnostics", {})
            per_seed.append(
                {
                    "seed": seed,
                    "delta_acc": float(result.get("delta_acc", 0.0)),
                    "base_acc": float(result.get("base", {}).get("acc", 0.0)),
                    "intervened_acc": float(result.get("intervened", {}).get("acc", 0.0)),
                    "fixed_count": int(diag.get("fixed_count", 0)),
                    "broken_count": int(diag.get("broken_count", 0)),
                    "paired_sign_test_pvalue": float(diag.get("paired_sign_test_pvalue", 1.0)),
                    "probe_result": str(probe_path),
                }
            )

        deltas = [row["delta_acc"] for row in per_seed]
        fixeds = [row["fixed_count"] for row in per_seed]
        brokens = [row["broken_count"] for row in per_seed]
        summary_rows.append(
            {
                "slug": config["slug"],
                "layers": [int(x) for x in config["layers"].split(",") if x],
                "beta": config["beta"],
                "mean_delta_acc": float(statistics.mean(deltas)),
                "median_delta_acc": float(statistics.median(deltas)),
                "std_delta_acc": float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0,
                "positive_seed_count": int(sum(1 for value in deltas if value > 0)),
                "nonnegative_seed_count": int(sum(1 for value in deltas if value >= 0)),
                "mean_fixed_count": float(statistics.mean(fixeds)),
                "mean_broken_count": float(statistics.mean(brokens)),
                "per_seed": per_seed,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["mean_delta_acc"],
            row["positive_seed_count"],
            -row["mean_broken_count"],
        ),
        reverse=True,
    )

    out = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
            "gpu_memory_gb": args.gpu_memory_gb,
            "calibration_size": args.calibration_size,
            "candidate_prefix": args.candidate_prefix,
            "seeds": seeds,
        },
        "results": summary_rows,
    }

    summary_path = output_dir / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    best = summary_rows[0]
    print(f"Saved sweep summary to: {summary_path}")
    print(
        "best config: "
        f"layers={best['layers']} beta={best['beta']} "
        f"mean_delta_acc={100.0 * best['mean_delta_acc']:.2f} points "
        f"positive_seeds={best['positive_seed_count']}/{len(seeds)}"
    )


if __name__ == "__main__":
    main()
