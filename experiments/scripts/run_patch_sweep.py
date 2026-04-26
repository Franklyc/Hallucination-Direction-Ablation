import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a targeted sweep over HDA patch configs.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--truthfulqa-csv", default="experiments/data/TruthfulQA.csv")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-gb", type=int, default=15)
    parser.add_argument("--candidate-prefix", default="newline", choices=["space", "newline", "none"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--modules", default="mlp", choices=["attn", "mlp", "both"])
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--diagnostic-top-k", type=int, default=20)
    parser.add_argument(
        "--configs",
        required=True,
        help="Semicolon-separated configs in the form label=18,19,20@2.0, label=18,19,20, or just 18,19,20",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/patch_sweep",
        help="Directory to save per-config outputs and aggregate summary",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_configs(raw: str):
    rows = []
    for idx, chunk in enumerate(raw.split(";"), start=1):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            label, spec = chunk.split("=", 1)
            label = label.strip()
        else:
            spec = chunk
            label = f"cfg{idx}"
        spec = spec.strip()
        if "@" in spec:
            layers, alpha_text = spec.rsplit("@", 1)
            alpha_override = float(alpha_text.strip())
        else:
            layers = spec
            alpha_override = None

        layer_values = [int(x.strip()) for x in layers.split(",") if x.strip()]
        layer_text = ",".join(str(x) for x in layer_values)
        slug_layers = "-".join(str(x) for x in layer_values)
        alpha_slug = ""
        if alpha_override is not None:
            alpha_slug = f"_a{str(alpha_override).replace('.', 'p')}"
        rows.append(
            {
                "label": label,
                "layers": layer_values,
                "layer_text": layer_text,
                "alpha": alpha_override,
                "slug": f"{label}_l{slug_layers}{alpha_slug}",
            }
        )
    if not rows:
        raise ValueError("No valid configs provided.")
    return rows


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
    output_dir = workdir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable

    configs = parse_configs(args.configs)
    summary_rows = []
    for config in configs:
        out_path = output_dir / f"{config['slug']}.json"
        if not (args.skip_existing and out_path.exists()):
            cmd = [
                python_exe,
                str(script_dir / "weight_patch_eval.py"),
                "--model",
                args.model,
                "--truthfulqa-csv",
                args.truthfulqa_csv,
                "--directions",
                args.directions,
                "--dtype",
                args.dtype,
                "--gpu-memory-gb",
                str(args.gpu_memory_gb),
                "--candidate-prefix",
                args.candidate_prefix,
                "--seed",
                str(args.seed),
                "--calibration-size",
                str(args.calibration_size),
                "--layers",
                config["layer_text"],
                "--alpha",
                str(config["alpha"] if config["alpha"] is not None else args.alpha),
                "--modules",
                args.modules,
                "--disable-thinking" if args.disable_thinking else "",
                "--bootstrap",
                str(args.bootstrap),
                "--diagnostic-top-k",
                str(args.diagnostic_top_k),
                "--output-json",
                str(out_path),
            ]
            cmd = [part for part in cmd if part != ""]
            run_command(cmd, workdir=workdir)

        payload = load_json(out_path)
        summary_rows.append(
            {
                "label": config["label"],
                "layers": config["layers"],
                "path": str(out_path),
                "alpha": float(payload["alpha"]),
                "delta_acc": float(payload["delta_acc"]),
                "base_acc": float(payload["base"]["acc"]),
                "patched_acc": float(payload["patched"]["acc"]),
                "fixed_count": int(payload.get("diagnostics", {}).get("fixed_count", 0)),
                "broken_count": int(payload.get("diagnostics", {}).get("broken_count", 0)),
            }
        )

    summary_rows.sort(
        key=lambda row: (row["delta_acc"], row["fixed_count"], -row["broken_count"]),
        reverse=True,
    )
    output = {
        "config": {
            "seed": args.seed,
            "directions": args.directions,
            "modules": args.modules,
            "alpha": args.alpha,
            "disable_thinking": args.disable_thinking,
            "candidate_prefix": args.candidate_prefix,
            "calibration_size": args.calibration_size,
        },
        "summary": {
            "n_configs": len(summary_rows),
            "best_delta_acc": float(summary_rows[0]["delta_acc"]),
            "worst_delta_acc": float(summary_rows[-1]["delta_acc"]),
            "mean_delta_acc": float(statistics.mean(row["delta_acc"] for row in summary_rows)),
        },
        "rows": summary_rows,
    }
    summary_path = output_dir / "aggregate_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved sweep summary to: {summary_path}")
    print(
        f"Best config: {summary_rows[0]['label']} "
        f"delta={100.0 * summary_rows[0]['delta_acc']:.2f} points"
    )


if __name__ == "__main__":
    main()
