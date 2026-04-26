import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import save_json


MMLU_TASKS = [
    "mmlu_high_school_biology",
    "mmlu_high_school_us_history",
    "mmlu_college_computer_science",
    "mmlu_formal_logic",
    "mmlu_business_ethics",
    "mmlu_high_school_macroeconomics",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the paired base-vs-patched regression package.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Base model id or local directory")
    parser.add_argument("--directions", required=True, help="Path to directions npz")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--gpu-memory-gb", type=int, default=15, help="Per-GPU memory cap in GiB")
    parser.add_argument("--seed", type=int, default=41, help="Evaluation seed")
    parser.add_argument("--calibration-size", type=int, default=200, help="TruthfulQA calibration split size")
    parser.add_argument("--candidate-prefix", default="newline", choices=["space", "newline", "none"])
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--alpha", type=float, required=True, help="Patch alpha")
    parser.add_argument("--modules", default="mlp", choices=["attn", "mlp", "both"], help="Which modules to patch")
    parser.add_argument(
        "--regression-root",
        default="experiments/artifacts/regression",
        help="Root output directory for the paired package",
    )
    parser.add_argument(
        "--patched-model-dir",
        default="experiments/artifacts/regression/patched_model",
        help="Local directory where the materialized patched model will be saved",
    )
    parser.add_argument(
        "--drift-jsonl",
        default="experiments/data/prepared/drift_benign_100.jsonl",
        help="Expanded benign drift prompt set",
    )
    parser.add_argument("--truthfulqa-csv", default="experiments/data/TruthfulQA.csv")
    parser.add_argument("--hellaswag-limit", type=int, default=500)
    parser.add_argument("--mmlu-limit", type=int, default=50)
    parser.add_argument("--gsm8k-limit", type=int, default=200)
    parser.add_argument("--hellaswag-batch-size", default="16")
    parser.add_argument("--mmlu-batch-size", default="16")
    parser.add_argument("--gsm8k-batch-size", default="1")
    parser.add_argument("--drift-max-new-tokens", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-truthfulqa", action="store_true")
    parser.add_argument("--skip-lm-eval", action="store_true")
    parser.add_argument("--skip-drift", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    return parser.parse_args()


def run(cmd, cwd: Path, env=None):
    print("RUN:", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_capture(cmd, cwd: Path, env=None):
    print("RUN:", " ".join(str(part) for part in cmd))
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=True, capture_output=True, text=True)


def ensure_dirs(root: Path):
    for relative in [
        "base/truthfulqa",
        "patched/truthfulqa",
        "base/hellaswag",
        "patched/hellaswag",
        "base/mmlu_slice",
        "patched/mmlu_slice",
        "base/gsm8k",
        "patched/gsm8k",
        "base/drift",
        "patched/drift",
        "compare",
        "cache",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def has_any(path: Path, pattern: str) -> bool:
    return any(path.rglob(pattern))


def build_manifest(args, root: Path, patched_model_dir: Path, gsm8k_task: str | None):
    manifest = {
        "base_model": args.model,
        "patched_model_dir": str(patched_model_dir),
        "directions": args.directions,
        "dtype": args.dtype,
        "gpu_memory_gb": args.gpu_memory_gb,
        "seed": args.seed,
        "calibration_size": args.calibration_size,
        "candidate_prefix": args.candidate_prefix,
        "layers": args.layers,
        "alpha": args.alpha,
        "modules": args.modules,
        "hellaswag_limit": args.hellaswag_limit,
        "hellaswag_batch_size": args.hellaswag_batch_size,
        "mmlu_tasks": MMLU_TASKS,
        "mmlu_limit": args.mmlu_limit,
        "mmlu_batch_size": args.mmlu_batch_size,
        "gsm8k_task": gsm8k_task,
        "gsm8k_limit": args.gsm8k_limit,
        "gsm8k_batch_size": args.gsm8k_batch_size,
        "drift_jsonl": args.drift_jsonl,
        "drift_max_new_tokens": args.drift_max_new_tokens,
    }
    save_json(root / "manifest.json", manifest)


def lm_eval_env():
    env = os.environ.copy()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def task_list(workspace: Path, root: Path, env):
    result = run_capture([sys.executable, "-m", "lm_eval", "ls", "tasks"], cwd=workspace, env=env)
    text = result.stdout
    (root / "available_tasks.txt").write_text(text, encoding="utf-8")
    return text


def detect_gsm8k_task(task_text: str):
    lowered = task_text.lower()
    if "|gsm8k|" in lowered or "\ngsm8k\n" in lowered or " gsm8k " in lowered:
        return "gsm8k"
    if "gsm8k_cot" in lowered:
        return "gsm8k_cot"
    raise ValueError("Neither gsm8k nor gsm8k_cot was found in lm-eval task list.")


def main():
    args = parse_args()
    workspace = Path(__file__).resolve().parents[2]
    root = Path(args.regression_root)
    if not root.is_absolute():
        root = workspace / root
    patched_model_dir = Path(args.patched_model_dir)
    if not patched_model_dir.is_absolute():
        patched_model_dir = workspace / patched_model_dir

    ensure_dirs(root)
    env = lm_eval_env()

    task_text = ""
    gsm8k_task = None
    if not args.skip_lm_eval:
        task_text = task_list(workspace, root, env)
        gsm8k_task = detect_gsm8k_task(task_text)

    build_manifest(args, root, patched_model_dir, gsm8k_task)

    if not args.skip_export:
        if not (patched_model_dir / "patch_manifest.json").exists():
            run(
                [
                    sys.executable,
                    str(workspace / "experiments" / "scripts" / "export_patched_model.py"),
                    "--model",
                    args.model,
                    "--directions",
                    args.directions,
                    "--dtype",
                    args.dtype,
                    "--gpu-memory-gb",
                    str(args.gpu_memory_gb),
                    "--layers",
                    args.layers,
                    "--alpha",
                    str(args.alpha),
                    "--modules",
                    args.modules,
                    "--output-dir",
                    str(patched_model_dir),
                ],
                cwd=workspace,
            )
        else:
            print(f"SKIP export, found: {patched_model_dir / 'patch_manifest.json'}")

    run(
        [
            sys.executable,
            str(workspace / "experiments" / "scripts" / "build_benign_drift_100.py"),
            "--output-jsonl",
            str(Path(args.drift_jsonl) if Path(args.drift_jsonl).is_absolute() else workspace / args.drift_jsonl),
        ],
        cwd=workspace,
    )

    if not args.skip_truthfulqa:
        base_truth = root / "base" / "truthfulqa" / "base_truthfulqa.json"
        patch_truth = root / "patched" / "truthfulqa" / "patched_truthfulqa.json"
        for model_path, output_json in [
            (args.model, base_truth),
            (str(patched_model_dir), patch_truth),
        ]:
            if output_json.exists():
                print(f"SKIP TruthfulQA, found: {output_json}")
            else:
                run(
                    [
                        sys.executable,
                        str(workspace / "experiments" / "scripts" / "truthfulqa_binary_eval.py"),
                        "--model",
                        model_path,
                        "--truthfulqa-csv",
                        args.truthfulqa_csv,
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
                        "--bootstrap",
                        str(args.bootstrap),
                        "--output-json",
                        str(output_json),
                    ],
                    cwd=workspace,
                )

    if not args.skip_lm_eval:
        common_args = [
            sys.executable,
            "-m",
            "lm_eval",
            "run",
            "--model",
            "hf",
            "--device",
            "cuda:0",
            "--apply_chat_template",
            "--log_samples",
        ]

        for model_label, model_path in [("base", args.model), ("patched", str(patched_model_dir))]:
            model_args = f"pretrained={model_path},dtype={args.dtype},trust_remote_code=True"

            if has_any(root / model_label / "hellaswag", "results_*.json"):
                print(f"SKIP HellaSwag for {model_label}, results already present.")
            else:
                run(
                    common_args
                    + [
                        "--model_args",
                        model_args,
                        "--batch_size",
                        args.hellaswag_batch_size,
                        "--tasks",
                        "hellaswag",
                        "--limit",
                        str(args.hellaswag_limit),
                        "--output_path",
                        str(root / model_label / "hellaswag"),
                        "--use_cache",
                        str(root / "cache" / f"{model_label}_hellaswag"),
                    ],
                    cwd=workspace,
                    env=env,
                )

            if has_any(root / model_label / "mmlu_slice", "results_*.json"):
                print(f"SKIP MMLU slice for {model_label}, results already present.")
            else:
                run(
                    common_args
                    + [
                        "--model_args",
                        model_args,
                        "--batch_size",
                        args.mmlu_batch_size,
                        "--tasks",
                        ",".join(MMLU_TASKS),
                        "--limit",
                        str(args.mmlu_limit),
                        "--output_path",
                        str(root / model_label / "mmlu_slice"),
                        "--use_cache",
                        str(root / "cache" / f"{model_label}_mmlu_slice"),
                    ],
                    cwd=workspace,
                    env=env,
                )

            if has_any(root / model_label / "gsm8k", "results_*.json"):
                print(f"SKIP GSM8K for {model_label}, results already present.")
            else:
                run(
                    [
                        sys.executable,
                        "-m",
                        "lm_eval",
                        "run",
                        "--model",
                        "hf",
                        "--model_args",
                        model_args,
                        "--tasks",
                        gsm8k_task,
                        "--limit",
                        str(args.gsm8k_limit),
                        "--batch_size",
                        args.gsm8k_batch_size,
                        "--device",
                        "cuda:0",
                        "--apply_chat_template",
                        "--gen_kwargs",
                        "do_sample=False,temperature=0.0",
                        "--output_path",
                        str(root / model_label / "gsm8k"),
                        "--log_samples",
                        "--use_cache",
                        str(root / "cache" / f"{model_label}_gsm8k"),
                    ],
                    cwd=workspace,
                    env=env,
                )

    if not args.skip_drift:
        drift_jsonl = Path(args.drift_jsonl)
        if not drift_jsonl.is_absolute():
            drift_jsonl = workspace / drift_jsonl
        for model_path, output_json in [
            (args.model, root / "base" / "drift" / "base_drift.json"),
            (str(patched_model_dir), root / "patched" / "drift" / "patched_drift.json"),
        ]:
            if output_json.exists():
                print(f"SKIP drift eval, found: {output_json}")
            else:
                run(
                    [
                        sys.executable,
                        str(workspace / "experiments" / "scripts" / "run_benign_drift_eval.py"),
                        "--model",
                        model_path,
                        "--drift-jsonl",
                        str(drift_jsonl),
                        "--dtype",
                        args.dtype,
                        "--gpu-memory-gb",
                        str(args.gpu_memory_gb),
                        "--max-new-tokens",
                        str(args.drift_max_new_tokens),
                        "--output-json",
                        str(output_json),
                    ],
                    cwd=workspace,
                )

    if not args.skip_compare:
        run(
            [
                sys.executable,
                str(workspace / "experiments" / "scripts" / "compare_regression_results.py"),
                "--regression-root",
                str(root),
                "--bootstrap",
                str(args.bootstrap),
                "--seed",
                str(args.seed),
            ],
            cwd=workspace,
        )


if __name__ == "__main__":
    main()
