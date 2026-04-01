"""
Colab quick start:

    !git clone https://github.com/Franklyc/Hallucination-Direction-Ablation.git
    %cd Hallucination-Direction-Ablation
    !pip install -U pip
    !pip install torch transformers accelerate bitsandbytes datasets numpy pandas scipy matplotlib tqdm
    !python experiments/scripts/run_heretic_simple_followup_colab.py --load-in-4bit --run-tag colab

The script focuses on the current highest-value HERETIC-simple follow-up:
re-confirm the best non-direct runtime direction, test a reject-specific branch,
and then run one attention-only patch transfer check around the same layer band.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    state_name: str
    directions_npz: Path
    direction_key: str
    layers: str
    beta: float
    answer_prediction_steps: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Colab-oriented HERETIC-simple follow-up driver. "
            "It prepares data, captures early-answer states, extracts directions, "
            "runs focused runtime sweeps, optionally runs a weight patch check, "
            "and writes a compact summary."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--gpu-memory-gb",
        type=int,
        default=14,
        help="Per-GPU memory cap in GiB",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit loading for capture/runtime sweeps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Main random seed used across preparation/capture/extraction",
    )
    parser.add_argument(
        "--calibration-questions",
        type=int,
        default=300,
        help="Question-level HERETIC-simple calibration split size",
    )
    parser.add_argument(
        "--dev-questions",
        type=int,
        default=100,
        help="Question-level HERETIC-simple dev split size",
    )
    parser.add_argument(
        "--capture-max-samples",
        type=int,
        default=400,
        help="Optional calibration row cap for faster Colab capture",
    )
    parser.add_argument(
        "--dev-max-samples",
        type=int,
        default=400,
        help="Optional dev row cap for runtime/patch eval; use 0 for full dev",
    )
    parser.add_argument(
        "--capture-answer-tokens",
        type=int,
        default=5,
        help="How many generated answer tokens to summarize in capture",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=20,
        help="Bootstrap stability rounds for direction extraction",
    )
    parser.add_argument(
        "--runtime-layers",
        default="19,20,21",
        help="Layer band used by the focused runtime/patch follow-up",
    )
    parser.add_argument(
        "--runtime-beta",
        type=float,
        default=0.5,
        help="Projection subtraction strength for runtime sweeps",
    )
    parser.add_argument(
        "--answer-prediction-steps",
        type=int,
        default=1,
        choices=[1, 3],
        help="How many early answer prediction steps to edit",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Generation budget for capture/runtime/patch eval",
    )
    parser.add_argument(
        "--patch-alpha",
        type=float,
        default=0.2,
        help="Patch strength for attention-only transfer check",
    )
    parser.add_argument(
        "--patch-modules",
        default="attn",
        choices=["attn", "mlp", "both"],
        help="Which write modules to patch",
    )
    parser.add_argument(
        "--patch-direction-key",
        default="pairmean_non_direct_minus_direct__normalized",
        help="Direction key used for the follow-up weight patch check",
    )
    parser.add_argument(
        "--patch-state",
        default="answer1to5",
        choices=["answer1", "answer1to5"],
        help="Which extracted state family to use for patching",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix for the artifact subdirectory",
    )
    parser.add_argument(
        "--skip-patch",
        action="store_true",
        help="Skip the patch transfer stage",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if their output files already exist",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_step(name: str, command: list[str], cwd: Path, output_sentinel: Path | None, force: bool):
    if output_sentinel is not None and output_sentinel.exists() and not force:
        print(f"[skip] {name}: {output_sentinel}")
        return
    print(f"[run] {name}")
    print(" ", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def runtime_delta_view(result: dict[str, Any]) -> dict[str, float]:
    base_bucket = result["base"]["bucket_success_rate"]
    target_bucket = result["target"]["bucket_success_rate"]
    delta = result["delta"]
    return {
        "delta_contradicted_rate": float(delta["contradicted_rate"]),
        "delta_supported_answer_rate": float(delta["supported_answer_rate"]),
        "delta_good_non_direct_rate": float(delta["good_non_direct_rate"]),
        "delta_bad_abstention_rate": float(delta["bad_abstention_rate"]),
        "delta_supported_direct_bucket": float(
            target_bucket["supported_direct"] - base_bucket["supported_direct"]
        ),
        "delta_insufficient_bucket": float(
            target_bucket["insufficient_should_abstain"] - base_bucket["insufficient_should_abstain"]
        ),
        "delta_reject_bucket": float(
            target_bucket["fabricated_premise_should_reject"]
            - base_bucket["fabricated_premise_should_reject"]
        ),
        "delta_clarify_bucket": float(
            target_bucket["ambiguous_should_clarify"] - base_bucket["ambiguous_should_clarify"]
        ),
        "output_changed_rate": float(delta["output_changed_rate"]),
    }


def patch_delta_view(result: dict[str, Any]) -> dict[str, float]:
    delta = result["delta"]
    target = result["target"]["bucket_success_rate"]
    base = result["base"]["bucket_success_rate"]
    return {
        "delta_target_contradicted_rate": float(delta["target_contradicted_rate"]),
        "delta_target_supported_answer_rate": float(delta["target_supported_answer_rate"]),
        "delta_target_good_non_direct_rate": float(delta["target_good_non_direct_rate"]),
        "delta_target_bad_abstention_rate": float(delta["target_bad_abstention_rate"]),
        "delta_target_supported_direct_bucket": float(
            target["supported_direct"] - base["supported_direct"]
        ),
        "delta_target_insufficient_bucket": float(
            target["insufficient_should_abstain"] - base["insufficient_should_abstain"]
        ),
        "delta_target_reject_bucket": float(
            target["fabricated_premise_should_reject"]
            - base["fabricated_premise_should_reject"]
        ),
        "delta_target_clarify_bucket": float(
            target["ambiguous_should_clarify"] - base["ambiguous_should_clarify"]
        ),
        "target_output_changed_rate": float(delta["target_output_changed_rate"]),
    }


def make_priority_score(view: dict[str, float]) -> float:
    return float(
        view.get("delta_reject_bucket", 0.0)
        + view.get("delta_good_non_direct_rate", 0.0)
        + 0.5 * view.get("delta_clarify_bucket", 0.0)
        + 0.5 * view.get("delta_insufficient_bucket", 0.0)
        + 0.25 * (-view.get("delta_contradicted_rate", 0.0))
        - max(0.0, -view.get("delta_supported_direct_bucket", 0.0))
        - max(0.0, -view.get("delta_supported_answer_rate", 0.0))
        - max(0.0, view.get("delta_bad_abstention_rate", 0.0))
    )


def choose_recommendation(runtime_rows: list[dict[str, Any]], patch_row: dict[str, Any] | None) -> list[str]:
    lines = []
    ranked = sorted(runtime_rows, key=lambda row: row["priority_score"], reverse=True)
    best = ranked[0] if ranked else None
    reject_rows = [row for row in runtime_rows if "reject" in row["name"]]
    reject_best = max(reject_rows, key=lambda row: row["priority_score"]) if reject_rows else None
    non_direct_rows = [row for row in runtime_rows if "non_direct" in row["name"]]
    non_direct_best = max(non_direct_rows, key=lambda row: row["priority_score"]) if non_direct_rows else None

    if reject_best and reject_best["view"]["delta_reject_bucket"] > 0.0:
        lines.append(
            "Primary next direction: continue the reject-focused extraction branch because it is the only one "
            "that materially moves the fabricated-premise bucket."
        )
    elif non_direct_best and non_direct_best["view"]["delta_good_non_direct_rate"] > 0.0:
        lines.append(
            "Primary next direction: keep the pairmean non-direct runtime route as the anchor, but treat it as "
            "a caution/clarify control rather than a full hallucination fix."
        )
    else:
        lines.append(
            "Primary next direction: pause wider sweeps and improve HERETIC-simple data construction, because the "
            "focused runtime branches are still weak on the held-out dev slice."
        )

    if reject_best and reject_best["view"]["delta_reject_bucket"] <= 0.0:
        lines.append(
            "Immediate gap: fabricated-premise rejection is still not moving, so the next branch should tighten "
            "reject-specific contexts before broader intervention search."
        )

    if patch_row is not None:
        if patch_row["view"]["delta_target_good_non_direct_rate"] > 0.0:
            lines.append(
                "Patch transfer is directionally positive, so a small alpha/layer sweep around the same runtime "
                "setting is justified."
            )
        else:
            lines.append(
                "Patch transfer is still null or harmful, so weight editing should remain secondary to runtime "
                "and data-design follow-up."
            )

    if best is not None:
        lines.append(
            f"Current top runtime config by the script's utility-risk score: `{best['name']}` "
            f"(score={best['priority_score']:.4f})."
        )
    return lines


def default_runtime_configs(run_dir: Path, args) -> list[RuntimeConfig]:
    answer1 = run_dir / "heretic_simple_directions_answer1_fast.npz"
    answer1to5 = run_dir / "heretic_simple_directions_answer1to5_fast.npz"
    return [
        RuntimeConfig(
            name="answer1to5_pairmean_non_direct",
            state_name="answer1to5",
            directions_npz=answer1to5,
            direction_key="pairmean_non_direct_minus_direct__normalized",
            layers=args.runtime_layers,
            beta=args.runtime_beta,
            answer_prediction_steps=args.answer_prediction_steps,
        ),
        RuntimeConfig(
            name="answer1to5_pairmean_reject",
            state_name="answer1to5",
            directions_npz=answer1to5,
            direction_key="pairmean_reject_minus_direct__normalized",
            layers=args.runtime_layers,
            beta=args.runtime_beta,
            answer_prediction_steps=args.answer_prediction_steps,
        ),
        RuntimeConfig(
            name="answer1to5_pairmean_clarify",
            state_name="answer1to5",
            directions_npz=answer1to5,
            direction_key="pairmean_clarify_minus_direct__normalized",
            layers=args.runtime_layers,
            beta=args.runtime_beta,
            answer_prediction_steps=args.answer_prediction_steps,
        ),
        RuntimeConfig(
            name="answer1to5_shuffled_control",
            state_name="answer1to5",
            directions_npz=answer1to5,
            direction_key="shuffled_non_direct_minus_direct__normalized",
            layers=args.runtime_layers,
            beta=args.runtime_beta,
            answer_prediction_steps=args.answer_prediction_steps,
        ),
        RuntimeConfig(
            name="answer1to5_pca_control",
            state_name="answer1to5",
            directions_npz=answer1to5,
            direction_key="pca_1__normalized",
            layers=args.runtime_layers,
            beta=args.runtime_beta,
            answer_prediction_steps=args.answer_prediction_steps,
        ),
        RuntimeConfig(
            name="answer1_pairmean_reject",
            state_name="answer1",
            directions_npz=answer1,
            direction_key="pairmean_reject_minus_direct__normalized",
            layers=args.runtime_layers,
            beta=args.runtime_beta,
            answer_prediction_steps=args.answer_prediction_steps,
        ),
    ]


def build_markdown_report(
    args,
    run_dir: Path,
    runtime_rows: list[dict[str, Any]],
    patch_row: dict[str, Any] | None,
    recommendations: list[str],
) -> str:
    lines = [
        "# HERETIC-simple focused Colab follow-up",
        "",
        "## Why this follow-up exists",
        "",
        "- Current HERETIC-simple signal is small and mostly improves abstain/clarify behavior rather than fabricated-premise rejection.",
        "- The highest-value next branch is therefore a reject-focused extraction check, not a broad new sweep.",
        "- Weight patching should only be re-tested around the current best runtime setting, and only after the runtime branch is re-confirmed.",
        "",
        "## Run setup",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Model: `{args.model}`",
        f"- 4-bit runtime: `{args.load_in_4bit}`",
        f"- Calibration row cap: `{args.capture_max_samples}`",
        f"- Dev row cap: `{args.dev_max_samples}`",
        f"- Runtime layers: `{args.runtime_layers}`",
        f"- Runtime beta: `{args.runtime_beta}`",
        f"- Answer prediction steps: `{args.answer_prediction_steps}`",
        "",
        "## Runtime ranking",
        "",
        "| config | score | d_contradicted | d_good_non_direct | d_reject_bucket | d_clarify_bucket | d_supported_direct_bucket | changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(runtime_rows, key=lambda item: item["priority_score"], reverse=True):
        view = row["view"]
        lines.append(
            "| "
            + row["name"]
            + f" | {row['priority_score']:.4f}"
            + f" | {view['delta_contradicted_rate']:+.4f}"
            + f" | {view['delta_good_non_direct_rate']:+.4f}"
            + f" | {view['delta_reject_bucket']:+.4f}"
            + f" | {view['delta_clarify_bucket']:+.4f}"
            + f" | {view['delta_supported_direct_bucket']:+.4f}"
            + f" | {view['output_changed_rate']:.4f} |"
        )

    lines.extend(["", "## Recommendation", ""])
    for item in recommendations:
        lines.append(f"- {item}")

    if patch_row is not None:
        view = patch_row["view"]
        lines.extend(
            [
                "",
                "## Patch check",
                "",
                f"- Config: `{patch_row['name']}`",
                f"- d_target_contradicted_rate: {view['delta_target_contradicted_rate']:+.4f}",
                f"- d_target_good_non_direct_rate: {view['delta_target_good_non_direct_rate']:+.4f}",
                f"- d_target_reject_bucket: {view['delta_target_reject_bucket']:+.4f}",
                f"- d_target_supported_direct_bucket: {view['delta_target_supported_direct_bucket']:+.4f}",
                f"- target_output_changed_rate: {view['target_output_changed_rate']:.4f}",
            ]
        )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"colab_heretic_simple_followup_{timestamp}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    run_dir = REPO_ROOT / "experiments" / "artifacts" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "experiments" / "data" / "heretic_simple"
    calibration_jsonl = data_dir / "calibration.jsonl"
    dev_jsonl = data_dir / "dev.jsonl"

    prepare_report = run_dir / "prepare_report.json"
    run_step(
        "prepare_heretic_simple_truthfulqa",
        [
            sys.executable,
            str(SCRIPT_DIR / "prepare_heretic_simple_truthfulqa.py"),
            "--seed",
            str(args.seed),
            "--calibration-questions",
            str(args.calibration_questions),
            "--dev-questions",
            str(args.dev_questions),
            "--report-json",
            str(prepare_report),
        ],
        cwd=REPO_ROOT,
        output_sentinel=prepare_report,
        force=args.force,
    )

    capture_jsonl = run_dir / "heretic_simple_calibration_capture.jsonl"
    capture_npz = run_dir / "heretic_simple_calibration_capture.npz"
    capture_meta = run_dir / "heretic_simple_calibration_capture_meta.json"
    capture_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_heretic_simple_generation_capture.py"),
        "--data-jsonl",
        str(calibration_jsonl),
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--gpu-memory-gb",
        str(args.gpu_memory_gb),
        "--capture-answer-tokens",
        str(args.capture_answer_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--output-jsonl",
        str(capture_jsonl),
        "--output-npz",
        str(capture_npz),
        "--metadata-json",
        str(capture_meta),
    ]
    if args.capture_max_samples > 0:
        capture_cmd.extend(["--max-samples", str(args.capture_max_samples)])
    if args.load_in_4bit:
        capture_cmd.append("--load-in-4bit")
    run_step(
        "run_heretic_simple_generation_capture",
        capture_cmd,
        cwd=REPO_ROOT,
        output_sentinel=capture_meta,
        force=args.force,
    )

    answer1_npz = run_dir / "heretic_simple_directions_answer1_fast.npz"
    answer1_meta = run_dir / "heretic_simple_directions_answer1_fast_meta.json"
    run_step(
        "extract_answer1_directions",
        [
            sys.executable,
            str(SCRIPT_DIR / "extract_heretic_simple_directions.py"),
            "--capture-jsonl",
            str(capture_jsonl),
            "--capture-npz",
            str(capture_npz),
            "--state-key",
            "answer_token_1",
            "--bootstrap-rounds",
            str(args.bootstrap_rounds),
            "--seed",
            str(args.seed),
            "--output-npz",
            str(answer1_npz),
            "--metadata-json",
            str(answer1_meta),
        ],
        cwd=REPO_ROOT,
        output_sentinel=answer1_meta,
        force=args.force,
    )

    answer1to5_npz = run_dir / "heretic_simple_directions_answer1to5_fast.npz"
    answer1to5_meta = run_dir / "heretic_simple_directions_answer1to5_fast_meta.json"
    run_step(
        "extract_answer1to5_directions",
        [
            sys.executable,
            str(SCRIPT_DIR / "extract_heretic_simple_directions.py"),
            "--capture-jsonl",
            str(capture_jsonl),
            "--capture-npz",
            str(capture_npz),
            "--state-key",
            "answer_token_1_to_5_mean",
            "--bootstrap-rounds",
            str(args.bootstrap_rounds),
            "--seed",
            str(args.seed),
            "--output-npz",
            str(answer1to5_npz),
            "--metadata-json",
            str(answer1to5_meta),
        ],
        cwd=REPO_ROOT,
        output_sentinel=answer1to5_meta,
        force=args.force,
    )

    runtime_rows = []
    for config in default_runtime_configs(run_dir, args):
        output_json = run_dir / f"{config.name}.runtime.json"
        runtime_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "heretic_simple_runtime_eval.py"),
            "--data-jsonl",
            str(dev_jsonl),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--gpu-memory-gb",
            str(args.gpu_memory_gb),
            "--directions-npz",
            str(config.directions_npz),
            "--direction-key",
            config.direction_key,
            "--layers",
            config.layers,
            "--beta",
            str(config.beta),
            "--answer-prediction-steps",
            str(config.answer_prediction_steps),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output-json",
            str(output_json),
        ]
        if args.dev_max_samples > 0:
            runtime_cmd.extend(["--max-samples", str(args.dev_max_samples)])
        if args.load_in_4bit:
            runtime_cmd.append("--load-in-4bit")
        run_step(
            f"runtime_eval::{config.name}",
            runtime_cmd,
            cwd=REPO_ROOT,
            output_sentinel=output_json,
            force=args.force,
        )
        result = read_json(output_json)
        view = runtime_delta_view(result)
        runtime_rows.append(
            {
                "name": config.name,
                "state_name": config.state_name,
                "direction_key": config.direction_key,
                "output_json": str(output_json),
                "priority_score": make_priority_score(view),
                "view": view,
            }
        )

    patch_row = None
    if not args.skip_patch:
        patch_npz = answer1to5_npz if args.patch_state == "answer1to5" else answer1_npz
        patch_output_json = run_dir / f"{args.patch_state}_{args.patch_direction_key}.patch.json"
        patch_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "heretic_simple_weight_patch_eval.py"),
            "--data-jsonl",
            str(dev_jsonl),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--gpu-memory-gb",
            str(args.gpu_memory_gb),
            "--directions-npz",
            str(patch_npz),
            "--direction-key",
            args.patch_direction_key,
            "--control-direction-key",
            "shuffled_non_direct_minus_direct__normalized",
            "--layers",
            args.runtime_layers,
            "--alpha",
            str(args.patch_alpha),
            "--modules",
            args.patch_modules,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output-json",
            str(patch_output_json),
        ]
        if args.dev_max_samples > 0:
            patch_cmd.extend(["--max-samples", str(args.dev_max_samples)])
        run_step(
            "weight_patch_followup",
            patch_cmd,
            cwd=REPO_ROOT,
            output_sentinel=patch_output_json,
            force=args.force,
        )
        patch_result = read_json(patch_output_json)
        patch_row = {
            "name": f"{args.patch_state}_{args.patch_direction_key}_{args.patch_modules}",
            "output_json": str(patch_output_json),
            "view": patch_delta_view(patch_result),
        }

    recommendations = choose_recommendation(runtime_rows, patch_row)
    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "runtime_rows": runtime_rows,
        "patch_row": patch_row,
        "recommendations": recommendations,
    }
    summary_json = run_dir / "followup_summary.json"
    save_json(summary_json, summary)

    markdown = build_markdown_report(args, run_dir, runtime_rows, patch_row, recommendations)
    summary_md = run_dir / "followup_summary.md"
    summary_md.write_text(markdown, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"[saved] {summary_json}")
    print(f"[saved] {summary_md}")


if __name__ == "__main__":
    main()
