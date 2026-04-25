import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "base": "#7f7f7f",
    "hda": "#4C78A8",
    "hda_light": "#9ecae9",
    "verifier": "#54A24B",
    "dola": "#E45756",
    "combo": "#B279A2",
    "control": "#F58518",
    "subset": "#8C564B",
    "attn": "#72B7B2",
}

FAMILY_COLORS = {
    "Qwen3": "#4C78A8",
    "Llama-3.2": "#F58518",
}

STATUS_COLORS = {
    "unchanged": "#b0b0b0",
    "corrected": "#54A24B",
    "worsened": "#E45756",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_preferred_json(paths):
    for path in paths:
        if path.exists():
            return load_json(path)
    raise FileNotFoundError(f"No candidate JSON found among: {paths}")


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: float) -> float:
    return 100.0 * float(value)


def fmt_alpha(value: float) -> str:
    return f"{value:.1f}"


def setup_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "font.size": 10,
            "font.family": "DejaVu Serif",
        }
    )


def contiguous_label(layers):
    layers = sorted(int(x) for x in layers)
    if not layers:
        return "NA"
    if len(layers) >= 30 and layers[0] == 0:
        return "all"
    if layers == list(range(layers[0], layers[-1] + 1)):
        if len(layers) == 1:
            return str(layers[0])
        return f"{layers[0]}-{layers[-1]}"
    return ",".join(str(x) for x in layers)


def load_patch_entry(path: Path, display_label: str = None):
    data = load_json(path)
    return {
        "file": path.name,
        "display_label": display_label or path.name,
        "layers": data["layers"],
        "layer_label": contiguous_label(data["layers"]),
        "modules": data["modules"],
        "alpha": float(data["alpha"]),
        "delta_points": pct(data["delta_acc"]),
        "n_eval": int(data["n_eval"]),
        "subset_only": int(data["n_eval"]) < 590,
        "path": str(path),
    }


def build_summary(root: Path):
    artifacts = root / "experiments" / "artifacts"

    qwen4b_reg_rows = load_csv(artifacts / "regression" / "compare" / "summary_table.csv")
    qwen4b_reg = {row["task"]: row for row in qwen4b_reg_rows}

    qwen4b_hda10 = load_json(artifacts / "analysis" / "hda_patch_l18to19_a2p8_10seed_summary.json")
    qwen4b_verifier = load_json(artifacts / "analysis" / "verifier_fixedfactual_extended_summary.json")

    patch_files = {
        "tuned": artifacts / "patch_taskalign_l18to19_a2p8_seed41_full_diag.json",
        "alpha1_same": artifacts / "patch_taskalign_l18to19_a1p0_seed41_full_diag.json",
        "core3": artifacts / "patch_taskalign_core3_a2p4_seed41_full_diag.json",
        "fullband": artifacts / "patch_taskalign_l18to24_mlp_a1p6_seed41_full_diag.json",
        "all_mlp": artifacts / "patch_taskalign_alllayers_mlp_a1p0_seed41_full_diag.json",
        "all_both_sub": artifacts / "patch_taskalign_alllayers_both_a1p0_sub.json",
    }
    patch_entries = {name: load_patch_entry(path) for name, path in patch_files.items()}

    qwen4b_gen = {
        "Base": load_preferred_json(
            [
                artifacts / "truthfulqa_qwen4b_generate_base_seed41_refresh.json",
                artifacts / "truthfulqa_qwen4b_generate_base_seed41.json",
            ]
        ),
        "DoLa": load_preferred_json(
            [
                artifacts / "truthfulqa_qwen4b_generate_dola_high_seed41_refresh.json",
                artifacts / "truthfulqa_qwen4b_generate_dola_high_seed41.json",
            ]
        ),
        "HDA": load_preferred_json(
            [
                artifacts / "truthfulqa_qwen4b_generate_hda_seed41_refresh.json",
                artifacts / "truthfulqa_qwen4b_generate_hda_seed41.json",
            ]
        ),
        "HDA + DoLa": load_preferred_json(
            [
                artifacts / "truthfulqa_qwen4b_generate_hda_dola_high_seed41_refresh.json",
                artifacts / "truthfulqa_qwen4b_generate_hda_dola_high_seed41.json",
            ]
        ),
    }

    open_bridge = {
        "Base": load_json(artifacts / "open_bridge_qwen4b_base_seed41.json"),
        "HDA": load_json(artifacts / "open_bridge_qwen4b_hda_seed41.json"),
    }

    qwen1p7_patch = load_json(artifacts / "patch_multiseed_qwen3_1p7b_nothink_l2426_a1p6" / "aggregate_summary.json")
    qwen0p6_patch = load_json(artifacts / "patch_multiseed_qwen3_0p6b_nothink_l2123_a2p4_both" / "aggregate_summary.json")
    llama1b_patch = load_json(artifacts / "patch_multiseed_llama32_1b_l9to11_a1p6" / "aggregate_summary.json")

    qwen1p7_reg = load_json(artifacts / "analysis" / "patch_regression_suite_qwen3_1p7b_nothink_l2426_a1p6_seed41.json")
    qwen0p6_reg = load_json(artifacts / "analysis" / "patch_regression_suite_qwen3_0p6b_nothink_l2123_a2p4_both_seed41.json")
    llama1b_reg_rows = load_csv(artifacts / "regression_llama32_1b_l9to11_a1p6_seed41" / "compare" / "summary_table.csv")
    llama1b_reg = {row["task"]: row for row in llama1b_reg_rows}

    thinking_diag = load_json(artifacts / "qwen3_thinking_mode_binary_diag.json")

    ablation_paths = [
        artifacts / "patch_taskalign_l18to19_a1p0_seed41_full_diag.json",
        artifacts / "patch_taskalign_l18to19_a2p8_seed41_full_diag.json",
        artifacts / "patch_taskalign_core3_a2p4_seed41_full_diag.json",
        artifacts / "patch_taskalign_l18to24_mlp_a0p8_full_diag.json",
        artifacts / "patch_taskalign_l18to24_mlp_a1p2_full_diag.json",
        artifacts / "patch_taskalign_l18to24_mlp_a1p6_seed41_full_diag.json",
        artifacts / "patch_taskalign_l20to24_mlp_a0p8_full_diag.json",
        artifacts / "patch_taskalign_l20to24_attn_a0p8_full_diag.json",
        artifacts / "patch_taskalign_l18to24_both_a0p8_full_diag.json",
        artifacts / "patch_taskalign_l20to24_both_a0p8_full_diag_repeat.json",
        artifacts / "patch_taskalign_alllayers_mlp_a1p0_seed41_full_diag.json",
        artifacts / "patch_taskalign_alllayers_both_a1p0_sub.json",
    ]
    ablation_entries = [load_patch_entry(path) for path in ablation_paths if path.exists()]

    tuned_diag = load_json(patch_files["tuned"])
    base_rows = tuned_diag["rows"]["base"]
    patched_rows = tuned_diag["rows"]["patched"]
    margin_rows = []
    corrected = 0
    worsened = 0
    unchanged = 0
    for base_row, patch_row in zip(base_rows, patched_rows):
        base_correct = base_row["pred"] == base_row["correct"]
        patch_correct = patch_row["pred"] == patch_row["correct"]
        if (not base_correct) and patch_correct:
            status = "corrected"
            corrected += 1
        elif base_correct and (not patch_correct):
            status = "worsened"
            worsened += 1
        else:
            status = "unchanged"
            unchanged += 1
        margin_rows.append(
            {
                "category": base_row["category"],
                "question": base_row["question"],
                "base_margin": float(base_row["margin_correct"]),
                "patched_margin": float(patch_row["margin_correct"]),
                "status": status,
            }
        )

    cross_models = [
        {
            "model": "Qwen3-4B",
            "family": "Qwen3",
            "truth_delta_points": pct(qwen4b_hda10["summary"]["mean_delta_acc"]),
            "truth_seeds": qwen4b_hda10["summary"]["n_seeds"],
            "positive_seeds": qwen4b_hda10["summary"]["positive_seed_count"],
            "mmlu_delta_points": pct(qwen4b_reg["mmlu_slice"]["delta"]),
            "hellaswag_delta_points": pct(qwen4b_reg["hellaswag"]["delta"]),
            "gsm8k_delta_points": pct(qwen4b_reg["gsm8k"]["delta"]),
            "worst_non_target_points": min(
                pct(qwen4b_reg["mmlu_slice"]["delta"]),
                pct(qwen4b_reg["hellaswag"]["delta"]),
                pct(qwen4b_reg["gsm8k"]["delta"]),
            ),
            "drift_similarity": float(qwen4b_reg["benign_drift_mean_similarity"]["patched_score"]),
            "material_drift_rate": float(qwen4b_reg["benign_drift_material_rate"]["patched_score"]),
            "notes": "Best balance; no non-target regression on the paired regression pack.",
        },
        {
            "model": "Qwen3-1.7B",
            "family": "Qwen3",
            "truth_delta_points": pct(qwen1p7_patch["summary"]["mean_delta_acc"]),
            "truth_seeds": qwen1p7_patch["summary"]["n_seeds"],
            "positive_seeds": qwen1p7_patch["summary"]["positive_seed_count"],
            "mmlu_delta_points": pct(next(row for row in qwen1p7_reg["task_table"] if row["name"] == "mmlu_zero_shot_letter")["delta"]),
            "hellaswag_delta_points": pct(next(row for row in qwen1p7_reg["task_table"] if row["name"] == "hellaswag_zero_shot_letter")["delta"]),
            "gsm8k_delta_points": pct(next(row for row in qwen1p7_reg["task_table"] if row["name"] == "gsm8k_final_number")["delta"]),
            "worst_non_target_points": min(
                pct(next(row for row in qwen1p7_reg["task_table"] if row["name"] == "mmlu_zero_shot_letter")["delta"]),
                pct(next(row for row in qwen1p7_reg["task_table"] if row["name"] == "hellaswag_zero_shot_letter")["delta"]),
                pct(next(row for row in qwen1p7_reg["task_table"] if row["name"] == "gsm8k_final_number")["delta"]),
            ),
            "drift_similarity": float(next(row for row in qwen1p7_reg["task_table"] if row["name"] == "benign_drift")["patched"]),
            "material_drift_rate": None,
            "notes": "Requires hard non-thinking mode; modest but reproducible gain.",
        },
        {
            "model": "Qwen3-0.6B",
            "family": "Qwen3",
            "truth_delta_points": pct(qwen0p6_patch["summary"]["mean_delta_acc"]),
            "truth_seeds": qwen0p6_patch["summary"]["n_seeds"],
            "positive_seeds": qwen0p6_patch["summary"]["positive_seed_count"],
            "mmlu_delta_points": pct(next(row for row in qwen0p6_reg["task_table"] if row["name"] == "mmlu_zero_shot_letter")["delta"]),
            "hellaswag_delta_points": pct(next(row for row in qwen0p6_reg["task_table"] if row["name"] == "hellaswag_zero_shot_letter")["delta"]),
            "gsm8k_delta_points": pct(next(row for row in qwen0p6_reg["task_table"] if row["name"] == "gsm8k_final_number")["delta"]),
            "worst_non_target_points": min(
                pct(next(row for row in qwen0p6_reg["task_table"] if row["name"] == "mmlu_zero_shot_letter")["delta"]),
                pct(next(row for row in qwen0p6_reg["task_table"] if row["name"] == "hellaswag_zero_shot_letter")["delta"]),
                pct(next(row for row in qwen0p6_reg["task_table"] if row["name"] == "gsm8k_final_number")["delta"]),
            ),
            "drift_similarity": float(next(row for row in qwen0p6_reg["task_table"] if row["name"] == "benign_drift")["patched"]),
            "material_drift_rate": None,
            "notes": "Truthfulness improves, but collateral regression appears on GSM8K.",
        },
        {
            "model": "Llama-3.2-1B",
            "family": "Llama-3.2",
            "truth_delta_points": pct(llama1b_patch["summary"]["mean_delta_acc"]),
            "truth_seeds": llama1b_patch["summary"]["n_seeds"],
            "positive_seeds": llama1b_patch["summary"]["positive_seed_count"],
            "mmlu_delta_points": pct(llama1b_reg["mmlu_slice"]["delta"]),
            "hellaswag_delta_points": pct(llama1b_reg["hellaswag"]["delta"]),
            "gsm8k_delta_points": pct(llama1b_reg["gsm8k"]["delta"]),
            "worst_non_target_points": min(
                pct(llama1b_reg["mmlu_slice"]["delta"]),
                pct(llama1b_reg["hellaswag"]["delta"]),
                pct(llama1b_reg["gsm8k"]["delta"]),
            ),
            "drift_similarity": float(llama1b_reg["benign_drift_mean_similarity"]["patched_score"]),
            "material_drift_rate": float(llama1b_reg["benign_drift_material_rate"]["patched_score"]),
            "notes": "Cross-family transfer works, but GSM8K drops on the regression pack.",
        },
    ]

    summary = {
        "qwen4b_main": {
            "base_accuracy": float(qwen4b_reg["truthfulqa_binary"]["base_score"]),
            "patched_accuracy": float(qwen4b_reg["truthfulqa_binary"]["patched_score"]),
            "hda_patch_delta_points": pct(qwen4b_hda10["summary"]["mean_delta_acc"]),
            "hda_patch_seeds": int(qwen4b_hda10["summary"]["n_seeds"]),
            "hda_patch_positive_seeds": int(qwen4b_hda10["summary"]["positive_seed_count"]),
            "hda_patch_seed_points": [pct(row["delta_acc"]) for row in qwen4b_hda10["per_seed"]],
            "hda_patch_seed_ids": [int(row["seed"]) for row in qwen4b_hda10["per_seed"]],
            "verifier_delta_points": pct(qwen4b_verifier["summary"]["mean_delta_acc"]),
            "verifier_seeds": int(qwen4b_verifier["summary"]["n_seeds"]),
            "verifier_positive_seeds": int(qwen4b_verifier["summary"]["positive_seed_count"]),
            "verifier_seed_points": [pct(row["delta_acc"]) for row in qwen4b_verifier["per_seed"]],
            "verifier_seed_ids": [int(row["seed"]) for row in qwen4b_verifier["per_seed"]],
            "paired_bootstrap_ci_low_points": pct(qwen4b_reg["truthfulqa_binary"]["paired_bootstrap_ci_low"]),
            "paired_bootstrap_ci_high_points": pct(qwen4b_reg["truthfulqa_binary"]["paired_bootstrap_ci_high"]),
            "drift_similarity": float(qwen4b_reg["benign_drift_mean_similarity"]["patched_score"]),
            "material_drift_rate": float(qwen4b_reg["benign_drift_material_rate"]["patched_score"]),
            "controls": [
                {
                    "label": "HDA tuned\n18-19 / mlp / a=2.8",
                    "delta_points": patch_entries["tuned"]["delta_points"],
                    "n_eval": patch_entries["tuned"]["n_eval"],
                    "color": COLORS["hda"],
                },
                {
                    "label": "18-20 / mlp / a=2.4",
                    "delta_points": patch_entries["core3"]["delta_points"],
                    "n_eval": patch_entries["core3"]["n_eval"],
                    "color": COLORS["hda_light"],
                },
                {
                    "label": "18-24 / mlp / a=1.6",
                    "delta_points": patch_entries["fullband"]["delta_points"],
                    "n_eval": patch_entries["fullband"]["n_eval"],
                    "color": COLORS["control"],
                },
                {
                    "label": "18-19 / mlp / a=1.0",
                    "delta_points": patch_entries["alpha1_same"]["delta_points"],
                    "n_eval": patch_entries["alpha1_same"]["n_eval"],
                    "color": COLORS["control"],
                },
                {
                    "label": "all / mlp / a=1.0",
                    "delta_points": patch_entries["all_mlp"]["delta_points"],
                    "n_eval": patch_entries["all_mlp"]["n_eval"],
                    "color": COLORS["control"],
                },
                {
                    "label": "all / both / a=1.0",
                    "delta_points": patch_entries["all_both_sub"]["delta_points"],
                    "n_eval": patch_entries["all_both_sub"]["n_eval"],
                    "color": COLORS["subset"],
                },
            ],
        },
        "qwen4b_dola_proxy": {
            name: {
                "accuracy": pct(payload["accuracy"]),
                "delta_points_vs_base": pct(payload["accuracy"] - qwen4b_gen["Base"]["accuracy"]),
            }
            for name, payload in qwen4b_gen.items()
        },
        "qwen4b_open_bridge": {
            name: payload["bucket_summary"]
            for name, payload in open_bridge.items()
        },
        "qwen4b_ablation": ablation_entries,
        "qwen4b_margin_shift": {
            "rows": margin_rows,
            "corrected": corrected,
            "worsened": worsened,
            "unchanged": unchanged,
            "fixed_count": int(tuned_diag["diagnostics"]["fixed_count"]),
            "broken_count": int(tuned_diag["diagnostics"]["broken_count"]),
        },
        "cross_model_patch": cross_models,
        "qwen_reasoning_diagnosis": thinking_diag,
    }
    return summary


def write_tables(summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    route_rows = [
        {
            "method": "Base",
            "protocol": "TruthfulQA binary logprob",
            "score_or_delta_points": summary["qwen4b_main"]["base_accuracy"] * 100.0,
            "notes": "Absolute base accuracy on the canonical split",
        },
        {
            "method": "HDA patch (best tuned)",
            "protocol": "TruthfulQA binary logprob",
            "score_or_delta_points": summary["qwen4b_main"]["hda_patch_delta_points"],
            "notes": f"{summary['qwen4b_main']['hda_patch_positive_seeds']}/{summary['qwen4b_main']['hda_patch_seeds']} positive split seeds",
        },
        {
            "method": "Verifier reranking",
            "protocol": "TruthfulQA binary logprob",
            "score_or_delta_points": summary["qwen4b_main"]["verifier_delta_points"],
            "notes": f"{summary['qwen4b_main']['verifier_positive_seeds']}/{summary['qwen4b_main']['verifier_seeds']} positive split seeds",
        },
        {
            "method": "HDA patch alpha=1",
            "protocol": "TruthfulQA binary logprob",
            "score_or_delta_points": next(row["delta_points"] for row in summary["qwen4b_main"]["controls"] if "a=1.0" in row["label"] and "18-19" in row["label"]),
            "notes": "Canonical-split control on the same band",
        },
        {
            "method": "HDA patch all layers / MLP / alpha=1",
            "protocol": "TruthfulQA binary logprob",
            "score_or_delta_points": next(row["delta_points"] for row in summary["qwen4b_main"]["controls"] if row["label"].startswith("all / mlp")),
            "notes": "Canonical-split control",
        },
    ]
    route_rows.extend(
        {
            "method": name,
            "protocol": "TruthfulQA binary generation proxy",
            "score_or_delta_points": row["accuracy"],
            "notes": f"delta vs proxy base = {row['delta_points_vs_base']:+.2f} points",
        }
        for name, row in summary["qwen4b_dola_proxy"].items()
    )

    with (output_dir / "final_qwen4b_route_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(route_rows[0].keys()))
        writer.writeheader()
        writer.writerows(route_rows)

    with (output_dir / "final_cross_model_patch_table.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "family",
            "truth_delta_points",
            "truth_seeds",
            "positive_seeds",
            "mmlu_delta_points",
            "hellaswag_delta_points",
            "gsm8k_delta_points",
            "worst_non_target_points",
            "drift_similarity",
            "material_drift_rate",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["cross_model_patch"])

    compact_summary = dict(summary)
    compact_summary["qwen4b_margin_shift"] = {
        key: value for key, value in summary["qwen4b_margin_shift"].items() if key != "rows"
    }
    with (output_dir / "final_project_summary.json").open("w", encoding="utf-8") as f:
        json.dump(compact_summary, f, indent=2, ensure_ascii=False)


def add_zero_line(ax):
    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--", zorder=0)


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))


def plot_qwen4b_routes(summary: dict, output_dir: Path):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    ax = axes[0]
    multi_seed = [
        (
            "HDA patch",
            summary["qwen4b_main"]["hda_patch_seed_points"],
            summary["qwen4b_main"]["hda_patch_seed_ids"],
            COLORS["hda"],
        ),
        (
            "Verifier reranking",
            summary["qwen4b_main"]["verifier_seed_points"],
            summary["qwen4b_main"]["verifier_seed_ids"],
            COLORS["verifier"],
        ),
    ]
    y_positions = np.arange(len(multi_seed))[::-1]
    add_zero_line(ax)
    for ypos, (label, values, seeds, color) in zip(y_positions, multi_seed):
        jitter = np.linspace(-0.14, 0.14, len(values))
        ax.scatter(values, ypos + jitter, s=42, color=color, edgecolor="black", linewidth=0.6, alpha=0.9, zorder=3)
        mean = float(np.mean(values))
        spread = ci95(values)
        ax.errorbar(mean, ypos, xerr=spread, fmt="D", color=color, mfc="white", mec="black", mew=0.8, ms=7.5, linewidth=1.2, capsize=4, zorder=4)
        ax.text(
            max(values) + 0.14,
            ypos,
            f"mean {mean:+.2f}\n{sum(v > 0 for v in values)}/{len(values)} positive",
            ha="left",
            va="center",
            fontsize=8.5,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in multi_seed])
    ax.set_xlabel("TruthfulQA delta over base (points)")
    ax.set_title("Seed-level stability")

    ax2 = axes[1]
    add_zero_line(ax2)
    controls = summary["qwen4b_main"]["controls"]
    control_y = np.arange(len(controls))[::-1]
    for ypos, row in zip(control_y, controls):
        marker = "D" if "tuned" in row["label"] else "s"
        ax2.scatter(row["delta_points"], ypos, s=62, marker=marker, color=row["color"], edgecolor="black", linewidth=0.8, zorder=3)
        note = f"{row['delta_points']:+.2f}"
        if row["n_eval"] < 590:
            note += f" ({row['n_eval']})"
        ax2.text(row["delta_points"] + 0.08, ypos, note, ha="left", va="center", fontsize=8.5)
    ax2.set_yticks(control_y)
    ax2.set_yticklabels([row["label"] for row in controls])
    ax2.set_xlabel("TruthfulQA delta on canonical split (points)")
    ax2.set_title("Selection and control structure")
    ax2.text(
        0.98,
        0.03,
        "Single-point controls use the canonical seed-41 split.\nThe last point is a 180-item subset control.",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )

    out_path = output_dir / "final_qwen4b_route_delta.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_hda_ablation_heatmap(summary: dict, output_dir: Path):
    entries = summary["qwen4b_ablation"]
    module_order = ["mlp", "both", "attn"]
    module_titles = {"mlp": "MLP write patch", "both": "Attention + MLP", "attn": "Attention only"}
    layer_order = ["18-19", "18-20", "18-24", "20-24", "all"]
    alpha_values = sorted({entry["alpha"] for entry in entries})
    vmax = max(abs(entry["delta_points"]) for entry in entries)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.8), constrained_layout=True)
    best = max(entries, key=lambda row: row["delta_points"])
    for ax, module in zip(axes, module_order):
        matrix = np.full((len(layer_order), len(alpha_values)), np.nan)
        subset_marks = {}
        for entry in entries:
            if entry["modules"] != module or entry["layer_label"] not in layer_order:
                continue
            ridx = layer_order.index(entry["layer_label"])
            cidx = alpha_values.index(entry["alpha"])
            matrix[ridx, cidx] = entry["delta_points"]
            subset_marks[(ridx, cidx)] = entry["subset_only"]
        im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(alpha_values)))
        ax.set_xticklabels([fmt_alpha(alpha) for alpha in alpha_values])
        ax.set_yticks(np.arange(len(layer_order)))
        ax.set_yticklabels(layer_order if ax is axes[0] else [])
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(module_titles[module], fontsize=10.5, pad=8)
        for ridx in range(len(layer_order)):
            for cidx in range(len(alpha_values)):
                if np.isnan(matrix[ridx, cidx]):
                    ax.text(cidx, ridx, "-", ha="center", va="center", color="#666666", fontsize=11)
                    continue
                value = matrix[ridx, cidx]
                text = f"{value:+.2f}" + ("*" if subset_marks.get((ridx, cidx), False) else "")
                ax.text(
                    cidx,
                    ridx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="black" if abs(value) < 0.8 * vmax else "white",
                    weight="bold",
                )
                if module == best["modules"] and layer_order[ridx] == best["layer_label"] and alpha_values[cidx] == best["alpha"]:
                    ax.add_patch(Rectangle((cidx - 0.5, ridx - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.0))
        ax.grid(False)
    axes[0].set_ylabel("Layer band")
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("TruthfulQA delta over base (points)")
    fig.text(0.5, 0.01, "Sparse ablation map on the canonical split; * denotes subset-only controls.", ha="center", fontsize=9)
    out_path = output_dir / "final_hda_ablation_heatmap.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_margin_shift(summary: dict, output_dir: Path):
    rows = summary["qwen4b_margin_shift"]["rows"]
    fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    x_vals = np.array([row["base_margin"] for row in rows], dtype=float)
    y_vals = np.array([row["patched_margin"] for row in rows], dtype=float)
    lim = float(np.quantile(np.abs(np.concatenate([x_vals, y_vals])), 0.995))
    lim = max(6.0, lim)
    for status in ["unchanged", "corrected", "worsened"]:
        subset = [row for row in rows if row["status"] == status]
        ax.scatter(
            [row["base_margin"] for row in subset],
            [row["patched_margin"] for row in subset],
            s=20 if status == "unchanged" else 34,
            color=STATUS_COLORS[status],
            edgecolor="none",
            alpha=0.55 if status == "unchanged" else 0.88,
            label=f"{status.title()} ({len(subset)})",
        )
    ax.plot([-lim, lim], [-lim, lim], color="#666666", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"Base margin: $\log P(\mathrm{true}) - \log P(\mathrm{false})$")
    ax.set_ylabel(r"Patched margin: $\log P(\mathrm{true}) - \log P(\mathrm{false})$")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.03,
        f"Corrected: {summary['qwen4b_margin_shift']['corrected']}  |  "
        f"Worsened: {summary['qwen4b_margin_shift']['worsened']}  |  "
        f"Unchanged: {summary['qwen4b_margin_shift']['unchanged']}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )
    out_path = output_dir / "final_qwen4b_margin_shift.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_dola_proxy(summary: dict, output_dir: Path):
    rows = summary["qwen4b_dola_proxy"]
    x = np.array([0, 1], dtype=float)
    dola_off = np.array([rows["Base"]["delta_points_vs_base"], rows["HDA"]["delta_points_vs_base"]], dtype=float)
    dola_on = np.array([rows["DoLa"]["delta_points_vs_base"], rows["HDA + DoLa"]["delta_points_vs_base"]], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    add_zero_line(ax)
    ax.plot(x, dola_off, marker="o", markersize=8, linewidth=2.0, color=COLORS["hda"], label="DoLa off")
    ax.plot(x, dola_on, marker="s", markersize=8, linewidth=2.0, color=COLORS["dola"], label="DoLa on")
    point_specs = [
        (0, dola_off[0], "Base", 0.00, 0.08, "center", "bottom"),
        (1, dola_off[1], "HDA", -0.06, 0.12, "right", "bottom"),
        (0, dola_on[0], "DoLa", 0.00, -0.10, "center", "top"),
        (1, dola_on[1], "HDA + DoLa", 0.08, -0.10, "left", "top"),
    ]
    for px, py, label, dx, dy, ha, va in point_specs:
        ax.text(px + dx, py + dy, f"{label}\n{py:+.2f}", ha=ha, va=va, fontsize=8.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Patch off", "Patch on"])
    ax.set_ylabel("Delta over proxy base (points)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.03,
        "Greedy A/B generation proxy on the canonical split.\nThe near-parallel lines show no additive HDA + DoLa interaction.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )
    out_path = output_dir / "final_qwen4b_dola_proxy.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_cross_model_tradeoff(summary: dict, output_dir: Path):
    rows = summary["cross_model_patch"]
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)

    x_min = min(row["truth_delta_points"] for row in rows) - 0.5
    x_max = max(row["truth_delta_points"] for row in rows) + 0.5
    y_min = min(row["worst_non_target_points"] for row in rows) - 0.8
    y_max = max(row["worst_non_target_points"] for row in rows) + 0.8

    ax.add_patch(Rectangle((0, 0), x_max, y_max, facecolor="#d9f0d3", alpha=0.18, zorder=0))
    ax.add_patch(Rectangle((0, y_min), x_max, abs(y_min), facecolor="#fee8c8", alpha=0.18, zorder=0))
    ax.add_patch(Rectangle((x_min, 0), abs(x_min), y_max, facecolor="#f0f0f0", alpha=0.16, zorder=0))
    ax.add_patch(Rectangle((x_min, y_min), abs(x_min), abs(y_min), facecolor="#fddede", alpha=0.16, zorder=0))

    for row in rows:
        drift_cost = row["material_drift_rate"]
        if drift_cost is None:
            drift_cost = max(0.0, 1.0 - row["drift_similarity"])
        size = 240 + 3200 * drift_cost
        ax.scatter(
            row["truth_delta_points"],
            row["worst_non_target_points"],
            s=size,
            color=FAMILY_COLORS[row["family"]],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.88,
        )
        ax.text(
            row["truth_delta_points"] + 0.06,
            row["worst_non_target_points"] + 0.08,
            f"{row['model']}\n{row['positive_seeds']}/{row['truth_seeds']} seeds",
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("TruthfulQA delta (points)")
    ax.set_ylabel("Worst non-target delta (points)")
    ax.text(0.98, 0.98, "Upper-right preferred", transform=ax.transAxes, ha="right", va="top", fontsize=10, weight="bold")
    ax.text(
        0.98,
        0.03,
        "Bubble size = drift cost (material drift rate when available,\notherwise 1 - benign similarity).",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )
    out_path = output_dir / "final_cross_model_tradeoff.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_reasoning_diag(summary: dict, output_dir: Path):
    rows = summary["qwen_reasoning_diagnosis"]
    models = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"]
    model_labels = {"Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-0.6B": "Qwen3-0.6B"}
    modes = ["default", "hard_no_think", "soft_no_think"]
    mode_labels = {"default": "Default", "hard_no_think": "Hard no-think", "soft_no_think": "Soft /no_think"}
    x = np.arange(len(modes))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)

    ax = axes[0]
    for model in models:
        values = [pct(next(item for item in rows if item["model"] == model and item["mode"] == mode)["accuracy"]) for mode in modes]
        color = FAMILY_COLORS["Qwen3"] if "1.7" in model else COLORS["control"]
        ax.plot(x, values, marker="o", linewidth=2.0, markersize=7, label=model_labels[model], color=color)
        for xpos, value in zip(x, values):
            ax.text(xpos, value + 1.2, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[mode] for mode in modes])
    ax.set_ylabel("100-item diagnostic accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Accuracy by reasoning mode")
    ax.legend(frameon=False, loc="upper left")

    ax2 = axes[1]
    width = 0.22
    positions = []
    tick_labels = []
    for midx, model in enumerate(models):
        for oidx, mode in enumerate(modes):
            xpos = midx * 0.9 + oidx * width
            row = next(item for item in rows if item["model"] == model and item["mode"] == mode)
            share_a = pct(row["share_a_pred"])
            share_b = 100.0 - share_a
            ax2.bar(xpos, share_a, width=width, color=COLORS["dola"], edgecolor="black", linewidth=0.6)
            ax2.bar(xpos, share_b, width=width, bottom=share_a, color=COLORS["hda"], edgecolor="black", linewidth=0.6)
            ax2.text(xpos, 102, mode_labels[mode].replace("Hard no-think", "Hard").replace("Default", "Def").replace("Soft /no_think", "Soft"), ha="center", va="bottom", fontsize=7.5, rotation=0)
            positions.append(xpos)
            tick_labels.append("")
    ax2.axhline(50.0, color="#444444", linewidth=1.0, linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Prediction share (%)")
    ax2.set_xticks([0.22, 1.12])
    ax2.set_xticklabels(["Qwen3-1.7B", "Qwen3-0.6B"])
    ax2.set_title('A/B prediction share: removing the "always A" collapse')
    ax2.text(0.02, 0.96, "red = predict A\nblue = predict B\nlabels = Def / Hard / Soft", transform=ax2.transAxes, ha="left", va="top", fontsize=8.5)

    out_path = output_dir / "final_qwen_reasoning_diagnosis.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def add_box(ax, x, y, w, h, text, facecolor="#ffffff", edgecolor="#555555", lw=1.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def add_arrow(ax, start, end, color="#555555", style="-|>", lw=1.2, linestyle="-"):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=lw, color=color, linestyle=linestyle)
    ax.add_patch(arrow)


def plot_protocol_schematic(output_dir: Path):
    fig, ax = plt.subplots(figsize=(10.6, 4.8), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 0.03, 0.38, 0.16, 0.18, "TruthfulQA source\nmultiple-choice rows", facecolor="#f7f7f7")
    add_box(ax, 0.25, 0.38, 0.18, 0.18, "Seed-indexed\n200 / 590 stratified split", facecolor="#f7f7f7")
    add_box(ax, 0.49, 0.63, 0.18, 0.16, "Calibration (200)\nDirection extraction\n+ config selection", facecolor="#deebf7", edgecolor=COLORS["hda"])
    add_box(ax, 0.49, 0.17, 0.18, 0.16, "Held-out eval (590)\nCandidate order fixed\nfor each split seed", facecolor="#f7f7f7")
    add_box(ax, 0.73, 0.64, 0.22, 0.14, "Frozen HDA patch\nbinary log-prob scorer", facecolor="#deebf7", edgecolor=COLORS["hda"])
    add_box(ax, 0.73, 0.42, 0.22, 0.14, "Verifier reranking\nsame split protocol", facecolor="#e5f5e0", edgecolor=COLORS["verifier"])
    add_box(ax, 0.73, 0.20, 0.22, 0.14, "DoLa proxy\n(greedy A/B generation)", facecolor="#fee0d2", edgecolor=COLORS["dola"])
    add_box(ax, 0.73, 0.02, 0.22, 0.12, "Canonical seed-41\npaired regression pack", facecolor="#f7f7f7")

    add_arrow(ax, (0.19, 0.47), (0.25, 0.47))
    add_arrow(ax, (0.43, 0.49), (0.49, 0.71))
    add_arrow(ax, (0.43, 0.45), (0.49, 0.25))
    add_arrow(ax, (0.67, 0.71), (0.73, 0.71), color=COLORS["hda"])
    add_arrow(ax, (0.67, 0.25), (0.73, 0.49), color=COLORS["verifier"])
    add_arrow(ax, (0.67, 0.25), (0.73, 0.27), color=COLORS["dola"], linestyle="--")
    add_arrow(ax, (0.67, 0.21), (0.73, 0.08))

    ax.text(0.34, 0.84, "Repeat over split seeds for robustness summaries", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(0.84, 0.88, "Main protocol", ha="center", va="center", fontsize=9, color=COLORS["hda"], weight="bold")
    ax.text(0.84, 0.58, "Alternative route", ha="center", va="center", fontsize=9, color=COLORS["verifier"], weight="bold")
    ax.text(0.84, 0.36, "Decode-time proxy", ha="center", va="center", fontsize=9, color=COLORS["dola"], weight="bold")

    out_path = output_dir / "final_protocol_schematic.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    root = repo_root()
    output_dir = root / "experiments" / "artifacts" / "route_plots"
    analysis_dir = root / "experiments" / "artifacts" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    summary = build_summary(root)
    write_tables(summary, analysis_dir)
    plot_qwen4b_routes(summary, output_dir)
    plot_hda_ablation_heatmap(summary, output_dir)
    plot_margin_shift(summary, output_dir)
    plot_dola_proxy(summary, output_dir)
    plot_cross_model_tradeoff(summary, output_dir)
    plot_reasoning_diag(summary, output_dir)
    plot_protocol_schematic(output_dir)
    print(f"Wrote final summary tables to: {analysis_dir}")
    print(f"Wrote final report plots to: {output_dir}")


if __name__ == "__main__":
    main()
