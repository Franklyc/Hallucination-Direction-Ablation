import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create route-specific summary plots for HDA experiments.")
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/route_plots",
        help="Directory to write plot PNGs",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct_points(value: float) -> float:
    return 100.0 * float(value)


def setup_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 10,
        }
    )


def binary_route_records(root: Path):
    original_probe = load_json(root / "experiments/artifacts/probe_taskalign_l18to24_b3p0_full_diag.json")
    patch_candidates = [
        ("Orthogonal patch (original)", load_json(root / "experiments/artifacts/patch_taskalign_l20to24_attn_a0p8_full_diag.json"), "attn"),
        ("Orthogonal patch (original)", load_json(root / "experiments/artifacts/patch_taskalign_l20to24_mlp_a0p8_full_diag.json"), "mlp"),
        ("Orthogonal patch (original)", load_json(root / "experiments/artifacts/patch_taskalign_l20to24_both_a0p8_full_diag_repeat.json"), "both"),
    ]
    best_patch_label, best_patch_json, best_patch_module = max(patch_candidates, key=lambda item: float(item[1]["delta_acc"]))

    upgraded_instruction = load_json(root / "experiments/artifacts/instruction_v3_l20to24_b3p0/aggregate_summary.json")
    upgraded_answer_state = load_json(root / "experiments/artifacts/multiseed_answerstate_l20to24_b3p0/aggregate_summary.json")
    verifier = load_json(root / "experiments/artifacts/verifier_fixedfactual_summary.json")

    records = [
        {
            "route": "Verifier reranking",
            "delta_points": pct_points(verifier["mean_delta_acc"]),
            "detail": f"4 seeds, best fixed prompt",
            "color": "#1b9e77",
        },
        {
            "route": "Activation probe (original)",
            "delta_points": pct_points(original_probe["delta_acc"]),
            "detail": "best utility, 1 seed",
            "color": "#4c78a8",
        },
        {
            "route": "Instruction extraction v3",
            "delta_points": pct_points(upgraded_instruction["summary"]["mean_delta_acc"]),
            "detail": "mean over 2 seeds",
            "color": "#72b7b2",
        },
        {
            "route": "Orthogonal patch (original)",
            "delta_points": pct_points(best_patch_json["delta_acc"]),
            "detail": f"best module = {best_patch_module}",
            "color": "#b279a2",
        },
        {
            "route": "Answer-state extraction",
            "delta_points": pct_points(upgraded_answer_state["summary"]["mean_delta_acc"]),
            "detail": "mean over 2 seeds",
            "color": "#e15759",
        },
    ]
    return records, verifier


def make_binary_route_overview(root: Path, output_dir: Path):
    records, _ = binary_route_records(root)
    fig, ax = plt.subplots(figsize=(11.4, 5.8), constrained_layout=True)

    routes = [row["route"] for row in records]
    deltas = [row["delta_points"] for row in records]
    colors = [row["color"] for row in records]
    y = np.arange(len(routes))

    ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.6, height=0.65)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(routes)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out delta accuracy (points)")
    ax.set_title("Best Result By Binary-Evaluated Route")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    x_min = min(-1.9, min(deltas) - 1.4)
    x_max = max(6.1, max(deltas) + 1.1)
    ax.set_xlim(x_min, x_max)

    for idx, row in enumerate(records):
        value = row["delta_points"]
        text = f"{value:+.2f} pts"
        detail = row["detail"]
        if value >= 0:
            ax.text(value + 0.12, idx, text, va="center", ha="left", fontsize=10, weight="bold")
            ax.text(value + 1.05, idx, detail, va="center", ha="left", fontsize=9, color="#444444")
        else:
            ax.text(-1.28, idx, text, va="center", ha="left", fontsize=10, weight="bold")
            ax.text(1.02, idx, detail, va="center", ha="left", fontsize=9, color="#444444")

    out_path = output_dir / "binary_route_best_results.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_verifier_multiseed(root: Path, output_dir: Path):
    verifier = load_json(root / "experiments/artifacts/verifier_fixedfactual_summary.json")
    per_seed = verifier["per_seed"]

    seeds = [str(row["seed"]) for row in per_seed]
    base = [pct_points(row["base_acc"]) for row in per_seed]
    rerank = [pct_points(row["verifier_acc"]) for row in per_seed]
    delta = [pct_points(row["delta_acc"]) for row in per_seed]
    pvals = [row["paired_sign_test_pvalue"] for row in per_seed]

    x = np.arange(len(seeds))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.bar(x - width / 2, base, width=width, color="#9aa5b1", edgecolor="black", linewidth=0.6, label="Base")
    ax.bar(x + width / 2, rerank, width=width, color="#2a9d8f", edgecolor="black", linewidth=0.6, label="Verifier rerank")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {seed}" for seed in seeds])
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Verifier Reranking Held-out Accuracy By Seed")
    ax.set_ylim(min(base) - 2.5, max(rerank) + 4.2)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    for idx, (base_val, rerank_val, delta_val, pval) in enumerate(zip(base, rerank, delta, pvals)):
        ax.text(idx + width / 2, rerank_val + 0.5, f"{delta_val:+.2f} pts", ha="center", va="bottom", fontsize=9, weight="bold")
        ax.text(idx + width / 2, rerank_val + 1.55, f"p={pval:.4g}", ha="center", va="bottom", fontsize=8, color="#444444")

    mean_delta = pct_points(verifier["mean_delta_acc"])
    ax.text(
        0.98,
        0.04,
        f"mean delta = {mean_delta:+.2f} pts\npositive seeds = {verifier['positive_seed_count']}/{verifier['n_seeds']}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )

    out_path = output_dir / "verifier_multiseed_detail.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_mechanistic_reviewed(root: Path, output_dir: Path):
    base_summary = load_json(root / "experiments/artifacts/mechanistic_annotation_pack_seed7_reviewed_summary.json")
    intervention = load_json(root / "experiments/artifacts/mechanistic_intervention_manual_review_seed7_subspace_r4_l31to35_b0p2_gen8_full80.json")

    labels = ["supported", "unsupported", "mixed"]
    colors = {"supported": "#2a9d8f", "unsupported": "#e76f51", "mixed": "#f4a261"}
    base_counts = [int(base_summary["label_counts"][label]) for label in labels]
    target_counts = [int(intervention["target_label_counts"][label]) for label in labels]
    total = int(base_summary["n_rows"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1.0]})

    bar_x = np.array([0, 1])
    bottoms = np.zeros(2)
    values_by_label = {
        "supported": [base_counts[0], target_counts[0]],
        "unsupported": [base_counts[1], target_counts[1]],
        "mixed": [base_counts[2], target_counts[2]],
    }
    for label in labels:
        vals = values_by_label[label]
        axes[0].bar(bar_x, vals, bottom=bottoms, color=colors[label], edgecolor="black", linewidth=0.5, width=0.62, label=label)
        for idx, val in enumerate(vals):
            if val > 0:
                axes[0].text(bar_x[idx], bottoms[idx] + val / 2, str(val), ha="center", va="center", fontsize=10, weight="bold")
        bottoms += np.array(vals)

    axes[0].set_xticks(bar_x)
    axes[0].set_xticklabels(["Base reviewed", "Intervened"])
    axes[0].set_ylabel("Count on reviewed seed set")
    axes[0].set_ylim(0, total)
    axes[0].set_title("Manual Reviewed Label Composition")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)

    delta_counts = [target - base for base, target in zip(base_counts, target_counts)]
    axes[1].bar(labels, delta_counts, color=[colors[label] for label in labels], edgecolor="black", linewidth=0.6, width=0.6)
    axes[1].axhline(0.0, color="#444444", linewidth=1.0)
    axes[1].set_ylabel("Count change")
    axes[1].set_title("Intervention Shift From Base")
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)
    y_abs = max(abs(min(delta_counts)), abs(max(delta_counts)), 1)
    axes[1].set_ylim(-y_abs - 0.6, y_abs + 0.8)
    for idx, val in enumerate(delta_counts):
        axes[1].text(idx, val + (0.08 if val >= 0 else -0.12), f"{val:+d}", ha="center", va="bottom" if val >= 0 else "top", fontsize=10, weight="bold")

    axes[1].text(
        0.98,
        0.06,
        f"n = {total}\nchanged outputs = {intervention['n_changed_outputs']}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )

    out_path = output_dir / "mechanistic_reviewed_intervention.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def heretic_metric_rows(root: Path):
    configs = [
        (
            "rank1_gen4_b0.5",
            "Rank-1 runtime\n(gen-only 4 tok, b=0.5)",
            load_json(root / "experiments/artifacts/heretic_to_open_annotation_seed_l30to34_b0p5_gen4.json"),
        ),
        (
            "rank1_fullseq_b0.2",
            "Rank-1 runtime\n(full sequence, b=0.2)",
            load_json(root / "experiments/artifacts/heretic_to_open_annotation_seed_l30to34_b0p2_fullseq.json"),
        ),
        (
            "rank4_gen4_b0.2",
            "Rank-4 subspace\n(gen-only 4 tok, b=0.2)",
            load_json(root / "experiments/artifacts/heretic_subspace_to_open_annotation_seed_r4_l30to34_b0p2_gen4.json"),
        ),
        (
            "supp_minus_insuff_fullseq",
            "Supported-minus-insufficient\n(full sequence, b=0.2)",
            load_json(root / "experiments/artifacts/heretic_supported_minus_insufficient_to_open_annotation_seed_l30to34_b0p2_fullseq.json"),
        ),
        (
            "weight_patch_attn",
            "Attention-only patch\n(alpha=0.2)",
            load_json(root / "experiments/artifacts/heretic_supported_minus_insufficient_weightpatch_open_annotation_seed_attn_l30to34_a0p2.json"),
        ),
    ]

    rows = []
    for config_id, label, payload in configs:
        delta = payload["delta"]
        target_values = [
            -pct_points(delta["target_hard_bad_rate"]),
            pct_points(delta["target_supported_rate"]),
            -pct_points(delta["target_unresolved_rate"]),
        ]
        random_values = [
            -pct_points(delta["random_hard_bad_rate"]),
            pct_points(delta["random_supported_rate"]),
            -pct_points(delta["random_unresolved_rate"]),
        ]
        rows.append(
            {
                "id": config_id,
                "label": label,
                "target_values": target_values,
                "random_values": random_values,
            }
        )
    return rows


def make_heretic_heatmap(root: Path, output_dir: Path):
    rows = heretic_metric_rows(root)
    metric_labels = ["Hallucination reduction", "Supported delta", "Unresolved reduction"]
    target_matrix = np.array([row["target_values"] for row in rows], dtype=float)
    random_matrix = np.array([row["random_values"] for row in rows], dtype=float)
    row_labels = [row["label"] for row in rows]

    vmax = max(np.abs(target_matrix).max(), np.abs(random_matrix).max(), 1.25)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2), constrained_layout=True)
    for ax, matrix, title in [
        (axes[0], target_matrix, "Target Direction / Patch"),
        (axes[1], random_matrix, "Random Control"),
    ]:
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=18, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    weight="bold",
                )

    colorbar = fig.colorbar(image, ax=axes, shrink=0.9)
    colorbar.set_label("Improvement (points, positive is better)")
    fig.suptitle("HERETIC-Style Open-Generation Transfer Tradeoff", fontsize=14, weight="bold")

    out_path = output_dir / "heretic_transfer_tradeoff_heatmap.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    setup_style()

    root = repo_root()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        make_binary_route_overview(root, output_dir),
        make_verifier_multiseed(root, output_dir),
        make_mechanistic_reviewed(root, output_dir),
        make_heretic_heatmap(root, output_dir),
    ]

    for path in generated:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
