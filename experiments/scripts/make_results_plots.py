import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_PATH = Path("experiments/artifacts/baseline_qwen4b_full_newline.json")
BEST_UTILITY_PATH = Path("experiments/artifacts/probe_taskalign_l18to24_b3p0_full_diag.json")
BEST_BALANCE_PATH = Path("experiments/artifacts/probe_taskalign_l20to24_b3p0_full_diag.json")
PATCH_ATTN_PATH = Path("experiments/artifacts/patch_taskalign_l20to24_attn_a0p8_full_diag.json")
PATCH_MLP_PATH = Path("experiments/artifacts/patch_taskalign_l20to24_mlp_a0p8_full_diag.json")
PATCH_BOTH_PATH = Path("experiments/artifacts/patch_taskalign_l20to24_both_a0p8_full_diag_repeat.json")

PROBE_FRONTIER_PATHS = [
    ("18-24, b2.5", Path("experiments/artifacts/probe_taskalign_l18to24_b2p5_full_diag.json"), Path("experiments/artifacts/drift_probe_taskalign_l18to24_b2p5_v2.json")),
    ("18-24, b2.75", Path("experiments/artifacts/probe_taskalign_l18to24_b2p75_full_diag.json"), Path("experiments/artifacts/drift_probe_taskalign_l18to24_b2p75_v2.json")),
    ("18-24, b3.0", Path("experiments/artifacts/probe_taskalign_l18to24_b3p0_full_diag.json"), Path("experiments/artifacts/drift_probe_taskalign_l18to24_b3p0_v2.json")),
    ("18-24, b4.0", Path("experiments/artifacts/probe_taskalign_l18to24_b4p0_full_diag.json"), Path("experiments/artifacts/drift_probe_taskalign_l18to24_b4p0_v2.json")),
    ("19-24, b3.0", Path("experiments/artifacts/probe_taskalign_l19to24_b3p0_full_diag.json"), Path("experiments/artifacts/drift_probe_taskalign_l19to24_b3p0_v2.json")),
    ("20-24, b3.0", Path("experiments/artifacts/probe_taskalign_l20to24_b3p0_full_diag.json"), Path("experiments/artifacts/drift_probe_taskalign_l20to24_b3p0_v2.json")),
    ("21-24, b3.0", Path("experiments/artifacts/probe_taskalign_l21to24_b3p0_full_diag.json"), Path("experiments/artifacts/drift_probe_taskalign_l21to24_b3p0_v2.json")),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create summary plots for the HDA experiments")
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/plots",
        help="Directory to write plot PNGs",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def accuracy_pct(value: float) -> float:
    return 100.0 * float(value)


def ci_to_pct(ci):
    return [accuracy_pct(ci[0]), accuracy_pct(ci[1])]


def make_overview_plot(output_dir: Path) -> Path:
    baseline = load_json(BASELINE_PATH)
    best_utility = load_json(BEST_UTILITY_PATH)
    best_balance = load_json(BEST_BALANCE_PATH)
    patch_attn = load_json(PATCH_ATTN_PATH)
    patch_mlp = load_json(PATCH_MLP_PATH)
    patch_both = load_json(PATCH_BOTH_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # Panel A: headline accuracy comparison.
    labels = ["Baseline", "Best probe", "Best balance"]
    values = [baseline["accuracy"], best_utility["intervened"]["acc"], best_balance["intervened"]["acc"]]
    cis = [baseline["ci95"], best_utility["intervened"]["ci95"], best_balance["intervened"]["ci95"]]
    x = np.arange(len(labels))
    heights = [accuracy_pct(v) for v in values]
    yerr = np.array([[accuracy_pct(v) - ci_to_pct(ci)[0] for v, ci in zip(values, cis)], [ci_to_pct(ci)[1] - accuracy_pct(v) for v, ci in zip(values, cis)]])
    axes[0].bar(x, heights, color=["#7f8c8d", "#2ecc71", "#3498db"], width=0.62, edgecolor="black", linewidth=0.6)
    axes[0].errorbar(x, heights, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Held-out accuracy (%)")
    axes[0].set_title("TruthfulQA held-out accuracy")
    axes[0].set_ylim(min(72.0, min(ci_to_pct(ci)[0] for ci in cis) - 1.0), max(80.5, max(ci_to_pct(ci)[1] for ci in cis) + 1.0))
    axes[0].grid(axis="y", alpha=0.2)

    # Panel B: probe utility-drift frontier.
    frontier_x = []
    frontier_y = []
    frontier_labels = []
    for label, probe_path, drift_path in PROBE_FRONTIER_PATHS:
        probe = load_json(probe_path)
        drift = load_json(drift_path)
        frontier_x.append(float(drift["summary"]["mean_similarity_ratio"]))
        frontier_y.append(100.0 * float(probe["delta_acc"]))
        frontier_labels.append(label)

    axes[1].scatter(frontier_x, frontier_y, s=70, color="#8e44ad", edgecolor="white", linewidth=0.6)
    for x_val, y_val, label in zip(frontier_x, frontier_y, frontier_labels):
        axes[1].annotate(label, (x_val, y_val), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Benign drift similarity")
    axes[1].set_ylabel("Delta accuracy (points)")
    axes[1].set_title("Utility-drift frontier")
    axes[1].set_xlim(0.84, 0.965)
    axes[1].grid(alpha=0.2)

    # Panel C: patch module ablation.
    module_labels = ["attn", "mlp", "both"]
    module_values = [100.0 * patch_attn["delta_acc"], 100.0 * patch_mlp["delta_acc"], 100.0 * patch_both["delta_acc"]]
    colors = ["#e67e22", "#95a5a6", "#34495e"]
    axes[2].bar(module_labels, module_values, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    axes[2].axhline(0.0, color="#555555", linewidth=0.8)
    axes[2].set_ylabel("Patch delta accuracy (points)")
    axes[2].set_title("Patch module ablation")
    axes[2].grid(axis="y", alpha=0.2)

    fig_path = output_dir / "results_overview.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def make_frontier_plot(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    frontier_x = []
    frontier_y = []
    frontier_labels = []
    frontier_sizes = []

    for label, probe_path, drift_path in PROBE_FRONTIER_PATHS:
        probe = load_json(probe_path)
        drift = load_json(drift_path)
        frontier_x.append(float(drift["summary"]["mean_similarity_ratio"]))
        frontier_y.append(100.0 * float(probe["delta_acc"]))
        frontier_labels.append(label)
        frontier_sizes.append(40 + 30 * len(probe.get("layers", [])))

    ax.scatter(frontier_x, frontier_y, s=frontier_sizes, c=np.linspace(0.2, 0.9, len(frontier_x)), cmap="viridis", edgecolor="white", linewidth=0.6)
    for x_val, y_val, label in zip(frontier_x, frontier_y, frontier_labels):
        ax.annotate(label, (x_val, y_val), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Benign generation similarity")
    ax.set_ylabel("Held-out delta accuracy (points)")
    ax.set_title("HDA utility-drift tradeoff")
    ax.set_xlim(0.84, 0.965)
    ax.grid(alpha=0.2)

    fig_path = output_dir / "tradeoff_frontier.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_path = make_overview_plot(output_dir)
    frontier_path = make_frontier_plot(output_dir)

    print(f"Saved overview figure to: {overview_path}")
    print(f"Saved frontier figure to: {frontier_path}")


if __name__ == "__main__":
    main()
