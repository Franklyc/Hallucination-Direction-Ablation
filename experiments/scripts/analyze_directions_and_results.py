import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze extracted directions and experiment outputs")
    parser.add_argument(
        "--directions",
        default="experiments/artifacts/directions_qwen4b_cal200_taskaligned.npz",
        help="Path to directions npz",
    )
    parser.add_argument(
        "--probe-results",
        nargs="*",
        default=[
            "experiments/artifacts/probe_taskalign_l18to24_b3p0_full_diag.json",
            "experiments/artifacts/probe_taskalign_l20to24_b3p0_full_diag.json",
            "experiments/artifacts/probe_taskalign_l18to24_b4p0_full_diag.json",
        ],
        help="Probe result JSON files to include in summary",
    )
    parser.add_argument(
        "--drift-results",
        nargs="*",
        default=[
            "experiments/artifacts/drift_probe_taskalign_l18to24_b2p5_v2.json",
            "experiments/artifacts/drift_probe_taskalign_l18to24_b2p75_v2.json",
            "experiments/artifacts/drift_probe_taskalign_l18to24_b3p0_v2.json",
            "experiments/artifacts/drift_probe_taskalign_l18to24_b4p0_v2.json",
            "experiments/artifacts/drift_probe_taskalign_l19to24_b3p0_v2.json",
            "experiments/artifacts/drift_probe_taskalign_l20to24_b3p0_v2.json",
            "experiments/artifacts/drift_probe_taskalign_l21to24_b3p0_v2.json",
        ],
        help="Drift result JSON files to include in summary",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/analysis",
        help="Directory for analysis outputs",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def refusal_rate_from_text(text: str) -> float:
    lowered = text.lower()
    markers = [
        "i don't know",
        "i do not know",
        "i'm sorry",
        "i am sorry",
        "can't help",
        "cannot help",
        "won't help",
        "refuse",
        "refusal",
        "not able to",
        "i can't",
        "i cannot",
    ]
    return 1.0 if any(marker in lowered for marker in markers) else 0.0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.directions)
    directions = data["directions"].astype(np.float64)
    norms = np.linalg.norm(directions, axis=1)

    denom = np.linalg.norm(directions, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    normalized = directions / denom
    cosine = normalized @ normalized.T

    np.save(output_dir / "direction_cosine_similarity.npy", cosine)
    np.save(output_dir / "direction_norms.npy", norms)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    heatmap = axes[0].imshow(cosine, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_title("Inter-layer cosine similarity")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Layer")
    fig.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(np.arange(len(norms)), norms, marker="o", linewidth=1.5)
    axes[1].set_title("Direction norm by layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("L2 norm")
    axes[1].grid(True, alpha=0.25)

    fig_path = output_dir / "direction_analysis.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    off_diag = cosine[~np.eye(cosine.shape[0], dtype=bool)]
    cosine_summary = {
        "mean_offdiag_cosine": float(off_diag.mean()) if off_diag.size else float("nan"),
        "median_offdiag_cosine": float(np.median(off_diag)) if off_diag.size else float("nan"),
        "max_abs_offdiag_cosine": float(np.max(np.abs(off_diag))) if off_diag.size else float("nan"),
        "top_norm_layer": int(np.argmax(norms)),
        "top_norm_value": float(norms.max()),
    }

    probe_rows = []
    for probe_path in args.probe_results:
        result = load_json(probe_path)
        probe_rows.append(
            {
                "path": probe_path,
                "layers": result.get("layers", []),
                "beta": result.get("beta"),
                "delta_acc": result.get("delta_acc"),
                "fixed_count": result.get("diagnostics", {}).get("fixed_count"),
                "broken_count": result.get("diagnostics", {}).get("broken_count"),
                "flip_count": result.get("diagnostics", {}).get("flip_count"),
            }
        )

    drift_rows = []
    for drift_path in args.drift_results:
        result = load_json(drift_path)
        rows = result.get("rows", [])
        refusal_rate = float(np.mean([refusal_rate_from_text(row.get("probe_text", "")) for row in rows])) if rows else float("nan")
        drift_rows.append(
            {
                "path": drift_path,
                "layers": result.get("layers", []),
                "beta": result.get("beta"),
                "mean_similarity_ratio": result.get("summary", {}).get("mean_similarity_ratio"),
                "exact_match_rate": result.get("summary", {}).get("exact_match_rate"),
                "refusal_keyword_rate": refusal_rate,
            }
        )

    summary = {
        "directions": args.directions,
        "direction_norm_summary": cosine_summary,
        "probe_summary": probe_rows,
        "drift_summary": drift_rows,
        "outputs": {
            "figure": str(fig_path),
            "cosine_npy": str(output_dir / "direction_cosine_similarity.npy"),
            "norms_npy": str(output_dir / "direction_norms.npy"),
        },
    }

    summary_path = output_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved figure to: {fig_path}")
    print(f"Saved summary to: {summary_path}")
    print(
        "direction summary: "
        f"mean_offdiag_cosine={cosine_summary['mean_offdiag_cosine']:.4f} "
        f"top_norm_layer={cosine_summary['top_norm_layer']}"
    )


if __name__ == "__main__":
    main()
