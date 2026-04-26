from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "w": "#4C78A8",
    "v": "#E45756",
    "residual": "#54A24B",
    "alpha1": "#F58518",
    "alpha28": "#B279A2",
    "guide": "#666666",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def draw_arrow(ax, start, end, color, label=None, lw=2.6, linestyle="-", zorder=3):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ax.arrow(
        start[0],
        start[1],
        dx,
        dy,
        length_includes_head=True,
        head_width=0.12,
        head_length=0.18,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        zorder=zorder,
    )
    if label:
        ax.text(
            end[0] + 0.06,
            end[1] + 0.06,
            label,
            color=color,
            fontsize=10,
            weight="bold",
            ha="left",
            va="bottom",
        )


def setup_axis(ax, title, xlim=(-1.0, 3.6), ylim=(-0.8, 2.6)):
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_alpha1_panel(ax, w, proj, alpha1):
    setup_axis(ax, r"Orthogonal Special Case ($\alpha = 1$)")
    ax.plot([0, 3.2], [0, 0], linestyle="--", color=COLORS["guide"], linewidth=1.2)
    ax.text(3.28, 0.02, r"extracted direction $\hat{v}$", color=COLORS["guide"], ha="left", va="bottom")
    ax.plot([proj[0], w[0]], [proj[1], w[1]], linestyle=":", color=COLORS["guide"], linewidth=1.1)
    draw_arrow(ax, (0, 0), tuple(w), COLORS["w"], r"$W$")
    draw_arrow(ax, (0, 0), tuple(proj), COLORS["v"], r"$(\hat{v}^{\top} W)\hat{v}$")
    draw_arrow(ax, (0, 0), tuple(alpha1), COLORS["alpha1"], r"$W'_{\alpha=1}$")
    ax.scatter([proj[0]], [proj[1]], color=COLORS["guide"], s=22, zorder=4)
    ax.text(
        0.03,
        0.05,
        r"$W' = W - \hat{v}(\hat{v}^{\top}W)$" "\n"
        r"clean nulling of the write along $\hat{v}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.3"},
    )


def render_alpha28_panel(ax, w, proj, residual, alpha1, alpha28):
    setup_axis(ax, r"Tuned Suppressive Operating Point ($\alpha = 2.8$)", xlim=(-5.4, 3.6))
    ax.plot([-5.2, 3.2], [0, 0], linestyle="--", color=COLORS["guide"], linewidth=1.2)
    ax.text(3.28, 0.02, r"extracted direction $\hat{v}$", color=COLORS["guide"], ha="left", va="bottom")
    draw_arrow(ax, (0, 0), tuple(w), COLORS["w"], r"$W$")
    draw_arrow(ax, (0, 0), tuple(proj), COLORS["v"], r"$(\hat{v}^{\top} W)\hat{v}$")
    draw_arrow(ax, (0, 0), tuple(residual), COLORS["residual"], r"residual")
    draw_arrow(ax, (0, 0), tuple(alpha28), COLORS["alpha28"], r"$W'_{\alpha=2.8}$")
    ax.annotate(
        "",
        xy=(alpha28[0], 0.0),
        xytext=(alpha1[0], 0.0),
        arrowprops={"arrowstyle": "<->", "color": COLORS["guide"], "linewidth": 1.1},
    )
    ax.text(
        (alpha28[0] + alpha1[0]) / 2,
        -0.15,
        "overshoot past zero\nalong $\\hat{v}$",
        ha="center",
        va="top",
        fontsize=9,
        color=COLORS["guide"],
    )
    ax.text(
        0.03,
        0.05,
        r"$W' = W - 2.8\,\hat{v}(\hat{v}^{\top}W)$" "\n"
        r"over-subtractive suppression, not a pure projector",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.3"},
    )


def main():
    out_dir = repo_root() / "experiments" / "artifacts" / "route_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 10,
        }
    )

    vhat = np.array([1.0, 0.0])
    w = np.array([2.6, 1.6])
    proj_scale = float(np.dot(w, vhat))
    proj = proj_scale * vhat
    residual = w - proj
    alpha1 = w - 1.0 * proj
    alpha28 = w - 2.8 * proj

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    render_alpha1_panel(axes[0], w, proj, alpha1)
    render_alpha28_panel(axes[1], w, proj, residual, alpha1, alpha28)

    fig.suptitle("Conceptual Geometry of the HDA Rank-One Patch", fontsize=14, weight="bold", y=1.02)
    out_path = out_dir / "orthogonal_style_suppression_geometry.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
    render_alpha1_panel(ax, w, proj, alpha1)
    out_path_alpha1 = out_dir / "orthogonal_special_case_alpha1.png"
    fig.savefig(out_path_alpha1, dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    render_alpha28_panel(ax, w, proj, residual, alpha1, alpha28)
    out_path_alpha28 = out_dir / "suppressive_operating_point_alpha28.png"
    fig.savefig(out_path_alpha28, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote conceptual geometry plot to: {out_path}")
    print(f"Wrote alpha=1 panel to: {out_path_alpha1}")
    print(f"Wrote alpha=2.8 panel to: {out_path_alpha28}")


if __name__ == "__main__":
    main()
