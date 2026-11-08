"""Chart mean training reward per iteration for the walking and stand/sit runs.

    python plot_learning_curves.py results/learning_curves.json results/learning_curves.png

The JSON holds ``Train/mean_reward`` exported from each run's TensorBoard log. Walking and stand/sit
have different reward functions, so they get separate panels.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [
    ("Walking", [("walk_norand", "No randomization", "#2a78d6"), ("walk_rand", "Randomized", "#eb6834")]),
    ("Stand / sit", [("sit_v1", "v1: reward bug", "#8a8984"), ("sit_v2", "v2: reward fixed", "#2a78d6"),
                     ("sit_v3", "v3: stronger pushes", "#eb6834"), ("sit_v4", "v4: may step", "#1baf7a")]),
]


def spread(values: list[float], gap: float) -> list[float]:
    """Move sorted label positions apart so that neighbours are at least ``gap`` apart; keeps input order."""
    order = sorted(range(len(values)), key=values.__getitem__)
    placed = [values[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = placed[k]
    return out


def main(src: str, dst: str) -> None:
    """Read the exported rewards in ``src`` and save the two-panel chart to ``dst``."""
    with open(src) as f:
        runs = json.load(f)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    for ax, (title, series) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ends = []
        for key, label, color in series:
            reward = runs[key]
            ax.plot(range(len(reward)), reward, color=color, linewidth=1.8, label=label)
            ends.append((len(reward) - 1, reward[-1], label, color))
        ymin, ymax = ax.get_ylim()
        x_max = max(e[0] for e in ends)
        for x_end in {e[0] for e in ends}:  # labels only compete with lines that end at the same iteration
            group = [e for e in ends if e[0] == x_end]
            for (x, y, label, color), y_text in zip(group, spread([e[1] for e in group], (ymax - ymin) * 0.075)):
                ax.plot([x, x + 25], [y, y_text], color=color, linewidth=1)
                ax.text(x + 35, y_text, label, va="center", color=INK_2)
        ax.set_xlim(0, x_max * 1.3)
        ax.set_xticks(range(0, x_max + 2, 500))
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left")
        ax.set_xlabel("training iteration")
        ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("mean episode reward")
    fig.tight_layout()
    fig.savefig(dst, dpi=150, facecolor=SURFACE)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
