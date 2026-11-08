"""Chart gap-study results: fall rate and velocity error against each fake-real change.

    python plot_gap.py results/gap_results.jsonl results/gap_chart.png
    python plot_gap.py results/sit_results.jsonl,results/sit_push_results.jsonl results/sit_compare_chart.png
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POLICIES = {  # name -> (label, color, marker)
    "norand": ("No randomization (500 iterations)", "#2a78d6", "o"),
    "rand": ("Domain randomization (1500 iterations)", "#eb6834", "s"),
    "sit": ("Sit, trained with 0.5 m/s pushes (1000 iterations)", "#2a78d6", "o"),
    "sit_push": ("Sit, trained with 0-1.5 m/s pushes (1500 iterations)", "#eb6834", "s"),
    "sit_step": ("Same, may step while standing (1500 iterations)", "#1baf7a", "^"),
}
AXES = {  # axis -> (title, x label, nominal, randomized training range or None)
    "friction": ("Floor friction", "friction coefficient", 1.0, (0.3, 1.25)),
    "payload_kg": ("Extra payload", "kg on the body", 0.0, (-1.0, 5.0)),
    "motor_gain": ("Weaker motors", "PD gain scale", 1.0, (0.6, 1.1)),
    "push_mps": ("Random pushes", "push speed every 4 s (m/s)", 0.0, (1.0, 1.0)),
    "imu_noise_x": ("IMU noise", "x stock noise level", 0.0, (1.0, 1.0)),
    "delay_steps": ("Action delay", "policy steps (20 ms)", 0.0, (0.0, 2.0)),
}
PCT = lambda v: f"{v:.0%}"  # noqa: E731
# metric rows per results kind: (key, y label, tick formatter or None for a shared linear scale)
WALK_METRICS = [("fall_rate", "Fall rate", PCT), ("vel_err_xy", "Velocity error (m/s)", None)]
SIT_METRICS = [("sit_success", "Held the sit", PCT), ("fall_rate", "Fall rate", PCT),
               ("sit_joint_err", "Sit joint error (rad)", None)]


def main(src: str, dst: str) -> None:
    """Read the comma-separated JSONL files in ``src`` and save the chart grid to ``dst``."""
    data = defaultdict(lambda: defaultdict(list))
    rows = [json.loads(line) for path in src.split(",") for line in open(path)]
    for r in rows:
        data[r["axis"]][r["policy"]].append((r["value"], r))
    METRICS = SIT_METRICS if "sit_success" in rows[0] else WALK_METRICS
    axes = [a for a in AXES if a in data]
    policies = [p for p in POLICIES if any(p in data[a] for a in axes)]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, grid = plt.subplots(len(METRICS), len(axes), figsize=(3.1 * len(axes), 2.8 * len(METRICS)), squeeze=False,
                             sharex="col", facecolor=SURFACE)
    for col, axis in enumerate(axes):
        title, xlabel, nominal, train = AXES[axis]
        if METRICS is SIT_METRICS and axis == "push_mps":
            train = (0.0, 0.0)  # the sit policies differ here; their legend labels give the push training
        for row, (metric, ylabel, fmt) in enumerate(METRICS):
            ax = grid[row][col]
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            lo, hi = train
            if hi > lo:
                ax.axvspan(lo, hi, color="#efeeea", zorder=0, linewidth=0)
            ax.axvline(nominal, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
            for p in policies:
                pts = sorted(data[axis].get(p, []), key=lambda t: t[0])
                if not pts:
                    continue
                label, color, marker = POLICIES[p]
                # None = too few robots left to score; drawn as a gap
                ys = [float("nan") if r[metric] is None else r[metric] for _, r in pts]
                ax.plot([v for v, _ in pts], ys, color=color, linewidth=2,
                        marker=marker, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=label, zorder=3)
            ax.set_ylim(bottom=0, top=1.0 if fmt is PCT else None)
            if fmt:
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _, f=fmt: f(v)))
            if row == 0:
                ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left")
            if row == len(METRICS) - 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
    for row, (metric, _, fmt) in enumerate(METRICS):
        if fmt is None and axes:
            top = max(r[metric] for a in axes for p in data[a].values() for _, r in p if r[metric] is not None) * 1.05
            for col in range(len(axes)):
                grid[row][col].set_ylim(0, max(top, 0.1))
    handles, labels = grid[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(policies), frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.text(0.99, 0.985, "dashed line = ideal simulator value · shaded = randomized training range",
             ha="right", va="top", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(dst, dpi=150, facecolor=SURFACE)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
