"""Context and OKR trajectories from existing machine-readable outputs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from _figure_common import BLUE, GRAY, PURPLE, panel_label, read_csv, save_figure, style, write_csv, write_qc_stub


STEM = Path(__file__).with_suffix("")
DIMENSIONS = (6.70, 3.90)


def build():
    candidate = read_csv(STEM.with_name(STEM.name + "_source_data.csv"))
    time_values = np.array([float(row["time_s"]) for row in candidate])
    dt = float(np.min(np.diff(np.unique(time_values))))
    epoch_samples = len(np.unique(time_values)) // 3
    reduced_start = epoch_samples * dt
    reduced_end = 2 * epoch_samples * dt

    style()
    fig, axes = plt.subplots(2, 1, figsize=DIMENSIONS, sharex=True)
    source = []
    specifications = (
        ("A", "g_okr", "Raw OKR gain", BLUE, "-"),
        ("B", "p_hist", "Reliability-context\n" r"probability, $p_t$", PURPLE, "--"),
    )
    for ax, (panel, metric, ylabel, color, linestyle) in zip(axes, specifications):
        rows = [row for row in candidate if row["panel"] == panel and row["metric"] == metric]
        time = np.array([float(row["time_s"]) for row in rows])
        mean = np.array([float(row["mean"]) for row in rows])
        low = np.array([float(row["realization_percentile_2p5"]) for row in rows])
        high = np.array([float(row["realization_percentile_97p5"]) for row in rows])
        ax.axvspan(reduced_start, reduced_end, color=GRAY, alpha=0.10, linewidth=0,
                   label="Reduced-reliability epoch")
        ax.fill_between(time, low, high, color=color, alpha=0.18, linewidth=0,
                        label="2.5th-97.5th realization percentiles")
        ax.plot(time, mean, color=color, linestyle=linestyle, linewidth=1.15,
                label="Across-realization mean")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        panel_label(ax, panel, x=-0.08, y=0.95)
        source.extend(rows)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(float(candidate[0]["time_s"]), float(candidate[-1]["time_s"]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01), handlelength=2.7)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.205, top=0.98, hspace=0.17)

    write_csv(STEM.with_name(STEM.name + "_source_data.csv"), source)
    save_figure(fig, STEM)
    write_qc_stub(STEM, DIMENSIONS, [
        "Simulation check: trajectories are read from the committed final source-data CSV; no raw simulation archive is required.",
        "Epoch check: reduced-reliability boundaries are reconstructed from the three equal recorded epochs and the source-data time step.",
    ])


if __name__ == "__main__":
    build()
