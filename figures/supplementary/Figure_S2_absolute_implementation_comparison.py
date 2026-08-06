"""Absolute implementation comparison from completed trial-epoch metrics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from _figure_common import (
    ANALYSIS, BLUE, EPOCHS, EPOCH_LABELS, GREEN, IDS30, ORANGE,
    bootstrap, panel_label, read_csv, save_figure, style, write_csv, write_qc_stub,
)


STEM = Path(__file__).with_suffix("")
DIMENSIONS = (6.70, 3.55)
VARIANTS = (
    "event_misaligned_fixed_eye_trace",
    "event_aligned_prerecorded_eye_trace",
    "per_trial_closed_loop",
)
LABELS = (
    "Event-misaligned fixed trace",
    "Event-aligned prerecorded trace",
    "Per-trial closed loop",
)
STYLES = ((ORANGE, "o", "-"), (GREEN, "s", "--"), (BLUE, "^", "-."))
METRICS = (
    ("angular_rmse", "Angular-motion RMSE\n(model units)"),
    ("object_rmse", "Object-motion RMSE\n(model units)"),
    ("false_positive_rms", "False-positive world-motion RMS\n(model units)"),
)


def build():
    rows = read_csv(ANALYSIS / "03_structural_controls" / "control_trial_epoch_metrics_v1.csv")
    source = []
    style()
    fig, axes = plt.subplots(1, 3, figsize=DIMENSIONS)
    x = np.arange(3)
    for p, (ax, (metric, ylabel)) in enumerate(zip(axes, METRICS)):
        for j, (variant, label, (color, marker, linestyle)) in enumerate(zip(VARIANTS, LABELS, STYLES)):
            means, lows, highs = [], [], []
            for e, epoch in enumerate(EPOCHS):
                values = [
                    float(row[metric]) for row in rows
                    if row["variant"] == variant and row["epoch"] == epoch
                    and row[metric] not in ("", "nan")
                ]
                mean, low, high = bootstrap(values, 20261000 + p * 20 + j * 3 + e)
                means.append(mean)
                lows.append(low)
                highs.append(high)
                source.append({
                    "panel": chr(ord("A") + p), "variant": variant,
                    "label": label, "epoch": epoch, "metric": metric,
                    "mean": mean, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                    "n_realizations": len(values), "realization_ids": IDS30,
                })
            means, lows, highs = map(np.asarray, (means, lows, highs))
            ax.errorbar(
                x + (j - 1) * 0.10, means, yerr=[means - lows, highs - means],
                color=color, marker=marker, markerfacecolor="white",
                linestyle=linestyle, linewidth=1.15, markersize=4.2,
                capsize=2.0, label=label,
            )
        ax.set_xticks(x, EPOCH_LABELS)
        ax.set_xlim(-0.18, 2.18)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        panel_label(ax, chr(ord("A") + p), x=-0.19, y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.012), handlelength=2.7)
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.255, top=0.955, wspace=0.50)
    write_csv(STEM.with_name(STEM.name + "_source_data.csv"), source)
    save_figure(fig, STEM)
    write_qc_stub(STEM, DIMENSIONS, [
        "Condition check: the three requested eye-path implementations are distinguished by color, marker, and line style.",
        "Data check: means and intervals are computed only from control_trial_epoch_metrics_v1.csv.",
    ])


if __name__ == "__main__":
    build()
