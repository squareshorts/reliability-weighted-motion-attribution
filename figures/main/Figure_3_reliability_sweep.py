"""Final reliability-sweep figure using the authoritative summary CSV."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from _figure_common import ANALYSIS, BLUE, GRAY, IDS100, PURPLE, panel_label, read_csv, save_figure, style, write_csv, write_qc_stub


STEM = Path(__file__).with_suffix("")
DIMENSIONS = (6.70, 3.60)


def build():
    rows = [
        row for row in read_csv(ANALYSIS / "04_reliability_sweep" / "reliability_sweep_extended_summary_v1.csv")
        if row["metric"] == "false_positive_rms"
    ]
    rows.sort(key=lambda row: float(row["kappa"]))
    kappa = np.array([float(row["kappa"]) for row in rows])
    series = [
        (
            np.array([float(row["manipulated_mean"]) for row in rows]),
            np.array([float(row["mean_bootstrap_ci_low"]) for row in rows]),
            np.array([float(row["mean_bootstrap_ci_high"]) for row in rows]),
            "Manipulated-epoch false-positive RMS\n(model units)",
        ),
        (
            np.array([float(row["paired_percent_change_mean"]) for row in rows]),
            np.array([float(row["paired_percent_change_ci_low"]) for row in rows]),
            np.array([float(row["paired_percent_change_ci_high"]) for row in rows]),
            "Paired change from baseline (%)",
        ),
    ]

    style()
    fig, axes = plt.subplots(1, 2, figsize=DIMENSIONS)
    for index, (ax, (mean, low, high, ylabel)) in enumerate(zip(axes, series)):
        ax.fill_between(kappa, low, high, color=BLUE, alpha=0.20, linewidth=0,
                        label="95% bootstrap band")
        ax.plot(kappa, mean, color=BLUE, linewidth=1.2, linestyle=(0, (3, 2)),
                marker="o", markersize=4.1, markerfacecolor="white",
                markeredgecolor=BLUE, label="Evaluated reliability settings")
        ax.axvline(0.15, color=PURPLE, linewidth=1.0, linestyle="--",
                   label=r"Canonical $\kappa=0.15$")
        if index == 1:
            ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.set_xlabel(r"Effective otolith reliability, $\kappa$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.075, 1.025)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        panel_label(ax, chr(ord("A") + index), x=-0.15, y=1.05)

        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        top_ticks = np.array([0.10, 0.15, 0.30, 0.50, 1.00])
        top.set_xticks(top_ticks, [f"{0.06 / value:.2f}" for value in top_ticks])
        top.set_xlabel(r"Generative otolith noise, $\sigma_{\rm oto}$", labelpad=5)
        top.tick_params(axis="x", labelsize=6.8, pad=2, labelrotation=35)
        top.spines["top"].set_visible(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.015), handlelength=2.8)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.255, top=0.77, wspace=0.38)

    source = [{**row, "realization_ids": IDS100} for row in rows]
    write_csv(STEM.with_name(STEM.name + "_source_data.csv"), source)
    save_figure(fig, STEM)
    write_qc_stub(STEM, DIMENSIONS, [
        "Axis check: Greek kappa and sigma_oto are typeset mathematically; the canonical kappa=0.15 marker is retained.",
        "Interpretation check: dashed segments only connect the evaluated settings; no linear model is fitted or asserted.",
        "Data check: means and bootstrap-band limits are direct copies of reliability_sweep_extended_summary_v1.csv.",
    ])


if __name__ == "__main__":
    build()
