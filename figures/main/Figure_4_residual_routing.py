"""Six-panel residual-routing mechanism figure from completed analysis outputs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from _figure_common import (
    ANALYSIS, BLUE, EPOCHS, EPOCH_LABELS, GRAY, GREEN, IDS30, ORANGE, PURPLE,
    bootstrap, panel_label, read_csv, save_figure, style, write_csv, write_qc_stub,
)


STEM = Path(__file__).with_suffix("")
DIMENSIONS = (8.00, 7.50)
INITS = ("canonical_corrected_initialization", "inflated_1p8_initialization")
INIT_LABELS = ("Canonical initialization", "Inflated 1.8-fold initialization")
INIT_SHORT = ("Canonical", "Inflated 1.8-fold")
INIT_STYLE = ((BLUE, "o", "-"), (ORANGE, "s", "--"))


def summary_value(summary, initialization, epoch, metric):
    return next(
        row for row in summary
        if row["initialization"] == initialization
        and row["epoch"] == epoch
        and row["metric"] == metric
    )


def add_epoch_panel(ax, summary, metric, ylabel, letter, source):
    x = np.arange(3)
    offsets = (-0.045, 0.045)
    for j, (initialization, label, (color, marker, linestyle)) in enumerate(
        zip(INITS, INIT_LABELS, INIT_STYLE)
    ):
        values, lows, highs = [], [], []
        for epoch in EPOCHS:
            row = summary_value(summary, initialization, epoch, metric)
            values.append(float(row["mean"]))
            lows.append(float(row["bootstrap_ci_low"]))
            highs.append(float(row["bootstrap_ci_high"]))
            source.append({
                "panel": letter, "initialization": initialization, "epoch": epoch,
                "metric": metric, "mean": row["mean"],
                "bootstrap_ci_low": row["bootstrap_ci_low"],
                "bootstrap_ci_high": row["bootstrap_ci_high"],
                "n_realizations": row["n_realizations"],
                "realization_ids": row["realization_ids"],
                "interval_design": "matched realizations; 95% bootstrap interval",
            })
        values = np.asarray(values)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        ax.errorbar(
            x + offsets[j], values, yerr=[values - lows, highs - values],
            color=color, marker=marker, markerfacecolor="white", markeredgecolor=color,
            linestyle=linestyle, linewidth=1.15, markersize=4.3, capsize=2.2,
            label=label,
        )
    ax.set_xticks(x, EPOCH_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.18, 2.18)
    panel_label(ax, letter, x=-0.20, y=1.07)


def paired_absolute_intervals(epoch_rows, initialization, metric, seed):
    lookup = {
        (int(row["realization_id"]), row["epoch"]): float(row[metric])
        for row in epoch_rows if row["variant"] == initialization
    }
    matrix = np.array([[lookup[(rid, epoch)] for rid in range(30)] for epoch in EPOCHS])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 30, size=(10_000, 30))
    boot = matrix[:, indices].mean(axis=2)
    low, high = np.quantile(boot, [0.025, 0.975], axis=1)
    return matrix.mean(axis=1), low, high


def build():
    summary = read_csv(ANALYSIS / "02_mechanism" / "residual_routing_summary_v1.csv")
    decomposition = read_csv(ANALYSIS / "02_mechanism" / "residual_routing_exact_decomposition_v1.csv")
    epoch_rows = read_csv(ANALYSIS / "02_mechanism" / "mechanism_trial_epoch_metrics_v1.csv")
    source = []

    style()
    fig = plt.figure(figsize=DIMENSIONS)
    outer = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.12), hspace=0.75,
                            wspace=0.80, width_ratios=(1.0, 1.0, 1.10))
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[0, 2])
    d_grid = outer[1, 0].subgridspec(2, 1, hspace=0.10)
    ax_d1 = fig.add_subplot(d_grid[0, 0])
    ax_d2 = fig.add_subplot(d_grid[1, 0], sharex=ax_d1)
    ax_e = fig.add_subplot(outer[1, 1])
    ax_f = fig.add_subplot(outer[1, 2])

    add_epoch_panel(ax_a, summary, "mean_epsilon_vis",
                    "Signed visual innovation", "A", source)
    add_epoch_panel(ax_b, summary, "mean_K_obj_vis",
                    "Object-component visual gain", "B", source)
    add_epoch_panel(ax_c, summary, "mean_abs_delta_v_obj_hat",
                    "Absolute one-step correction", "C", source)
    ax_a.axhline(0, color=GRAY, linewidth=0.75)

    # D: separate subaxes preserve the sign and scale of the two covariance terms.
    for dax, metric, ylabel, top_axis in (
        (ax_d1, "mean_P_minus_obj_obj", r"$P^-_{\rm obj,obj}$", True),
        (ax_d2, "mean_P_minus_a_obj", r"$P^-_{a,\rm obj}$", False),
    ):
        x = np.arange(3)
        for j, (initialization, label, (color, marker, linestyle)) in enumerate(
            zip(INITS, INIT_LABELS, INIT_STYLE)
        ):
            values, lows, highs = [], [], []
            for epoch in EPOCHS:
                row = summary_value(summary, initialization, epoch, metric)
                values.append(float(row["mean"]))
                lows.append(float(row["bootstrap_ci_low"]))
                highs.append(float(row["bootstrap_ci_high"]))
                source.append({
                    "panel": "D", "initialization": initialization, "epoch": epoch,
                    "metric": metric, "mean": row["mean"],
                    "bootstrap_ci_low": row["bootstrap_ci_low"],
                    "bootstrap_ci_high": row["bootstrap_ci_high"],
                    "n_realizations": row["n_realizations"],
                    "realization_ids": row["realization_ids"],
                    "interval_design": "matched realizations; 95% bootstrap interval",
                })
            values, lows, highs = map(np.asarray, (values, lows, highs))
            dax.errorbar(
                x + (-0.04, 0.04)[j], values, yerr=[values - lows, highs - values],
                color=color, marker=marker, markerfacecolor="white", linestyle=linestyle,
                linewidth=1.05, markersize=3.8, capsize=1.8,
            )
        dax.set_ylabel(ylabel)
        dax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        dax.set_axisbelow(True)
        if top_axis:
            panel_label(dax, "D", x=-0.20, y=1.07)
            dax.tick_params(axis="x", labelbottom=False)
        else:
            dax.axhline(0, color=GRAY, linewidth=0.7)
            dax.set_xticks(x, EPOCH_LABELS)
            dax.set_xlim(-0.18, 2.18)

    # E: exact matched-sample decomposition, shown as directly labeled rows.
    term_specs = (
        ("mean_innovation_term", "Innovation", BLUE, "o", "-"),
        ("mean_gain_term", "Gain", GREEN, "s", "--"),
        ("mean_interaction_term", "Interaction", PURPLE, "D", ":"),
    )
    y_positions = {INITS[0]: [5.0, 4.0, 3.0], INITS[1]: [1.4, 0.4, -0.6]}
    for initialization, shape in zip(INITS, ("o", "s")):
        for y, (term, term_label, color, _, linestyle) in zip(y_positions[initialization], term_specs):
            values = np.array([
                float(row[term]) for row in decomposition
                if row["initialization"] == initialization
            ])
            mean, low, high = bootstrap(values, 20260850 + list(INITS).index(initialization) * 10 + list(t[0] for t in term_specs).index(term))
            ax_e.hlines(y, low, high, color=color, linewidth=1.2, linestyle=linestyle)
            ax_e.plot([low, high], [y, y], linestyle="none", marker="|", color=color,
                      markersize=5, markeredgewidth=0.8)
            ax_e.plot(mean, y, marker=shape, markerfacecolor="white", markeredgecolor=color,
                      markersize=4.8, linestyle="none")
            source.append({
                "panel": "E", "initialization": initialization,
                "epoch": "reduced_reliability_minus_baseline", "metric": term,
                "mean": mean, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "n_realizations": 30, "realization_ids": IDS30,
                "interval_design": "paired matched-sample 95% bootstrap interval",
            })
    ax_e.axvline(0, color=GRAY, linewidth=0.75)
    ax_e.axhline(2.2, color="#D0D0D0", linewidth=0.6)
    ax_e.set_yticks([5.0, 4.0, 3.0, 1.4, 0.4, -0.6],
                    ["Innovation", "Gain", "Interaction"] * 2)
    ax_e.set_xlim(-0.0014, 0.0015)
    ax_e.set_ylim(-1.2, 6.4)
    # Center the text directly above the data points so it acts as a clear group header
    ax_e.text(0.0005, 5.7, "Canonical initialization", ha="center", va="bottom",
              fontsize=7.8, fontweight="bold", color="#333333")
    ax_e.text(0.0005, 2.1, "Inflated 1.8-fold initialization", ha="center", va="bottom",
              fontsize=7.8, fontweight="bold", color="#333333")
    ax_e.set_xlabel("Contribution to signed\ncorrection change")
    ax_e.tick_params(axis="y", length=0, pad=10, labelsize=7.0)
    ax_e.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax_e.set_axisbelow(True)
    panel_label(ax_e, "E", x=-0.20, y=1.07)

    # F: absolute epoch values expose elevated inflated-initialization baseline and convergence.
    x_f = np.array([0, 1])
    for j, (initialization, label, (color, marker, linestyle)) in enumerate(
        zip(INITS, INIT_LABELS, INIT_STYLE)
    ):
        means, lows, highs = paired_absolute_intervals(
            epoch_rows, initialization, "false_positive_rms", 20260940 + j
        )
        means, lows, highs = means[:2], lows[:2], highs[:2]
        ax_f.errorbar(
            x_f + (-0.025, 0.025)[j], means,
            yerr=[means - lows, highs - means], color=color, marker=marker,
            markerfacecolor="white", linestyle=linestyle, linewidth=1.2,
            markersize=4.8, capsize=2.2, label=label,
        )
        for index, (mean, low, high) in enumerate(zip(means, lows, highs)):
            source.append({
                "panel": "F", "initialization": initialization,
                "epoch": EPOCHS[index], "metric": "false_positive_rms_absolute",
                "mean": mean, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "n_realizations": 30, "realization_ids": IDS30,
                "interval_design": "paired resampling of matched realizations; 95% bootstrap interval",
            })
        ax_f.annotate(f"{means[0]:.5f}", (x_f[0] + (-0.025, 0.025)[j], means[0]),
                      xytext=((-20, -18) if j == 0 else (20, 15)), textcoords="offset points",
                      color=color, fontsize=6.8, ha=("right" if j == 0 else "left"))
    ax_f.text(1.25, 0.08510, "both 0.08510", color="#444444", fontsize=6.5,
              ha="left", va="center")
    ax_f.set_xticks(x_f, ("Baseline", "Reduced\nreliability"))
    ax_f.set_xlim(-0.12, 1.80)
    ax_f.set_ylabel("Absolute false-positive RMS\n(model units)", labelpad=15)
    ax_f.set_ylim(0.045, 0.111)
    ax_f.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax_f.set_axisbelow(True)
    panel_label(ax_f, "F", x=-0.20, y=1.07)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.525), handlelength=2.8)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.945, bottom=0.105)

    write_csv(STEM.with_name(STEM.name + "_source_data.csv"), source)
    save_figure(fig, STEM)
    write_qc_stub(STEM, DIMENSIONS, [
        "Panel F check: absolute baseline and reduced-reliability means are shown and directly labeled for both initializations.",
        "Mechanism check: panels A-C use signed innovation, gain, and absolute correction without asserting increases in absolute innovation, gain, or correction.",
        "Decomposition check: panel E uses the exact matched-sample terms from residual_routing_exact_decomposition_v1.csv.",
    ])


if __name__ == "__main__":
    build()
