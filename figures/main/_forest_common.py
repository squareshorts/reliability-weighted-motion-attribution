"""Shared structural-control forest plot implementation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _figure_common import ANALYSIS, BLUE, GREEN, GRAY, ORANGE, PURPLE, save_figure, style, write_csv, write_qc_stub


ORDER = [
    ("Alignment", "event_misaligned_fixed_eye_trace", "Event-misaligned fixed trace"),
    ("Alignment", "event_aligned_prerecorded_eye_trace", "Event-aligned prerecorded trace"),
    ("Alignment", "per_trial_closed_loop", "Per-trial closed loop"),
    ("Initialization", "canonical_forward_initialization", "Canonical initialization"),
    ("Initialization", "inflated_1p8_forward_initialization", "Inflated 1.8-fold initialization"),
    ("Variance policy", "previous_belief_weighted_otolith_variance", "Previous-belief-weighted variance"),
    ("Variance policy", "oracle_otolith_variance", "Oracle variance"),
    ("Variance policy", "fixed_baseline_context_variance", "Fixed baseline-context variance"),
    ("Eye path", "eye_compensation", "Eye compensation"),
    ("Eye path", "no_eye_compensation", "No eye compensation"),
    ("Eye path", "eye_velocity_forced_zero", "Eye velocity forced to zero"),
    ("Object-event controls", "object_events_removed", "Object events removed"),
]

GROUP_STYLE = {
    "Alignment": (BLUE, "o", "-"),
    "Initialization": (PURPLE, "s", "--"),
    "Variance policy": (ORANGE, "D", "-."),
    "Eye path": (GREEN, "^", ":"),
    "Object-event controls": (GRAY, "v", "--"),
}


def build_forest(stem: Path, metric: str, xlabel: str, dimensions=(6.70, 5.25)):
    import csv
    with (ANALYSIS / "03_structural_controls" / "control_paired_contrasts_v1.csv").open(encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    lookup = {(row["variant"], row["metric"]): row for row in raw}
    plotted = [(g, v, lab, lookup[(v, metric)]) for g, v, lab in ORDER if (v, metric) in lookup]

    style()
    fig, ax = plt.subplots(figsize=dimensions)
    fig.subplots_adjust(left=0.46, right=0.975, top=0.975, bottom=0.13)

    positions = []
    y = 0.0
    last_group = None
    for group, variant, label, row in plotted:
        if group != last_group:
            if last_group is not None:
                y += 0.55
            positions.append((y, group, True))
            y += 0.78
            last_group = group
        positions.append((y, label, False))
        y += 0.82

    row_positions = [p[0] for p in positions if not p[2]]
    source = []
    for yi, (group, variant, label, row) in zip(row_positions, plotted):
        point = float(row["paired_absolute_change_mean"])
        lo = float(row["paired_absolute_change_ci_low"])
        hi = float(row["paired_absolute_change_ci_high"])
        color, marker, linestyle = GROUP_STYLE[group]
        ax.hlines(yi, lo, hi, color=color, linewidth=1.25, linestyle=linestyle)
        ax.plot([lo, hi], [yi, yi], linestyle="none", marker="|", color=color,
                markersize=5.5, markeredgewidth=0.8)
        ax.plot(point, yi, marker=marker, markersize=5.3, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.05, linestyle="none")
        source.append({
            "group": group, "variant": variant, "label": label, "metric": metric,
            "point_estimate": row["paired_absolute_change_mean"],
            "bootstrap_ci_low": row["paired_absolute_change_ci_low"],
            "bootstrap_ci_high": row["paired_absolute_change_ci_high"],
            "n_realizations": row["n_realizations"],
            "realization_ids": row["seed_identities"],
        })

    ax.axvline(0, color="#4D4D4D", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_yticks([p[0] for p in positions], [p[1] for p in positions])
    for tick, (_, group_or_label, is_group) in zip(ax.get_yticklabels(), positions):
        if is_group:
            tick.set_fontweight("bold")
            tick.set_color(GROUP_STYLE[group_or_label][0])
            tick.set_fontsize(8.2)
        else:
            tick.set_fontsize(8.0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_ylim(y - 0.30, -0.45)

    all_lows = np.array([float(p[3]["paired_absolute_change_ci_low"]) for p in plotted])
    all_highs = np.array([float(p[3]["paired_absolute_change_ci_high"]) for p in plotted])
    lo = min(float(all_lows.min()), 0.0)
    hi = max(float(all_highs.max()), 0.0)
    span = hi - lo
    ax.set_xlim(lo - 0.08 * span, hi + 0.10 * span)

    write_csv(stem.with_name(stem.name + "_source_data.csv"), source)
    save_figure(fig, stem)
    write_qc_stub(stem, dimensions, [
        "Forest-plot check: zero-reference line retained; row-level sample-size annotations omitted.",
        "Data check: plotted points and interval endpoints are direct copies of control_paired_contrasts_v1.csv.",
    ])
