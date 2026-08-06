"""Shared helpers for the final publication figures."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FINAL = Path(__file__).resolve().parent
ANALYSIS = FINAL.parents[1]
IDS30 = ";".join(map(str, range(30)))
IDS100 = ";".join(map(str, range(100)))
EPOCHS = ("baseline", "reduced_reliability", "recovery")
EPOCH_LABELS = ("Baseline", "Reduced\nreliability", "Recovery")

# Paul Tol's colorblind-safe bright palette, paired with shape and line style.
BLUE = "#4477AA"
ORANGE = "#EE7733"
GREEN = "#228833"
PURPLE = "#AA3377"
CYAN = "#66CCEE"
GRAY = "#666666"
LIGHT_GRID = "#D9D9D9"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, seed: int, n: int = 10_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def save_figure(fig, stem: Path) -> None:
    """Save fixed-size vector outputs and a nominal 600-dpi PNG."""
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def panel_label(ax, letter: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.0,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def finish_axis(ax) -> None:
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def write_qc_stub(stem: Path, dimensions_in: tuple[float, float], notes: list[str]) -> None:
    width_mm = dimensions_in[0] * 25.4
    height_mm = dimensions_in[1] * 25.4
    lines = [
        f"# QC: {stem.name}",
        "",
        f"- Dimensions: {dimensions_in[0]:.2f} x {dimensions_in[1]:.2f} in "
        f"({width_mm:.1f} x {height_mm:.1f} mm).",
        "- Typography: DejaVu Sans; 8.0-10.0 pt publication-scale text.",
        "- Formats: PDF and SVG preserve vector text/line art; PNG exported at 600 dpi.",
        "- Clipping/overlap: pending final rendered-page inspection.",
        "- Source-data agreement: pending automated final comparison.",
    ]
    lines.extend(f"- {note}" for note in notes)
    stem.with_name(stem.name + "_QC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
