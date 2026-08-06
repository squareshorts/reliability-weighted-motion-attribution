"""Draw the final causal model architecture from scratch."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath

from _figure_common import BLUE, GREEN, GRAY, ORANGE, PURPLE, save_figure, style, write_csv, write_qc_stub


STEM = Path(__file__).with_suffix("")
DIMENSIONS = (6.70, 7.25)

CATEGORIES = {
    "generative signal": ("#E8F1F8", BLUE),
    "observation": ("#FFF1E6", ORANGE),
    "estimate": ("#E9F5EC", GREEN),
    "adaptive parameter": ("#F5EAF2", PURPLE),
    "reliability context": ("#F0F0F0", GRAY),
    "gaze feedback": ("#FFF5CC", "#8C7200"),
}


def node(ax, source, key, label, x, y, w, h, category, boxstyle="round"):
    face, edge = CATEGORIES[category]
    rounding = 0.8 if boxstyle == "round" else 0.0
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.18,rounding_size={rounding}",
        linewidth=0.95, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.6, zorder=4)
    source.append({
        "record_type": "node", "id": key, "label": label.replace("\n", " "),
        "category": category, "x": x, "y": y, "width": w, "height": h,
        "from": "", "to": "", "route": "", "edge_label": "",
    })


def edge(ax, source, start, end, route, label="", label_xy=None, linestyle="-"):
    path = MplPath(route, [MplPath.MOVETO] + [MplPath.LINETO] * (len(route) - 1))
    arrow = FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=8.5, linewidth=0.78,
        color="#3F3F3F", linestyle=linestyle, zorder=2,
        joinstyle="round", capstyle="round",
    )
    ax.add_patch(arrow)
    if label and label_xy is not None:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=6.8,
                color="#303030", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    source.append({
        "record_type": "edge", "id": f"{start}_to_{end}", "label": "",
        "category": "gating" if linestyle != "-" else "signal",
        "x": "", "y": "", "width": "", "height": "",
        "from": start, "to": end,
        "route": ";".join(f"{x:.2f},{y:.2f}" for x, y in route),
        "edge_label": label,
    })


def build():
    style()
    fig, ax = plt.subplots(figsize=DIMENSIONS)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.995, bottom=0.065)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 100)
    ax.axis("off")

    lanes = [
        (74, 98, "1  Generative state\nand observations"),
        (51, 71, "2  Estimation"),
        (25, 48, "3  Reliability and adaptation"),
        (2, 22, "4  Gaze feedback"),
    ]
    for i, (bottom, top, label) in enumerate(lanes):
        ax.add_patch(Rectangle((0, bottom), 100, top - bottom,
                               facecolor=("#FAFAFA" if i % 2 == 0 else "#F6F6F6"),
                               edgecolor="#D7D7D7", linewidth=0.65, zorder=0))
        ax.text(2.0, top - 1.2, label, ha="left", va="top",
                fontsize=7.2, fontweight="bold", color="#555555", zorder=6,
                bbox={"facecolor": "#FAFAFA", "edgecolor": "none", "pad": 0.5})

    source = []
    # Lane 1
    node(ax, source, "latent", "Latent states\n" r"$\omega,\ a,\ v_{\rm obj}$",
         12, 86, 16, 7, "generative signal")
    node(ax, source, "canal", "Canal\nobservation", 34, 94, 15, 6.5, "observation")
    node(ax, source, "otolith", "Otolith\nobservation", 34, 86, 15, 6.5, "observation")
    node(ax, source, "retinal", "Raw retinal\nobservation", 34, 77.5, 15, 6.5, "observation")

    # Lane 2
    node(ax, source, "eye_comp", "Eye-compensated\nvisual observation", 13, 61, 18, 7.5, "observation")
    node(ax, source, "comparator",
         "Visual comparator\n$\\epsilon_{\\rm vis}=\\tilde y_{\\rm vis}-\\hat y_{\\rm vis}$",
         39, 61, 19, 8.5, "estimate")
    node(ax, source, "kalman_update", "Kalman\nstate update", 70, 61, 14, 7, "estimate")
    node(ax, source, "estimated_states", "Estimated states\n" r"$\hat\omega,\ \hat a,\ \hat v_{\rm obj}$",
         91, 61, 16, 7.5, "estimate")

    # Lane 3
    node(ax, source, "otolith_innovation", "Otolith\ninnovation", 11, 40, 14, 7, "estimate")
    node(ax, source, "context", "Reliability-context\ninference $p_t$", 31, 40, 17, 7,
         "reliability context")
    node(ax, source, "hypothesis_B", "Variance\nhypothesis $B$", 52, 44, 11, 5,
         "reliability context")
    node(ax, source, "hypothesis_R", "Variance\nhypothesis $R$", 52, 36.5, 11, 5,
         "reliability context")
    node(ax, source, "weighted_variance", "Previous-belief-\nweighted otolith variance\n$R_{\\rm oto,t}$",
         76, 40, 20, 8.5, "reliability context")
    node(ax, source, "forward_prediction", "Forward visual\nprediction $\\hat y_{\\rm vis}$",
         42, 29.5, 17, 7, "adaptive parameter")
    node(ax, source, "adaptation", "Context-gated\nforward-model adaptation",
         67, 29.5, 22, 7.5, "adaptive parameter")

    # Lane 4
    node(ax, source, "vor_okr", "VOR and OKR\npathways", 30, 11.5, 15, 6.5, "gaze feedback")
    node(ax, source, "eye_plant", "Eye plant", 52, 11.5, 13, 6, "gaze feedback")
    node(ax, source, "eye_velocity", "Eye velocity\n$\\dot e$", 75, 11.5, 13, 6, "gaze feedback")

    # Generative branches and observations.
    edge(ax, source, "latent", "canal", [(20, 86), (23, 86), (23, 94), (26.5, 94)])
    edge(ax, source, "latent", "otolith", [(20, 86), (26.5, 86)])
    edge(ax, source, "latent", "retinal", [(20, 86), (23, 86), (23, 77.5), (26.5, 77.5)],
         "$A_{\\rm true}x$", (18.0, 80.5))
    edge(ax, source, "canal", "kalman_update", [(41.5, 94), (70, 94), (70, 64.5)],
         "$y_{\\rm canal}$", (56, 95.7))
    edge(ax, source, "otolith", "kalman_update", [(41.5, 86), (64, 86), (64, 66.5), (67, 64.5)],
         "$y_{\\rm oto}$", (54, 87.5))
    edge(ax, source, "otolith", "otolith_innovation",
         [(26.5, 86), (23.5, 86), (23.5, 49), (11, 49), (11, 43.5)])
    edge(ax, source, "retinal", "eye_comp",
         [(34, 74.25), (34, 72.0), (25, 72.0), (25, 61), (22, 61)])

    # Visual estimation path.
    edge(ax, source, "eye_comp", "comparator", [(22, 61), (29.5, 61)],
         "$\\tilde y_{\\rm vis}$", (25.7, 63.0))
    edge(ax, source, "comparator", "kalman_update", [(48.5, 61), (63, 61)],
         "$K_{\\rm vis}\\epsilon_{\\rm vis}$", (55.8, 63.0))
    edge(ax, source, "kalman_update", "estimated_states", [(77, 61), (83, 61)])
    edge(ax, source, "estimated_states", "forward_prediction",
         [(91, 57.25), (91, 23.5), (42, 23.5), (42, 26)],
         "estimated state", (67, 22.2))
    edge(ax, source, "forward_prediction", "comparator", [(42, 33), (42, 56.75)],
         "$\\hat y_{\\rm vis}$", (45.3, 47.5))

    # Reliability and adaptation path.
    edge(ax, source, "otolith_innovation", "context", [(18, 40), (22.5, 40)])
    edge(ax, source, "context", "hypothesis_B", [(39.5, 40), (44, 40), (44, 44), (46.5, 44)])
    edge(ax, source, "context", "hypothesis_R", [(39.5, 40), (44, 40), (44, 36.5), (46.5, 36.5)])
    edge(ax, source, "hypothesis_B", "weighted_variance", [(57.5, 44), (62, 44), (62, 40), (66, 40)])
    edge(ax, source, "hypothesis_R", "weighted_variance", [(57.5, 36.5), (62, 36.5), (62, 40), (66, 40)])
    edge(ax, source, "weighted_variance", "kalman_update", [(76, 44.25), (76, 53.5), (72, 57.5)],
         "$R_{\\rm oto,t}$", (80.0, 50.7))
    edge(ax, source, "context", "adaptation", [(31, 36.5), (31, 24.7), (67, 24.7), (67, 25.75)],
         "context gate", (47.5, 23.4), linestyle="--")
    edge(ax, source, "comparator", "adaptation",
         [(47, 56.75), (47, 52.0), (90, 52.0), (90, 29.5), (78, 29.5)],
         "$\\epsilon_{\\rm vis}$", (92.2, 42.0))
    edge(ax, source, "adaptation", "forward_prediction", [(56, 29.5), (50.5, 29.5)],
         "$A_t$", (53.0, 31.4))

    # Gaze feedback.  The forward model never points to the retinal observation.
    edge(ax, source, "estimated_states", "vor_okr", [(96, 57.25), (96, 19.0), (30, 19.0), (30, 14.75)])
    edge(ax, source, "vor_okr", "eye_plant", [(37.5, 11.5), (45.5, 11.5)],
         "commands", (41.5, 13.3))
    edge(ax, source, "eye_plant", "eye_velocity", [(58.5, 11.5), (68.5, 11.5)])
    edge(ax, source, "eye_velocity", "eye_comp",
         [(75, 14.5), (75, 17.0), (1, 17.0), (1, 61), (4, 61)],
         "+ eye-velocity\nrestoration", (10.0, 33.2))
    edge(ax, source, "eye_velocity", "retinal",
         [(75, 8.5), (75, 3.2), (99, 3.2), (99, 77.5), (41.5, 77.5)],
         "$-$ eye-velocity contribution", (68.5, 79.1))

    # Compact category key below the lanes.
    key_positions = [(5, -1.4), (37, -1.4), (68, -1.4),
                     (5, -3.8), (37, -3.8), (68, -3.8)]
    for (x, y), (category, (face, edge_color)) in zip(key_positions, CATEGORIES.items()):
        ax.add_patch(Rectangle((x - 1.0, y - 0.9), 2.0, 1.8, facecolor=face,
                               edgecolor=edge_color, linewidth=0.8, clip_on=False))
        ax.text(x + 1.5, y, category, fontsize=6.8, va="center", ha="left", clip_on=False)

    write_csv(STEM.with_name(STEM.name + "_source_data.csv"), source)
    save_figure(fig, STEM)
    write_qc_stub(STEM, DIMENSIONS, [
        "Diagram-specific check: all connectors use explicit routed coordinates; no connector is intended to cross a node or text label.",
        "Causal semantics: forward prediction feeds only the comparator; OKR/VOR feed only the eye plant and retinal-feedback path.",
    ])


if __name__ == "__main__":
    build()
