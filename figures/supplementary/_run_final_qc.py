"""Automated integrity checks for the final figure package."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader


FINAL = Path(__file__).resolve().parent
ANALYSIS = FINAL.parents[1]
SPECS = {
    "Figure_1_model_architecture": (6.70, 7.25),
    "Figure_2_structural_robustness": (6.70, 5.25),
    "Figure_3_reliability_sweep": (6.70, 3.60),
    "Figure_4_residual_routing": (6.70, 6.55),
    "Figure_S1_structural_object_rmse": (6.70, 4.95),
    "Figure_S2_absolute_implementation_comparison": (6.70, 3.55),
    "Figure_S3_context_okr": (6.70, 3.90),
}


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def segment_intersects_rect(a, b, rect):
    x1, y1, x2, y2 = rect
    if a[0] == b[0]:
        return x1 < a[0] < x2 and max(min(a[1], b[1]), y1) < min(max(a[1], b[1]), y2)
    if a[1] == b[1]:
        return y1 < a[1] < y2 and max(min(a[0], b[0]), x1) < min(max(a[0], b[0]), x2)
    # Figure 1 has one intentionally diagonal, unobstructed segment.
    for t in np.linspace(0.01, 0.99, 100):
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if x1 < x < x2 and y1 < y < y2:
            return True
    return False


report = {"status": "PASS", "figures": {}, "data_checks": {}}
for stem, dimensions in SPECS.items():
    expected = [
        FINAL / f"{stem}.pdf", FINAL / f"{stem}.svg", FINAL / f"{stem}.png",
        FINAL / f"{stem}.py", FINAL / f"{stem}_source_data.csv", FINAL / f"{stem}_QC.md",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected), stem

    pdf = PdfReader(str(FINAL / f"{stem}.pdf"))
    assert len(pdf.pages) == 1
    page = pdf.pages[0]
    pdf_size = (float(page.mediabox.width) / 72, float(page.mediabox.height) / 72)
    assert np.allclose(pdf_size, dimensions, atol=0.002)
    text = page.extract_text() or ""
    assert len(text.strip()) > 20

    svg = (FINAL / f"{stem}.svg").read_text(encoding="utf-8")
    assert "<text" in svg and "<image" not in svg

    with Image.open(FINAL / f"{stem}.png") as png:
        png_size = png.size
        dpi = tuple(float(value) for value in png.info.get("dpi", (0, 0)))
    expected_pixels = tuple(round(value * 600) for value in dimensions)
    assert png_size == expected_pixels, (stem, png_size, expected_pixels)
    assert all(abs(value - 600) < 0.01 for value in dpi), (stem, dpi)

    report["figures"][stem] = {
        "pdf_dimensions_in": [round(value, 4) for value in pdf_size],
        "png_pixels": list(png_size),
        "png_dpi_metadata": [round(value, 4) for value in dpi],
        "pdf_text_extractable": True,
        "svg_text_elements": True,
        "svg_embedded_raster_images": False,
    }

# Figure 1: ensure every route avoids every non-endpoint node rectangle.
architecture = rows(FINAL / "Figure_1_model_architecture_source_data.csv")
nodes = {
    row["id"]: (
        float(row["x"]) - float(row["width"]) / 2,
        float(row["y"]) - float(row["height"]) / 2,
        float(row["x"]) + float(row["width"]) / 2,
        float(row["y"]) + float(row["height"]) / 2,
    )
    for row in architecture if row["record_type"] == "node"
}
crossings = []
for row in architecture:
    if row["record_type"] != "edge":
        continue
    route = [tuple(map(float, point.split(","))) for point in row["route"].split(";")]
    for node_id, rect in nodes.items():
        if node_id in (row["from"], row["to"]):
            continue
        if any(segment_intersects_rect(a, b, rect) for a, b in zip(route, route[1:])):
            crossings.append((row["id"], node_id))
assert not crossings, crossings
report["data_checks"]["figure_1_nonendpoint_box_crossings"] = 0

# Figures 2 and S1: all plotted contrast values are exact upstream copies.
control = rows(ANALYSIS / "03_structural_controls" / "control_paired_contrasts_v1.csv")
control_lookup = {(row["variant"], row["metric"]): row for row in control}
for stem in ("Figure_2_structural_robustness", "Figure_S1_structural_object_rmse"):
    for row in rows(FINAL / f"{stem}_source_data.csv"):
        upstream = control_lookup[(row["variant"], row["metric"])]
        assert close(row["point_estimate"], upstream["paired_absolute_change_mean"])
        assert close(row["bootstrap_ci_low"], upstream["paired_absolute_change_ci_low"])
        assert close(row["bootstrap_ci_high"], upstream["paired_absolute_change_ci_high"])
report["data_checks"]["forest_intervals_match_upstream"] = True

# Figure 3: direct agreement with the completed reliability summary.
reliability = [
    row for row in rows(ANALYSIS / "04_reliability_sweep" / "reliability_sweep_extended_summary_v1.csv")
    if row["metric"] == "false_positive_rms"
]
final_reliability = rows(FINAL / "Figure_3_reliability_sweep_source_data.csv")
assert len(reliability) == len(final_reliability)
for final_row, upstream in zip(final_reliability, reliability):
    for field in ("kappa", "sigma_oto", "manipulated_mean", "mean_bootstrap_ci_low",
                  "mean_bootstrap_ci_high", "paired_percent_change_mean",
                  "paired_percent_change_ci_low", "paired_percent_change_ci_high"):
        assert close(final_row[field], upstream[field])
report["data_checks"]["reliability_intervals_match_upstream"] = True

# Figure 4: summary panels, exact decomposition, and requested absolute values.
summary = rows(ANALYSIS / "02_mechanism" / "residual_routing_summary_v1.csv")
summary_lookup = {(row["initialization"], row["epoch"], row["metric"]): row for row in summary}
figure4 = rows(FINAL / "Figure_4_residual_routing_source_data.csv")
for row in figure4:
    if row["panel"] in ("A", "B", "C", "D"):
        upstream = summary_lookup[(row["initialization"], row["epoch"], row["metric"])]
        assert close(row["mean"], upstream["mean"])
        assert close(row["bootstrap_ci_low"], upstream["bootstrap_ci_low"])
        assert close(row["bootstrap_ci_high"], upstream["bootstrap_ci_high"])

decomp = rows(ANALYSIS / "02_mechanism" / "residual_routing_exact_decomposition_v1.csv")
for initialization in ("canonical_corrected_initialization", "inflated_1p8_initialization"):
    subset = [row for row in decomp if row["initialization"] == initialization]
    for term in ("mean_innovation_term", "mean_gain_term", "mean_interaction_term"):
        final_row = next(row for row in figure4 if row["panel"] == "E"
                         and row["initialization"] == initialization and row["metric"] == term)
        assert close(final_row["mean"], np.mean([float(row[term]) for row in subset]))
    for row in subset:
        summed = float(row["mean_innovation_term"]) + float(row["mean_gain_term"]) + float(row["mean_interaction_term"])
        assert close(summed, row["mean_observed_change"], tol=1e-14)

epoch = rows(ANALYSIS / "02_mechanism" / "mechanism_trial_epoch_metrics_v1.csv")
for row in [row for row in figure4 if row["panel"] == "F"]:
    values = [float(item["false_positive_rms"]) for item in epoch
              if item["variant"] == row["initialization"] and item["epoch"] == row["epoch"]]
    assert close(row["mean"], np.mean(values))
f_values = {(row["initialization"], row["epoch"]): float(row["mean"])
            for row in figure4 if row["panel"] == "F"}
assert round(f_values[("canonical_corrected_initialization", "baseline")], 5) == 0.05778
assert round(f_values[("canonical_corrected_initialization", "reduced_reliability")], 5) == 0.08510
assert round(f_values[("inflated_1p8_initialization", "baseline")], 5) == 0.09682
assert round(f_values[("inflated_1p8_initialization", "reduced_reliability")], 5) == 0.08510
for initialization in ("canonical_corrected_initialization", "inflated_1p8_initialization"):
    abs_innovation = [float(summary_lookup[(initialization, epoch_name, "mean_abs_epsilon_vis")]["mean"])
                      for epoch_name in ("baseline", "reduced_reliability")]
    abs_correction = [float(summary_lookup[(initialization, epoch_name, "mean_abs_delta_v_obj_hat")]["mean"])
                      for epoch_name in ("baseline", "reduced_reliability")]
    assert abs_innovation[1] <= abs_innovation[0]
    assert abs_correction[1] <= abs_correction[0]
report["data_checks"]["figure_4_exact_decomposition_identity"] = True
report["data_checks"]["figure_4_panel_f_requested_values"] = True
report["data_checks"]["figure_4_absolute_innovation_and_correction_do_not_increase"] = True

# S2: every plotted mean equals the corresponding raw-data aggregation.
s2 = rows(FINAL / "Figure_S2_absolute_implementation_comparison_source_data.csv")
control_epoch = rows(ANALYSIS / "03_structural_controls" / "control_trial_epoch_metrics_v1.csv")
for row in s2:
    values = [
        float(item[row["metric"]]) for item in control_epoch
        if item["variant"] == row["variant"] and item["epoch"] == row["epoch"]
        and item[row["metric"]] not in ("", "nan")
    ]
    assert close(row["mean"], np.mean(values))
report["data_checks"]["supplement_s2_means_match_upstream"] = True

# S3: committed final source data are complete and self-describing.
s3 = rows(FINAL / "Figure_S3_context_okr_source_data.csv")
assert len(s3) == 5000
assert {row["panel"] for row in s3} == {"A", "B"}
assert {row["metric"] for row in s3} == {"g_okr", "p_hist"}
assert all(row.get("realization_ids") for row in s3)
report["data_checks"]["supplement_s3_source_data_complete"] = True

(FINAL / "_automated_qc.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, sort_keys=True))
