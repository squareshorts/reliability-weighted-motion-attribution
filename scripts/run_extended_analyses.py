r"""Run the extended analyses for the otolith reliability study.

Run from the repository root with:

    python scripts/run_extended_analyses.py

The script is deterministic, uses paired realization identities 0..29 for the
principal analyses and 0..99 for the continuous reliability sweep, and writes
only to results/extended and figures/extended.
"""
from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
for path in (CODE_DIR, SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cerebellum.config import Config  # noqa: E402
from cerebellum import metrics as mx  # noqa: E402
from cerebellum import stimulus as stim  # noqa: E402
from cerebellum.experiment import run_trial  # noqa: E402
from model_variants import (  # noqa: E402
    MODEL_DEFINITIONS,
    canonical_core_run,
    run_model,
    single_timescale_mapping,
)


RESULTS = REPO_ROOT / "results" / "extended"
FIGURES = REPO_ROOT / "figures" / "extended"
DIAGNOSTICS = RESULTS / "validation"
LOGS = RESULTS / "logs"
for directory in (RESULTS, FIGURES, DIAGNOSTICS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

N_PRINCIPAL = 30
N_SWEEP = 100
N_BOOT = 10_000
BOOTSTRAP_SEED = 20260828
KAPPA_GRID = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)
EPOCHS = {
    "baseline": lambda c: np.arange(0, c.micro_start),
    "reduced_reliability": lambda c: np.arange(c.micro_start, c.micro_end),
    "recovery": lambda c: np.arange(c.micro_end, c.T),
}
MODEL_ORDER = tuple(MODEL_DEFINITIONS)
MODEL_SHORT = {
    "A_full_canonical": "A Full",
    "B_fixed_context": "B Fixed context",
    "C_no_adaptation": "C No adaptation",
    "D_single_timescale": "D Single rate",
    "E_residual_only": "E Residual proxy",
    "F_oracle_self_motion": "F Oracle self",
    "G_no_eye_feedback": "G No eye",
}
COLORS = {
    "baseline": "#2878B5",
    "reduced_reliability": "#D9534F",
    "recovery": "#3A923A",
}


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    try:
        with (LOGS / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except OSError:
        pass


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values, seed_offset: int = 0, n_boot: int = N_BOOT) -> tuple[float, float]:
    arr = finite(values)
    if not len(arr):
        return (np.nan, np.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    # Chunked means keep memory modest for the 100-realization sweep.
    means = np.empty(n_boot, dtype=float)
    chunk = 1000
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        idx = rng.integers(0, len(arr), size=(stop - start, len(arr)))
        means[start:stop] = arr[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(values, seed_offset: int = 0) -> dict:
    arr = finite(values)
    lo, hi = bootstrap_ci(arr, seed_offset)
    return {
        "mean": float(arr.mean()) if len(arr) else np.nan,
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "q25": float(np.quantile(arr, 0.25)) if len(arr) else np.nan,
        "q75": float(np.quantile(arr, 0.75)) if len(arr) else np.nan,
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(arr)),
    }


def active_masks(out: dict, cfg: Config, epoch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = EPOCHS[epoch](cfg)
    epoch_mask = np.zeros(cfg.T, dtype=bool)
    epoch_mask[indices] = True
    active = np.abs(out["x_true"][2]) > 1e-3
    return epoch_mask, epoch_mask & active, epoch_mask & ~active


def epoch_metrics(out: dict, cfg: Config, model_id: str, realization_id: int) -> list[dict]:
    rows = []
    for epoch in EPOCHS:
        mask, present, free = active_masks(out, cfg, epoch)
        estimate = out["v_obj_hat"]
        assumed_sigma = out.get("assumed_sigma_oto", np.full(cfg.T, np.nan))
        innov_vis = out.get("innov_vis", np.full(cfg.T, np.nan))
        object_gain = out.get("K_obj_vis", np.full(cfg.T, np.nan))
        object_correction = out.get("delta_v_obj_vis", np.full(cfg.T, np.nan))
        rows.append({
            "model_id": model_id,
            "model_definition": MODEL_DEFINITIONS.get(model_id, model_id),
            "realization_id": realization_id,
            "rng_seed": 100 + realization_id,
            "epoch": epoch,
            "angular_rmse": mx.rmse(out["x_hat"][0, mask], out["x_true"][0, mask]),
            "object_rmse": mx.rmse(estimate[present], out["x_true"][2, present]),
            "false_positive_rms": float(np.sqrt(np.mean(estimate[free] ** 2))),
            "mean_context_belief": float(np.mean(out["p_hist"][mask])),
            "median_context_belief": float(np.median(out["p_hist"][mask])),
            "mean_assumed_sigma_oto": float(np.nanmean(assumed_sigma[mask])) if np.isfinite(assumed_sigma[mask]).any() else np.nan,
            "mean_signed_innovation": float(np.nanmean(innov_vis[mask])) if np.isfinite(innov_vis[mask]).any() else np.nan,
            "mean_abs_innovation": float(np.nanmean(np.abs(innov_vis[mask]))) if np.isfinite(innov_vis[mask]).any() else np.nan,
            "mean_object_gain": float(np.nanmean(object_gain[mask])) if np.isfinite(object_gain[mask]).any() else np.nan,
            "mean_signed_object_correction": float(np.nanmean(object_correction[mask])) if np.isfinite(object_correction[mask]).any() else np.nan,
            "mean_abs_object_correction": float(np.nanmean(np.abs(object_correction[mask]))) if np.isfinite(object_correction[mask]).any() else np.nan,
            "metric_homology": out.get("metric_homology", "recurrent latent object-motion state"),
            "object_present_samples": int(present.sum()),
            "object_free_samples": int(free.sum()),
        })
    return rows


def paired_summary(seed_df: pd.DataFrame, group_col: str, group_value: str, metric: str, offset: int) -> dict:
    sub = seed_df[(seed_df[group_col] == group_value)]
    pivot = sub.pivot(index="realization_id", columns="epoch", values=metric)
    before = pivot["baseline"].to_numpy(float)
    after = pivot["reduced_reliability"].to_numpy(float)
    delta = after - before
    interpretable_pct = bool(np.nanmean(np.abs(before)) >= 1e-6)
    pct = 100.0 * delta / before if interpretable_pct else np.full_like(delta, np.nan)
    dlo, dhi = bootstrap_ci(delta, offset)
    plo, phi = bootstrap_ci(pct, offset + 1)
    return {
        "baseline_mean": float(np.mean(before)),
        "reduced_mean": float(np.mean(after)),
        "paired_absolute_change_mean": float(np.mean(delta)),
        "paired_absolute_change_ci_low": dlo,
        "paired_absolute_change_ci_high": dhi,
        "mean_within_realization_percent_change": float(np.nanmean(pct)) if interpretable_pct else np.nan,
        "percent_change_ci_low": plo,
        "percent_change_ci_high": phi,
        "percent_metric_interpretable": interpretable_pct,
        "sign_consistency_positive": float(np.mean(delta > 0)),
        "n_realizations": int(len(delta)),
        "seed_identities": ";".join(str(int(x)) for x in pivot.index),
    }


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig, filename: str) -> None:
    fig.savefig(FIGURES / filename, facecolor="white")
    plt.close(fig)


def run_principal_models():
    log("Running canonical reproduction and Models A-G (30 paired realizations)")
    cfg = Config(n_trials=N_PRINCIPAL)
    seed_rows = []
    canonical_runs = []
    single_runs = []
    equivalence = []
    no_eye_equivalence = []
    model_runs_for_equivalence = {}
    for model_index, model_id in enumerate(MODEL_ORDER):
        log(f"  {model_id}")
        for rid in range(N_PRINCIPAL):
            out = run_model(cfg, rid, model_id)
            seed_rows.extend(epoch_metrics(out, cfg, model_id, rid))
            if model_id == "A_full_canonical":
                core = canonical_core_run(cfg, rid)
                shared = ("x_hat", "p_hist", "R_oto", "eye_vel", "innov_vis")
                max_error = max(float(np.max(np.abs(out[name] - core[name]))) for name in shared)
                equivalence.append({
                    "realization_id": rid,
                    "rng_seed": 100 + rid,
                    "max_abs_shared_trace_error": max_error,
                    "bitwise_x_hat_equal": bool(np.array_equal(out["x_hat"], core["x_hat"])),
                })
                canonical_runs.append(out)
            elif model_id == "D_single_timescale":
                single_runs.append(out)
            elif model_id == "G_no_eye_feedback":
                no_eye_equivalence.append({
                    "realization_id": rid,
                    "max_abs_x_hat_difference": float(np.max(np.abs(out["x_hat"] - canonical_runs[rid]["x_hat"]))),
                    "max_abs_compensated_visual_difference": float(np.max(np.abs(out["y_vis_comp"] - canonical_runs[rid]["y_vis_comp"]))),
                    "raw_retinal_paths_equal": bool(np.array_equal(out["y_vis_raw"], canonical_runs[rid]["y_vis_raw"])),
                })
            if rid == 0:
                model_runs_for_equivalence[model_id] = out

    seed_df = pd.DataFrame(seed_rows)
    write_csv(RESULTS / "model_comparison_seedlevel.csv", seed_df)
    write_csv(DIAGNOSTICS / "instrumentation_equivalence.csv", equivalence)
    if not all(row["bitwise_x_hat_equal"] for row in equivalence):
        raise AssertionError("Extended-analysis instrumentation changed canonical x_hat")

    summary_rows = []
    offset = 100
    for model_id in MODEL_ORDER:
        for metric in ("angular_rmse", "object_rmse", "false_positive_rms"):
            row = {
                "model_id": model_id,
                "model_label": MODEL_SHORT[model_id],
                "model_definition": MODEL_DEFINITIONS[model_id],
                "metric": metric,
                **paired_summary(seed_df, "model_id", model_id, metric, offset),
            }
            if model_id == "E_residual_only" and metric in ("object_rmse", "false_positive_rms"):
                row["metric_homology"] = "instantaneous residual-based external-motion proxy"
            else:
                row["metric_homology"] = "canonical state metric"
            summary_rows.append(row)
            offset += 3
    summary_df = pd.DataFrame(summary_rows)
    write_csv(RESULTS / "model_comparison.csv", summary_df)

    # Exact cancellation check for the eye plant.
    eye_equivalence = {
        "max_abs_x_hat_difference_seed0": no_eye_equivalence[0]["max_abs_x_hat_difference"],
        "max_abs_compensated_visual_difference_seed0": no_eye_equivalence[0]["max_abs_compensated_visual_difference"],
        "max_abs_x_hat_difference_all_30": max(row["max_abs_x_hat_difference"] for row in no_eye_equivalence),
        "max_abs_compensated_visual_difference_all_30": max(row["max_abs_compensated_visual_difference"] for row in no_eye_equivalence),
        "raw_retinal_paths_differ": bool(any(not row["raw_retinal_paths_equal"] for row in no_eye_equivalence)),
    }
    write_json(DIAGNOSTICS / "eye_cancellation_check.json", eye_equivalence)
    write_csv(DIAGNOSTICS / "eye_cancellation_seedlevel.csv", no_eye_equivalence)
    if eye_equivalence["max_abs_x_hat_difference_all_30"] > 1e-12:
        raise AssertionError("Zero-eye control changed canonical compensated estimator")
    plot_model_comparison(summary_df)
    return cfg, seed_df, summary_df, canonical_runs, single_runs, equivalence, eye_equivalence


def plot_model_comparison(summary_df: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), sharex=True)
    metrics = (
        ("angular_rmse", "Angular self-motion RMSE"),
        ("object_rmse", "Object/external-motion RMSE"),
        ("false_positive_rms", "False-positive world-motion RMS"),
    )
    x = np.arange(len(MODEL_ORDER))
    for ax, (metric, title) in zip(axes, metrics):
        sub = summary_df[summary_df.metric == metric].set_index("model_id").loc[list(MODEL_ORDER)]
        y = sub["mean_within_realization_percent_change"].to_numpy(float)
        lo = sub["percent_change_ci_low"].to_numpy(float)
        hi = sub["percent_change_ci_high"].to_numpy(float)
        ok = np.isfinite(y)
        ax.errorbar(x[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]], fmt="o", capsize=3, color="#333333")
        for missing_x in x[~ok]:
            ax.text(missing_x, 0.5, "n/a", ha="center", va="bottom", fontsize=8, color="#666666")
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_title(title)
        ax.set_xticks(x, [m.split("_", 1)[0] for m in MODEL_ORDER], rotation=0)
        ax.set_ylabel("Mean within-realization change (%)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Structural model comparison", fontsize=12, fontweight="bold")
    mapping = "A Full canonical | B Fixed context | C No adaptation | D Single timescale | E Residual-only proxy | F Oracle self-motion | G No eye feedback"
    fig.text(0.5, 0.03, mapping, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(bottom=0.18, wspace=0.35)
    save_figure(fig, "model_comparison.png")


def _context_level_worker(kappa: float) -> list[dict]:
    seed_rows = []
    cfg = Config(n_trials=N_SWEEP, g_micro=kappa)
    for rid in range(N_SWEEP):
        out = run_model(cfg, rid, "A_full_canonical")
        metrics = epoch_metrics(out, cfg, "A_full_canonical", rid)
        lookup = {row["epoch"]: row for row in metrics}
        reduced = lookup["reduced_reliability"]
        baseline = lookup["baseline"]
        mask = np.zeros(cfg.T, dtype=bool)
        mask[cfg.micro_start:cfg.micro_end] = True
        row = {
            "kappa": kappa,
            "realization_id": rid,
            "rng_seed": 100 + rid,
            "true_generative_sigma_oto": cfg.sigma_oto_base / kappa,
            "internal_baseline_sigma_hypothesis": cfg.sigma_oto_earth_hyp,
            "internal_reduced_sigma_hypothesis": cfg.sigma_oto_micro_hyp,
            "mean_context_belief": float(np.mean(out["p_hist"][mask])),
            "median_context_belief": float(np.median(out["p_hist"][mask])),
            "mean_effective_assumed_sigma_oto": float(np.mean(out["assumed_sigma_oto"][mask])),
            "median_effective_assumed_sigma_oto": float(np.median(out["assumed_sigma_oto"][mask])),
        }
        for metric in ("angular_rmse", "object_rmse", "false_positive_rms"):
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"reduced_{metric}"] = reduced[metric]
            row[f"paired_absolute_change_{metric}"] = reduced[metric] - baseline[metric]
            row[f"within_realization_percent_change_{metric}"] = 100.0 * (reduced[metric] - baseline[metric]) / baseline[metric]
        seed_rows.append(row)
    return seed_rows


def run_context_mapping():
    log("Running the 100-realization continuous reliability/context mapping")
    seed_rows = []
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_context_level_worker, kappa): kappa for kappa in KAPPA_GRID}
        for future in as_completed(futures):
            kappa = futures[future]
            seed_rows.extend(future.result())
            log(f"  completed kappa={kappa:.2f}")
    seed_rows.sort(key=lambda row: (row["kappa"], row["realization_id"]))
    seed_df = pd.DataFrame(seed_rows)
    write_csv(RESULTS / "context_mapping_seedlevel.csv", seed_df)

    rows = []
    for ki, kappa in enumerate(KAPPA_GRID):
        sub = seed_df[seed_df.kappa == kappa]
        true_sigma = float(sub.true_generative_sigma_oto.iloc[0])
        if math.isclose(true_sigma, Config().sigma_oto_earth_hyp, abs_tol=1e-12):
            relation = "matches_internal_baseline_hypothesis"
        elif math.isclose(true_sigma, Config().sigma_oto_micro_hyp, abs_tol=1e-12):
            relation = "matches_internal_reduced_hypothesis"
        elif Config().sigma_oto_earth_hyp < true_sigma < Config().sigma_oto_micro_hyp:
            relation = "interpolates_between_internal_hypotheses"
        elif true_sigma > Config().sigma_oto_micro_hyp:
            relation = "extrapolates_beyond_internal_reduced_hypothesis"
        else:
            relation = "extrapolates_beyond_internal_baseline_hypothesis"
        row = {
            "kappa": kappa,
            "true_generative_sigma_oto": true_sigma,
            "internal_baseline_sigma_hypothesis": Config().sigma_oto_earth_hyp,
            "internal_reduced_sigma_hypothesis": Config().sigma_oto_micro_hyp,
            "hypothesis_relation": relation,
            "n_realizations": len(sub),
            "seed_identities": ";".join(str(x) for x in range(N_SWEEP)),
        }
        for col in (
            "mean_context_belief", "median_context_belief",
            "mean_effective_assumed_sigma_oto", "median_effective_assumed_sigma_oto",
        ):
            stats = summarize(sub[col], 1000 + ki * 40 + len(row))
            row[col] = stats["mean"] if col.startswith("mean") else stats["median"]
            row[f"{col}_ci_low"] = stats["ci_low"]
            row[f"{col}_ci_high"] = stats["ci_high"]
        for metric in ("angular_rmse", "object_rmse", "false_positive_rms"):
            for prefix in ("baseline", "reduced"):
                vals = sub[f"{prefix}_{metric}"]
                stats = summarize(vals, 1200 + ki * 50 + len(row))
                row[f"{prefix}_{metric}_mean"] = stats["mean"]
                row[f"{prefix}_{metric}_ci_low"] = stats["ci_low"]
                row[f"{prefix}_{metric}_ci_high"] = stats["ci_high"]
            for change_type in ("paired_absolute_change", "within_realization_percent_change"):
                vals = sub[f"{change_type}_{metric}"]
                stats = summarize(vals, 1400 + ki * 50 + len(row))
                row[f"{change_type}_{metric}_mean"] = stats["mean"]
                row[f"{change_type}_{metric}_ci_low"] = stats["ci_low"]
                row[f"{change_type}_{metric}_ci_high"] = stats["ci_high"]
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    write_csv(RESULTS / "context_mapping.csv", summary_df)
    plot_context_mapping(summary_df)
    return seed_df, summary_df


def plot_context_mapping(df: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    x = df.true_generative_sigma_oto.to_numpy(float)
    ax = axes[0, 0]
    ax.errorbar(x, df.mean_context_belief, yerr=[df.mean_context_belief - df.mean_context_belief_ci_low, df.mean_context_belief_ci_high - df.mean_context_belief], marker="o", capsize=2)
    ax.set(xlabel="True generative otolith noise SD (model units)", ylabel="Mean approximate context belief", ylim=(-0.03, 1.03))
    ax.axvline(0.06, color="#777777", ls="--", lw=0.8)
    ax.axvline(0.40, color="#777777", ls="--", lw=0.8)
    ax.set_title("Binary-context belief")

    ax = axes[0, 1]
    ax.errorbar(x, df.mean_effective_assumed_sigma_oto, yerr=[df.mean_effective_assumed_sigma_oto - df.mean_effective_assumed_sigma_oto_ci_low, df.mean_effective_assumed_sigma_oto_ci_high - df.mean_effective_assumed_sigma_oto], marker="o", capsize=2, color="#7B3F98")
    ax.plot(x, x, color="#999999", ls=":", label="Matched observer")
    ax.axhline(0.40, color="#777777", ls="--", lw=0.8, label="Internal reduced hypothesis")
    ax.set(xlabel="True generative otolith noise SD (model units)", ylabel="Mean internally assumed noise SD")
    ax.set_title("Effective assumed variance mixture")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for metric, label, color in (
        ("angular_rmse", "Angular", "#2878B5"),
        ("object_rmse", "Object", "#E07A1F"),
        ("false_positive_rms", "False positive", "#D9534F"),
    ):
        y = df[f"within_realization_percent_change_{metric}_mean"]
        lo = df[f"within_realization_percent_change_{metric}_ci_low"]
        hi = df[f"within_realization_percent_change_{metric}_ci_high"]
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=2, label=label, color=color)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set(xlabel="True generative otolith noise SD (model units)", ylabel="Mean within-realization change (%)")
    ax.set_title("Reliability effects")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for metric, label, color in (
        ("angular_rmse", "Angular RMSE", "#2878B5"),
        ("object_rmse", "Object RMSE", "#E07A1F"),
        ("false_positive_rms", "False-positive RMS", "#D9534F"),
    ):
        y = df[f"reduced_{metric}_mean"]
        lo = df[f"reduced_{metric}_ci_low"]
        hi = df[f"reduced_{metric}_ci_high"]
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=2, label=label, color=color)
    ax.set(xlabel="True generative otolith noise SD (model units)", ylabel="Manipulated-epoch error (model units)")
    ax.set_title("Absolute errors")
    ax.legend(frameon=False)
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
    fig.suptitle("Generative reliability versus two-hypothesis context mapping", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "context_mapping.png")


def _window_value(out: dict, cfg: Config, metric: str, indices: np.ndarray) -> float:
    mask = np.zeros(cfg.T, dtype=bool)
    mask[indices] = True
    active = np.abs(out["x_true"][2]) > 1e-3
    if metric == "context_belief":
        return float(np.mean(out["p_hist"][mask]))
    if metric == "assumed_sigma_oto":
        return float(np.mean(out["assumed_sigma_oto"][mask]))
    if metric == "angular_rmse":
        return mx.rmse(out["x_hat"][0, mask], out["x_true"][0, mask])
    if metric == "object_rmse":
        use = mask & active
        return mx.rmse(out["v_obj_hat"][use], out["x_true"][2, use]) if use.any() else np.nan
    if metric == "false_positive_rms":
        use = mask & ~active
        return float(np.sqrt(np.mean(out["v_obj_hat"][use] ** 2))) if use.any() else np.nan
    if metric == "fast_adaptation_norm":
        return float(np.mean(np.linalg.norm(out["A_fast_effective"][:, mask], axis=0)))
    if metric == "slow_adaptation_norm":
        return float(np.mean(np.linalg.norm(out["A_slow_deviation_effective"][:, mask], axis=0)))
    if metric == "combined_adaptation_norm":
        return float(np.mean(np.linalg.norm(out["A_combined_deviation"][:, mask], axis=0)))
    raise ValueError(metric)


def _return_time(time_s, values, baseline, fraction: float, dwell_bins: int = 3) -> float:
    """First sustained entry within the requested fraction of baseline distance."""
    y = np.asarray(values, dtype=float)
    t = np.asarray(time_s, dtype=float)
    valid = np.isfinite(y)
    if not valid.any():
        return np.nan
    first = int(np.flatnonzero(valid)[0])
    initial_distance = abs(y[first] - baseline)
    if initial_distance <= 1e-12:
        return 0.0
    tolerance = (1.0 - fraction) * initial_distance
    inside = np.abs(y - baseline) <= tolerance
    for idx in range(first, len(y) - dwell_bins + 1):
        if np.all(inside[idx:idx + dwell_bins]):
            return float(t[idx])
    return np.nan


def run_recovery(canonical_runs: list[dict], cfg: Config):
    log("Computing time-resolved recovery and hysteresis summaries")
    bin_samples = int(round(0.10 / cfg.dt))
    start = cfg.micro_end
    bins = [(i, min(i + bin_samples, cfg.T)) for i in range(start, cfg.T, bin_samples)]
    metrics = (
        "context_belief", "assumed_sigma_oto", "angular_rmse", "object_rmse",
        "false_positive_rms", "fast_adaptation_norm", "slow_adaptation_norm",
        "combined_adaptation_norm",
    )
    timecourse_rows = []
    for bi, (lo_i, hi_i) in enumerate(bins):
        indices = np.arange(lo_i, hi_i)
        row = {
            "time_from_recovery_onset_s": float((lo_i - start) * cfg.dt),
            "bin_start_index": lo_i,
            "bin_end_index_exclusive": hi_i,
            "bin_width_s": float((hi_i - lo_i) * cfg.dt),
            "n_realizations": len(canonical_runs),
        }
        for mi, metric in enumerate(metrics):
            vals = [_window_value(out, cfg, metric, indices) for out in canonical_runs]
            stats = summarize(vals, 2000 + bi * 20 + mi)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]
            row[f"{metric}_n"] = stats["n"]
        timecourse_rows.append(row)
    time_df = pd.DataFrame(timecourse_rows)
    write_csv(RESULTS / "recovery_timecourse.csv", time_df)

    windows = {
        "baseline_ensemble": np.arange(0, cfg.micro_start),
        "recovery_onset_0_0p25s": np.arange(start, min(start + int(round(0.25 / cfg.dt)), cfg.T)),
        "early_recovery_0_2s": np.arange(start, min(start + int(round(2.0 / cfg.dt)), cfg.T)),
        "late_recovery_last_2s": np.arange(max(start, cfg.T - int(round(2.0 / cfg.dt))), cfg.T),
        "final_recovery_last_0p5s": np.arange(max(start, cfg.T - int(round(0.5 / cfg.dt))), cfg.T),
    }
    seed_rows = []
    summary_rows = []
    for wi, (window, indices) in enumerate(windows.items()):
        for mi, metric in enumerate(metrics):
            vals = []
            for rid, out in enumerate(canonical_runs):
                value = _window_value(out, cfg, metric, indices)
                vals.append(value)
                seed_rows.append({
                    "row_type": "window_seedlevel",
                    "window": window,
                    "metric": metric,
                    "realization_id": rid,
                    "rng_seed": 100 + rid,
                    "value": value,
                })
            stats = summarize(vals, 3000 + wi * 30 + mi)
            summary_rows.append({
                "row_type": "window_summary",
                "window": window,
                "metric": metric,
                "mean": stats["mean"],
                "sd": stats["sd"],
                "bootstrap_ci_low": stats["ci_low"],
                "bootstrap_ci_high": stats["ci_high"],
                "n_realizations": stats["n"],
                "window_start_s": float(indices[0] * cfg.dt),
                "window_end_s": float((indices[-1] + 1) * cfg.dt),
            })

    seed_df = pd.DataFrame(seed_rows)
    for mi, metric in enumerate(metrics):
        base = seed_df[(seed_df.window == "baseline_ensemble") & (seed_df.metric == metric)].sort_values("realization_id").value.to_numpy(float)
        late = seed_df[(seed_df.window == "late_recovery_last_2s") & (seed_df.metric == metric)].sort_values("realization_id").value.to_numpy(float)
        valid = np.isfinite(base) & np.isfinite(late)
        delta = late[valid] - base[valid]
        lo, hi = bootstrap_ci(delta, 4000 + mi)
        summary_rows.append({
            "row_type": "paired_baseline_vs_late_recovery",
            "window": "late_recovery_last_2s_minus_baseline_ensemble",
            "metric": metric,
            "baseline_mean": float(np.mean(base[valid])),
            "late_recovery_mean": float(np.mean(late[valid])),
            "paired_difference_mean": float(np.mean(delta)),
            "paired_difference_ci_low": lo,
            "paired_difference_ci_high": hi,
            "interval_includes_zero": bool(lo <= 0 <= hi),
            "n_realizations": int(valid.sum()),
        })

    # Nonparametric ensemble return times from the 0.10-s source curves.
    for mi, metric in enumerate(metrics):
        baseline_vals = seed_df[(seed_df.window == "baseline_ensemble") & (seed_df.metric == metric)].value
        baseline = float(np.nanmean(baseline_vals))
        y = time_df[f"{metric}_mean"].to_numpy(float)
        t = time_df.time_from_recovery_onset_s.to_numpy(float)
        summary_rows.append({
            "row_type": "nonparametric_return_time",
            "window": "recovery_timecourse_0p1s_bins_0p3s_dwell",
            "metric": metric,
            "baseline_mean": baseline,
            "recovery_onset_mean": float(y[0]) if len(y) else np.nan,
            "final_recovery_mean": float(y[-1]) if len(y) else np.nan,
            "time_to_50_percent_return_s": _return_time(t, y, baseline, 0.50),
            "time_to_90_percent_return_s": _return_time(t, y, baseline, 0.90),
            "definition": "first 0.10-s bin beginning a sustained 0.30-s interval within 50% or 10% of the onset-to-baseline distance",
        })

    summary_df = pd.DataFrame(summary_rows)
    write_csv(RESULTS / "recovery_summary.csv", summary_df)
    write_csv(RESULTS / "recovery_summary_seedlevel.csv", seed_df)
    plot_recovery(time_df, cfg)
    return time_df, summary_df, seed_df


def plot_recovery(df: pd.DataFrame, cfg: Config) -> None:
    set_plot_style()
    t = df.time_from_recovery_onset_s.to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)

    ax = axes[0, 0]
    for metric, label, color in (
        ("context_belief", "Context belief", "#7B3F98"),
        ("assumed_sigma_oto", "Assumed otolith SD", "#C17C00"),
    ):
        y = df[f"{metric}_mean"].to_numpy(float)
        lo = df[f"{metric}_ci_low"].to_numpy(float)
        hi = df[f"{metric}_ci_high"].to_numpy(float)
        ax.plot(t, y, label=label, color=color)
        ax.fill_between(t, lo, hi, color=color, alpha=0.18)
    ax.set_ylabel("Belief / model-unit SD")
    ax.set_title("Context and assumed reliability")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for metric, label, color in (
        ("angular_rmse", "Angular RMSE", "#2878B5"),
        ("object_rmse", "Object RMSE", "#E07A1F"),
        ("false_positive_rms", "False-positive RMS", "#D9534F"),
    ):
        y = df[f"{metric}_mean"].to_numpy(float)
        lo = df[f"{metric}_ci_low"].to_numpy(float)
        hi = df[f"{metric}_ci_high"].to_numpy(float)
        ax.plot(t, y, label=label, color=color)
        ax.fill_between(t, lo, hi, color=color, alpha=0.13)
    ax.set_ylabel("Error (model units)")
    ax.set_title("Time-resolved error")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for metric, label, color in (
        ("fast_adaptation_norm", "Fast", "#2878B5"),
        ("slow_adaptation_norm", "Slow deviation", "#E07A1F"),
        ("combined_adaptation_norm", "Combined deviation", "#3A923A"),
    ):
        y = df[f"{metric}_mean"].to_numpy(float)
        lo = df[f"{metric}_ci_low"].to_numpy(float)
        hi = df[f"{metric}_ci_high"].to_numpy(float)
        ax.plot(t, y, label=label, color=color)
        ax.fill_between(t, lo, hi, color=color, alpha=0.12)
    ax.set(xlabel="Time after reliability restoration (s)", ylabel="Forward-model deviation norm")
    ax.set_title("Adaptive-state persistence")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    y = df["false_positive_rms_mean"].to_numpy(float)
    lo = df["false_positive_rms_ci_low"].to_numpy(float)
    hi = df["false_positive_rms_ci_high"].to_numpy(float)
    ax.plot(t, y, color="#D9534F")
    ax.fill_between(t, lo, hi, color="#D9534F", alpha=0.18)
    ax.set(xlabel="Time after reliability restoration (s)", ylabel="False-positive RMS (model units)")
    ax.set_title("Object-free recovery detail")

    for ax in axes.ravel():
        ax.grid(alpha=0.18)
        ax.set_xlim(0, max(t) if len(t) else 1)
    fig.suptitle("Recovery after restoration of baseline generative reliability", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "recovery_dynamics.png")


def event_latencies(out: dict, cfg: Config, epoch: str, threshold: float) -> list[float]:
    active = np.abs(out["x_true"][2]) > 1e-3
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    epoch_indices = set(int(i) for i in EPOCHS[epoch](cfg))
    horizon = int(round(1.0 / cfg.dt))
    values = []
    for onset in starts:
        if int(onset) not in epoch_indices:
            continue
        window = np.abs(out["v_obj_hat"][onset:min(cfg.T, onset + horizon)])
        hits = np.flatnonzero(window > threshold)
        values.append(np.nan if not len(hits) else float(hits[0] * cfg.dt))
    return values


def run_criterion_sensitivity(canonical_runs: list[dict], cfg: Config):
    log("Evaluating criterion thresholds 0.10-0.30")
    seed_rows = []
    pooled = {}
    for ti, threshold in enumerate(THRESHOLDS):
        for epoch in ("baseline", "reduced_reliability"):
            pooled[(threshold, epoch)] = []
            for rid, out in enumerate(canonical_runs):
                lats = np.asarray(event_latencies(out, cfg, epoch, threshold), dtype=float)
                pooled[(threshold, epoch)].extend(lats[np.isfinite(lats)].tolist())
                finite_lats = lats[np.isfinite(lats)]
                seed_rows.append({
                    "threshold": threshold,
                    "threshold_as_percent_of_event_amplitude": 100.0 * threshold / 0.40,
                    "epoch": epoch,
                    "realization_id": rid,
                    "rng_seed": 100 + rid,
                    "event_count": int(len(lats)),
                    "crossing_count": int(np.isfinite(lats).sum()),
                    "crossing_proportion": float(np.isfinite(lats).mean()) if len(lats) else np.nan,
                    "conditional_mean_latency_s": float(np.mean(finite_lats)) if len(finite_lats) else np.nan,
                    "conditional_median_latency_s": float(np.median(finite_lats)) if len(finite_lats) else np.nan,
                })
    seed_df = pd.DataFrame(seed_rows)
    write_csv(RESULTS / "criterion_sensitivity_seedlevel.csv", seed_df)

    rows = []
    for ti, threshold in enumerate(THRESHOLDS):
        for ei, epoch in enumerate(("baseline", "reduced_reliability")):
            sub = seed_df[(seed_df.threshold == threshold) & (seed_df.epoch == epoch)]
            prop = summarize(sub.crossing_proportion, 5000 + ti * 20 + ei)
            lat = summarize(sub.conditional_mean_latency_s, 5100 + ti * 20 + ei)
            pooled_lats = finite(pooled[(threshold, epoch)])
            rows.append({
                "row_type": "epoch_summary",
                "threshold": threshold,
                "threshold_as_percent_of_event_amplitude": 100.0 * threshold / 0.40,
                "epoch": epoch,
                "n_realizations": len(sub),
                "total_events": int(sub.event_count.sum()),
                "total_crossings": int(sub.crossing_count.sum()),
                "pooled_crossing_probability": float(sub.crossing_count.sum() / sub.event_count.sum()),
                "mean_seed_crossing_probability": prop["mean"],
                "crossing_probability_ci_low": prop["ci_low"],
                "crossing_probability_ci_high": prop["ci_high"],
                "conditional_latency_mean_s": float(np.mean(pooled_lats)) if len(pooled_lats) else np.nan,
                "conditional_latency_median_s": float(np.median(pooled_lats)) if len(pooled_lats) else np.nan,
                "conditional_latency_q25_s": float(np.quantile(pooled_lats, 0.25)) if len(pooled_lats) else np.nan,
                "conditional_latency_q75_s": float(np.quantile(pooled_lats, 0.75)) if len(pooled_lats) else np.nan,
                "mean_seed_conditional_latency_s": lat["mean"],
                "mean_seed_conditional_latency_ci_low": lat["ci_low"],
                "mean_seed_conditional_latency_ci_high": lat["ci_high"],
            })
        base = seed_df[(seed_df.threshold == threshold) & (seed_df.epoch == "baseline")].sort_values("realization_id")
        red = seed_df[(seed_df.threshold == threshold) & (seed_df.epoch == "reduced_reliability")].sort_values("realization_id")
        prop_delta = red.crossing_proportion.to_numpy(float) - base.crossing_proportion.to_numpy(float)
        lat_base = base.conditional_mean_latency_s.to_numpy(float)
        lat_red = red.conditional_mean_latency_s.to_numpy(float)
        valid = np.isfinite(lat_base) & np.isfinite(lat_red)
        lat_delta = lat_red[valid] - lat_base[valid]
        p_lo, p_hi = bootstrap_ci(prop_delta, 5200 + ti * 4)
        l_lo, l_hi = bootstrap_ci(lat_delta, 5201 + ti * 4)
        rows.append({
            "row_type": "paired_contrast",
            "threshold": threshold,
            "threshold_as_percent_of_event_amplitude": 100.0 * threshold / 0.40,
            "epoch": "reduced_reliability_minus_baseline",
            "paired_crossing_probability_difference": float(np.nanmean(prop_delta)),
            "paired_crossing_probability_difference_ci_low": p_lo,
            "paired_crossing_probability_difference_ci_high": p_hi,
            "paired_conditional_latency_difference_s": float(np.mean(lat_delta)) if len(lat_delta) else np.nan,
            "paired_conditional_latency_difference_ci_low": l_lo,
            "paired_conditional_latency_difference_ci_high": l_hi,
            "latency_effect_direction": "delayed" if len(lat_delta) and np.mean(lat_delta) > 0 else "earlier_or_no_change",
            "latency_conclusion_robust_at_threshold": bool(l_lo > 0),
            "n_paired_realizations_for_latency": int(len(lat_delta)),
            "noncrossings_retained_in_probability": True,
        })
    summary_df = pd.DataFrame(rows)
    write_csv(RESULTS / "criterion_sensitivity.csv", summary_df)
    plot_criterion(summary_df)
    return seed_df, summary_df


def plot_criterion(df: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    epochs = df[df.row_type == "epoch_summary"]
    for epoch in ("baseline", "reduced_reliability"):
        sub = epochs[epochs.epoch == epoch].sort_values("threshold")
        axes[0].errorbar(sub.threshold, sub.mean_seed_crossing_probability,
                         yerr=[sub.mean_seed_crossing_probability - sub.crossing_probability_ci_low,
                               sub.crossing_probability_ci_high - sub.mean_seed_crossing_probability],
                         marker="o", capsize=3, label=epoch.replace("_", " ").title(), color=COLORS[epoch])
    axes[0].set(xlabel="Criterion |object estimate| (model units)", ylabel="Crossing probability", ylim=(-0.03, 1.03), title="Crossing probability")
    axes[0].legend(frameon=False)

    contrast = df[df.row_type == "paired_contrast"].sort_values("threshold")
    y = contrast.paired_conditional_latency_difference_s.to_numpy(float)
    lo = contrast.paired_conditional_latency_difference_ci_low.to_numpy(float)
    hi = contrast.paired_conditional_latency_difference_ci_high.to_numpy(float)
    axes[1].errorbar(contrast.threshold, y, yerr=[y - lo, hi - y], marker="o", capsize=3, color="#333333")
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set(xlabel="Criterion |object estimate| (model units)", ylabel="Reduced - baseline conditional latency (s)", title="Paired latency difference")
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle("Criterion-threshold sensitivity", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "criterion_sensitivity.png")


def run_adaptation_timescales(canonical_runs: list[dict], single_runs: list[dict], cfg: Config):
    log("Quantifying fast, slow, and single-timescale adaptive trajectories")
    bin_samples = int(round(0.10 / cfg.dt))
    bins = [(i, min(i + bin_samples, cfg.T)) for i in range(0, cfg.T, bin_samples)]
    rows = []
    run_sets = {"A_full_canonical": canonical_runs, "D_single_timescale": single_runs}
    for model_id, runs in run_sets.items():
        for bi, (lo_i, hi_i) in enumerate(bins):
            idx = np.arange(lo_i, hi_i)
            per_signal = {"fast": [], "slow": [], "combined": []}
            for out in runs:
                per_signal["fast"].append(float(np.mean(np.linalg.norm(out["A_fast_effective"][:, idx], axis=0))))
                per_signal["slow"].append(float(np.mean(np.linalg.norm(out["A_slow_deviation_effective"][:, idx], axis=0))))
                per_signal["combined"].append(float(np.mean(np.linalg.norm(out["A_combined_deviation"][:, idx], axis=0))))
            row = {
                "row_type": "timecourse",
                "model_id": model_id,
                "time_s": float(lo_i * cfg.dt),
                "bin_width_s": float((hi_i - lo_i) * cfg.dt),
                "n_realizations": len(runs),
            }
            for si, signal in enumerate(("fast", "slow", "combined")):
                stats = summarize(per_signal[signal], 6000 + bi * 10 + si + (1000 if model_id.startswith("D") else 0))
                row[f"{signal}_norm_mean"] = stats["mean"]
                row[f"{signal}_norm_ci_low"] = stats["ci_low"]
                row[f"{signal}_norm_ci_high"] = stats["ci_high"]
            rows.append(row)

    transition_windows = {
        "baseline_late": np.arange(cfg.micro_start - 50, cfg.micro_start),
        "reduced_early": np.arange(cfg.micro_start, cfg.micro_start + 50),
        "reduced_late": np.arange(cfg.micro_end - 50, cfg.micro_end),
        "recovery_early": np.arange(cfg.micro_end, cfg.micro_end + 50),
        "recovery_late": np.arange(cfg.T - 50, cfg.T),
    }
    seed_rows = []
    for model_id, runs in run_sets.items():
        for window, idx in transition_windows.items():
            for rid, out in enumerate(runs):
                seed_rows.append({
                    "model_id": model_id,
                    "window": window,
                    "realization_id": rid,
                    "rng_seed": 100 + rid,
                    "fast_norm": float(np.mean(np.linalg.norm(out["A_fast_effective"][:, idx], axis=0))),
                    "slow_norm": float(np.mean(np.linalg.norm(out["A_slow_deviation_effective"][:, idx], axis=0))),
                    "combined_norm": float(np.mean(np.linalg.norm(out["A_combined_deviation"][:, idx], axis=0))),
                })
    seed_df = pd.DataFrame(seed_rows)
    write_csv(RESULTS / "adaptation_timescales_seedlevel.csv", seed_df)
    for model_id in run_sets:
        for window in transition_windows:
            sub = seed_df[(seed_df.model_id == model_id) & (seed_df.window == window)]
            row = {"row_type": "window_summary", "model_id": model_id, "window": window, "n_realizations": len(sub)}
            for si, signal in enumerate(("fast_norm", "slow_norm", "combined_norm")):
                stats = summarize(sub[signal], 7000 + len(rows) * 5 + si)
                row[f"{signal}_mean"] = stats["mean"]
                row[f"{signal}_ci_low"] = stats["ci_low"]
                row[f"{signal}_ci_high"] = stats["ci_high"]
            rows.append(row)
    result_df = pd.DataFrame(rows)
    write_csv(RESULTS / "adaptation_timescales.csv", result_df)
    plot_adaptation(result_df, cfg)
    return result_df, seed_df


def plot_adaptation(df: pd.DataFrame, cfg: Config) -> None:
    set_plot_style()
    tc = df[df.row_type == "timecourse"]
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.8), sharex=True)
    for ax, signal, title in zip(axes, ("fast", "slow", "combined"), ("Fast process", "Slow-process deviation", "Combined adaptive deviation")):
        for model_id, label, color in (
            ("A_full_canonical", "Canonical two-rate", "#2878B5"),
            ("D_single_timescale", "Single-timescale control", "#D9534F"),
        ):
            sub = tc[tc.model_id == model_id].sort_values("time_s")
            y = sub[f"{signal}_norm_mean"].to_numpy(float)
            if not np.isfinite(y).any() or (signal in ("fast", "slow") and model_id == "D_single_timescale" and np.nanmax(np.abs(y)) == 0):
                continue
            lo = sub[f"{signal}_norm_ci_low"].to_numpy(float)
            hi = sub[f"{signal}_norm_ci_high"].to_numpy(float)
            ax.plot(sub.time_s, y, label=label, color=color)
            ax.fill_between(sub.time_s, lo, hi, color=color, alpha=0.14)
        ax.axvline(cfg.micro_start * cfg.dt, color="#777777", ls="--", lw=0.8)
        ax.axvline(cfg.micro_end * cfg.dt, color="#777777", ls="--", lw=0.8)
        ax.set_ylabel("Deviation norm")
        ax.set_title(title)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Fast, slow, and single-timescale adaptation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "adaptation_timescales.png")


def _events_with_sign(out: dict, cfg: Config, epoch: str) -> list[tuple[int, float]]:
    obj = out["x_true"][2]
    active = np.abs(obj) > 1e-3
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    allowed = set(int(i) for i in EPOCHS[epoch](cfg))
    return [(int(onset), float(np.sign(obj[onset]))) for onset in starts if int(onset) in allowed]


def run_innovation_dynamics(canonical_runs: list[dict], cfg: Config):
    log("Extracting signed innovation, correction, and recurrent-state dynamics")
    rows = []
    seed_rows = []
    # Epoch summaries split by object presence.
    trace_metrics = {
        "signed_visual_innovation": lambda o: o["innov_vis"],
        "absolute_visual_innovation": lambda o: np.abs(o["innov_vis"]),
        "object_kalman_gain": lambda o: o["K_obj_vis"],
        "signed_object_correction": lambda o: o["delta_v_obj_vis"],
        "absolute_object_correction": lambda o: np.abs(o["delta_v_obj_vis"]),
        "previsual_object_state": lambda o: o["x_previsual"][2],
        "posterior_object_state": lambda o: o["x_postvisual"][2],
    }
    for epoch in ("baseline", "reduced_reliability"):
        for period in ("object_event", "object_free"):
            for metric_name, extractor in trace_metrics.items():
                vals = []
                for rid, out in enumerate(canonical_runs):
                    mask, present, free = active_masks(out, cfg, epoch)
                    use = present if period == "object_event" else free
                    value = float(np.mean(extractor(out)[use]))
                    vals.append(value)
                    seed_rows.append({
                        "analysis_type": "epoch_period_summary",
                        "epoch": epoch,
                        "period": period,
                        "metric": metric_name,
                        "realization_id": rid,
                        "rng_seed": 100 + rid,
                        "value": value,
                    })
                stats = summarize(vals, 8000 + len(rows))
                rows.append({
                    "analysis_type": "epoch_period_summary",
                    "epoch": epoch,
                    "period": period,
                    "metric": metric_name,
                    "mean": stats["mean"],
                    "bootstrap_ci_low": stats["ci_low"],
                    "bootstrap_ci_high": stats["ci_high"],
                    "n_realizations": stats["n"],
                })

    # Event-aligned, sign-normalized trajectories. Uncertainty is across seeds,
    # after averaging events within seed at each relative sample.
    pre = int(round(0.20 / cfg.dt))
    post = int(round(0.80 / cfg.dt))
    rel_samples = np.arange(-pre, post + 1)
    variables = {
        "sign_normalized_innovation": lambda o, s: o["innov_vis"] * s,
        "object_gain": lambda o, s: o["K_obj_vis"],
        "sign_normalized_correction": lambda o, s: o["delta_v_obj_vis"] * s,
        "sign_normalized_previsual_object": lambda o, s: o["x_previsual"][2] * s,
        "sign_normalized_posterior_object": lambda o, s: o["x_postvisual"][2] * s,
    }
    for epoch in ("baseline", "reduced_reliability"):
        seed_curves = {name: [] for name in variables}
        cumulative_curves = []
        next_pred_curves = []
        for rid, out in enumerate(canonical_runs):
            event_arrays = {name: [] for name in variables}
            cumulative_events = []
            next_events = []
            for onset, sign in _events_with_sign(out, cfg, epoch):
                indices = onset + rel_samples
                if indices[0] < 0 or indices[-1] >= cfg.T:
                    continue
                for name, getter in variables.items():
                    event_arrays[name].append(getter(out, sign)[indices])
                correction = out["delta_v_obj_vis"][indices] * sign
                cumulative_events.append(np.cumsum(np.where(rel_samples >= 0, correction, 0.0)))
                next_pred = np.r_[out["x_pred"][2, indices[1:]], np.nan] * sign
                next_events.append(next_pred)
            for name in variables:
                seed_curves[name].append(np.nanmean(event_arrays[name], axis=0))
            cumulative_curves.append(np.nanmean(cumulative_events, axis=0))
            next_array = np.asarray(next_events, dtype=float)
            next_count = np.sum(np.isfinite(next_array), axis=0)
            next_curve = np.divide(
                np.nansum(next_array, axis=0), next_count,
                out=np.full(next_array.shape[1], np.nan), where=next_count > 0,
            )
            next_pred_curves.append(next_curve)
        seed_curves["cumulative_sign_normalized_correction"] = cumulative_curves
        seed_curves["sign_normalized_subsequent_predicted_object"] = next_pred_curves
        for si, rel in enumerate(rel_samples):
            for vi, (name, curves) in enumerate(seed_curves.items()):
                vals = np.asarray(curves)[:, si]
                stats = summarize(vals, 9000 + si * 20 + vi + (3000 if epoch == "reduced_reliability" else 0))
                rows.append({
                    "analysis_type": "event_aligned",
                    "epoch": epoch,
                    "period": "object_event",
                    "metric": name,
                    "relative_time_s": float(rel * cfg.dt),
                    "mean": stats["mean"],
                    "bootstrap_ci_low": stats["ci_low"],
                    "bootstrap_ci_high": stats["ci_high"],
                    "n_realizations": stats["n"],
                    "sign_normalization": "multiplied by true event sign; positive is event-congruent",
                })

    # Object-free timecourses within each epoch, including cumulative signed
    # correction accumulated only over object-free samples.
    bin_samples = int(round(0.10 / cfg.dt))
    for epoch in ("baseline", "reduced_reliability"):
        epoch_idx = EPOCHS[epoch](cfg)
        start = int(epoch_idx[0])
        end = int(epoch_idx[-1] + 1)
        for lo_i in range(start, end, bin_samples):
            hi_i = min(lo_i + bin_samples, end)
            vals_by_metric = {name: [] for name in (
                "signed_visual_innovation", "signed_object_correction",
                "cumulative_signed_object_correction", "previsual_object_state",
                "posterior_object_state", "subsequent_predicted_object_state",
            )}
            for out in canonical_runs:
                active = np.abs(out["x_true"][2]) > 1e-3
                free = ~active
                correction_free = np.where(free[start:end], out["delta_v_obj_vis"][start:end], 0.0)
                cumulative = np.cumsum(correction_free)
                use = free[lo_i:hi_i]
                if not use.any():
                    for name in vals_by_metric:
                        vals_by_metric[name].append(np.nan)
                    continue
                local = np.arange(lo_i, hi_i)[use]
                vals_by_metric["signed_visual_innovation"].append(float(np.mean(out["innov_vis"][local])))
                vals_by_metric["signed_object_correction"].append(float(np.mean(out["delta_v_obj_vis"][local])))
                cum_local = local - start
                vals_by_metric["cumulative_signed_object_correction"].append(float(np.mean(cumulative[cum_local])))
                vals_by_metric["previsual_object_state"].append(float(np.mean(out["x_previsual"][2, local])))
                vals_by_metric["posterior_object_state"].append(float(np.mean(out["x_postvisual"][2, local])))
                next_idx = np.minimum(local + 1, cfg.T - 1)
                vals_by_metric["subsequent_predicted_object_state"].append(float(np.mean(out["x_pred"][2, next_idx])))
            for vi, (name, vals) in enumerate(vals_by_metric.items()):
                stats = summarize(vals, 13000 + (lo_i - start) * 10 + vi + (2000 if epoch == "reduced_reliability" else 0))
                rows.append({
                    "analysis_type": "object_free_timecourse",
                    "epoch": epoch,
                    "period": "object_free",
                    "metric": name,
                    "relative_time_s": float((lo_i - start) * cfg.dt),
                    "mean": stats["mean"],
                    "bootstrap_ci_low": stats["ci_low"],
                    "bootstrap_ci_high": stats["ci_high"],
                    "n_realizations": stats["n"],
                })

    # Exact matched-sample product decomposition.
    base_idx = EPOCHS["baseline"](cfg)
    red_idx = EPOCHS["reduced_reliability"](cfg)
    decomposition_seed = []
    for rid, out in enumerate(canonical_runs):
        kb = out["K_obj_vis"][base_idx]
        kr = out["K_obj_vis"][red_idx]
        eb = out["innov_vis"][base_idx]
        er = out["innov_vis"][red_idx]
        innovation_term = kb * (er - eb)
        gain_term = eb * (kr - kb)
        interaction_term = (kr - kb) * (er - eb)
        observed = kr * er - kb * eb
        residual = observed - innovation_term - gain_term - interaction_term
        decomposition_seed.append({
            "analysis_type": "matched_product_decomposition_seedlevel",
            "realization_id": rid,
            "rng_seed": 100 + rid,
            "mean_observed_change": float(np.mean(observed)),
            "mean_innovation_term": float(np.mean(innovation_term)),
            "mean_gain_term": float(np.mean(gain_term)),
            "mean_interaction_term": float(np.mean(interaction_term)),
            "max_abs_identity_residual": float(np.max(np.abs(residual))),
            "temporal_matching": "same relative sample index within paired baseline and reduced epochs",
        })
    for term in ("mean_observed_change", "mean_innovation_term", "mean_gain_term", "mean_interaction_term"):
        vals = [row[term] for row in decomposition_seed]
        stats = summarize(vals, 15000 + len(rows))
        rows.append({
            "analysis_type": "matched_product_decomposition_summary",
            "epoch": "reduced_reliability_minus_baseline",
            "period": "all_samples",
            "metric": term,
            "mean": stats["mean"],
            "bootstrap_ci_low": stats["ci_low"],
            "bootstrap_ci_high": stats["ci_high"],
            "n_realizations": stats["n"],
        })

    result_df = pd.DataFrame(rows)
    seed_df = pd.DataFrame(seed_rows + decomposition_seed)
    write_csv(RESULTS / "innovation_dynamics.csv", result_df)
    write_csv(RESULTS / "innovation_dynamics_seedlevel.csv", seed_df)
    plot_innovation(result_df)
    return result_df, seed_df


def plot_innovation(df: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, metric, title in (
        (axes[0, 0], "sign_normalized_innovation", "Event-aligned visual innovation"),
        (axes[0, 1], "sign_normalized_correction", "Event-aligned object correction"),
    ):
        for epoch in ("baseline", "reduced_reliability"):
            sub = df[(df.analysis_type == "event_aligned") & (df.metric == metric) & (df.epoch == epoch)].sort_values("relative_time_s")
            x = sub.relative_time_s.to_numpy(float)
            y = sub["mean"].to_numpy(float)
            lo = sub.bootstrap_ci_low.to_numpy(float)
            hi = sub.bootstrap_ci_high.to_numpy(float)
            ax.plot(x, y, label=epoch.replace("_", " ").title(), color=COLORS[epoch])
            ax.fill_between(x, lo, hi, color=COLORS[epoch], alpha=0.15)
        ax.axvline(0, color="#777777", ls="--", lw=0.8)
        ax.axhline(0, color="#777777", lw=0.6)
        ax.set(xlabel="Time from object-event onset (s)", ylabel="Event-sign-normalized value", title=title)
        ax.legend(frameon=False)

    ax = axes[1, 0]
    for epoch in ("baseline", "reduced_reliability"):
        sub = df[(df.analysis_type == "object_free_timecourse") & (df.metric == "cumulative_signed_object_correction") & (df.epoch == epoch)].sort_values("relative_time_s")
        x = sub.relative_time_s.to_numpy(float)
        y = sub["mean"].to_numpy(float)
        lo = sub.bootstrap_ci_low.to_numpy(float)
        hi = sub.bootstrap_ci_high.to_numpy(float)
        ax.plot(x, y, label=epoch.replace("_", " ").title(), color=COLORS[epoch])
        ax.fill_between(x, lo, hi, color=COLORS[epoch], alpha=0.15)
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set(xlabel="Time within epoch (s)", ylabel="Cumulative signed correction (model units)", title="Object-free cumulative correction")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    decomp = df[df.analysis_type == "matched_product_decomposition_summary"]
    order = ["mean_innovation_term", "mean_gain_term", "mean_interaction_term", "mean_observed_change"]
    labels = ["Innovation", "Gain", "Interaction", "Observed"]
    sub = decomp.set_index("metric").loc[order]
    y = sub["mean"].to_numpy(float)
    lo = sub.bootstrap_ci_low.to_numpy(float)
    hi = sub.bootstrap_ci_high.to_numpy(float)
    ax.bar(np.arange(4), y, color=["#2878B5", "#E07A1F", "#7B3F98", "#555555"])
    ax.errorbar(np.arange(4), y, yerr=[y - lo, hi - y], fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set_xticks(np.arange(4), labels)
    ax.set(ylabel="Mean change in signed correction", title="Matched product decomposition")
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Signed innovation and recurrent object-state dynamics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "innovation_dynamics.png")


def _safe_corr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def run_replay_controls(cfg: Config, canonical_runs: list[dict]):
    log("Running canonical and replay controls")
    specs = (
        ("canonical_closed_loop", "corrected_closed_loop"),
        ("event_aligned_replay", "per_trial_eye_trace"),
        ("event_agnostic_replay", "legacy_fixed_eye_trace"),
    )
    seed_rows = []
    for label, engine in specs:
        log(f"  {label}")
        for rid in range(N_PRINCIPAL):
            if label == "canonical_closed_loop":
                out = canonical_runs[rid]
            else:
                out = run_trial(cfg, engine, rid)
            seed_rows.extend(epoch_metrics(out, cfg, label, rid))
            eye = np.asarray(out.get("eye_vel", np.zeros(cfg.T)), float)
            xhat = np.asarray(out["x_hat"], float)
            y_path = np.asarray(out.get("y_vis_raw", out.get("y_not", np.full(cfg.T, np.nan))), float)
            seed_rows.append({
                "model_id": label,
                "realization_id": rid,
                "rng_seed": 100 + rid,
                "epoch": "coupling_diagnostic",
                "eye_next_vs_current_estimated_omega_correlation": _safe_corr(eye[1:], xhat[0, :-1]),
                "eye_vs_current_retinal_path_correlation": _safe_corr(eye, y_path),
                "eye_vs_object_schedule_correlation": _safe_corr(eye, out["x_true"][2]),
            })
    seed_df = pd.DataFrame(seed_rows)
    write_csv(RESULTS / "replay_controls_seedlevel.csv", seed_df)

    rows = []
    metric_seed = seed_df[seed_df.epoch.isin(EPOCHS)]
    offset = 16000
    for label, _ in specs:
        for metric in ("angular_rmse", "object_rmse", "false_positive_rms"):
            rows.append({
                "row_type": "metric_verification",
                "control": label,
                "metric": metric,
                **paired_summary(metric_seed, "model_id", label, metric, offset),
            })
            offset += 3
    coupling = seed_df[seed_df.epoch == "coupling_diagnostic"]
    for label, _ in specs:
        sub = coupling[coupling.model_id == label]
        for metric in (
            "eye_next_vs_current_estimated_omega_correlation",
            "eye_vs_current_retinal_path_correlation",
            "eye_vs_object_schedule_correlation",
        ):
            stats = summarize(sub[metric], offset)
            rows.append({
                "row_type": "coupling_diagnostic",
                "control": label,
                "metric": metric,
                "mean": stats["mean"],
                "bootstrap_ci_low": stats["ci_low"],
                "bootstrap_ci_high": stats["ci_high"],
                "n_realizations": stats["n"],
            })
            offset += 2

    structure = {
        "canonical_closed_loop": {
            "jointly_generated": "object schedule, sensory noise, estimator state, retinal input, and eye trajectory within the same realization",
            "replayed": "none",
            "event_alignment": "same current realization",
            "independent_components": "independent channel noise draws within a shared realization RNG",
            "conditioned_on_current_estimator": "eye command uses current posterior angular estimate",
            "conditioned_on_current_eye": "raw retinal input subtracts current eye; exact compensation adds the same eye back",
        },
        "event_aligned_replay": {
            "jointly_generated": "current event schedule and current estimator/sensory stream",
            "replayed": "eye trajectory generated by a separate simulation using the same object schedule",
            "event_alignment": "aligned to object-event schedule only",
            "independent_components": "eye-source sensory RNG seed 10000+trial is independent of current sensory RNG 100+trial",
            "conditioned_on_current_estimator": "no",
            "conditioned_on_current_eye": "replay raw visual path subtracts the replayed eye and does not exactly restore it",
        },
        "event_agnostic_replay": {
            "jointly_generated": "current event schedule and current estimator/sensory stream",
            "replayed": "one deterministic fixed eye trajectory shared across trials",
            "event_alignment": "not aligned to current events",
            "independent_components": "fixed eye source is independent of current event/sensory realization",
            "conditioned_on_current_estimator": "no",
            "conditioned_on_current_eye": "replay raw visual path subtracts the replayed eye and does not exactly restore it",
        },
    }
    for label, values in structure.items():
        rows.append({"row_type": "structural_description", "control": label, **values})
    result_df = pd.DataFrame(rows)
    write_csv(RESULTS / "replay_controls_verified.csv", result_df)
    return result_df, seed_df


SENSITIVITY_SPECS = {
    "sigma_oto_base": [0.030, 0.045, 0.060, 0.075, 0.090],
    "sigma_oto_earth_hyp": [0.030, 0.045, 0.060, 0.075, 0.090],
    "sigma_oto_micro_hyp": [0.20, 0.30, 0.40, 0.50, 0.60],
    "sigma_canal": [0.030, 0.045, 0.060, 0.075, 0.090],
    "llr_leak": [0.90, 0.95, 0.98, 0.99, 0.995],
    "llr_clip": [4.0, 6.0, 8.0, 10.0, 12.0],
    "eta_fast": [0.0015, 0.00225, 0.0030, 0.00375, 0.0045],
    "retention_fast": [0.96, 0.97, 0.98, 0.99, 0.995],
    "eta_slow": [0.00015, 0.000225, 0.00030, 0.000375, 0.00045],
    "retention_slow": [0.9990, 0.99925, 0.9995, 0.99975, 0.9999],
    "tau_eye": [0.50, 0.75, 1.00, 1.25, 1.50],
    "g_vor0": [0.50, 0.75, 1.00, 1.10, 1.20],
}


def _sensitivity_level_worker(parameter: str, level: float, level_index: int) -> list[dict]:
    default = Config(n_trials=N_PRINCIPAL)
    seed_rows = []
    canonical_value = getattr(default, parameter)
    cfg = replace(default, **{parameter: level})
    for rid in range(N_PRINCIPAL):
        out = run_model(cfg, rid, "A_full_canonical")
        epoch = {row["epoch"]: row for row in epoch_metrics(out, cfg, parameter, rid)}
        row = {
            "parameter": parameter,
            "canonical_value": canonical_value,
            "tested_value": level,
            "level_index": level_index,
            "level_label": ("canonical" if math.isclose(float(level), float(canonical_value), rel_tol=0, abs_tol=1e-12) else f"level_{level_index+1}"),
            "realization_id": rid,
            "rng_seed": 100 + rid,
        }
        for metric in ("angular_rmse", "object_rmse", "false_positive_rms"):
            b = epoch["baseline"][metric]
            m = epoch["reduced_reliability"][metric]
            row[f"baseline_{metric}"] = b
            row[f"reduced_{metric}"] = m
            row[f"paired_change_{metric}"] = m - b
            row[f"percent_change_{metric}"] = 100.0 * (m - b) / b
        seed_rows.append(row)
    return seed_rows


def run_targeted_sensitivity():
    log("Running targeted one-at-a-time parameter sensitivity (30 paired seeds per level)")
    default = Config(n_trials=N_PRINCIPAL)
    seed_rows = []
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for parameter, levels in SENSITIVITY_SPECS.items():
            for li, level in enumerate(levels):
                futures[pool.submit(_sensitivity_level_worker, parameter, level, li)] = (parameter, li, level)
        completed_by_parameter = {parameter: 0 for parameter in SENSITIVITY_SPECS}
        for future in as_completed(futures):
            parameter, li, level = futures[future]
            seed_rows.extend(future.result())
            completed_by_parameter[parameter] += 1
            if completed_by_parameter[parameter] == len(SENSITIVITY_SPECS[parameter]):
                log(f"  completed {parameter}")
    seed_rows.sort(key=lambda row: (list(SENSITIVITY_SPECS).index(row["parameter"]), row["level_index"], row["realization_id"]))
    seed_df = pd.DataFrame(seed_rows)
    write_csv(RESULTS / "targeted_parameter_sensitivity_seedlevel.csv", seed_df)

    rows = []
    for pi, (parameter, levels) in enumerate(SENSITIVITY_SPECS.items()):
        canonical_value = getattr(default, parameter)
        for li, level in enumerate(levels):
            sub = seed_df[(seed_df.parameter == parameter) & np.isclose(seed_df.tested_value, level)]
            for mi, metric in enumerate(("angular_rmse", "object_rmse", "false_positive_rms")):
                d = sub[f"paired_change_{metric}"].to_numpy(float)
                pct = sub[f"percent_change_{metric}"].to_numpy(float)
                d_lo, d_hi = bootstrap_ci(d, 18000 + pi * 100 + li * 10 + mi)
                p_lo, p_hi = bootstrap_ci(pct, 18001 + pi * 100 + li * 10 + mi)
                rows.append({
                    "parameter": parameter,
                    "canonical_value": canonical_value,
                    "tested_value": level,
                    "level_index": li,
                    "is_canonical_level": bool(math.isclose(float(level), float(canonical_value), rel_tol=0, abs_tol=1e-12)),
                    "metric": metric,
                    "baseline_mean": float(sub[f"baseline_{metric}"].mean()),
                    "reduced_mean": float(sub[f"reduced_{metric}"].mean()),
                    "paired_change_mean": float(np.mean(d)),
                    "paired_change_ci_low": d_lo,
                    "paired_change_ci_high": d_hi,
                    "mean_within_realization_percent_change": float(np.mean(pct)),
                    "percent_change_ci_low": p_lo,
                    "percent_change_ci_high": p_hi,
                    "positive_sign_consistency": float(np.mean(d > 0)),
                    "qualitative_positive_conclusion_survives": bool(np.mean(d) > 0),
                    "paired_interval_excludes_zero_positive": bool(d_lo > 0),
                    "n_realizations": len(sub),
                })
    result_df = pd.DataFrame(rows)
    write_csv(RESULTS / "targeted_parameter_sensitivity.csv", result_df)
    plot_targeted_sensitivity(result_df)
    return result_df, seed_df


def plot_targeted_sensitivity(df: pd.DataFrame) -> None:
    set_plot_style()
    fp = df[df.metric == "false_positive_rms"].copy()
    parameters = list(SENSITIVITY_SPECS)
    matrix = np.full((len(parameters), 5), np.nan)
    annotation = np.empty((len(parameters), 5), dtype=object)
    for i, parameter in enumerate(parameters):
        sub = fp[fp.parameter == parameter].sort_values("level_index")
        matrix[i, :len(sub)] = sub.mean_within_realization_percent_change.to_numpy(float)
        annotation[i, :len(sub)] = [f"{v:.1f}" for v in sub.mean_within_realization_percent_change]
    vmax = float(np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(parameters)), parameters)
    ax.set_xticks(np.arange(5), ["Low 1", "Low 2", "Canonical*", "High 1", "High 2"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                color = "white" if abs(matrix[i, j]) > 0.55 * vmax else "black"
                ax.text(j, i, annotation[i, j], ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean within-realization false-positive change (%)")
    ax.set_title("Targeted parameter sensitivity")
    ax.set_xlabel("Ordered tested value (canonical is center; see source CSV for exact values)")
    fig.tight_layout()
    save_figure(fig, "targeted_parameter_sensitivity.png")




def main() -> None:
    start = time.perf_counter()
    if (LOGS / "run.log").exists():
        (LOGS / "run.log").unlink()
    log("Starting deterministic extended analyses")
    cfg, model_seed_df, model_df, canonical_runs, single_runs, equivalence, eye_equivalence = run_principal_models()
    context_seed_df, context_df = run_context_mapping()
    recovery_time_df, recovery_df, recovery_seed_df = run_recovery(canonical_runs, cfg)
    criterion_seed_df, criterion_df = run_criterion_sensitivity(canonical_runs, cfg)
    adaptation_df, adaptation_seed_df = run_adaptation_timescales(canonical_runs, single_runs, cfg)
    innovation_df, innovation_seed_df = run_innovation_dynamics(canonical_runs, cfg)
    replay_df, replay_seed_df = run_replay_controls(cfg, canonical_runs)
    sensitivity_df, sensitivity_seed_df = run_targeted_sensitivity()
    runtime = time.perf_counter() - start
    write_json(LOGS / "runtime.json", {
        "runtime_seconds": runtime,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "principal_realizations": N_PRINCIPAL,
        "sweep_realizations": N_SWEEP,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": BOOTSTRAP_SEED,
    })
    log(f"Completed extended analyses in {runtime:.1f} seconds")


if __name__ == "__main__":
    main()
