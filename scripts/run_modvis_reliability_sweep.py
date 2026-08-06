"""Run the paired MODVIS effective-otolith-reliability sweep.

The sweep uses the corrected closed-loop implementation, corrected forward
model initialization, and common random numbers across reliability levels.
It writes trial-level metrics, summaries, adjacent paired contrasts, a
machine-readable run record, and a concise analysis report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from cerebellum import metrics as mx
from cerebellum import model as model
from cerebellum import stimulus as stim
from cerebellum.config import Config
from cerebellum.experiment import run_trial


KAPPA_GRID = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
              0.80, 0.90, 1.00)
EPOCH_LABELS = {
    "pre": "baseline",
    "micro": "reduced_reliability",
    "post": "recovery",
}
PRIMARY_METRICS = ("angular_rmse", "object_rmse", "false_positive_rms")
IMPLEMENTATION = "corrected_closed_loop"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return _hash_bytes(arr.view(np.uint8).tobytes())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _hash_bytes(encoded)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                  n_boot: int) -> tuple[float, float]:
    return mx.bootstrap_mean_ci(values, rng, n_boot=n_boot, alpha=0.05)


def _simulate_task(task):
    kappa, trial_id, base_config, config_id = task
    cfg = replace(Config(**base_config), g_micro=float(kappa))
    if cfg.forward_initialization != "corrected":
        raise AssertionError("MODVIS sweep requires corrected initialization")
    model.assert_initial_forward_model(cfg)
    out = run_trial(cfg, IMPLEMENTATION, trial_id)

    active = np.abs(out["x_true"][2]) > 1e-3
    rows = []
    for internal_epoch, idx in stim.epoch_indices(cfg).items():
        mask = np.zeros(cfg.T, dtype=bool)
        mask[idx] = True
        present = mask & active
        free = mask & ~active
        crossings = mx.criterion_crossings(out["x_hat"][2], active, idx, cfg)
        finite_crossings = crossings[np.isfinite(crossings)]
        rows.append({
            "seed": 100 + trial_id,
            "seed_identity": trial_id,
            "kappa": float(kappa),
            "epoch": EPOCH_LABELS[internal_epoch],
            "angular_rmse": mx.rmse(
                out["x_hat"][0, mask], out["x_true"][0, mask]),
            "object_rmse": mx.rmse(
                out["x_hat"][2, present], out["x_true"][2, present]),
            "false_positive_rms": float(
                np.sqrt(np.mean(out["x_hat"][2, free] ** 2))),
            "criterion_crossing_time": (
                float(finite_crossings.mean())
                if len(finite_crossings) else np.nan),
            "valid_object_free_samples": int(free.sum()),
            "object_present_samples": int(present.sum()),
            "object_event_count": int(len(crossings)),
            "criterion_crossing_count": int(len(finite_crossings)),
            "implementation": IMPLEMENTATION,
            "configuration_id": config_id,
        })

    stream_hashes = {
        "seed_identity": trial_id,
        "kappa": float(kappa),
        "event_schedule_hash": _hash_array(out["object_motion"]),
        "self_motion_hash": _hash_array(out["x_true"][:2]),
        "canal_noise_hash": _hash_array(out["canal_standard_normal"]),
        "otolith_noise_hash": _hash_array(out["otolith_standard_normal"]),
        "visual_noise_hash": _hash_array(out["visual_standard_normal"]),
    }
    parameter_state = cfg.to_dict()
    parameter_state.pop("g_micro")
    return rows, stream_hashes, _json_hash(parameter_state)


def _metric_lookup(rows: list[dict]) -> dict[tuple[int, float, str], dict]:
    return {
        (int(row["seed_identity"]), float(row["kappa"]), row["epoch"]): row
        for row in rows
    }


def _summarize(rows: list[dict], n_boot: int, bootstrap_seed: int):
    lookup = _metric_lookup(rows)
    seed_ids = sorted({int(row["seed_identity"]) for row in rows})
    rng = np.random.default_rng(bootstrap_seed)
    summary = []
    for kappa in KAPPA_GRID:
        for metric in PRIMARY_METRICS:
            baseline = np.asarray([
                lookup[(seed, kappa, "baseline")][metric] for seed in seed_ids
            ], dtype=float)
            reduced = np.asarray([
                lookup[(seed, kappa, "reduced_reliability")][metric]
                for seed in seed_ids
            ], dtype=float)
            absolute = reduced - baseline
            percent = 100.0 * absolute / baseline
            mean_lo, mean_hi = _bootstrap_ci(reduced, rng, n_boot)
            abs_lo, abs_hi = _bootstrap_ci(absolute, rng, n_boot)
            pct_lo, pct_hi = _bootstrap_ci(percent, rng, n_boot)
            summary.append({
                "kappa": kappa,
                "metric": metric,
                "n_realizations": len(seed_ids),
                "baseline_mean": float(baseline.mean()),
                "manipulated_mean": float(reduced.mean()),
                "manipulated_sd": float(reduced.std(ddof=1)),
                "monte_carlo_se": float(
                    reduced.std(ddof=1) / np.sqrt(len(reduced))),
                "mean_bootstrap_ci_low": mean_lo,
                "mean_bootstrap_ci_high": mean_hi,
                "paired_absolute_change_mean": float(absolute.mean()),
                "paired_absolute_change_ci_low": abs_lo,
                "paired_absolute_change_ci_high": abs_hi,
                "paired_percent_change_mean": float(percent.mean()),
                "paired_percent_change_ci_low": pct_lo,
                "paired_percent_change_ci_high": pct_hi,
                "bootstrap_resamples": n_boot,
                "bootstrap_seed": bootstrap_seed,
            })
    return summary


def _adjacent_contrasts(rows: list[dict], n_boot: int, bootstrap_seed: int):
    lookup = _metric_lookup(rows)
    seed_ids = sorted({int(row["seed_identity"]) for row in rows})
    rng = np.random.default_rng(bootstrap_seed + 1)
    contrasts = []
    for lower, upper in zip(KAPPA_GRID[:-1], KAPPA_GRID[1:]):
        for metric in PRIMARY_METRICS:
            low_values = np.asarray([
                lookup[(seed, lower, "reduced_reliability")][metric]
                for seed in seed_ids
            ], dtype=float)
            high_values = np.asarray([
                lookup[(seed, upper, "reduced_reliability")][metric]
                for seed in seed_ids
            ], dtype=float)
            delta = high_values - low_values
            lo, hi = _bootstrap_ci(delta, rng, n_boot)
            contrasts.append({
                "kappa_a": upper,
                "kappa_b": lower,
                "contrast": "kappa_a_minus_kappa_b",
                "metric": metric,
                "mean_paired_difference": float(delta.mean()),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "n_pairs": len(delta),
                "bootstrap_resamples": n_boot,
                "bootstrap_seed": bootstrap_seed + 1,
            })
    return contrasts


def _trend_assessment(rows: list[dict], summary: list[dict],
                      n_boot: int, bootstrap_seed: int):
    lookup = _metric_lookup(rows)
    seed_ids = sorted({int(row["seed_identity"]) for row in rows})
    x = 1.0 - np.asarray(KAPPA_GRID, dtype=float)
    slopes = []
    for seed in seed_ids:
        y = np.asarray([
            lookup[(seed, kappa, "reduced_reliability")]
            ["false_positive_rms"] for kappa in KAPPA_GRID
        ])
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    slopes = np.asarray(slopes)
    rng = np.random.default_rng(bootstrap_seed + 2)
    slope_lo, slope_hi = _bootstrap_ci(slopes, rng, n_boot)

    fp_rows = {
        float(row["kappa"]): row for row in summary
        if row["metric"] == "false_positive_rms"
    }
    level_means = np.asarray([
        fp_rows[kappa]["manipulated_mean"] for kappa in KAPPA_GRID
    ])
    kappa_ranks = np.argsort(np.argsort(np.asarray(KAPPA_GRID))).astype(float)
    mean_ranks = np.argsort(np.argsort(level_means)).astype(float)
    spearman = float(np.corrcoef(kappa_ranks, mean_ranks)[0, 1])

    y20 = np.asarray([
        lookup[(seed, 0.20, "reduced_reliability")]["false_positive_rms"]
        for seed in seed_ids
    ])
    y30 = np.asarray([
        lookup[(seed, 0.30, "reduced_reliability")]["false_positive_rms"]
        for seed in seed_ids
    ])
    bump = y30 - y20
    bump_lo, bump_hi = _bootstrap_ci(bump, rng, n_boot)
    if bump_lo > 0:
        local_result = "persistent local nonmonotonicity"
    elif bump_lo <= 0 <= bump_hi:
        local_result = "previous bump is consistent with Monte Carlo variability"
    else:
        local_result = "previous local reversal is absent"
    return {
        "seed_level_slope_mean": float(slopes.mean()),
        "seed_level_slope_sd": float(slopes.std(ddof=1)),
        "seed_level_slope_bootstrap_ci_low": slope_lo,
        "seed_level_slope_bootstrap_ci_high": slope_hi,
        "slope_definition": (
            "least-squares slope of manipulated-epoch false-positive RMS "
            "against unreliability (1-kappa), estimated within each seed"),
        "descriptive_spearman_kappa_vs_level_means": spearman,
        "kappa_0.30_minus_0.20_mean": float(bump.mean()),
        "kappa_0.30_minus_0.20_bootstrap_ci_low": bump_lo,
        "kappa_0.30_minus_0.20_bootstrap_ci_high": bump_hi,
        "local_reversal_interpretation": local_result,
        "n_seed_level_slopes": len(slopes),
    }


def _qc(rows, stream_hashes, parameter_hashes, n_trials):
    required_numeric = [
        "angular_rmse", "object_rmse", "false_positive_rms",
        "criterion_crossing_time", "valid_object_free_samples",
        "object_present_samples", "object_event_count",
    ]
    numeric = np.asarray([
        [float(row[name]) for name in required_numeric] for row in rows
    ])
    key_pairs = [(int(row["seed_identity"]), float(row["kappa"]),
                  row["epoch"]) for row in rows]
    counts = {
        str(kappa): len({
            int(row["seed_identity"]) for row in rows
            if float(row["kappa"]) == kappa
        }) for kappa in KAPPA_GRID
    }
    stream_fields = (
        "event_schedule_hash", "self_motion_hash", "canal_noise_hash",
        "otolith_noise_hash", "visual_noise_hash",
    )
    paired_stream_failures = {}
    for field in stream_fields:
        failures = []
        for seed in range(n_trials):
            values = {
                row[field] for row in stream_hashes
                if int(row["seed_identity"]) == seed
            }
            if len(values) != 1:
                failures.append(seed)
        paired_stream_failures[field] = failures
    epoch_lengths = {
        epoch: sorted({
            int(row["valid_object_free_samples"])
            + int(row["object_present_samples"])
            for row in rows if row["epoch"] == epoch
        })
        for epoch in EPOCH_LABELS.values()
    }
    return {
        "row_count": len(rows),
        "expected_row_count": n_trials * len(KAPPA_GRID) * 3,
        "missing_value_count": int(np.isnan(numeric).sum()),
        "infinite_value_count": int(np.isinf(numeric).sum()),
        "zero_length_object_free_mask_count": sum(
            int(row["valid_object_free_samples"]) == 0 for row in rows),
        "seed_counts_by_kappa": counts,
        "unequal_seed_counts": len(set(counts.values())) != 1,
        "duplicate_seed_kappa_epoch_count": len(key_pairs) - len(set(key_pairs)),
        "paired_stream_failures": paired_stream_failures,
        "epoch_lengths": epoch_lengths,
        "parameter_hashes_excluding_kappa": sorted(set(parameter_hashes)),
        "only_kappa_varied": (
            len(set(parameter_hashes)) == 1
            and not any(paired_stream_failures.values())),
        "corrected_initialization": True,
    }


def _reproducibility_check(base_config, config_id, rows):
    lookup = _metric_lookup(rows)
    tasks = [
        (kappa, seed, base_config, config_id)
        for seed in (0, 37, 99) for kappa in (0.15, 0.30, 1.00)
    ]
    exact = True
    checked = 0
    for task in tasks:
        repeated_rows, _, _ = _simulate_task(task)
        for repeated in repeated_rows:
            original = lookup[(
                int(repeated["seed_identity"]), float(repeated["kappa"]),
                repeated["epoch"])]
            for metric in PRIMARY_METRICS:
                exact &= repeated[metric] == original[metric]
            exact &= (
                repeated["criterion_crossing_time"]
                == original["criterion_crossing_time"])
            checked += 1
    return {
        "subset_simulations_repeated": len(tasks),
        "epoch_rows_compared": checked,
        "exact_metric_match": bool(exact),
    }


def _old_sweep_snapshot(path: Path):
    if not path.exists():
        return {"available": False}
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for kappa in (0.20, 0.30):
        matches = [
            row for row in rows
            if row.get("epoch") == "micro"
            and abs(float(row.get("value", "nan")) - kappa) < 1e-12
        ]
        if matches:
            selected[f"kappa_{kappa:.2f}_false_positive_mean"] = float(
                matches[0]["false_positive_rms"])
    return {"available": True, **selected}


def _source_state(workspace: Path):
    relative_paths = [
        "cerebellum/config.py",
        "cerebellum/experiment.py",
        "cerebellum/model.py",
        "cerebellum/stimulus.py",
        "cerebellum/metrics.py",
        "scripts/analysis/run_modvis_reliability_sweep.py",
    ]
    hashes = {
        rel: _file_hash(workspace / rel) for rel in relative_paths
    }
    return _json_hash(hashes), hashes


def _package_versions():
    result = {}
    for package in ("numpy", "matplotlib", "pytest"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _write_analysis(path: Path, runtime_seconds: float, trend: dict,
                    old_sweep: dict, qc: dict, reproducibility: dict,
                    n_trials: int, n_boot: int):
    old_text = "The earlier sweep table was not available."
    if old_sweep.get("available"):
        old_text = (
            "The earlier five-realization table reported manipulated-epoch "
            f"false-positive means of "
            f"{old_sweep.get('kappa_0.20_false_positive_mean', float('nan')):.5f} "
            f"at κ=0.20 and "
            f"{old_sweep.get('kappa_0.30_false_positive_mean', float('nan')):.5f} "
            "at κ=0.30."
        )
    text = f"""# MODVIS paired reliability sweep analysis

The canonical sweep completed {n_trials * len(KAPPA_GRID)} corrected
closed-loop simulations ({n_trials} common-random-number identities at each of
{len(KAPPA_GRID)} effective otolith reliability levels) in
{runtime_seconds:.2f} seconds. The full κ grid was
{", ".join(f"{value:.2f}" for value in KAPPA_GRID)}.

The mean within-seed slope of manipulated-epoch false-positive RMS against
unreliability (1-κ) was {trend["seed_level_slope_mean"]:.6f} model units per
unit unreliability (95% paired bootstrap interval
{trend["seed_level_slope_bootstrap_ci_low"]:.6f} to
{trend["seed_level_slope_bootstrap_ci_high"]:.6f}; {n_boot} resamples).
The descriptive Spearman correlation between κ and the 11 level means was
{trend["descriptive_spearman_kappa_vs_level_means"]:.3f}.

The paired κ=0.30 minus κ=0.20 contrast was
{trend["kappa_0.30_minus_0.20_mean"]:.6f} model units (95% paired bootstrap
interval {trend["kappa_0.30_minus_0.20_bootstrap_ci_low"]:.6f} to
{trend["kappa_0.30_minus_0.20_bootstrap_ci_high"]:.6f}). Under the prespecified
decision rule, the {trend["local_reversal_interpretation"]}.

{old_text}

Intervals quantify Monte Carlo variability under this specified model. They
are not population-level uncertainty intervals for biological observers.

## Quality control

- Trial-level rows: {qc["row_count"]} (expected {qc["expected_row_count"]}).
- Missing numeric values: {qc["missing_value_count"]}.
- Infinite numeric values: {qc["infinite_value_count"]}.
- Zero-length object-free masks: {qc["zero_length_object_free_mask_count"]}.
- Duplicate seed-κ-epoch combinations:
  {qc["duplicate_seed_kappa_epoch_count"]}.
- Only κ varied across paired levels: {qc["only_kappa_varied"]}.
- Corrected forward-model initialization: {qc["corrected_initialization"]}.
- Exact repeated-subset metric match:
  {reproducibility["exact_metric_match"]} across
  {reproducibility["subset_simulations_repeated"]} repeated simulations.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/modvis_reliability_sweep_100")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument(
        "--old-sweep",
        default="results/corrected_model/tables/reliability_loading_sweep.csv")
    args = parser.parse_args()
    if args.n_trials != 100:
        raise SystemExit("The MODVIS canonical sweep requires exactly 100 trials")
    if args.bootstrap_resamples < 2000:
        raise SystemExit("At least 2,000 bootstrap resamples are required")

    workspace = Path(__file__).resolve().parents[2]
    output = (workspace / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = Config(n_trials=args.n_trials,
                  bootstrap_samples=args.bootstrap_resamples)
    if base.forward_initialization != "corrected":
        raise AssertionError("base configuration is not corrected")
    base_config = base.to_dict()
    config_identity = dict(base_config)
    config_identity["g_micro"] = "<varied kappa>"
    config_id = f"modvis-kappa-{_json_hash(config_identity)[:12]}"
    tasks = [
        (kappa, seed, base_config, config_id)
        for kappa in KAPPA_GRID for seed in range(args.n_trials)
    ]

    started = time.perf_counter()
    if args.workers == 1:
        results = [_simulate_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_simulate_task, tasks, chunksize=5))
    runtime_seconds = time.perf_counter() - started

    rows = [row for item in results for row in item[0]]
    rows.sort(key=lambda row: (
        float(row["kappa"]), int(row["seed_identity"]), row["epoch"]))
    stream_hashes = [item[1] for item in results]
    parameter_hashes = [item[2] for item in results]
    summary = _summarize(
        rows, args.bootstrap_resamples, args.bootstrap_seed)
    contrasts = _adjacent_contrasts(
        rows, args.bootstrap_resamples, args.bootstrap_seed)
    trend = _trend_assessment(
        rows, summary, args.bootstrap_resamples, args.bootstrap_seed)
    qc = _qc(rows, stream_hashes, parameter_hashes, args.n_trials)
    reproducibility = _reproducibility_check(base_config, config_id, rows)
    qc["reproducibility"] = reproducibility

    critical_failures = [
        qc["row_count"] != qc["expected_row_count"],
        qc["missing_value_count"] != 0,
        qc["infinite_value_count"] != 0,
        qc["zero_length_object_free_mask_count"] != 0,
        qc["unequal_seed_counts"],
        qc["duplicate_seed_kappa_epoch_count"] != 0,
        not qc["only_kappa_varied"],
        not reproducibility["exact_metric_match"],
    ]
    if any(critical_failures):
        raise RuntimeError(f"sweep quality control failed: {qc}")

    trial_path = output / "trial_level_reliability_sweep.csv"
    summary_path = output / "reliability_sweep_summary.csv"
    contrast_path = output / "reliability_sweep_adjacent_contrasts.csv"
    analysis_path = output / "reliability_sweep_analysis.md"
    config_path = output / "reliability_sweep_run_config.json"
    _write_csv(trial_path, rows)
    _write_csv(summary_path, summary)
    _write_csv(contrast_path, contrasts)

    old_sweep = _old_sweep_snapshot(workspace / args.old_sweep)
    _write_analysis(
        analysis_path, runtime_seconds, trend, old_sweep, qc,
        reproducibility, args.n_trials, args.bootstrap_resamples)
    source_state_id, source_hashes = _source_state(workspace)
    run_config = {
        "analysis": "MODVIS paired effective-otolith-reliability sweep",
        "timestamp": datetime.now().astimezone().isoformat(),
        "runtime_seconds": runtime_seconds,
        "core_simulations": len(tasks),
        "repeated_qc_simulations": reproducibility[
            "subset_simulations_repeated"],
        "workers": args.workers,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "source_state_identifier": source_state_id,
        "source_file_sha256": source_hashes,
        "seed_policy": "seed identity 0..99; event/sensory RNG seed = 100 + identity",
        "seed_identities": list(range(args.n_trials)),
        "rng_seeds": list(range(100, 100 + args.n_trials)),
        "kappa_grid": list(KAPPA_GRID),
        "trial_count_per_kappa": args.n_trials,
        "common_random_numbers": True,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "implementation": IMPLEMENTATION,
        "configuration_id": config_id,
        "model_configuration": config_identity,
        "trend_assessment": trend,
        "old_sweep_snapshot": old_sweep,
        "quality_control": qc,
        "output_sha256": {
            trial_path.name: _file_hash(trial_path),
            summary_path.name: _file_hash(summary_path),
            contrast_path.name: _file_hash(contrast_path),
            analysis_path.name: _file_hash(analysis_path),
        },
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "runtime_seconds": runtime_seconds,
        "core_simulations": len(tasks),
        "trend": trend,
        "quality_control_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
