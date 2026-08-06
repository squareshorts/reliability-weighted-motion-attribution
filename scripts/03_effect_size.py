import sys
import json
import csv
from pathlib import Path
import numpy as np

ROOT = Path("c:/work/cerebellum").resolve()
sys.path.insert(0, str(ROOT))

from cerebellum.config import Config
from analysis_modvis_completion import run_analysis_v1 as run_v1
import cerebellum.metrics as mx

def compute_paired_dz_ci(values_after, values_before, rng, n_boot=2000, alpha=0.05):
    diffs = np.asarray(values_after) - np.asarray(values_before)
    if not len(diffs):
        return np.nan, (np.nan, np.nan)
        
    def compute_dz(d):
        sd = np.std(d, ddof=1)
        if sd > 0:
            return np.mean(d) / sd
        return np.nan
        
    dz = compute_dz(diffs)
    
    indices = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_dzs = []
    for idx in indices:
        boot_d = diffs[idx]
        b_dz = compute_dz(boot_d)
        if np.isfinite(b_dz):
            boot_dzs.append(b_dz)
            
    if not boot_dzs:
        return dz, (np.nan, np.nan)
        
    boot_dzs = np.array(boot_dzs)
    ci = np.quantile(boot_dzs, [alpha / 2, 1 - alpha / 2])
    return float(dz), (float(ci[0]), float(ci[1]))

def compute_percent_change_ci(values_after, values_before, rng, n_boot=2000, alpha=0.05):
    b = np.asarray(values_before)
    a = np.asarray(values_after)
    pcts = 100.0 * (a - b) / b
    
    mean_pct = np.mean(pcts)
    
    indices = rng.integers(0, len(pcts), size=(n_boot, len(pcts)))
    boot_means = np.mean(pcts[indices], axis=1)
    
    ci = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(mean_pct), (float(ci[0]), float(ci[1]))

def run_effect_sizes():
    rng = np.random.default_rng(20260731) # BOOTSTRAP_SEED from run_v1
    
    with open(ROOT / "analysis_modvis_completion" / "final_analysis_resolution" / "00_canonical_seed_metrics.csv") as f:
        rows = list(csv.DictReader(f))
        
    canonical = [r for r in rows if r["variant"] == "canonical"]
    ids = sorted({int(r["realization_id"]) for r in canonical})
    lookup = {(int(r["realization_id"]), r["epoch"]): r for r in canonical}
    
    metrics = ["angular_rmse", "object_rmse", "false_positive_rms", "criterion_crossing_time"]
    
    out_rows = []
    
    pcts = {}
    
    for metric in metrics:
        b = np.array([float(lookup[(i, "baseline")].get(metric, np.nan)) for i in ids])
        m = np.array([float(lookup[(i, "reduced_reliability")].get(metric, np.nan)) for i in ids])
        
        valid = np.isfinite(b) & np.isfinite(m)
        if not valid.any():
            continue
        
        b, m = b[valid], m[valid]
        
        pcts[metric] = 100.0 * (m - b) / b
        
        dz, (dz_lo, dz_hi) = compute_paired_dz_ci(m, b, rng)
        pct, (pct_lo, pct_hi) = compute_percent_change_ci(m, b, rng)
        
        out_rows.append({
            "metric": metric,
            "paired_cohens_dz": dz,
            "dz_ci_low": dz_lo,
            "dz_ci_high": dz_hi,
            "mean_percentage_change": pct,
            "pct_ci_low": pct_lo,
            "pct_ci_high": pct_hi
        })
        
    # Contrast 1: Object vs Angular
    valid = np.isfinite(pcts["object_rmse"]) & np.isfinite(pcts["angular_rmse"])
    p_obj = pcts["object_rmse"][valid]
    p_ang = pcts["angular_rmse"][valid]
    
    diff = p_obj - p_ang
    mean_diff = np.mean(diff)
    indices = rng.integers(0, len(diff), size=(10000, len(diff)))
    boot_diffs = np.mean(diff[indices], axis=1)
    diff_ci = np.quantile(boot_diffs, [0.025, 0.975])
    
    out_rows.append({
        "metric": "object_minus_angular_pct",
        "paired_cohens_dz": np.nan, "dz_ci_low": np.nan, "dz_ci_high": np.nan,
        "mean_percentage_change": float(mean_diff),
        "pct_ci_low": float(diff_ci[0]),
        "pct_ci_high": float(diff_ci[1])
    })
    
    # Contrast 2: FP vs Angular
    valid = np.isfinite(pcts["false_positive_rms"]) & np.isfinite(pcts["angular_rmse"])
    p_fp = pcts["false_positive_rms"][valid]
    p_ang = pcts["angular_rmse"][valid]
    
    diff = p_fp - p_ang
    mean_diff = np.mean(diff)
    indices = rng.integers(0, len(diff), size=(10000, len(diff)))
    boot_diffs = np.mean(diff[indices], axis=1)
    diff_ci = np.quantile(boot_diffs, [0.025, 0.975])
    
    out_rows.append({
        "metric": "fp_minus_angular_pct",
        "paired_cohens_dz": np.nan, "dz_ci_low": np.nan, "dz_ci_high": np.nan,
        "mean_percentage_change": float(mean_diff),
        "pct_ci_low": float(diff_ci[0]),
        "pct_ci_high": float(diff_ci[1])
    })

    with open(ROOT / "analysis_modvis_completion" / "final_analysis_resolution" / "03_paired_effect_sizes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)
        
    print("Effect sizes calculated and saved.")

if __name__ == "__main__":
    run_effect_sizes()
