import csv
import numpy as np
from pathlib import Path

ROOT = Path("c:/work/cerebellum").resolve()
FINAL = ROOT / "analysis_modvis_completion" / "final_analysis_resolution"

def boot_ci(vals, n=2000, rng=None):
    if rng is None: rng = np.random.default_rng(20260731)
    if not len(vals): return np.nan, (np.nan, np.nan)
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(vals), len(vals))
        means.append(np.mean(vals[idx]))
    return np.mean(vals), np.quantile(means, [0.025, 0.975])

def generate_reliability_tables():
    with open(ROOT / "analysis_modvis_completion" / "04_reliability_sweep_new" / "trial_level_reliability_sweep.csv") as f:
        rows = list(csv.DictReader(f))
        
    canonical = [r for r in rows if r["epoch"] == "reduced_reliability"]
    
    data = {}
    for r in canonical:
        k = float(r["kappa"])
        s = int(r["seed_identity"])
        if k not in data: data[k] = {}
        data[k][s] = float(r["false_positive_rms"])
        
    baseline_canonical = [r for r in rows if r["epoch"] == "baseline"]
    baseline_data = {}
    for r in baseline_canonical:
        k = float(r["kappa"])
        s = int(r["seed_identity"])
        if k not in baseline_data: baseline_data[k] = {}
        baseline_data[k][s] = float(r["false_positive_rms"])
        
    kappas = sorted(data.keys())
    seeds = sorted(data[kappas[0]].keys())
    
    rng = np.random.default_rng(20260731)
    
    # Generate seed slopes
    slope_rows = []
    slopes = []
    for s in seeds:
        x = np.array([1.0 - k for k in kappas])
        y = np.array([data[k][s] for k in kappas])
        slope = float(np.polyfit(x, y, 1)[0])
        slopes.append(slope)
        
    slopes = np.array(slopes)
    mean_slope, ci_slope = boot_ci(slopes, rng=rng)
    
    for s, slope in zip(seeds, slopes):
        slope_rows.append({
            "relationship": "1_minus_kappa",
            "realization_id": s,
            "rng_seed": 100 + s,
            "within_seed_slope": slope,
            "mean_slope": mean_slope,
            "mean_slope_bootstrap_ci_low": ci_slope[0],
            "mean_slope_bootstrap_ci_high": ci_slope[1],
            "n_realizations": len(seeds)
        })
        
    with open(FINAL / "04_reliability_sweep_seed_slopes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=slope_rows[0].keys())
        writer.writeheader()
        writer.writerows(slope_rows)
        
    # Generate level summary
    summary_rows = []
    for k in kappas:
        b_vals = np.array([baseline_data[k][s] for s in seeds])
        m_vals = np.array([data[k][s] for s in seeds])
        
        _, m_ci = boot_ci(m_vals, rng=rng)
        
        deltas = m_vals - b_vals
        _, d_ci = boot_ci(deltas, rng=rng)
        
        pcts = 100.0 * deltas / b_vals
        _, p_ci = boot_ci(pcts, rng=rng)
        
        summary_rows.append({
            "kappa": k,
            "metric": "false_positive_rms",
            "n_realizations": len(seeds),
            "baseline_mean": np.mean(b_vals),
            "manipulated_mean": np.mean(m_vals),
            "manipulated_sd": np.std(m_vals, ddof=1),
            "monte_carlo_se": np.std(m_vals, ddof=1) / np.sqrt(len(seeds)),
            "mean_bootstrap_ci_low": m_ci[0],
            "mean_bootstrap_ci_high": m_ci[1],
            "paired_absolute_change_mean": np.mean(deltas),
            "paired_absolute_change_ci_low": d_ci[0],
            "paired_absolute_change_ci_high": d_ci[1],
            "paired_percent_change_mean": np.mean(pcts),
            "paired_percent_change_ci_low": p_ci[0],
            "paired_percent_change_ci_high": p_ci[1],
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 20260731,
            "realization_ids": ";".join(str(s) for s in seeds),
            "sigma_oto": "0.6",
            "otolith_observation_variance": "0.36"
        })
        
    with open(FINAL / "04_reliability_sweep_level_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

if __name__ == "__main__":
    generate_reliability_tables()
