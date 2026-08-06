import argparse
import csv
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import qmc, spearmanr, rankdata, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

from cerebellum import stimulus as stim
from cerebellum.config import Config
from cerebellum.model import simulate_closed_loop

PARAM_BOUNDS = {
    "sigma_canal": (0.03, 0.12),
    "sigma_oto_base": (0.03, 0.12),
    "sigma_vis": (0.04, 0.16),
    "sigma_not": (0.025, 0.10),
    "q_omega": (5e-5, 2e-4),
    "q_a": (1e-4, 4e-4),
    "q_obj": (4e-4, 1.6e-3),
    "eta_fast": (1.5e-3, 6e-3),
    "eta_slow": (1.5e-4, 6e-4),
    "tau_not": (0.075, 0.30),
    "B_omega": (0.4, 1.2),  
    "B_a": (0.3, 0.9),      
}

def _pregenerate_seeds(cfg, seeds):
    pregen = []
    for s in seeds:
        rng = np.random.default_rng(s)
        obj = stim.random_object_events(cfg, rng)
        pregen.append({
            "seed": s,
            "object_motion": obj,
            "canal_z": rng.standard_normal((1, cfg.T)),
            "oto_z": rng.standard_normal((1, cfg.T)),
            "visual_z": rng.standard_normal(cfg.T),
        })
    return pregen

def simulate_closed_loop_pregen(cfg, pregen_seed_data):
    rng = np.random.default_rng(pregen_seed_data["seed"])
    obj = pregen_seed_data["object_motion"]
    return simulate_closed_loop(cfg, rng, obj, trial_id=pregen_seed_data["seed"])

def _run_config(args):
    sample_id, config_dict, params, pregen_data_list, n_boot, boot_indices = args
    
    for k, v in params.items():
        if k not in ["B_omega", "B_a"]:
            config_dict[k] = v
            
    if "B_omega" in params and "B_a" in params:
        config_dict["B"] = (params["B_omega"], params["B_a"], 0.0)
    
    cfg_base = replace(Config(**config_dict), g_micro=1.0)
    cfg_red = replace(Config(**config_dict), g_micro=0.15)
    
    effects = []
    fp_bases = []
    fp_reds = []
    
    failed = False
    error_msg = ""
    
    for pregen in pregen_data_list:
        try:
            out_base = simulate_closed_loop_pregen(cfg_base, pregen)
            out_red = simulate_closed_loop_pregen(cfg_red, pregen)
            
            active = np.abs(pregen["object_motion"]) > 1e-3
            mask_base = np.zeros(cfg_base.T, dtype=bool)
            mask_base[cfg_base.micro_start:cfg_base.micro_end] = True
            free = mask_base & ~active
            
            fp_b = float(np.sqrt(np.mean(out_base["x_hat"][2, free] ** 2))) if free.any() else 0.0
            fp_r = float(np.sqrt(np.mean(out_red["x_hat"][2, free] ** 2))) if free.any() else 0.0
            
            effects.append(fp_r - fp_b)
            fp_bases.append(fp_b)
            fp_reds.append(fp_r)
        except Exception as e:
            failed = True
            error_msg = str(e) + "\n" + traceback.format_exc()
            break
            
    if failed:
        return {"sample_id": sample_id, "failed": True, "error": error_msg}
        
    effects = np.array(effects)
    
    boot_means = np.array([np.mean(effects[idx]) for idx in boot_indices])
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    mean_eff = np.mean(effects)
    prop_pred = np.sum(effects > 0) / len(effects)
    excludes_zero = ci_lower > 0 or ci_upper < 0
    
    if mean_eff > 0 and prop_pred >= 0.90 and excludes_zero and ci_lower > 0:
        status = "ROBUST"
    elif mean_eff <= 0 and excludes_zero and ci_upper < 0:
        status = "REVERSED"
    elif mean_eff > 0 and (prop_pred < 0.90 or not excludes_zero):
        status = "ATTENUATED"
    else:
        status = "ATTENUATED"
        
    return {
        "sample_id": sample_id,
        "failed": False,
        "error": "",
        "raw_effects": effects.tolist(),
        "mean_effect": float(mean_eff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "prop_predicted": float(prop_pred),
        "status": status,
        "mean_fp_base": float(np.mean(fp_bases)),
        "mean_fp_red": float(np.mean(fp_reds))
    }

def calc_prcc(X, y):
    n, d = X.shape
    X_rank = np.apply_along_axis(rankdata, 0, X)
    y_rank = rankdata(y)
    prccs = []
    for j in range(d):
        mask = np.ones(d, dtype=bool)
        mask[j] = False
        X_other = X_rank[:, mask]
        X_other_with_intercept = np.column_stack([np.ones(n), X_other])
        
        beta_X, _, _, _ = np.linalg.lstsq(X_other_with_intercept, X_rank[:, j], rcond=None)
        r_X = X_rank[:, j] - X_other_with_intercept @ beta_X
        
        beta_y, _, _, _ = np.linalg.lstsq(X_other_with_intercept, y_rank, rcond=None)
        r_y = y_rank - X_other_with_intercept @ beta_y
        
        r, _ = pearsonr(r_X, r_y)
        prccs.append(r)
    return prccs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="analysis_modvis_completion/limitations_closure/corrective_rerun/02_parameter_sensitivity")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = Config(n_trials=1)
    
    # 1. LHS
    sampler = qmc.LatinHypercube(d=len(PARAM_BOUNDS), seed=args.seed)
    sample = sampler.random(n=args.n_samples)
    l_bounds = [PARAM_BOUNDS[k][0] for k in PARAM_BOUNDS.keys()]
    u_bounds = [PARAM_BOUNDS[k][1] for k in PARAM_BOUNDS.keys()]
    param_sets_scaled = qmc.scale(sample, l_bounds, u_bounds)
    
    param_list = []
    for i in range(args.n_samples):
        p_dict = {k: param_sets_scaled[i, j] for j, k in enumerate(PARAM_BOUNDS.keys())}
        p_dict["sample_id"] = i
        param_list.append(p_dict)
        
    with open(out_dir / "parameter_samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PARAM_BOUNDS.keys()) + ["sample_id"])
        writer.writeheader()
        writer.writerows(param_list)

    # 2. Pre-generate seeds
    seeds = list(range(3000, 3000 + args.n_seeds))
    pregen_data_list = _pregenerate_seeds(base_cfg, seeds)
    
    n_boot = 2000
    rng_boot = np.random.default_rng(999)
    boot_indices = [rng_boot.choice(args.n_seeds, size=args.n_seeds, replace=True) for _ in range(n_boot)]
    
    # Get reference effect
    ref_task = (-1, base_cfg.to_dict(), {}, pregen_data_list, n_boot, boot_indices)
    ref_out = _run_config(ref_task)
    ref_mean_effect = ref_out["mean_effect"] if not ref_out["failed"] else 0.0

    # 3. Tasks
    tasks = []
    config_dict = base_cfg.to_dict()
    for p_dict in param_list:
        sample_id = p_dict["sample_id"]
        sim_params = {k: v for k, v in p_dict.items() if k != "sample_id"}
        tasks.append((sample_id, config_dict.copy(), sim_params, pregen_data_list, n_boot, boot_indices))

    print(f"Running {len(tasks)} sensitivity configs across {args.n_seeds} seeds each...")
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(_run_config, tasks))
        
    t1 = time.time()
    print(f"Finished in {t1 - t0:.2f} s")

    # 4. Results
    result_rows = []
    exceptions_list = []
    failed_samples = 0
    
    for i in range(args.n_samples):
        res = results[i]
        row = param_list[i].copy()
        
        if res["failed"]:
            failed_samples += 1
            exceptions_list.append({"sample_id": res["sample_id"], "error": res["error"]})
            row.update({"mean_effect": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, 
                        "prop_predicted": np.nan, "status": "FAILED",
                        "mean_fp_base": np.nan, "mean_fp_red": np.nan,
                        "is_positive": np.nan, "is_attenuated": np.nan,
                        "raw_effects": []})
        else:
            is_pos = res["mean_effect"] > 0
            is_att = res["mean_effect"] < 0.25 * ref_mean_effect
            row.update({
                "mean_effect": res["mean_effect"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "prop_predicted": res["prop_predicted"],
                "status": res["status"],
                "mean_fp_base": res["mean_fp_base"],
                "mean_fp_red": res["mean_fp_red"],
                "is_positive": is_pos,
                "is_attenuated": is_att,
                "raw_effects": res["raw_effects"]
            })
        result_rows.append(row)
        
    with open(out_dir / "sensitivity_results.csv", "w", newline="") as f:
        # Don't write raw_effects to CSV as it breaks easy parsing
        fieldnames = [k for k in result_rows[0].keys() if k != "raw_effects"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in result_rows:
            r_csv = {k: v for k, v in r.items() if k != "raw_effects"}
            writer.writerow(r_csv)
            
    with open(out_dir / "parameter_seed_effects.json", "w") as f:
        json.dump([
            {"sample_id": r["sample_id"], "raw_effects": r.get("raw_effects", [])}
            for r in result_rows
        ], f, indent=2)

    if exceptions_list:
        with open(out_dir / "exceptions_log.json", "w") as f:
            json.dump(exceptions_list, f, indent=2)

    # 5. Importance Analysis
    valid_rows = [r for r in result_rows if r["status"] != "FAILED"]
    corrs = {}
    interactions_diagnostics = {}
    
    if len(valid_rows) > 0:
        X = np.array([[r[k] for k in PARAM_BOUNDS.keys()] for r in valid_rows])
        y = np.array([r["mean_effect"] for r in valid_rows])
        
        # Spearman & PRCC
        prccs = calc_prcc(X, y)
        for j, k in enumerate(PARAM_BOUNDS.keys()):
            c, p = spearmanr(X[:, j], y)
            corrs[k] = {"spearman_rho": c, "spearman_p": p, "prcc": prccs[j]}
            
        # Cross-validated permutation importance
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        perm_importances = []
        for train_idx, test_idx in kf.split(X):
            rf_cv = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_cv.fit(X[train_idx], y[train_idx])
            r = permutation_importance(rf_cv, X[test_idx], y[test_idx], n_repeats=5, random_state=42)
            perm_importances.append(r.importances_mean)
        
        cv_perm_mean = np.mean(perm_importances, axis=0)
        for j, k in enumerate(PARAM_BOUNDS.keys()):
            corrs[k]["cv_permutation_importance"] = cv_perm_mean[j]
            
        # Diagnostics for strongest two-way interactions
        rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_full.fit(X, y)
        
        # Build interaction terms and measure their PRCC controlling for main effects
        keys = list(PARAM_BOUNDS.keys())
        d = len(keys)
        n_val = X.shape[0]
        
        # Base matrix is X
        inter_prccs = []
        for (i, j) in combinations(range(d), 2):
            X_inter = X[:, i] * X[:, j]
            # Control for X
            X_combined = np.column_stack([X, X_inter])
            # calculate PRCC of the last column
            X_combined_rank = np.apply_along_axis(rankdata, 0, X_combined)
            y_rank = rankdata(y)
            X_other_with_intercept = np.column_stack([np.ones(n_val), X_combined_rank[:, :-1]])
            
            beta_X, _, _, _ = np.linalg.lstsq(X_other_with_intercept, X_combined_rank[:, -1], rcond=None)
            r_X = X_combined_rank[:, -1] - X_other_with_intercept @ beta_X
            
            beta_y, _, _, _ = np.linalg.lstsq(X_other_with_intercept, y_rank, rcond=None)
            r_y = y_rank - X_other_with_intercept @ beta_y
            
            r_val, _ = pearsonr(r_X, r_y)
            inter_prccs.append((keys[i], keys[j], r_val))
            
        inter_prccs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_interactions = inter_prccs[:10]
        
        for k1, k2, r_val in top_interactions:
            interactions_diagnostics[f"{k1}_x_{k2}"] = {"prcc_interaction": r_val}
            
        with open(out_dir / "importance_analysis.json", "w") as f:
            json.dump({
                "main_effects": corrs,
                "top_interactions": interactions_diagnostics
            }, f, indent=2)

    # 6. Evaluation Criteria
    total = args.n_samples
    n_valid = len(valid_rows)
    
    n_pos = len([r for r in valid_rows if r["is_positive"]])
    n_robust = len([r for r in valid_rows if r["status"] == "ROBUST"])
    n_reversed = len([r for r in valid_rows if r["status"] == "REVERSED"])
    n_attenuated = len([r for r in valid_rows if r["is_attenuated"]])
    
    summary = {
        "n_samples": total,
        "failed_samples": failed_samples,
        "failed_percentage": (failed_samples / total) * 100.0,
        "valid_samples": n_valid,
        "positive_effect_percentage": (n_pos / n_valid * 100.0) if n_valid else 0.0,
        "robust_positive_percentage": (n_robust / n_valid * 100.0) if n_valid else 0.0,
        "robust_reversal_percentage": (n_reversed / n_valid * 100.0) if n_valid else 0.0,
        "attenuation_percentage": (n_attenuated / n_valid * 100.0) if n_valid else 0.0,
        "ref_mean_effect": float(ref_mean_effect)
    }
    
    with open(out_dir / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
