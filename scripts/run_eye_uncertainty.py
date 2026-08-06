import argparse
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from cerebellum.config import Config
from cerebellum import stimulus as stim
from cerebellum.model import initial_forward_models, _kalman_scalar_update, _otolith_variance, assert_initial_forward_model
from cerebellum.eyes import EyePlant
from cerebellum.context import ContextEstimator

def simulate_closed_loop_perturbed(cfg: Config, rng_state_not_used_for_draws, 
                                   pregen, 
                                   eye_noise_rms=0.0, eye_gain=1.0, 
                                   eye_delay_steps=0, eye_bias=0.0):
    """
    Identical to simulate_closed_loop, but injects perturbation into the eye 
    velocity estimate used for retinal slip compensation. 
    Consumes pregenerated arrays.
    """
    object_motion = np.asarray(pregen["object_motion"], dtype=float)
    if cfg.forward_initialization == "corrected":
        assert_initial_forward_model(cfg)

    u = stim.efference(cfg)
    x_true = stim.latent_selfmotion(cfg, u)
    x_true[2] = object_motion
    
    # We must replicate the logic of stim.vestibular_obs but using pregen
    canal_noise = pregen["canal_z"] * cfg.sigma_canal
    y_canal = x_true[0:1, :] + canal_noise
    
    # Otolith noise calculation based on true sigma (which changes during microgravity)
    sigma_oto_true = np.ones(cfg.T) * cfg.sigma_oto_base
    sigma_oto_true[cfg.micro_start:cfg.micro_end] /= cfg.g_micro
    
    # In 1D, L = R^0.5 = sigma_oto_true
    y_oto = np.zeros((1, cfg.T))
    for t in range(cfg.T):
        y_oto[0, t] = x_true[1, t] + sigma_oto_true[t] * pregen["oto_z"][0, t]
        
    visual_noise = pregen["visual_z"] * cfg.sigma_vis
    eye_noise_z = pregen["eye_noise_z"]

    parts = initial_forward_models(cfg)
    A_fast_e = parts["A_fast_e"].copy()
    A_fast_m = parts["A_fast_m"].copy()
    A_slow_e = parts["A_slow_e"].copy()
    A_slow_m = parts["A_slow_m"].copy()
    A_anchor = np.array([cfg.A0])

    ctx = ContextEstimator(cfg)
    B = np.asarray(cfg.B)
    H_canal = np.array([[1.0, 0.0, 0.0]])
    H_oto = np.array([[0.0, 1.0, 0.0]])
    A_true = np.array(cfg.A0)
    plant = EyePlant(cfg)

    x_hat = np.zeros((3, cfg.T))
    P = np.eye(3) * 0.5
    sigma_adaptive = cfg.sigma_oto_base
    not_state = 0.0
    not_alpha = cfg.dt / (cfg.tau_not + cfg.dt)

    eye_history = np.zeros(cfg.T)

    for t in range(cfg.T):
        # 1. True eye state generates retinal slip
        eye_t = 0.0 if cfg.force_zero_eye else plant.eye_vel
        eye_history[t] = eye_t
        y_retinal = float(A_true @ x_true[:, t] - eye_t + visual_noise[t])
        not_state += not_alpha * (y_retinal - not_state)

        # 2. State prediction
        if t:
            du = (u[t] - u[t - 1]) if cfg.consistent_efference else u[t]
            x_pred = x_hat[:, t - 1] + B * du
            P_pred = P + cfg.Q
        else:
            x_pred = np.zeros(3)
            P_pred = np.eye(3) * 0.5

        # 3. Vestibular updates
        x_upd, P_upd, _ = _kalman_scalar_update(
            x_pred, P_pred, H_canal, y_canal[0, t],
            np.array([[cfg.sigma_canal ** 2]]))

        p_prior = ctx.p
        R_oto_t = _otolith_variance(cfg, p_prior, sigma_oto_true[t], sigma_adaptive)
        Saa = float(P_upd[1, 1])
        x_upd, P_upd, innov_o = _kalman_scalar_update(
            x_upd, P_upd, H_oto, y_oto[0, t], np.array([[R_oto_t]]))
        
        if cfg.oto_variance_policy == "adaptive_legacy":
            sigma_adaptive += cfg.eta_sigma * (innov_o ** 2 - sigma_adaptive ** 2)
            sigma_adaptive = float(np.clip(sigma_adaptive, cfg.sigma_o_min, cfg.sigma_o_max))
        p_t = ctx.update(innov_o, Saa)

        if cfg.context_gating:
            w_e, w_m = 1.0 - p_t, p_t
        else:
            w_e, w_m = 1.0, 0.0
        A_earth = A_slow_e + A_fast_e
        A_micro = A_slow_m + A_fast_m
        A_mix = w_e * A_earth + w_m * A_micro

        # 4. Perturbed eye estimate for compensation
        idx = max(0, t - eye_delay_steps)
        eye_t_delayed = eye_history[idx]
        eye_t_est = eye_t_delayed * eye_gain + eye_bias + eye_noise_z[t] * eye_noise_rms
        
        y_comp = y_retinal + eye_t_est if cfg.eye_compensation else y_retinal
        
        x_upd, P, innov_v = _kalman_scalar_update(
            x_upd, P_upd, A_mix, y_comp,
            np.array([[cfg.sigma_vis ** 2]]))
        x_hat[:, t] = x_upd

        # 5. Forward model update
        grad = -2.0 * innov_v * x_hat[:, t].reshape(1, 3)
        if cfg.enable_fast:
            A_fast_e = cfg.retention_fast * A_fast_e - cfg.eta_fast * w_e * grad
            A_fast_m = cfg.retention_fast * A_fast_m - cfg.eta_fast * w_m * grad
        if cfg.enable_slow:
            A_slow_e = A_anchor + cfg.retention_slow * (A_slow_e - A_anchor) - cfg.eta_slow * w_e * grad
            A_slow_m = A_anchor + cfg.retention_slow * (A_slow_m - A_anchor) - cfg.eta_slow * w_m * grad

        # 6. Eye plant advance
        if t < cfg.T - 1 and not cfg.force_zero_eye:
            plant.advance(not_state, x_upd[0])

    return {
        "v_obj_hat": x_hat[2].copy(),
        "eye_history": eye_history.copy(),
    }


def _pregenerate_streams(cfg, seed):
    rng = np.random.default_rng(seed)
    obj = stim.random_object_events(cfg, rng)
    return {
        "object_motion": obj,
        "canal_z": rng.standard_normal((1, cfg.T)),
        "oto_z": rng.standard_normal((1, cfg.T)),
        "visual_z": rng.standard_normal(cfg.T),
        "eye_noise_z": rng.standard_normal(cfg.T)
    }

def run_paired_seed(args):
    seed, condition = args
    cfg_base = Config(g_micro=1.0)
    cfg_red = Config(g_micro=0.15)
    
    pregen = _pregenerate_streams(cfg_base, seed)
    
    out_base = simulate_closed_loop_perturbed(
        cfg_base, None, pregen, 
        eye_noise_rms=condition["noise"],
        eye_gain=condition["gain"],
        eye_delay_steps=condition["delay_steps"],
        eye_bias=condition["bias"]
    )
    
    out_red = simulate_closed_loop_perturbed(
        cfg_red, None, pregen,
        eye_noise_rms=condition["noise"],
        eye_gain=condition["gain"],
        eye_delay_steps=condition["delay_steps"],
        eye_bias=condition["bias"]
    )
    
    # Calculate FP RMS
    active = np.abs(pregen["object_motion"]) > 1e-3
    mask = np.zeros(cfg_base.T, dtype=bool)
    mask[cfg_base.micro_start:cfg_base.micro_end] = True
    free = mask & ~active
    fp_base = float(np.sqrt(np.mean(out_base["v_obj_hat"][free] ** 2))) if free.any() else 0.0
    fp_red = float(np.sqrt(np.mean(out_red["v_obj_hat"][free] ** 2))) if free.any() else 0.0
    
    effect = fp_red - fp_base
    
    return {
        "seed": seed,
        "fp_base": fp_base,
        "fp_red": fp_red,
        "effect": effect
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="analysis_modvis_completion/limitations_closure/corrective_rerun/01_eye_signal_uncertainty")
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Estimate baseline eye velocity statistics for relative perturbations
    cfg_stats = Config()
    pregen_stats = _pregenerate_streams(cfg_stats, 42)
    out_stats = simulate_closed_loop_perturbed(cfg_stats, None, pregen_stats)
    rms_eye = float(np.sqrt(np.mean(out_stats["eye_history"]**2)))
    peak_eye = float(np.max(np.abs(out_stats["eye_history"])))
    
    print(f"Eye stats: RMS={rms_eye:.4f}, Peak={peak_eye:.4f}")
    
    # Exact reference for relative comparisons
    reference_cond = {"noise": 0.0, "gain": 1.0, "delay_steps": 0, "bias": 0.0}
    
    conditions = []
    for n_pct in [0, 2.5, 5, 10, 20, 40]:
        conditions.append({"type": "noise", "val": n_pct, "noise": (n_pct/100.0)*rms_eye, "gain": 1.0, "delay_steps": 0, "bias": 0.0})
    for g in [0.8, 0.9, 1.0, 1.1, 1.2]:
        conditions.append({"type": "gain", "val": g, "noise": 0.0, "gain": g, "delay_steps": 0, "bias": 0.0})
    for d_ms in [0, 10, 20, 40, 80, 120]:
        d_steps = int(d_ms / 10) # dt=0.01s = 10ms
        conditions.append({"type": "delay", "val": d_ms, "noise": 0.0, "gain": 1.0, "delay_steps": d_steps, "bias": 0.0})
    for b_pct in [0, 2.5, -2.5, 5, -5, 10, -10]:
        conditions.append({"type": "bias", "val": b_pct, "noise": 0.0, "gain": 1.0, "delay_steps": 0, "bias": (b_pct/100.0)*peak_eye})
        
    for n_pct in [0, 10, 20, 40]:
        for g in [0.8, 0.9, 1.0, 1.1, 1.2]:
            for d_ms in [0, 20, 40, 80, 120]:
                d_steps = int(d_ms / 10)
                conditions.append({"type": "combined", "val": f"N{n_pct}_G{g}_D{d_ms}", 
                                   "noise": (n_pct/100.0)*rms_eye, "gain": g, "delay_steps": d_steps, "bias": 0.0})
                                   
    seeds = list(range(1000, 1050)) # 50 paired seeds
    
    # Store bootstrap indices to ensure reproducibility
    n_boot = 2000
    boot_seed = 999
    rng_boot = np.random.default_rng(boot_seed)
    boot_indices = [rng_boot.choice(len(seeds), size=len(seeds), replace=True) for _ in range(n_boot)]
    
    with open(out_dir / "bootstrap_info.json", "w") as f:
        json.dump({
            "method": "percentile",
            "n_resamples": n_boot,
            "seed": boot_seed,
            "description": "2000 resamples of length 50 with replacement, representing indices of the 50 paired seeds. Used across all eye and parameter uncertainty conditions.",
            "indices": [list(map(int, idx)) for idx in boot_indices]
        }, f)
    
    ref_tasks = [(s, reference_cond) for s in seeds]
    with ProcessPoolExecutor() as executor:
        ref_results = list(executor.map(run_paired_seed, ref_tasks))
    ref_mean_effect = np.mean([r["effect"] for r in ref_results])
    
    all_results = []
    
    t0 = time.time()
    for c_idx, cond in enumerate(conditions):
        tasks = [(s, cond) for s in seeds]
        with ProcessPoolExecutor() as executor:
            res = list(executor.map(run_paired_seed, tasks))
            
        effects = np.array([r["effect"] for r in res])
        
        boot_means = np.array([np.mean(effects[idx]) for idx in boot_indices])
            
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        
        mean_eff = np.mean(effects)
        median_eff = np.median(effects)
        sd_eff = np.std(effects)
        se_eff = sd_eff / np.sqrt(len(effects))
        n_pred = np.sum(effects > 0)
        prop_pred = n_pred / len(effects)
        
        excludes_zero = ci_lower > 0 or ci_upper < 0
        predicted_sign = mean_eff > 0
        
        all_results.append({
            "raw_effects": effects.tolist(),
            "type": cond["type"],
            "val": cond["val"],
            "noise": cond["noise"],
            "gain": cond["gain"],
            "delay_steps": cond["delay_steps"],
            "bias": cond["bias"],
            "mean_effect": float(mean_eff),
            "median_effect": float(median_eff),
            "sd_effect": float(sd_eff),
            "se_effect": float(se_eff),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n_predicted": int(n_pred),
            "raw_effects": effects.tolist(),
            "type": cond["type"],
            "val": cond["val"],
            "noise": cond["noise"],
            "gain": cond["gain"],
            "delay_steps": cond["delay_steps"],
            "bias": cond["bias"],
            "mean_effect": float(mean_eff),
            "median_effect": float(median_eff),
            "sd_effect": float(sd_eff),
            "se_effect": float(se_eff),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n_predicted": int(n_pred),
            "prop_predicted": float(prop_pred),
            "relative_to_ref": float(mean_eff / ref_mean_effect) if ref_mean_effect != 0 else 0.0,
            "excludes_zero": bool(excludes_zero)
        })
        
        if (c_idx+1) % 20 == 0:
            print(f"Processed {c_idx+1}/{len(conditions)} conditions...")
            
    print(f"Finished in {time.time()-t0:.2f}s")
    
    with open(out_dir / "eye_uncertainty_raw_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
        
    with open(out_dir / "eye_uncertainty_seed_effects.json", "w") as f:
        json.dump([
            {"type": r["type"], "val": r["val"], "raw_effects": r["raw_effects"]} 
            for r in all_results
        ], f, indent=2)
        
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
