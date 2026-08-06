"""Trial orchestration for the three explicit eye-path implementations."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from . import model as M
from . import stimulus as stim


VARIANTS = ("legacy_fixed_eye_trace", "per_trial_eye_trace",
            "corrected_closed_loop")


def configured(cfg, variant):
    """Return a configuration with scientifically explicit variant defaults."""
    if variant == "legacy_fixed_eye_trace":
        return replace(cfg, simulation_variant=variant,
                       forward_initialization="legacy",
                       forward_rule="legacy_tracking",
                       oto_variance_policy="adaptive_legacy",
                       eye_compensation=False,
                       legacy_visual_nonlinearity=True)
    if variant == "per_trial_eye_trace":
        return replace(cfg, simulation_variant=variant,
                       forward_initialization="legacy",
                       forward_rule="legacy_tracking",
                       oto_variance_policy="adaptive_legacy",
                       eye_compensation=False,
                       legacy_visual_nonlinearity=True)
    if variant == "corrected_closed_loop":
        return replace(cfg, simulation_variant=variant,
                       forward_initialization=cfg.forward_initialization,
                       forward_rule=cfg.forward_rule,
                       oto_variance_policy=cfg.oto_variance_policy,
                       eye_compensation=cfg.eye_compensation,
                       legacy_visual_nonlinearity=False)
    raise ValueError(f"unknown variant: {variant}")


def _legacy_baseline(cfg):
    base = M.simulate(cfg, np.random.default_rng(cfg.seed),
                      stim.fixed_object_motion(cfg))
    learned = {k: base[k] for k in
               ("A_fast_e", "A_fast_m", "A_slow_e", "A_slow_m")}
    eye, gain = M.okr_pass(cfg, base["y_not"], base["omega_hat"])
    return learned, eye, gain


def run_trial(cfg, variant, trial, legacy_cache=None, object_free=False):
    """Run one trial with matched event and sensory RNG state across variants."""
    cfg = configured(cfg, variant)
    rng = np.random.default_rng(100 + trial)
    object_motion = (np.zeros(cfg.T) if object_free
                     else stim.random_object_events(cfg, rng))

    if variant == "corrected_closed_loop":
        out = M.simulate_closed_loop(cfg, rng, object_motion,
                                     trial_id=trial)
    else:
        learned, fixed_eye, fixed_gain = legacy_cache or _legacy_baseline(cfg)
        if variant == "legacy_fixed_eye_trace":
            eye, eye_id = fixed_eye, -1
            gain = fixed_gain
        else:
            eye_rng = np.random.default_rng(10_000 + trial)
            eye_source = M.simulate(cfg, eye_rng, object_motion, init_A=learned)
            eye, gain = M.okr_pass(
                cfg, eye_source["y_not"], eye_source["omega_hat"])
            eye_id = trial
        out = M.simulate(cfg, rng, object_motion, eye_vel=eye, init_A=learned)
        out.update({"eye_vel": eye, "g_okr": gain,
                    "object_motion": object_motion.copy(),
                    "trial_id": trial, "object_trial_id": trial,
                    "eye_trial_id": eye_id,
                    "oracle_variance": False})
    out["variant"] = np.array(variant)
    out["object_free_control"] = bool(object_free)
    return out


def save_trial(path, out):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in out.items()
                    if isinstance(v, (np.ndarray, np.number, int, float, bool, str))}
    np.savez_compressed(path, **serializable)


def run_ensemble(cfg, variant, output_root, object_free=False, result_label=None):
    """Run and save all trial-level data; return written paths."""
    cfg = configured(cfg, variant)
    root = Path(output_root) / variant / "trial_level"
    root.mkdir(parents=True, exist_ok=True)
    legacy_cache = None if variant == "corrected_closed_loop" else _legacy_baseline(cfg)
    paths = []
    for trial in range(cfg.n_trials):
        out = run_trial(cfg, variant, trial, legacy_cache, object_free)
        if result_label is not None:
            out["variant"] = np.array(result_label)
        suffix = "_object_free" if object_free else ""
        path = root / f"trial_{trial:03d}{suffix}.npz"
        save_trial(path, out)
        paths.append(path)
    metadata = {"variant": variant, "result_label": result_label or variant,
                "n_trials": cfg.n_trials,
                "seed_rule": "event/sensory RNG seed = 100 + trial",
                "object_free": object_free, "config": cfg.to_dict()}
    with open(root.parent / ("metadata_object_free.json" if object_free else "metadata.json"),
              "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return paths
