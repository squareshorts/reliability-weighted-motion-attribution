from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'results' / 'extended'
OUT.mkdir(parents=True, exist_ok=True)

T = 2500
DT = 0.01
MICRO_START = int(T / 3)
MICRO_END = int(2 * T / 3)
EVENT_LEN = int(0.8 / DT)
N_EVENTS_PER_EPOCH = 3
SIGMA_CANAL = 0.06
SIGMA_OTO_BASE = 0.06
SIGMA_VIS = 0.08
KAPPA_REDUCED = 0.15
A_TRUE = np.array([1.0, 0.9, 1.0])
SIGMA_OBJ = 0.40
PI_OBJ = (N_EVENTS_PER_EPOCH * EVENT_LEN) / MICRO_START
SEEDS = range(100, 130)
N_BOOT = 10_000


def timebase():
    return np.arange(T) * DT


def efference():
    t = timebase()
    return 0.2 * np.sin(2 * np.pi * 0.12 * t)


def latent_selfmotion(u):
    t = timebase()
    x = np.zeros((3, T))
    x[0] = 0.6 * np.sin(2 * np.pi * 0.35 * t) + 0.20 * np.sin(2 * np.pi * 0.09 * t + 0.5)
    x[1] = 0.35 * np.sin(2 * np.pi * 0.18 * t + 1.2) + 0.08 * np.sin(2 * np.pi * 0.03 * t)
    x[0] += 0.8 * u
    x[1] += 0.6 * u
    return x


def random_object_events(rng):
    v = np.zeros(T)
    epochs = [np.arange(0, MICRO_START), np.arange(MICRO_START, MICRO_END), np.arange(MICRO_END, T)]
    for idx in epochs:
        lo = idx[0] + 50
        hi = idx[-1] - EVENT_LEN - 50
        starts = rng.choice(np.arange(lo, hi), size=N_EVENTS_PER_EPOCH, replace=False)
        for s in starts:
            v[s:s + EVENT_LEN] = rng.choice([0.4, -0.4])
    return v


def bootstrap_mean(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    draws = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def generate_seed(seed):
    rng = np.random.default_rng(seed)
    obj = random_object_events(rng)
    u = efference()
    x = latent_selfmotion(u)
    x[2] = obj

    kappa = np.ones(T)
    kappa[MICRO_START:MICRO_END] = KAPPA_REDUCED
    sigma_oto = SIGMA_OTO_BASE / kappa

    y_canal = x[0] + rng.standard_normal(T) * SIGMA_CANAL
    y_oto = x[1] + rng.standard_normal(T) * sigma_oto
    visual_noise = rng.standard_normal(T) * SIGMA_VIS
    y_vis_comp = A_TRUE @ x + visual_noise

    residual = y_vis_comp - y_canal - 0.9 * y_oto

    s0 = SIGMA_VIS**2 + SIGMA_CANAL**2 + (0.9**2) * sigma_oto**2
    s1 = s0 + SIGMA_OBJ**2
    n0 = np.exp(-0.5 * residual**2 / s0) / np.sqrt(2 * np.pi * s0)
    n1 = np.exp(-0.5 * residual**2 / s1) / np.sqrt(2 * np.pi * s1)
    p_move = PI_OBJ * n1 / (PI_OBJ * n1 + (1.0 - PI_OBJ) * n0)
    bayes = p_move * (SIGMA_OBJ**2 / s1) * residual

    estimates = {
        'direct sensory subtraction': residual,
        'static Bayesian causal attribution': bayes,
    }
    epoch_bounds = {
        'baseline': (0, MICRO_START),
        'reduced': (MICRO_START, MICRO_END),
        'recovery': (MICRO_END, T),
    }
    rows = []
    for model, est in estimates.items():
        by_epoch = {}
        for epoch, (lo, hi) in epoch_bounds.items():
            idx = np.arange(lo, hi)
            active = idx[np.abs(obj[idx]) > 1e-3]
            free = idx[np.abs(obj[idx]) <= 1e-3]
            by_epoch[epoch] = {
                'fp_rms': float(np.sqrt(np.mean(est[free] ** 2))),
                'obj_rmse': float(np.sqrt(np.mean((est[idx] - obj[idx]) ** 2))),
                'event_rmse': float(np.sqrt(np.mean((est[active] - obj[active]) ** 2))),
            }
        b = by_epoch['baseline']
        r = by_epoch['reduced']
        q = by_epoch['recovery']
        rows.append({
            'seed': seed,
            'model': model,
            'baseline_fp_rms': b['fp_rms'],
            'reduced_fp_rms': r['fp_rms'],
            'recovery_fp_rms': q['fp_rms'],
            'fp_pct_change': 100.0 * (r['fp_rms'] - b['fp_rms']) / b['fp_rms'],
            'baseline_obj_rmse': b['obj_rmse'],
            'reduced_obj_rmse': r['obj_rmse'],
            'recovery_obj_rmse': q['obj_rmse'],
            'obj_pct_change': 100.0 * (r['obj_rmse'] - b['obj_rmse']) / b['obj_rmse'],
            'baseline_event_rmse': b['event_rmse'],
            'reduced_event_rmse': r['event_rmse'],
        })
    return rows


rows = [row for seed in SEEDS for row in generate_seed(seed)]
seed_path = OUT / 'independent_baselines_seedlevel.csv'
with seed_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

summary = []
for mi, model in enumerate(['direct sensory subtraction', 'static Bayesian causal attribution']):
    r = [x for x in rows if x['model'] == model]
    out = {'model': model}
    fields = [
        ('baseline_fp_rms', 'baseline_fp_rms'),
        ('reduced_fp_rms', 'reduced_fp_rms'),
        ('recovery_fp_rms', 'recovery_fp_rms'),
        ('fp_pct_change', 'fp_pct_change'),
        ('baseline_obj_rmse', 'baseline_obj_rmse'),
        ('reduced_obj_rmse', 'reduced_obj_rmse'),
        ('recovery_obj_rmse', 'recovery_obj_rmse'),
        ('obj_pct_change', 'obj_pct_change'),
        ('baseline_event_rmse', 'baseline_event_rmse'),
        ('reduced_event_rmse', 'reduced_event_rmse'),
    ]
    for j, (name, key) in enumerate(fields):
        mean, lo, hi = bootstrap_mean([x[key] for x in r], 7200 + 100 * mi + j)
        out[f'{name}_mean'] = mean
        out[f'{name}_lo'] = lo
        out[f'{name}_hi'] = hi
    recdiff = [x['recovery_fp_rms'] - x['baseline_fp_rms'] for x in r]
    mean, lo, hi = bootstrap_mean(recdiff, 7990 + mi)
    out['recovery_minus_baseline_fp_mean'] = mean
    out['recovery_minus_baseline_fp_lo'] = lo
    out['recovery_minus_baseline_fp_hi'] = hi
    summary.append(out)

summary_path = OUT / 'independent_baselines_summary.csv'
with summary_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

print(seed_path)
print(summary_path)
