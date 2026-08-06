"""Performance metrics and adaptation-constant estimation."""
from __future__ import annotations
import numpy as np


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))


def detection_latency(v_obj_hat, obj_active, idx, cfg):
    """Mean latency from object-motion onset to first threshold crossing."""
    onsets = np.where(np.diff(obj_active.astype(int)) == 1)[0]
    lats = []
    for o in onsets:
        if o not in idx:
            continue
        for td in range(o, min(o + int(1.0 / cfg.dt), cfg.T)):
            if abs(v_obj_hat[td]) > cfg.detect_threshold:
                lats.append((td - o) * cfg.dt)
                break
    return lats


def criterion_crossings(v_obj_hat, obj_active, idx, cfg):
    """One latency per event; missing crossings are NaN, never zero."""
    active = np.asarray(obj_active, dtype=bool)
    onsets = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    allowed = np.zeros(cfg.T, dtype=bool)
    allowed[np.asarray(idx)] = True
    values = []
    horizon = int(round(1.0 / cfg.dt))
    for onset in onsets:
        if not allowed[onset]:
            continue
        hits = np.flatnonzero(np.abs(v_obj_hat[onset:min(cfg.T, onset + horizon)])
                              > cfg.detect_threshold)
        values.append(np.nan if not len(hits) else float(hits[0] * cfg.dt))
    return np.asarray(values, dtype=float)


def cohens_d_paired(after, before):
    delta = np.asarray(after, float) - np.asarray(before, float)
    sd = np.std(delta, ddof=1)
    return float(np.mean(delta) / sd) if len(delta) > 1 and sd > 0 else np.nan


def bootstrap_mean_ci(values, rng, n_boot=2000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (np.nan, np.nan)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


def percent_change(after, before):
    return float(100.0 * (after - before) / before) if before != 0 else np.nan


def transition_times(t, y, start, end=None):
    """Nonparametric 25/50/75% times, settling time, and total change."""
    t = np.asarray(t); y = np.asarray(y, float)
    mask = t >= start
    if end is not None:
        mask &= t < end
    tt, yy = t[mask] - start, y[mask]
    if len(yy) < 2:
        return {k: np.nan for k in ("t25", "t50", "t75", "settling_time", "total_change")}
    n_tail = max(5, len(yy) // 10)
    y0, yinf = float(yy[0]), float(np.mean(yy[-n_tail:]))
    change = yinf - y0
    out = {"total_change": float(change)}
    for frac in (0.25, 0.50, 0.75):
        target = y0 + frac * change
        hits = np.flatnonzero(yy >= target) if change >= 0 else np.flatnonzero(yy <= target)
        out[f"t{int(frac * 100)}"] = np.nan if not len(hits) else float(tt[hits[0]])
    tol = max(abs(change) * 0.05, 1e-12)
    outside = np.flatnonzero(np.abs(yy - yinf) > tol)
    out["settling_time"] = 0.0 if not len(outside) else (
        np.nan if outside[-1] == len(yy) - 1 else float(tt[outside[-1] + 1]))
    return out


def fit_exponential_transition(t, y, start, end, min_r2=0.8,
                               taus=None):
    """Fit y=a+b*exp(-time/tau) and reject uninterpretable fits.

    A fit is supported only when its R2 reaches ``min_r2``, the observed
    transition is non-negligible relative to sample variability, and it beats
    a constant model.  Raw/nonparametric summaries should always accompany it.
    """
    if taus is None:
        taus = np.geomspace(0.02, max(0.04, end - start) * 4, 500)
    t = np.asarray(t); y = np.asarray(y, float)
    m = (t >= start) & (t < end) & np.isfinite(y)
    tt, yy = t[m] - start, y[m]
    if len(yy) < 20:
        return {"supported": False, "reason": "fewer than 20 samples", "tau": np.nan, "r2": np.nan}
    sst = float(np.sum((yy - yy.mean()) ** 2))
    best = None
    for tau in taus:
        X = np.column_stack((np.ones(len(tt)), np.exp(-tt / tau)))
        beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        pred = X @ beta
        sse = float(np.sum((yy - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(tau), beta, pred)
    sse, tau, beta, _ = best
    X_linear = np.column_stack((np.ones(len(tt)), tt))
    linear_pred = X_linear @ np.linalg.lstsq(X_linear, yy, rcond=None)[0]
    linear_sse = float(np.sum((yy - linear_pred) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    transition = float((beta[0]) - (beta[0] + beta[1]))
    noise = float(np.std(np.diff(yy))) if len(yy) > 1 else np.nan
    duration = float(end - start)
    supported = bool(np.isfinite(r2) and r2 >= min_r2
                     and abs(transition) > 2 * noise
                     and tau < duration
                     and sse < 0.95 * linear_sse)
    reason = "supported" if supported else "poor fit or negligible transition"
    return {"supported": supported, "reason": reason, "tau": tau,
            "r2": float(r2),
            "linear_r2": float(1.0 - linear_sse / sst) if sst > 0 else np.nan,
            "asymptote": float(beta[0]),
            "initial": float(beta.sum()), "transition": transition,
            "start": float(start), "end": float(end),
            "model": "a + b*exp(-(t-start)/tau)"}


def fit_tau(t, y, t0, taus=None):
    """Least-squares single-exponential adaptation time constant after t0."""
    if taus is None:
        taus = np.linspace(0.5, 60.0, 400)
    m = t >= t0
    tt, yy = t[m] - t0, y[m]
    ok = np.isfinite(yy)
    tt, yy = tt[ok], yy[ok]
    if len(tt) < 20:
        return np.nan
    y0 = yy[0]
    y_inf = np.nanmean(yy[-max(10, len(yy) // 10):])
    best_tau, best_err = np.nan, np.inf
    for tau in taus:
        pred = y_inf + (y0 - y_inf) * np.exp(-tt / tau)
        err = np.mean((yy - pred) ** 2)
        if err < best_err:
            best_err, best_tau = err, tau
    return float(best_tau)


def time_to_fraction(t, y, t0, frac=0.5, direction="up"):
    m = t >= t0
    tt, yy = t[m] - t0, y[m]
    if direction == "up":
        target = yy[0] + frac * (np.nanmax(yy) - yy[0])
        idx = np.where(yy >= target)[0]
    else:
        target = yy[0] + frac * (np.nanmin(yy) - yy[0])
        idx = np.where(yy <= target)[0]
    return np.nan if len(idx) == 0 else float(tt[idx[0]])


def sliding_gain(eye_vel, slip, dt, win_s=2.0):
    win = int(win_s / dt)
    g = np.full_like(eye_vel, np.nan, dtype=float)
    for i in range(win, len(eye_vel)):
        e, s = eye_vel[i - win:i], slip[i - win:i]
        g[i] = np.sqrt(np.mean(e ** 2)) / (np.sqrt(np.mean(s ** 2)) + 1e-9)
    return g
