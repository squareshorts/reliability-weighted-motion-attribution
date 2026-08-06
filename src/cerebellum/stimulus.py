"""Generation of latent states and sensory observations."""
from __future__ import annotations
import numpy as np


def efference(cfg) -> np.ndarray:
    """Active self-motion command (efference copy)."""
    t = cfg.time
    return 0.2 * np.sin(2 * np.pi * 0.12 * t)


def latent_selfmotion(cfg, u: np.ndarray) -> np.ndarray:
    """Angular velocity (omega) and linear/tilt acceleration (a) traces.

    The object-motion channel is filled in separately per condition/trial.
    """
    t = cfg.time
    x = np.zeros((3, cfg.T))
    x[0] = 0.6 * np.sin(2 * np.pi * 0.35 * t) + 0.20 * np.sin(2 * np.pi * 0.09 * t + 0.5)
    x[1] = 0.35 * np.sin(2 * np.pi * 0.18 * t + 1.2) + 0.08 * np.sin(2 * np.pi * 0.03 * t)
    x[0] += 0.8 * u
    x[1] += 0.6 * u
    return x


def fixed_object_motion(cfg) -> np.ndarray:
    """The deterministic object-motion trace used for the illustrative run."""
    v = np.zeros(cfg.T)
    v[600:950] = 0.45
    v[1500:1750] = -0.35
    return v


def random_object_events(cfg, rng: np.random.Generator) -> np.ndarray:
    """Balanced object-motion events, n per epoch, used for trial statistics."""
    v = np.zeros(cfg.T)
    epochs = epoch_indices(cfg)
    for idx in epochs.values():
        lo = idx[0] + 50
        hi = idx[-1] - cfg.event_len - 50
        starts = rng.choice(np.arange(lo, hi), size=cfg.n_events_per_epoch, replace=False)
        for s in starts:
            v[s:s + cfg.event_len] = rng.choice([0.4, -0.4])
    return v


def epoch_indices(cfg):
    return {
        "pre": np.arange(0, cfg.micro_start),
        "micro": np.arange(cfg.micro_start, cfg.micro_end),
        "post": np.arange(cfg.micro_end, cfg.T),
    }


def vestibular_obs(cfg, x_true: np.ndarray, rng: np.random.Generator,
                   return_standard_normals: bool = False):
    """Canal and otolith measurements.

    ``return_standard_normals`` exposes the common-random-number identities
    used by paired simulation sweeps without changing the generated samples.
    """
    g = cfg.gravity()
    sigma_oto = cfg.sigma_oto_base / np.clip(g, 1e-3, None)
    canal_z = rng.standard_normal(cfg.T)
    oto_z = rng.standard_normal(cfg.T)
    y_canal = x_true[0] + canal_z * cfg.sigma_canal
    y_oto = x_true[1] + oto_z * sigma_oto
    if return_standard_normals:
        return y_canal, y_oto, sigma_oto, canal_z, oto_z
    return y_canal, y_oto, sigma_oto


def visual_raw(cfg, x_true: np.ndarray, rng: np.random.Generator, eye_vel=None):
    """World/self retinal motion minus eye velocity, weak static nonlinearity."""
    A_true = np.array([[1.0, 0.9, 1.0]])
    y = (A_true @ x_true).flatten() + rng.standard_normal(cfg.T) * cfg.sigma_vis
    y += 0.15 * np.tanh(y)
    if eye_vel is not None:
        y = y - eye_vel
    return y


def lowpass_1st_order(signal: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """First-order low-pass emulating NOT/DTN wide-field integration."""
    out = np.zeros_like(signal)
    alpha = dt / (tau + dt)
    for t in range(1, len(signal)):
        out[t] = out[t - 1] + alpha * (signal[t] - out[t - 1])
    return out
