"""Core state-space inference with two-timescale cerebellar learning.

The latent state is x = [omega, a_lin, v_obj].  A linear-Gaussian Kalman
filter fuses canal, otolith and (forward-model-predicted) visual channels.
Gravity context is inferred online and gates a mixture of Earth/microgravity
forward models, each decomposed into fast and slow cerebellar components.
"""
from __future__ import annotations
import numpy as np

from .config import Config
from .context import ContextEstimator
from . import stimulus as stim
from .eyes import EyePlant


def _kalman_scalar_update(x, P, H, y, R):
    """Single scalar-measurement Kalman update. H is (1,3)."""
    S = H @ P @ H.T + R
    K = (P @ H.T) / S            # (3,1)
    innov = float(np.ravel(y - (H @ x))[0])   # scalar
    x = x + K.flatten() * innov
    P = (np.eye(3) - K @ H) @ P
    return x, P, innov


def initial_forward_models(cfg):
    """Return fast/slow components with an explicit initialization policy."""
    A0 = np.array([cfg.A0])
    if cfg.forward_initialization == "corrected":
        Af = np.zeros_like(A0)
    elif cfg.forward_initialization == "legacy":
        Af = np.array([cfg.A_fast0])
    else:
        raise ValueError(f"unknown forward_initialization: {cfg.forward_initialization}")
    return {"A_fast_e": Af.copy(), "A_fast_m": Af.copy(),
            "A_slow_e": A0.copy(), "A_slow_m": A0.copy()}


def assert_initial_forward_model(cfg, atol=1e-12):
    """Assert that both context models sum to declared A0 (corrected policy)."""
    parts = initial_forward_models(cfg)
    target = np.array([cfg.A0])
    for suffix in ("e", "m"):
        effective = parts[f"A_slow_{suffix}"] + parts[f"A_fast_{suffix}"]
        if not np.allclose(effective, target, atol=atol, rtol=0):
            raise AssertionError(f"initial {suffix} effective model {effective} != A0 {target}")
    return True


def simulate(cfg: Config, rng, object_motion, eye_vel=None, learn=True, init_A=None):
    """Run one inference pass.

    ``init_A`` optionally supplies pre-trained forward-model components
    (as returned in the output dict of a previous run); otherwise learning
    starts from the untrained defaults.  Returns traces and learned components.
    """
    u = stim.efference(cfg)
    x_true = stim.latent_selfmotion(cfg, u)
    x_true[2] = object_motion

    y_canal, y_oto, sigma_oto = stim.vestibular_obs(cfg, x_true, rng)
    y_vis = stim.visual_raw(cfg, x_true, rng, eye_vel=eye_vel)
    y_not = stim.lowpass_1st_order(y_vis, cfg.tau_not, cfg.dt)

    # forward-model components (context-specific fast + slow)
    if init_A is None:
        init_A = initial_forward_models(cfg)
    A_fast_e = init_A["A_fast_e"].copy()
    A_fast_m = init_A["A_fast_m"].copy()
    A_slow_e = init_A["A_slow_e"].copy()
    A_slow_m = init_A["A_slow_m"].copy()

    ctx = ContextEstimator(cfg)
    B = np.array(cfg.B)
    H_canal = np.array([[1.0, 0.0, 0.0]])
    H_oto = np.array([[0.0, 1.0, 0.0]])

    x_hat = np.zeros((3, cfg.T))
    P = np.eye(3) * 0.5
    sigma_o_est = cfg.sigma_oto_base
    p_hist = np.zeros(cfg.T)
    innov_o_hist = np.zeros(cfg.T)

    for t in range(cfg.T):
        # predict
        if t > 0:
            du = (u[t] - u[t - 1]) if cfg.consistent_efference else u[t]
            x_pred = x_hat[:, t - 1] + B * du
            P_pred = P + cfg.Q
        else:
            x_pred = np.zeros(3)
            P_pred = np.eye(3) * 0.5

        # canal update
        x_upd, P_upd, _ = _kalman_scalar_update(
            x_pred, P_pred, H_canal, y_canal[t], np.array([[cfg.sigma_canal ** 2]]))

        # otolith update (with reliability tracking)
        Saa = float(P_upd[1, 1])   # state part of otolith innovation variance
        x_upd, P_upd, innov_o = _kalman_scalar_update(
            x_upd, P_upd, H_oto, y_oto[t], np.array([[sigma_o_est ** 2]]))
        sigma_o_est += cfg.eta_sigma * (innov_o ** 2 - sigma_o_est ** 2)
        sigma_o_est = float(np.clip(sigma_o_est, cfg.sigma_o_min, cfg.sigma_o_max))
        innov_o_hist[t] = innov_o

        # context inference (gates forward models)
        p = ctx.update(innov_o, Saa)
        p_hist[t] = p

        # forward-model mixture
        A_earth = A_slow_e + A_fast_e
        A_micro = A_slow_m + A_fast_m
        A_mix = (1 - p) * A_earth + p * A_micro

        # visual update
        x_upd, P, innov_n = _kalman_scalar_update(
            x_upd, P_upd, A_mix, y_not[t], np.array([[cfg.sigma_not ** 2]]))
        x_hat[:, t] = x_upd

        # two-timescale cerebellar learning
        if learn:
            grad = -2.0 * innov_n * x_hat[:, t].reshape(1, 3)
            A_fast_e -= cfg.eta_fast * (1 - p) * grad
            A_fast_m -= cfg.eta_fast * p * grad
            A_slow_e -= cfg.eta_slow * (1 - p) * (A_slow_e - A_fast_e)
            A_slow_m -= cfg.eta_slow * p * (A_slow_m - A_fast_m)

    return {
        "x_hat": x_hat,
        "v_obj_hat": x_hat[2].copy(),
        "omega_hat": x_hat[0].copy(),
        "a_hat": x_hat[1].copy(),
        "x_true": x_true,
        "p_hist": p_hist,
        "innov_o": innov_o_hist,
        "y_not": y_not,
        "A_fast_e": A_fast_e, "A_fast_m": A_fast_m,
        "A_slow_e": A_slow_e, "A_slow_m": A_slow_m,
    }


def okr_pass(cfg: Config, y_not_base: np.ndarray, head_omega: np.ndarray):
    """Generate eye velocity and gain trace from a retinal-slip signal.

    Uses the EyePlant (VOR + OKR). With cfg.use_vor=False and head_omega
    ignored this reproduces the original OKR-only reflex.
    """
    plant = EyePlant(cfg)
    eye_vel = np.zeros(cfg.T)
    g_hist = np.zeros(cfg.T)
    g_hist[0] = plant.g_okr
    for t in range(1, cfg.T):
        eye_vel[t] = plant.step(y_not_base[t - 1], head_omega[t - 1])
        g_hist[t] = plant.g_okr
    return eye_vel, g_hist


def _otolith_variance(cfg, p_prior, true_sigma, sigma_adaptive):
    """Measurement variance available to the estimator at sample t.

    Only the explicitly labelled ``oracle`` policy reads ``true_sigma``.
    The corrected default uses the context belief from t-1, because p_t is not
    available until the current otolith innovation has been evaluated.
    """
    if cfg.oto_variance_policy == "belief_weighted":
        return ((1.0 - p_prior) * cfg.sigma_oto_earth_hyp ** 2
                + p_prior * cfg.sigma_oto_micro_hyp ** 2)
    if cfg.oto_variance_policy == "fixed":
        return cfg.fixed_oto_std ** 2
    if cfg.oto_variance_policy == "oracle":
        return float(true_sigma ** 2)
    if cfg.oto_variance_policy == "adaptive_legacy":
        return float(sigma_adaptive ** 2)
    raise ValueError(f"unknown oto_variance_policy: {cfg.oto_variance_policy}")


def simulate_closed_loop(cfg: Config, rng, object_motion, learn=True, init_A=None,
                         trial_id=0):
    """Appendix-compliant, genuinely coupled trial simulation.

    Causal update order for each sample ``t``:

    1. ``x_true[t]`` and the eye state produced at ``t-1`` generate retinal
       observation ``y_vis[t] = A_true*x_true[t] - eye[t] + noise[t]``.
    2. The state predicted from ``t-1`` receives canal and otolith updates at
       ``t``.  ``R_oto[t]`` uses context belief ``p[t-1]`` (or an explicit
       fixed/oracle control), then the innovation produces ``p[t]``.
    3. The visual prediction and innovation are evaluated at ``t`` using
       ``y_vis[t] + eye[t]`` when compensation is enabled.
    4. Forward models are updated from the visual innovation at ``t``.
    5. VOR/OKR commands evaluated at ``t`` advance the eye plant to ``t+1``.

    Thus no eye sample from another trial is available or reusable.
    """
    object_motion = np.asarray(object_motion, dtype=float)
    if object_motion.shape != (cfg.T,):
        raise ValueError(f"object_motion must have shape ({cfg.T},)")
    if cfg.forward_initialization == "corrected" and init_A is None:
        assert_initial_forward_model(cfg)

    u = stim.efference(cfg)
    x_true = stim.latent_selfmotion(cfg, u)
    x_true[2] = object_motion
    y_canal, y_oto, sigma_oto_true, canal_z, oto_z = stim.vestibular_obs(
        cfg, x_true, rng, return_standard_normals=True)
    visual_z = rng.standard_normal(cfg.T)
    visual_noise = visual_z * cfg.sigma_vis

    parts = initial_forward_models(cfg) if init_A is None else init_A
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

    trace_names = ("eye_vel", "y_vis_raw", "y_vis_comp", "y_not",
                   "predicted_vis", "innov_vis", "p_hist", "innov_o",
                   "R_oto", "vor_command", "okr_command", "g_okr",
                   "g_vor", "A_effective")
    tr = {name: np.zeros(cfg.T) for name in trace_names}

    for t in range(cfg.T):
        # Eye velocity at t was produced by commands at t-1.
        eye_t = 0.0 if cfg.force_zero_eye else plant.eye_vel
        tr["eye_vel"][t] = eye_t
        y_retinal = float(A_true @ x_true[:, t] - eye_t + visual_noise[t])
        tr["y_vis_raw"][t] = y_retinal
        not_state += not_alpha * (y_retinal - not_state)
        tr["y_not"][t] = not_state

        if t:
            du = (u[t] - u[t - 1]) if cfg.consistent_efference else u[t]
            x_pred = x_hat[:, t - 1] + B * du
            P_pred = P + cfg.Q
        else:
            x_pred = np.zeros(3)
            P_pred = np.eye(3) * 0.5

        x_upd, P_upd, _ = _kalman_scalar_update(
            x_pred, P_pred, H_canal, y_canal[t],
            np.array([[cfg.sigma_canal ** 2]]))

        p_prior = ctx.p
        R_oto_t = _otolith_variance(
            cfg, p_prior, sigma_oto_true[t], sigma_adaptive)
        tr["R_oto"][t] = R_oto_t
        Saa = float(P_upd[1, 1])
        x_upd, P_upd, innov_o = _kalman_scalar_update(
            x_upd, P_upd, H_oto, y_oto[t], np.array([[R_oto_t]]))
        tr["innov_o"][t] = innov_o
        if cfg.oto_variance_policy == "adaptive_legacy":
            sigma_adaptive += cfg.eta_sigma * (innov_o ** 2 - sigma_adaptive ** 2)
            sigma_adaptive = float(np.clip(
                sigma_adaptive, cfg.sigma_o_min, cfg.sigma_o_max))
        p_t = ctx.update(innov_o, Saa)
        tr["p_hist"][t] = p_t

        if cfg.context_gating:
            w_e, w_m = 1.0 - p_t, p_t
        else:
            w_e, w_m = 1.0, 0.0
        A_earth = A_slow_e + A_fast_e
        A_micro = A_slow_m + A_fast_m
        A_mix = w_e * A_earth + w_m * A_micro
        tr["A_effective"][t] = float(np.linalg.norm(A_mix))

        y_comp = y_retinal + eye_t if cfg.eye_compensation else y_retinal
        tr["y_vis_comp"][t] = y_comp
        tr["predicted_vis"][t] = float((A_mix @ x_upd).item())
        x_upd, P, innov_v = _kalman_scalar_update(
            x_upd, P_upd, A_mix, y_comp,
            np.array([[cfg.sigma_vis ** 2]]))
        x_hat[:, t] = x_upd
        tr["innov_vis"][t] = innov_v

        if learn:
            grad = -2.0 * innov_v * x_hat[:, t].reshape(1, 3)
            if cfg.forward_rule == "canonical_two_rate":
                if cfg.enable_fast:
                    A_fast_e = cfg.retention_fast * A_fast_e - cfg.eta_fast * w_e * grad
                    A_fast_m = cfg.retention_fast * A_fast_m - cfg.eta_fast * w_m * grad
                if cfg.enable_slow:
                    A_slow_e = A_anchor + cfg.retention_slow * (A_slow_e - A_anchor) - cfg.eta_slow * w_e * grad
                    A_slow_m = A_anchor + cfg.retention_slow * (A_slow_m - A_anchor) - cfg.eta_slow * w_m * grad
            elif cfg.forward_rule == "legacy_tracking":
                if cfg.enable_fast:
                    A_fast_e -= cfg.eta_fast * w_e * grad
                    A_fast_m -= cfg.eta_fast * w_m * grad
                if cfg.enable_slow:
                    A_slow_e -= cfg.eta_slow * w_e * (A_slow_e - A_fast_e)
                    A_slow_m -= cfg.eta_slow * w_m * (A_slow_m - A_fast_m)
            else:
                raise ValueError(f"unknown forward_rule: {cfg.forward_rule}")

        tr["g_okr"][t] = plant.g_okr
        tr["g_vor"][t] = plant.g_vor
        if t < cfg.T - 1 and not cfg.force_zero_eye:
            _, vor_t, okr_t = plant.advance(not_state, x_upd[0])
            tr["vor_command"][t] = vor_t
            tr["okr_command"][t] = okr_t

    return {
        "x_hat": x_hat, "v_obj_hat": x_hat[2].copy(),
        "omega_hat": x_hat[0].copy(), "a_hat": x_hat[1].copy(),
        "x_true": x_true, "object_motion": object_motion.copy(),
        "sigma_oto_true": sigma_oto_true,
        "y_canal": y_canal, "y_oto": y_oto,
        "canal_standard_normal": canal_z,
        "otolith_standard_normal": oto_z,
        "visual_standard_normal": visual_z,
        "A_fast_e": A_fast_e, "A_fast_m": A_fast_m,
        "A_slow_e": A_slow_e, "A_slow_m": A_slow_m,
        "trial_id": int(trial_id), "object_trial_id": int(trial_id),
        "eye_trial_id": int(trial_id), "oracle_variance": cfg.oto_variance_policy == "oracle",
        **tr,
    }
