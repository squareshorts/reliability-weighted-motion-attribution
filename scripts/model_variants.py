"""Structural model variants for extended reliability analyses.

The module mirrors the canonical observer update order while exposing additional
state needed for structural comparisons and time-resolved analyses.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cerebellum.config import Config  # noqa: E402
from cerebellum.context import ContextEstimator  # noqa: E402
from cerebellum.eyes import EyePlant  # noqa: E402
from cerebellum import model as M  # noqa: E402
from cerebellum import stimulus as stim  # noqa: E402


MODEL_DEFINITIONS = {
    "A_full_canonical": "Canonical three-state observer, two-rate adaptation, inferred context, and online eye plant.",
    "B_fixed_context": "Generative otolith noise changes, but p=0 and assumed otolith variance remains at the baseline hypothesis.",
    "C_no_adaptation": "Canonical observer with both forward-model adaptive processes frozen at their initial values.",
    "D_single_timescale": "One adaptive deviation state preserving the canonical aggregate instantaneous learning rate and its learning-rate-weighted one-step retention.",
    "E_residual_only": "Two-state self-motion observer with instantaneous visual residual used as an external-motion proxy; no recurrent object-motion state.",
    "F_oracle_self_motion": "Diagnostic observer supplied true self-motion at the visual stage, with near-zero self-state covariance, while retaining recurrent object inference.",
    "G_no_eye_feedback": "Canonical estimator with the eye plant forced to zero; exact compensation predicts numerical equivalence to Model A.",
}


def single_timescale_mapping(cfg: Config) -> dict[str, float | str]:
    """Map two canonical processes to one without outcome-based fitting.

    After a unit gradient impulse the canonical aggregate deviation is
    -(eta_fast + eta_slow). One sample later its retained value is
    -(eta_fast*rho_fast + eta_slow*rho_slow). The mapping preserves both.
    """
    eta = cfg.eta_fast + cfg.eta_slow
    retention = (
        cfg.eta_fast * cfg.retention_fast
        + cfg.eta_slow * cfg.retention_slow
    ) / eta
    return {
        "eta_single": float(eta),
        "retention_single": float(retention),
        "mapping_rule": (
            "eta_single=eta_fast+eta_slow; "
            "rho_single=(eta_fast*rho_fast+eta_slow*rho_slow)/eta_single"
        ),
    }


def generate_object_motion(cfg: Config, realization_id: int) -> tuple[np.random.Generator, np.ndarray]:
    """Create the canonical paired event/sensory stream for one realization."""
    rng = np.random.default_rng(100 + realization_id)
    object_motion = stim.random_object_events(cfg, rng)
    return rng, object_motion


def run_model(cfg: Config, realization_id: int, model_id: str) -> dict:
    """Run one requested structural model with canonical common random numbers."""
    run_cfg = replace(cfg, force_zero_eye=True) if model_id == "G_no_eye_feedback" else cfg
    rng, object_motion = generate_object_motion(run_cfg, realization_id)
    if model_id == "E_residual_only":
        return _simulate_residual_only(run_cfg, rng, object_motion, realization_id)
    if model_id not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model_id: {model_id}")
    return _simulate_three_state(run_cfg, rng, object_motion, realization_id, model_id)


def _simulate_three_state(
    cfg: Config,
    rng: np.random.Generator,
    object_motion: np.ndarray,
    realization_id: int,
    model_id: str,
) -> dict:
    """Instrumented three-state observer with minimal structural switches."""
    object_motion = np.asarray(object_motion, dtype=float)
    if object_motion.shape != (cfg.T,):
        raise ValueError(f"object_motion must have shape ({cfg.T},)")
    if cfg.forward_initialization == "corrected":
        M.assert_initial_forward_model(cfg)

    u = stim.efference(cfg)
    x_true = stim.latent_selfmotion(cfg, u)
    x_true[2] = object_motion
    y_canal, y_oto, sigma_oto_true, canal_z, oto_z = stim.vestibular_obs(
        cfg, x_true, rng, return_standard_normals=True
    )
    visual_z = rng.standard_normal(cfg.T)
    visual_noise = visual_z * cfg.sigma_vis

    parts = M.initial_forward_models(cfg)
    A_fast_e = parts["A_fast_e"].copy()
    A_fast_m = parts["A_fast_m"].copy()
    A_slow_e = parts["A_slow_e"].copy()
    A_slow_m = parts["A_slow_m"].copy()
    A_anchor = np.array([cfg.A0], dtype=float)

    mapping = single_timescale_mapping(cfg)
    D_single_e = np.zeros_like(A_anchor)
    D_single_m = np.zeros_like(A_anchor)

    ctx = ContextEstimator(cfg)
    B = np.asarray(cfg.B, dtype=float)
    H_canal = np.array([[1.0, 0.0, 0.0]])
    H_oto = np.array([[0.0, 1.0, 0.0]])
    A_true = np.asarray(cfg.A0, dtype=float)
    plant = EyePlant(cfg)

    x_hat = np.zeros((3, cfg.T))
    P = np.eye(3) * 0.5
    sigma_adaptive = cfg.sigma_oto_base
    not_state = 0.0
    not_alpha = cfg.dt / (cfg.tau_not + cfg.dt)

    scalars = (
        "eye_vel", "y_vis_raw", "y_vis_comp", "not_state",
        "predicted_vis", "innov_vis", "K_obj_vis", "delta_v_obj_vis",
        "p_hist", "context_log_odds", "innov_o", "R_oto",
        "assumed_sigma_oto", "vor_command", "okr_command", "g_okr", "g_vor",
        "P_pre_obj_obj", "P_pre_omega_obj", "P_pre_a_obj",
    )
    tr = {name: np.zeros(cfg.T, dtype=float) for name in scalars}
    vectors = (
        "x_pred", "x_previsual", "x_postvisual", "H_vis",
        "A_fast_effective", "A_slow_deviation_effective",
        "A_combined_deviation", "A_effective",
    )
    tr.update({name: np.zeros((3, cfg.T), dtype=float) for name in vectors})

    freeze_context = model_id == "B_fixed_context"
    learn = model_id not in ("C_no_adaptation",)
    single_rate = model_id == "D_single_timescale"
    oracle_self = model_id == "F_oracle_self_motion"

    for t in range(cfg.T):
        eye_t = 0.0 if cfg.force_zero_eye else plant.eye_vel
        tr["eye_vel"][t] = eye_t
        y_retinal = float(A_true @ x_true[:, t] - eye_t + visual_noise[t])
        tr["y_vis_raw"][t] = y_retinal
        not_state += not_alpha * (y_retinal - not_state)
        tr["not_state"][t] = not_state

        if t:
            du = (u[t] - u[t - 1]) if cfg.consistent_efference else u[t]
            x_pred = x_hat[:, t - 1] + B * du
            P_pred = P + cfg.Q
        else:
            x_pred = np.zeros(3)
            P_pred = np.eye(3) * 0.5
        tr["x_pred"][:, t] = x_pred

        x_upd, P_upd, _ = M._kalman_scalar_update(
            x_pred, P_pred, H_canal, y_canal[t],
            np.array([[cfg.sigma_canal ** 2]])
        )

        p_prior = 0.0 if freeze_context else ctx.p
        if freeze_context:
            R_oto_t = cfg.sigma_oto_earth_hyp ** 2
        else:
            R_oto_t = M._otolith_variance(
                cfg, p_prior, sigma_oto_true[t], sigma_adaptive
            )
        Saa = float(P_upd[1, 1])
        x_upd, P_upd, innov_o = M._kalman_scalar_update(
            x_upd, P_upd, H_oto, y_oto[t], np.array([[R_oto_t]])
        )
        if cfg.oto_variance_policy == "adaptive_legacy" and not freeze_context:
            sigma_adaptive += cfg.eta_sigma * (innov_o ** 2 - sigma_adaptive ** 2)
            sigma_adaptive = float(np.clip(
                sigma_adaptive, cfg.sigma_o_min, cfg.sigma_o_max
            ))
        if freeze_context:
            p_t = 0.0
            log_odds = -np.inf
        else:
            p_t = ctx.update(innov_o, Saa)
            log_odds = ctx.L

        tr["p_hist"][t] = p_t
        tr["context_log_odds"][t] = log_odds
        tr["innov_o"][t] = innov_o
        tr["R_oto"][t] = R_oto_t
        tr["assumed_sigma_oto"][t] = np.sqrt(R_oto_t)

        if cfg.context_gating and not freeze_context:
            w_e, w_m = 1.0 - p_t, p_t
        else:
            w_e, w_m = 1.0, 0.0

        if single_rate:
            A_earth = A_anchor + D_single_e
            A_micro = A_anchor + D_single_m
            fast_mix = w_e * D_single_e + w_m * D_single_m
            slow_dev_mix = np.zeros_like(A_anchor)
        else:
            A_earth = A_slow_e + A_fast_e
            A_micro = A_slow_m + A_fast_m
            fast_mix = w_e * A_fast_e + w_m * A_fast_m
            slow_dev_mix = (
                w_e * (A_slow_e - A_anchor)
                + w_m * (A_slow_m - A_anchor)
            )
        A_mix = w_e * A_earth + w_m * A_micro
        tr["A_fast_effective"][:, t] = fast_mix.ravel()
        tr["A_slow_deviation_effective"][:, t] = slow_dev_mix.ravel()
        tr["A_combined_deviation"][:, t] = (A_mix - A_anchor).ravel()
        tr["A_effective"][:, t] = A_mix.ravel()
        tr["H_vis"][:, t] = A_mix.ravel()

        if oracle_self:
            x_upd[:2] = x_true[:2, t]
            P_upd[:2, :] = 0.0
            P_upd[:, :2] = 0.0
            P_upd[0, 0] = 1e-12
            P_upd[1, 1] = 1e-12
        tr["x_previsual"][:, t] = x_upd

        y_comp = y_retinal + eye_t if cfg.eye_compensation else y_retinal
        predicted_vis = float((A_mix @ x_upd).item())
        innov_v = float(y_comp - predicted_vis)
        S_vis = A_mix @ P_upd @ A_mix.T + np.array([[cfg.sigma_vis ** 2]])
        K_vis = (P_upd @ A_mix.T) / S_vis
        delta = K_vis.flatten() * innov_v
        x_upd = x_upd + delta
        P = (np.eye(3) - K_vis @ A_mix) @ P_upd
        x_hat[:, t] = x_upd

        tr["y_vis_comp"][t] = y_comp
        tr["predicted_vis"][t] = predicted_vis
        tr["innov_vis"][t] = innov_v
        tr["K_obj_vis"][t] = K_vis[2, 0]
        tr["delta_v_obj_vis"][t] = delta[2]
        tr["P_pre_obj_obj"][t] = P_upd[2, 2]
        tr["P_pre_omega_obj"][t] = P_upd[0, 2]
        tr["P_pre_a_obj"][t] = P_upd[1, 2]
        tr["x_postvisual"][:, t] = x_upd

        if learn:
            # Canonical implementation uses the post-visual posterior here.
            grad = -2.0 * innov_v * x_hat[:, t].reshape(1, 3)
            if single_rate:
                rho = float(mapping["retention_single"])
                eta = float(mapping["eta_single"])
                D_single_e = rho * D_single_e - eta * w_e * grad
                D_single_m = rho * D_single_m - eta * w_m * grad
            else:
                if cfg.enable_fast:
                    A_fast_e = (
                        cfg.retention_fast * A_fast_e
                        - cfg.eta_fast * w_e * grad
                    )
                    A_fast_m = (
                        cfg.retention_fast * A_fast_m
                        - cfg.eta_fast * w_m * grad
                    )
                if cfg.enable_slow:
                    A_slow_e = (
                        A_anchor
                        + cfg.retention_slow * (A_slow_e - A_anchor)
                        - cfg.eta_slow * w_e * grad
                    )
                    A_slow_m = (
                        A_anchor
                        + cfg.retention_slow * (A_slow_m - A_anchor)
                        - cfg.eta_slow * w_m * grad
                    )

        tr["g_okr"][t] = plant.g_okr
        tr["g_vor"][t] = plant.g_vor
        if t < cfg.T - 1 and not cfg.force_zero_eye:
            _, vor_t, okr_t = plant.advance(not_state, x_upd[0])
            tr["vor_command"][t] = vor_t
            tr["okr_command"][t] = okr_t

    return {
        "model_id": model_id,
        "realization_id": int(realization_id),
        "rng_seed": 100 + int(realization_id),
        "x_hat": x_hat,
        "v_obj_hat": x_hat[2].copy(),
        "x_true": x_true,
        "object_motion": object_motion.copy(),
        "sigma_oto_true": sigma_oto_true,
        "y_canal": y_canal,
        "y_oto": y_oto,
        "canal_standard_normal": canal_z,
        "otolith_standard_normal": oto_z,
        "visual_standard_normal": visual_z,
        "single_timescale_mapping": mapping,
        **tr,
    }


def _simulate_residual_only(
    cfg: Config,
    rng: np.random.Generator,
    object_motion: np.ndarray,
    realization_id: int,
) -> dict:
    """Two-state self-motion observer plus instantaneous external residual.

    The external-motion proxy is the compensated visual observation remaining
    after the two-state pre-visual self-motion prediction, divided by the true
    object coefficient (1.0). It is not propagated to the next sample.
    """
    u = stim.efference(cfg)
    x_true3 = stim.latent_selfmotion(cfg, u)
    x_true3[2] = object_motion
    y_canal, y_oto, sigma_oto_true, canal_z, oto_z = stim.vestibular_obs(
        cfg, x_true3, rng, return_standard_normals=True
    )
    visual_z = rng.standard_normal(cfg.T)
    visual_noise = visual_z * cfg.sigma_vis

    A_anchor = np.asarray(cfg.A0[:2], dtype=float).reshape(1, 2)
    A_fast_e = np.zeros_like(A_anchor)
    A_fast_m = np.zeros_like(A_anchor)
    A_slow_e = A_anchor.copy()
    A_slow_m = A_anchor.copy()
    ctx = ContextEstimator(cfg)
    plant = EyePlant(cfg)
    B = np.asarray(cfg.B[:2], dtype=float)
    Q = cfg.Q[:2, :2]
    H_canal = np.array([[1.0, 0.0]])
    H_oto = np.array([[0.0, 1.0]])

    x_hat2 = np.zeros((2, cfg.T))
    residual_proxy = np.zeros(cfg.T)
    P = np.eye(2) * 0.5
    not_state = 0.0
    not_alpha = cfg.dt / (cfg.tau_not + cfg.dt)

    scalars = (
        "eye_vel", "y_vis_raw", "y_vis_comp", "not_state",
        "predicted_vis", "innov_vis", "p_hist", "context_log_odds",
        "innov_o", "R_oto", "assumed_sigma_oto", "g_okr", "g_vor",
    )
    tr = {name: np.zeros(cfg.T, dtype=float) for name in scalars}
    tr.update({
        "x_pred": np.zeros((3, cfg.T)),
        "x_previsual": np.zeros((3, cfg.T)),
        "x_postvisual": np.zeros((3, cfg.T)),
        "H_vis": np.zeros((3, cfg.T)),
        "A_fast_effective": np.zeros((3, cfg.T)),
        "A_slow_deviation_effective": np.zeros((3, cfg.T)),
        "A_combined_deviation": np.zeros((3, cfg.T)),
        "A_effective": np.zeros((3, cfg.T)),
        "K_obj_vis": np.full(cfg.T, np.nan),
        "delta_v_obj_vis": np.full(cfg.T, np.nan),
        "P_pre_obj_obj": np.full(cfg.T, np.nan),
        "P_pre_omega_obj": np.full(cfg.T, np.nan),
        "P_pre_a_obj": np.full(cfg.T, np.nan),
        "vor_command": np.zeros(cfg.T),
        "okr_command": np.zeros(cfg.T),
    })

    for t in range(cfg.T):
        eye_t = 0.0 if cfg.force_zero_eye else plant.eye_vel
        y_retinal = float(np.asarray(cfg.A0) @ x_true3[:, t] - eye_t + visual_noise[t])
        not_state += not_alpha * (y_retinal - not_state)
        tr["eye_vel"][t] = eye_t
        tr["y_vis_raw"][t] = y_retinal
        tr["not_state"][t] = not_state

        if t:
            du = (u[t] - u[t - 1]) if cfg.consistent_efference else u[t]
            x_pred = x_hat2[:, t - 1] + B * du
            P_pred = P + Q
        else:
            x_pred = np.zeros(2)
            P_pred = np.eye(2) * 0.5
        tr["x_pred"][:2, t] = x_pred

        x_upd, P_upd, _ = _kalman_scalar_update_nd(
            x_pred, P_pred, H_canal, y_canal[t], cfg.sigma_canal ** 2
        )
        p_prior = ctx.p
        R_oto_t = (
            (1.0 - p_prior) * cfg.sigma_oto_earth_hyp ** 2
            + p_prior * cfg.sigma_oto_micro_hyp ** 2
        )
        Saa = float(P_upd[1, 1])
        x_upd, P_upd, innov_o = _kalman_scalar_update_nd(
            x_upd, P_upd, H_oto, y_oto[t], R_oto_t
        )
        p_t = ctx.update(innov_o, Saa)
        w_e, w_m = (1.0 - p_t, p_t) if cfg.context_gating else (1.0, 0.0)
        A_earth = A_slow_e + A_fast_e
        A_micro = A_slow_m + A_fast_m
        A_mix = w_e * A_earth + w_m * A_micro
        fast_mix = w_e * A_fast_e + w_m * A_fast_m
        slow_dev = w_e * (A_slow_e - A_anchor) + w_m * (A_slow_m - A_anchor)

        y_comp = y_retinal + eye_t if cfg.eye_compensation else y_retinal
        predicted = float((A_mix @ x_upd).item())
        innov_v = y_comp - predicted
        residual_proxy[t] = innov_v / float(cfg.A0[2])

        x_post, P, _ = _kalman_scalar_update_nd(
            x_upd, P_upd, A_mix, y_comp, cfg.sigma_vis ** 2
        )
        x_hat2[:, t] = x_post
        grad = -2.0 * innov_v * x_post.reshape(1, 2)
        A_fast_e = cfg.retention_fast * A_fast_e - cfg.eta_fast * w_e * grad
        A_fast_m = cfg.retention_fast * A_fast_m - cfg.eta_fast * w_m * grad
        A_slow_e = (
            A_anchor + cfg.retention_slow * (A_slow_e - A_anchor)
            - cfg.eta_slow * w_e * grad
        )
        A_slow_m = (
            A_anchor + cfg.retention_slow * (A_slow_m - A_anchor)
            - cfg.eta_slow * w_m * grad
        )

        tr["x_previsual"][:2, t] = x_upd
        tr["x_postvisual"][:2, t] = x_post
        tr["H_vis"][:2, t] = A_mix.ravel()
        tr["A_fast_effective"][:2, t] = fast_mix.ravel()
        tr["A_slow_deviation_effective"][:2, t] = slow_dev.ravel()
        tr["A_combined_deviation"][:2, t] = (A_mix - A_anchor).ravel()
        tr["A_effective"][:2, t] = A_mix.ravel()
        tr["y_vis_comp"][t] = y_comp
        tr["predicted_vis"][t] = predicted
        tr["innov_vis"][t] = innov_v
        tr["p_hist"][t] = p_t
        tr["context_log_odds"][t] = ctx.L
        tr["innov_o"][t] = innov_o
        tr["R_oto"][t] = R_oto_t
        tr["assumed_sigma_oto"][t] = np.sqrt(R_oto_t)
        tr["g_okr"][t] = plant.g_okr
        tr["g_vor"][t] = plant.g_vor
        if t < cfg.T - 1 and not cfg.force_zero_eye:
            _, vor_t, okr_t = plant.advance(not_state, x_post[0])
            tr["vor_command"][t] = vor_t
            tr["okr_command"][t] = okr_t

    x_hat3 = np.vstack((x_hat2, residual_proxy))
    tr["x_previsual"][2] = residual_proxy
    tr["x_postvisual"][2] = residual_proxy
    return {
        "model_id": "E_residual_only",
        "realization_id": int(realization_id),
        "rng_seed": 100 + int(realization_id),
        "x_hat": x_hat3,
        "v_obj_hat": residual_proxy,
        "x_true": x_true3,
        "object_motion": np.asarray(object_motion).copy(),
        "sigma_oto_true": sigma_oto_true,
        "y_canal": y_canal,
        "y_oto": y_oto,
        "canal_standard_normal": canal_z,
        "otolith_standard_normal": oto_z,
        "visual_standard_normal": visual_z,
        "metric_homology": "instantaneous residual proxy; no recurrent object state",
        **tr,
    }


def _kalman_scalar_update_nd(x, P, H, y, variance):
    S = float((H @ P @ H.T).item() + variance)
    K = (P @ H.T) / S
    innov = float(y - (H @ x).item())
    x_new = x + K.ravel() * innov
    P_new = (np.eye(len(x)) - K @ H) @ P
    return x_new, P_new, innov


def canonical_core_run(cfg: Config, realization_id: int) -> dict:
    """Run the unmodified canonical function for the reproduction gate."""
    rng, object_motion = generate_object_motion(cfg, realization_id)
    return M.simulate_closed_loop(
        cfg, rng, object_motion, trial_id=realization_id
    )
